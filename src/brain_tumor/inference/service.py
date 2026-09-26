"""Single shared InferenceService — §19. Imported directly by FastAPI + Streamlit.

Production wiring (system-integration phase): frozen CLS-001 + SEG-001,
frozen calibration (T/tau1/tau2 from outputs/CLS-001/calibration_frozen.json),
frozen seg norm, threshold 0.5, extract() localization in original pixels.
Every result is validated against brain_tumor.contracts (FROZEN — this module
must satisfy the contract, never amend it).

UNC-001 consistency is exposed as a separate descriptive method/endpoint. It
does NOT alter classification_state or system_state: gating on consistency
would require a contract amendment (PI decision, deferred).
Service probe seeds are fixed (base 7003 + k); batch analyses use per-case
indexing — both documented, both deterministic.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_tumor.classification.models import build_classifier
from brain_tumor.contracts import (
    CANONICAL_CLASSES,
    BrainTumorResult,
    LocalizationResult,
    derive_system_state,
)
from brain_tumor.localization.extract import extract
from brain_tumor.preprocessing.pipeline import (
    build_cls_transform,
    build_seg_pair_transform,
)
from brain_tumor.segmentation.unet import UNet

CLASSES = CANONICAL_CLASSES
CONSISTENCY_K = 8
CONSISTENCY_SEED_BASE = 7003
CONSISTENCY_SIGMA = 0.05

# Request ID propagation (set by API middleware / Streamlit)
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get current request ID from context."""
    return _request_id_var.get()


def set_request_id(req_id: str | None = None) -> str:
    """Set request ID in context; generate if not provided. Returns the ID."""
    if req_id is None:
        req_id = uuid.uuid4().hex[:12]
    _request_id_var.set(req_id)
    return req_id


_log = logging.getLogger("brain_tumor.inference")


def _timed(label: str):
    """Decorator: emit JSON log with stage latency in ms and request ID."""

    def deco(fn):
        @wraps(fn)
        def wrap(self, *a, **kw):
            t0 = time.perf_counter()
            req_id = get_request_id()
            try:
                return fn(self, *a, **kw)
            finally:
                _log.info(
                    json.dumps(
                        {
                            "stage": label,
                            "ms": round((time.perf_counter() - t0) * 1000, 2),
                            "request_id": req_id,
                        }
                    )
                )

        return wrap

    return deco


def _usable_ckpt(p: Path | None) -> Path | None:
    """Return p only if it holds real weights.

    Fresh clones without `git lfs pull` contain 134-byte LFS pointer text at
    the checkpoint path; feeding that to torch.load raises UnpicklingError.
    Pointers degrade to unavailable (same as missing) with a warning, so the
    UI/API/tests report unavailable instead of crashing. Genuine load errors
    on real-sized files still raise (fail fast on corruption).
    """
    if p is None:
        return None
    fp = Path(p)
    if not fp.is_file() or fp.stat().st_size < 1024 * 1024:
        if fp.is_file():
            _log.warning("checkpoint %s looks like an LFS pointer; treating as unavailable", fp)
        return None
    with open(fp, "rb") as f:
        if f.read(32).startswith(b"version https://git-lfs.github.com/"):
            _log.warning("checkpoint %s is an LFS pointer; treating as unavailable", fp)
            return None
    return fp


class InferenceService:
    """Frozen-model inference. Offline-first. CPU default; device optional.

    Production representation: FP16 autocast is applied to every forward
    when device is CUDA (validated rung: val deltas ~0, deterministic).
    On CPU the context is a no-op and execution is FP32-identical.
    """

    def __init__(
        self,
        classifier_ckpt: Path | None = None,
        segmenter_ckpt: Path | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.classifier_ckpt = classifier_ckpt
        self.segmenter_ckpt = segmenter_ckpt
        root = Path(__file__).resolve().parents[3]
        cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
        self.T, self.tau1, self.tau2 = cal["temperature"], cal["tau1"], cal["tau2"]
        norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
        self.seg_norm = norm

        self.clf = None
        if _usable_ckpt(classifier_ckpt) is not None:
            clf = build_classifier("convnext_tiny")
            # weights_only=True: no pickle execution surface (own frozen file,
            # state_dict-only content — verified loadable under the restriction).
            clf.load_state_dict(
                torch.load(classifier_ckpt, map_location=device, weights_only=True)["state"]
            )
            clf.eval()
            clf.to(device)
            self.clf = clf
        self.cls_tf = build_cls_transform(False)

        self.seg = None
        if _usable_ckpt(segmenter_ckpt) is not None:
            seg = UNet()
            seg.load_state_dict(
                torch.load(segmenter_ckpt, map_location=device, weights_only=True)["state"]
            )
            seg.eval()
            seg.to(device)
            self.seg = seg
        self.seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])

    @property
    def segmentation_available(self) -> bool:
        return self.seg is not None

    @classmethod
    def from_registry(cls, project_root: Path | None = None) -> InferenceService:
        root = project_root or Path(__file__).resolve().parents[3]
        clf = root / "checkpoints" / "CLS-001" / "best.pt"
        seg = root / "checkpoints" / "SEG-001" / "best.pt"
        return cls(
            classifier_ckpt=clf if clf.exists() else None,
            segmenter_ckpt=seg if seg.exists() else None,
        )

    def _autocast(self):
        """FP16 on CUDA (frozen deployment representation); no-op on CPU."""
        import contextlib

        if self.device != "cpu" and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # ---- primitives ----

    @_timed("classify")
    def classify(self, image: Any) -> dict:
        if self.clf is None:
            raise RuntimeError("classifier_unavailable")
        from PIL import Image as _Image

        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        with torch.no_grad(), self._autocast():
            probs = torch.softmax(
                self.clf(self.cls_tf(pil.convert("RGB")).unsqueeze(0).to(self.device)) / self.T,
                dim=1,
            )[0]
        top2 = probs.topk(2).values
        conf, margin = float(top2[0]), float(top2[0] - top2[1])
        pred = CLASSES[int(probs.argmax())]
        return {
            "predicted_class": pred,
            "probabilities": {c: round(float(p), 6) for c, p in zip(CLASSES, probs, strict=True)},
            "confidence": conf,
            "classification_state": "confident"
            if (conf >= self.tau1 and margin >= self.tau2)
            else "uncertain",
        }

    def _segment_prob(self, pil_gray) -> tuple[np.ndarray, int, int]:
        ow, oh = pil_gray.size
        with torch.no_grad(), self._autocast():
            img_t = self.seg_tf(pil_gray, pil_gray)[0]
            prob = torch.sigmoid(self.seg(img_t.unsqueeze(0).to(self.device)))[0, 0].cpu().numpy()
        return prob, ow, oh

    @_timed("segment")
    def segment(self, image: Any = None) -> dict:
        if not self.segmentation_available:
            # Frozen contract (endpoints.md /segment): the degraded branch still
            # carries localization + warnings, exactly like analyze()'s path.
            return {
                "segmentation_state": "unavailable",
                "localization": LocalizationResult(bbox=None, centroid=None, area_pixels=0),
                "warnings": ["segmentation_unavailable"],
            }
        from PIL import Image as _Image

        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        prob, ow, oh = self._segment_prob(pil.convert("L"))
        loc, warnings = extract(prob, (oh, ow))
        state = "empty" if loc.area_pixels == 0 else "nonempty"
        return {
            "segmentation_state": state,
            "localization": LocalizationResult(
                bbox=loc.bbox, centroid=loc.centroid, area_pixels=int(loc.area_pixels)
            ),
            "warnings": list(warnings),
        }

    def localize(self, image: Any = None) -> LocalizationResult:
        if self.segmentation_available:
            return self.segment(image)["localization"]
        return LocalizationResult(bbox=None, centroid=None, area_pixels=0)

    @_timed("consistency")
    def consistency(
        self, image: Any = None, k: int = CONSISTENCY_K, clean_pred: str | None = None
    ) -> dict:
        """UNC-001 frozen rule, descriptive only. Fixed seeds (base + k).

        Deployment rung BENCH-004 tested a batched single-forward execution:
        numerically identical (0 mismatches, N=1000) but SLOWER on T4
        (0.258s vs 0.159s sequential), so the sequential loop is retained.
        Same seeds, same rule either way.

        clean_pred: optional precomputed clean prediction. When supplied the
        internal classify() call is skipped (callers that already ran
        analyze/classify thread it through to avoid a redundant forward);
        when None, behavior is identical to the original implementation.
        """
        if self.clf is None:
            raise RuntimeError("classifier_unavailable")
        import numpy as _np
        from PIL import Image as _Image

        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        base = pil.convert("RGB")
        if clean_pred is None:
            clean_pred = self.classify(base)["predicted_class"]
        agree = 0
        with torch.no_grad(), self._autocast():
            for kk in range(k):
                rng = _np.random.RandomState(CONSISTENCY_SEED_BASE + kk)
                a = _np.asarray(base).astype(_np.float32) / 255.0
                a = _np.clip(a + rng.normal(0, CONSISTENCY_SIGMA, a.shape), 0, 1)
                noisy = _Image.fromarray((a * 255).astype(_np.uint8))
                probs = torch.softmax(
                    self.clf(self.cls_tf(noisy).unsqueeze(0).to(self.device)) / self.T, dim=1
                )[0]
                if CLASSES[int(probs.argmax())] == clean_pred:
                    agree += 1
        frac = agree / k
        return {
            "k": k,
            "sigma": CONSISTENCY_SIGMA,
            "clean_pred": clean_pred,
            "agreement_fraction": round(frac, 4),
            "flagged": bool(frac < 1.0),
        }

    def quality(self, data: bytes) -> dict:
        """Descriptive input-quality pre-check (Workstream 1). Never alters
        inference outputs or system_state; SB-1 behavior unchanged."""
        from brain_tumor.quality.gate import assess

        return assess(data)

    def reliability(
        self,
        image: Any = None,
        raw: bytes | None = None,
        result: BrainTumorResult | None = None,
        con: dict | None = None,
        qual: dict | None = None,
    ) -> dict:
        """REL-001 descriptive fusion (observer only).

        Runs the identical analyze()/consistency()/quality() paths and fuses
        their already-computed outputs via reliability_report(). Cannot alter
        predictions, probabilities, segmentation, localization, or
        system_state — disabling it leaves every SB-1 output bit-identical.

        result/con/qual: optional precomputed outputs of analyze(),
        consistency(), and quality(). Any argument left as None is computed
        internally exactly as before, so the default call path is unchanged;
        threaded arguments only skip redundant recomputation (bit-identical
        outputs — determinism covered by tests).
        """
        from brain_tumor.reliability.engine import reliability_report

        if result is None:
            result = self.analyze(image)
        payload = result.model_dump()
        if con is None:
            con = self.consistency(image)
        if qual is None:
            qual = self.quality(raw) if raw is not None else None
        return reliability_report(
            classification={
                "predicted_class": payload["predicted_class"],
                "confidence": payload["confidence"],
                "classification_state": payload["classification_state"],
            },
            consistency=con,
            quality=qual,
            segmentation_state=payload["segmentation_state"],
            localization=payload["localization"],
            system_state=payload["system_state"],
        )

    def explain(
        self,
        image: Any = None,
        raw: bytes | None = None,
        result: BrainTumorResult | None = None,
        con: dict | None = None,
    ) -> dict:
        """EXPL-001 unified explanation (observer only).

        Runs the identical analyze() path (outputs/system_state provably
        unchanged), then attaches descriptive explanation: UNC-001
        consistency, Grad-CAM heatmap + overlay, segmentation overlay,
        quality (when raw bytes supplied), and the text report.
        Explanation forwards run FP32 on all devices (fidelity choice,
        documented in EXPL-001 evidence).

        result/con: optional precomputed analyze()/consistency() outputs
        (API-symmetric with reliability(); None => computed internally,
        default path unchanged).
        """
        import base64
        import io as _io

        from PIL import Image as _Image
        from PIL import ImageDraw as _Draw

        from brain_tumor.explain.gradcam import (
            cam_mass_in_bbox,
            gradcam_heatmap,
            overlay,
            resolve_target_layer,
            upsample_cam,
        )
        from brain_tumor.explain.report import build_report

        if result is None:
            result = self.analyze(image)
        payload = result.model_dump()
        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        base = pil.convert("RGB")
        gray = pil.convert("L")
        ow, oh = pil.size
        payload["consistency"] = con if con is not None else self.consistency(base)
        if raw is not None:
            payload["quality"] = self.quality(raw)

        def _png(arr) -> str:
            buf = _io.BytesIO()
            _Image.fromarray(arr).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        # Grad-CAM observer (FP32, hooks removed after use inside helper).
        x = self.cls_tf(base).unsqueeze(0).to(self.device)
        with torch.enable_grad():
            cam_small, target_name = gradcam_heatmap(
                self.clf,
                resolve_target_layer(self.clf),
                x,
                CLASSES.index(payload["predicted_class"]),
            )
        cam = upsample_cam(cam_small, (ow, oh))
        payload["gradcam"] = {
            "target_layer": target_name,
            "heatmap_png_b64": _png((cam * 255).astype("uint8")),
            "overlay_png_b64": _png(overlay(np.asarray(gray), cam)),
            "stats": {"max": round(float(cam.max()), 4), "mean": round(float(cam.mean()), 4)},
        }

        # Segmentation visualization (mirrors segment() exactly).
        if self.segmentation_available and payload["predicted_class"] != "notumor":
            prob, _, _ = self._segment_prob(gray)
            mask = (prob > 0.5).astype("uint8") * 255
            mask_full = np.asarray(_Image.fromarray(mask).resize((ow, oh), _Image.NEAREST))
            vis = base.copy()
            d = _Draw.Draw(vis, "RGBA")
            if payload["localization"]["bbox"] is not None:
                d.rectangle(payload["localization"]["bbox"], outline=(255, 0, 0), width=3)
                cx, cy = payload["localization"]["centroid"]
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 255, 0))
            red = np.zeros((oh, ow, 3), dtype=np.float32)
            red[..., 0] = mask_full
            blend = (
                (np.asarray(vis).astype(np.float32) * 0.65 + red * 0.35)
                .clip(0, 255)
                .astype("uint8")
            )
            payload["segmentation_vis"] = {
                "mask_png_b64": _png(mask_full),
                "overlay_png_b64": _png(blend),
                "focus_in_bbox": cam_mass_in_bbox(cam, payload["localization"]["bbox"]),
            }
        else:
            payload["segmentation_vis"] = None
        return build_report(payload)

    @_timed("analyze")
    def analyze(self, image: Any = None, consistency_probes: bool = True) -> BrainTumorResult:
        c = self.classify(image)
        warnings: list[str] = []
        if c["classification_state"] == "uncertain":
            warnings.append("low_confidence")
        if c["predicted_class"] == "notumor":
            seg_state, loc = "empty", LocalizationResult(bbox=None, centroid=None, area_pixels=0)
        elif not self.segmentation_available:
            seg_state, loc = (
                "unavailable",
                LocalizationResult(bbox=None, centroid=None, area_pixels=0),
            )
            warnings.append("segmentation_unavailable")
        else:
            s = self.segment(image)
            seg_state, loc = s["segmentation_state"], s["localization"]
            if seg_state == "empty":
                warnings.append("segmentation_empty")
            allowed = {
                "segmentation_empty",
                "low_confidence",
                "multiple_components",
                "oversized_component_filtered",
                "segmentation_unavailable",
            }
            warnings.extend(w for w in s["warnings"] if w in allowed)
        system_state = derive_system_state(
            c["predicted_class"], c["classification_state"], seg_state, warnings
        )
        return BrainTumorResult(
            predicted_class=c["predicted_class"],
            probabilities=c["probabilities"],
            confidence=c["confidence"],
            classification_state=c["classification_state"],
            segmentation_state=seg_state,
            localization=loc,
            warnings=warnings,
            system_state=system_state,
        )

    @_timed("analyze_batch")
    def analyze_batch(
        self, images: list[Any], consistency_probes: bool = True
    ) -> list[BrainTumorResult]:
        """Batch analyze multiple images with tensor-level batching.

        This provides true GPU batching by stacking preprocessed tensors,
        running inference once per stage on the full batch, then unpacking results.
        """
        if not images:
            return []

        from PIL import Image as _Image

        # Preprocess all images for classification
        cls_tensors = []
        pil_images = []
        for img in images:
            pil = img if isinstance(img, _Image.Image) else _Image.open(img)
            pil_images.append(pil.convert("RGB"))
            cls_tensors.append(self.cls_tf(pil.convert("RGB")))

        # Batch classification
        cls_batch = torch.stack(cls_tensors).to(self.device)
        with torch.no_grad(), self._autocast():
            logits = self.clf(cls_batch) / self.T
            probs = torch.softmax(logits, dim=1)
            top2 = probs.topk(2, dim=1).values
            confs = top2[:, 0].cpu().numpy()
            margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            pred_indices = probs.argmax(dim=1).cpu().numpy()

        results = []
        for i, pil in enumerate(pil_images):
            pred = CLASSES[int(pred_indices[i])]
            conf = float(confs[i])
            margin = float(margins[i])
            cls_state = "confident" if (conf >= self.tau1 and margin >= self.tau2) else "uncertain"
            prob_dict = {c: round(float(p), 6) for c, p in zip(CLASSES, probs[i].cpu().numpy())}
            c_result = {
                "predicted_class": pred,
                "probabilities": prob_dict,
                "confidence": conf,
                "classification_state": cls_state,
            }

            warnings: list[str] = []
            if cls_state == "uncertain":
                warnings.append("low_confidence")

            if pred == "notumor":
                seg_state, loc = "empty", LocalizationResult(bbox=None, centroid=None, area_pixels=0)
            elif not self.segmentation_available:
                seg_state, loc = "unavailable", LocalizationResult(bbox=None, centroid=None, area_pixels=0)
                warnings.append("segmentation_unavailable")
            else:
                # Segment this image (could be batched too for optimization)
                s = self.segment(pil)
                seg_state, loc = s["segmentation_state"], s["localization"]
                if seg_state == "empty":
                    warnings.append("segmentation_empty")
                allowed = {
                    "segmentation_empty",
                    "low_confidence",
                    "multiple_components",
                    "oversized_component_filtered",
                    "segmentation_unavailable",
                }
                warnings.extend(w for w in s["warnings"] if w in allowed)

            system_state = derive_system_state(pred, cls_state, seg_state, warnings)
            results.append(
                BrainTumorResult(
                    predicted_class=pred,
                    probabilities=prob_dict,
                    confidence=conf,
                    classification_state=cls_state,
                    segmentation_state=seg_state,
                    localization=loc,
                    warnings=warnings,
                    system_state=system_state,
                )
            )

        return results
