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

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_tumor.classification.models import build_classifier
from brain_tumor.contracts import (
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

CLASSES = ("glioma", "meningioma", "pituitary", "notumor")
CONSISTENCY_K = 8
CONSISTENCY_SEED_BASE = 7003
CONSISTENCY_SIGMA = 0.05


class InferenceService:
    """Frozen-model inference. Offline-first. CPU default; device optional.

    Production representation: FP16 autocast is applied to every forward
    when device is CUDA (validated rung: val deltas ~0, deterministic).
    On CPU the context is a no-op and execution is FP32-identical.
    """

    def __init__(self, classifier_ckpt: Path | None = None,
                 segmenter_ckpt: Path | None = None,
                 device: str = "cpu"):
        self.device = device
        self.classifier_ckpt = classifier_ckpt
        self.segmenter_ckpt = segmenter_ckpt
        root = Path(__file__).resolve().parents[3]
        cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
        self.T, self.tau1, self.tau2 = cal["temperature"], cal["tau1"], cal["tau2"]
        norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
        self.seg_norm = norm

        self.clf = None
        if classifier_ckpt is not None and Path(classifier_ckpt).exists():
            clf = build_classifier("convnext_tiny")
            clf.load_state_dict(torch.load(classifier_ckpt, map_location=device,
                                           weights_only=False)["state"])
            clf.eval()
            clf.to(device)
            self.clf = clf
        self.cls_tf = build_cls_transform(False)

        self.seg = None
        if segmenter_ckpt is not None and Path(segmenter_ckpt).exists():
            seg = UNet()
            seg.load_state_dict(torch.load(segmenter_ckpt, map_location=device,
                                           weights_only=False)["state"])
            seg.eval()
            seg.to(device)
            self.seg = seg
        self.seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])

    @property
    def segmentation_available(self) -> bool:
        return self.seg is not None

    @classmethod
    def from_registry(cls, project_root: Path | None = None) -> "InferenceService":
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

    def classify(self, image: Any) -> dict:
        if self.clf is None:
            raise RuntimeError("classifier_unavailable")
        from PIL import Image as _Image
        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        with torch.no_grad(), self._autocast():
            probs = torch.softmax(
                self.clf(self.cls_tf(pil.convert("RGB")).unsqueeze(0).to(self.device))
                / self.T, dim=1)[0]
        top2 = probs.topk(2).values
        conf, margin = float(top2[0]), float(top2[0] - top2[1])
        pred = CLASSES[int(probs.argmax())]
        return {
            "predicted_class": pred,
            "probabilities": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
            "confidence": conf,
            "classification_state": "confident"
            if (conf >= self.tau1 and margin >= self.tau2) else "uncertain",
        }

    def _segment_prob(self, pil_gray) -> tuple[np.ndarray, int, int]:
        from PIL import Image as _Image
        ow, oh = pil_gray.size
        with torch.no_grad(), self._autocast():
            img_t = self.seg_tf(pil_gray, pil_gray)[0]
            prob = torch.sigmoid(
                self.seg(img_t.unsqueeze(0).to(self.device)))[0, 0].cpu().numpy()
        return prob, ow, oh

    def segment(self, image: Any = None) -> dict:
        if not self.segmentation_available:
            return {"segmentation_state": "unavailable"}
        from PIL import Image as _Image
        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        prob, ow, oh = self._segment_prob(pil.convert("L"))
        loc, warnings = extract(prob, (oh, ow))
        state = "empty" if loc.area_pixels == 0 else "nonempty"
        return {"segmentation_state": state,
                "localization": LocalizationResult(
                    bbox=loc.bbox, centroid=loc.centroid,
                    area_pixels=int(loc.area_pixels)),
                "warnings": list(warnings)}

    def localize(self, image: Any = None) -> LocalizationResult:
        if self.segmentation_available:
            return self.segment(image)["localization"]
        return LocalizationResult(bbox=None, centroid=None, area_pixels=0)

    def consistency(self, image: Any = None, k: int = CONSISTENCY_K) -> dict:
        """UNC-001 frozen rule, descriptive only. Fixed seeds (base + k).

        Deployment rung BENCH-004 tested a batched single-forward execution:
        numerically identical (0 mismatches, N=1000) but SLOWER on T4
        (0.258s vs 0.159s sequential), so the sequential loop is retained.
        Same seeds, same rule either way.
        """
        if self.clf is None:
            raise RuntimeError("classifier_unavailable")
        from PIL import Image as _Image
        import numpy as _np
        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        base = pil.convert("RGB")
        clean_pred = self.classify(base)["predicted_class"]
        agree = 0
        with torch.no_grad(), self._autocast():
            for kk in range(k):
                rng = _np.random.RandomState(CONSISTENCY_SEED_BASE + kk)
                a = _np.asarray(base).astype(_np.float32) / 255.0
                a = _np.clip(a + rng.normal(0, CONSISTENCY_SIGMA, a.shape), 0, 1)
                noisy = _Image.fromarray((a * 255).astype(_np.uint8))
                probs = torch.softmax(
                    self.clf(self.cls_tf(noisy).unsqueeze(0).to(self.device))
                    / self.T, dim=1)[0]
                if CLASSES[int(probs.argmax())] == clean_pred:
                    agree += 1
        frac = agree / k
        return {"k": k, "sigma": CONSISTENCY_SIGMA, "clean_pred": clean_pred,
                "agreement_fraction": round(frac, 4), "flagged": bool(frac < 1.0)}

    def quality(self, data: bytes) -> dict:
        """Descriptive input-quality pre-check (Workstream 1). Never alters
        inference outputs or system_state; SB-1 behavior unchanged."""
        from brain_tumor.quality.gate import assess
        return assess(data)

    def explain(self, image: Any = None, raw: bytes | None = None) -> dict:
        """EXPL-001 unified explanation (observer only).

        Runs the identical analyze() path (outputs/system_state provably
        unchanged), then attaches descriptive explanation: UNC-001
        consistency, Grad-CAM heatmap + overlay, segmentation overlay,
        quality (when raw bytes supplied), and the text report.
        Explanation forwards run FP32 on all devices (fidelity choice,
        documented in EXPL-001 evidence).
        """
        import base64
        import io as _io
        from PIL import Image as _Image, ImageDraw as _Draw
        from brain_tumor.explain.gradcam import (
            cam_mass_in_bbox, gradcam_heatmap, overlay, resolve_target_layer,
            upsample_cam)
        from brain_tumor.explain.report import build_report

        result = self.analyze(image)
        payload = result.model_dump()
        pil = image if isinstance(image, _Image.Image) else _Image.open(image)
        base = pil.convert("RGB")
        gray = pil.convert("L")
        ow, oh = pil.size
        payload["consistency"] = self.consistency(base)
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
                self.clf, resolve_target_layer(self.clf), x,
                CLASSES.index(payload["predicted_class"]))
        cam = upsample_cam(cam_small, (ow, oh))
        payload["gradcam"] = {
            "target_layer": target_name,
            "heatmap_png_b64": _png((cam * 255).astype("uint8")),
            "overlay_png_b64": _png(overlay(np.asarray(gray), cam)),
            "stats": {"max": round(float(cam.max()), 4),
                      "mean": round(float(cam.mean()), 4)},
        }

        # Segmentation visualization (mirrors segment() exactly).
        if self.segmentation_available and payload["predicted_class"] != "notumor":
            prob, _, _ = self._segment_prob(gray)
            mask = (prob > 0.5).astype("uint8") * 255
            mask_full = np.asarray(
                _Image.fromarray(mask).resize((ow, oh), _Image.NEAREST))
            vis = base.copy()
            d = _Draw.Draw(vis, "RGBA")
            if payload["localization"]["bbox"] is not None:
                d.rectangle(payload["localization"]["bbox"], outline=(255, 0, 0), width=3)
                cx, cy = payload["localization"]["centroid"]
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 255, 0))
            red = np.zeros((oh, ow, 3), dtype=np.float32)
            red[..., 0] = mask_full
            blend = (np.asarray(vis).astype(np.float32) * 0.65
                     + red * 0.35).clip(0, 255).astype("uint8")
            payload["segmentation_vis"] = {
                "mask_png_b64": _png(mask_full),
                "overlay_png_b64": _png(blend),
                "focus_in_bbox": cam_mass_in_bbox(
                    cam, payload["localization"]["bbox"]),
            }
        else:
            payload["segmentation_vis"] = None
        return build_report(payload)

    def analyze(self, image: Any = None, consistency_probes: bool = True) -> BrainTumorResult:
        c = self.classify(image)
        warnings: list[str] = []
        if c["classification_state"] == "uncertain":
            warnings.append("low_confidence")
        if c["predicted_class"] == "notumor":
            seg_state, loc = "empty", LocalizationResult(
                bbox=None, centroid=None, area_pixels=0)
        elif not self.segmentation_available:
            seg_state, loc = "unavailable", LocalizationResult(
                bbox=None, centroid=None, area_pixels=0)
            warnings.append("segmentation_unavailable")
        else:
            s = self.segment(image)
            seg_state, loc = s["segmentation_state"], s["localization"]
            if seg_state == "empty":
                warnings.append("segmentation_empty")
            allowed = {"segmentation_empty", "low_confidence", "multiple_components",
                       "oversized_component_filtered", "segmentation_unavailable"}
            warnings.extend(w for w in s["warnings"] if w in allowed)
        system_state = derive_system_state(
            c["predicted_class"], c["classification_state"], seg_state, warnings)
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
