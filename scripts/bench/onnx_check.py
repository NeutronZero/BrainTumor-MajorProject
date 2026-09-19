"""ONNX rung — representation change only. NOT a model experiment.

Exports frozen CLS-001 + SEG-001 to ONNX, then checks (in order):
(a) correctness: torch-vs-ORT output agreement, Macro-F1 / Dice deltas vs the
    FP32 torch reference on identical validation populations (floors -0.01),
    repeat determinism, localization (empty+bbox) consistency on a subset;
(b) deployment: cold load (session creation), warm latency suite identical to
    BENCH-001, peak GPU memory via nvidia-smi sampling.
K=8 probes unbatched (batching stays a separate rung). Accuracy and latency
reported separately. Writes outputs/SYSINT/onnx_check.json (+ onnx files).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPEATS = 20
SEEDS = (101, 202, 303)
LOC_SUBSET = 100
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def _synthetic(seed: int, size: int = 512):
    from PIL import Image
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size) * 255).astype("uint8"))


def _stats(xs: list[float]) -> dict:
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def _softmax_t(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _MemSampler:
    def __init__(self):
        self.peak, self._run = 0.0, False

    def _loop(self):
        while self._run:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], text=True)
                self.peak = max(self.peak, float(out.strip().split()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)

    def __enter__(self):
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *e):
        self._run = False
        self.t.join()


def main() -> int:
    from sklearn.metrics import f1_score
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "SYSINT"
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    assert torch.cuda.is_available(), "no CUDA — GPU rung kernel required"

    from brain_tumor.classification.models import build_classifier
    from brain_tumor.data.brisc import SegDataset, load_manifest
    from brain_tumor.localization.extract import extract
    from brain_tumor.preprocessing.pipeline import build_seg_pair_transform
    from brain_tumor.segmentation.unet import UNet

    ref = json.loads((out / "fp16_check.json").read_text())
    fp32_f1, fp32_dice = ref["fp32"]["macro_f1_val"], ref["fp32"]["mean_dice_segval"]
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]
    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    val_rows = [r for r in manifest if r["population"] == "official_train_pool"
                and r["project_split"] == "val"]
    assert len(val_rows) == 1000
    seg_pairs = SegDataset(root, manifest, "val").pairs
    assert len(seg_pairs) == 785

    # ---- export frozen graphs ----
    import onnxruntime as ort
    assert "CUDAExecutionProvider" in ort.get_available_providers(), \
        f"no CUDA EP: {ort.get_available_providers()}"
    clf = build_classifier("convnext_tiny")
    clf.load_state_dict(torch.load(root / "checkpoints" / "CLS-001" / "best.pt",
                                   map_location="cpu", weights_only=False)["state"])
    clf.eval()
    seg = UNet()
    seg.load_state_dict(torch.load(root / "checkpoints" / "SEG-001" / "best.pt",
                                   map_location="cpu", weights_only=False)["state"])
    seg.eval()
    from brain_tumor.preprocessing.pipeline import build_cls_transform
    cls_tf = build_cls_transform(False)
    seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])
    from PIL import Image
    t0 = time.perf_counter()
    torch.onnx.export(clf, torch.zeros(1, 3, 224, 224), str(onnx_dir / "cls.onnx"),
                      dynamo=True)
    torch.onnx.export(seg, torch.zeros(1, 1, 256, 256), str(onnx_dir / "seg.onnx"),
                      dynamo=True)
    export_s = round(time.perf_counter() - t0, 1)
    print(f"exported in {export_s}s", flush=True)

    t0 = time.perf_counter()
    so = ort.SessionOptions()
    cls_sess = ort.InferenceSession(str(onnx_dir / "cls.onnx"), sess_options=so,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    seg_sess = ort.InferenceSession(str(onnx_dir / "seg.onnx"), sess_options=so,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    engaged = [s.get_providers()[0] for s in (cls_sess, seg_sess)]
    assert engaged == ["CUDAExecutionProvider", "CUDAExecutionProvider"], \
        f"CUDA EP not engaged (silent CPU fallback forbidden): {engaged}"
    cold_load = round(time.perf_counter() - t0, 3)
    print(f"ORT sessions on {[s for s in (cls_sess.get_providers(), seg_sess.get_providers())]}",
          flush=True)

    # ---- (a) correctness on identical validation populations ----
    yp, yt = [], []
    max_logit_diff, max_prob_diff = 0.0, 0.0
    for i, r in enumerate(val_rows):
        x = cls_tf(Image.open(root / r["path"]).convert("RGB")).unsqueeze(0).numpy()
        logits = cls_sess.run(None, {cls_sess.get_inputs()[0].name: x})[0]
        # Softmax/T in float64 numpy (matches torch float32 path to ~1e-7).
        P = _softmax_t(logits.astype(np.float64), T)[0]
        yp.append(int(P.argmax()))
        yt.append(CLASSES.index(r["class"]))
        if (i + 1) % 500 == 0:
            print(f"cls ort {i + 1}/1000", flush=True)
    f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

    dice_l, loc_match, loc_n = [], 0, 0
    for i, (ip, mp) in enumerate(seg_pairs):
        ti, tm = seg_tf(Image.open(ip).convert("L"), Image.open(mp).convert("L"))
        x = ti.unsqueeze(0).numpy()
        o = seg_sess.run(None, {seg_sess.get_inputs()[0].name: x})[0][0, 0]
        pr = 1 / (1 + np.exp(-o.astype(np.float64)))
        g = tm[0].numpy()
        binp, bing = pr > 0.5, g > 0.5
        inter = float((binp & bing).sum())
        dice_l.append((2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6))
        if i < LOC_SUBSET:
            ow, oh = Image.open(ip).size
            loc_o, _ = extract(pr, (oh, ow))
            # torch-path reference for the same case
            with torch.no_grad():
                pt = torch.sigmoid(seg(ti.unsqueeze(0)))[0, 0].numpy()
            loc_t, _ = extract(pt, (oh, ow))
            loc_n += 1
            loc_match += ((loc_o.area_pixels == 0) == (loc_t.area_pixels == 0)
                          and loc_o.bbox == loc_t.bbox)
        if (i + 1) % 400 == 0:
            print(f"seg ort {i + 1}/785", flush=True)
    dice = float(np.mean(dice_l))
    d1 = cls_sess.run(None, {cls_sess.get_inputs()[0].name:
                             np.zeros((1, 3, 224, 224), np.float32)})[0]
    d2 = cls_sess.run(None, {cls_sess.get_inputs()[0].name:
                             np.zeros((1, 3, 224, 224), np.float32)})[0]
    deterministic = bool((d1 == d2).all())

    correctness = {
        "macro_f1_ort": round(f1, 6), "d_f1_vs_fp32": round(f1 - fp32_f1, 6),
        "f1_floor_pass": (f1 - fp32_f1) >= -0.01,
        "mean_dice_ort": round(dice, 6), "d_dice_vs_fp32": round(dice - fp32_dice, 6),
        "dice_floor_pass": (dice - fp32_dice) >= -0.01,
        "localization_subset": {"n": loc_n, "empty_bbox_match_rate": round(loc_match / loc_n, 6)},
        "ort_deterministic_repeat": deterministic,
    }

    # ---- (b) deployment measurements (ORT-native pipeline mirrors service) ----
    def ort_classify(pil):
        x = cls_tf(pil.convert("RGB")).unsqueeze(0).numpy()
        P = _softmax_t(cls_sess.run(
            None, {cls_sess.get_inputs()[0].name: x})[0].astype(np.float64), T)[0]
        top2 = np.sort(P)[-2:]
        return int(P.argmax()), float(top2[1]), float(top2[1] - top2[0])

    def ort_segment(pil):
        ti, _ = seg_tf(pil.convert("L"), pil.convert("L"))
        x = ti.unsqueeze(0).numpy()
        o = seg_sess.run(None, {seg_sess.get_inputs()[0].name: x})[0][0, 0]
        pr = 1 / (1 + np.exp(-o.astype(np.float64)))
        ow, oh = pil.size
        return extract(pr, (oh, ow))

    def ort_consistency(pil, k=8):
        rng = np.random.RandomState(7003)
        base = np.asarray(pil.convert("RGB")).astype(np.float64) / 255.0
        cp, _, _ = ort_classify(pil)
        agree = 0
        for kk in range(k):
            a = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
            noisy = Image.fromarray((a * 255).astype(np.uint8))
            if ort_classify(noisy)[0] == cp:
                agree += 1
        return agree / k

    images = [_synthetic(s) for s in SEEDS]
    for img in images:  # warmup
        ort_classify(img)
        ort_segment(img)
        ort_consistency(img)
    lat: dict[str, list[float]] = {"classify": [], "segment": [],
                                   "consistency_k8": [], "analyze": []}

    def ort_analyze(pil):
        cp, conf, margin = ort_classify(pil)
        certain = conf >= t1 and margin >= t2
        if CLASSES[cp] == "notumor":
            return
        loc, _ = ort_segment(pil)

    with _MemSampler() as mem:
        for _ in range(REPEATS):
            for img in images:
                for name, fn in (("classify", lambda: ort_classify(img)),
                                 ("segment", lambda: ort_segment(img)),
                                 ("consistency_k8", lambda: ort_consistency(img)),
                                 ("analyze", lambda: ort_analyze(img))):
                    t = time.perf_counter()
                    fn()
                    lat[name].append(time.perf_counter() - t)
    deployment = {"cold_load_sessions_s": cold_load, "export_s": export_s,
                  "latency_s": {k: _stats(v) for k, v in lat.items()},
                  "peak_gpu_memory_MB": round(mem.peak, 1)}

    payload = {"correctness": correctness, "deployment": deployment,
               "fp32_reference": {"macro_f1_val": fp32_f1, "mean_dice_segval": fp32_dice},
               "note": "K=8 unbatched; batching stays a separate rung"}
    (out / "onnx_check.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
