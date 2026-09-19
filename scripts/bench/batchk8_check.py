"""K=8 batching rung — sole change: 8 sequential forwards -> 1 batched forward.

Verifies on validation (N=1000): per-case agreement fractions and flags
identical between the frozen sequential reference (inline, pre-batching
implementation, identical seeds) and the batched service method.
Reconfirms untouched CLS/SEG paths (Macro-F1/Dice vs recorded FP32 reference),
determinism, peak memory, and latency with CORRECTED accounting:
  analyze_core = classify + segment (service.analyze)
  analyze_full = analyze_core + consistency (what /analyze returns)
Autocast FP16 both arms (candidate representation). No model/rule change.
Writes outputs/SYSINT/batchk8_check.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPEATS = 20
SEEDS = (101, 202, 303)
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
K, SEED_BASE, SIGMA = 8, 7003, 0.05


def _synthetic(seed: int, size: int = 512):
    from PIL import Image
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size) * 255).astype("uint8"))


def _stats(xs: list[float]) -> dict:
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def sequential_consistency(svc, base, k=K):
    """Frozen pre-batching reference implementation (identical seeds)."""
    from PIL import Image
    clean_pred = svc.classify(base)["predicted_class"]
    agree = 0
    with torch.no_grad():
        for kk in range(k):
            rng = np.random.RandomState(SEED_BASE + kk)
            a = np.asarray(base).astype(np.float32) / 255.0
            a = np.clip(a + rng.normal(0, SIGMA, a.shape), 0, 1)
            noisy = Image.fromarray((a * 255).astype(np.uint8))
            probs = torch.softmax(
                svc.clf(svc.cls_tf(noisy).unsqueeze(0).to(svc.device)) / svc.T, dim=1)[0]
            if CLASSES[int(probs.argmax())] == clean_pred:
                agree += 1
    return round(agree / k, 4), bool(agree / k < 1.0)


def main() -> int:
    from sklearn.metrics import f1_score
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "SYSINT"
    assert torch.cuda.is_available(), "no CUDA — GPU rung kernel required"

    from brain_tumor.data.brisc import SegDataset, load_manifest
    from brain_tumor.inference.service import InferenceService
    from brain_tumor.preprocessing.pipeline import build_seg_pair_transform
    from PIL import Image

    svc = InferenceService(
        classifier_ckpt=root / "checkpoints" / "CLS-001" / "best.pt",
        segmenter_ckpt=root / "checkpoints" / "SEG-001" / "best.pt",
        device="cuda")
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    val_rows = [r for r in manifest if r["population"] == "official_train_pool"
                and r["project_split"] == "val"]
    assert len(val_rows) == 1000
    ref = json.loads((out / "fp16_check.json").read_text())
    amp = torch.autocast(device_type="cuda", dtype=torch.float16)

    # ---- untouched-path reconfirmation (expect exact match) ----
    with amp, torch.no_grad():
        yp, yt = [], []
        for i, r in enumerate(val_rows):
            img = svc.cls_tf(Image.open(root / r["path"]).convert("RGB")).unsqueeze(0).to("cuda")
            yp.append(int(torch.softmax(svc.clf(img) / svc.T, dim=1)[0].argmax()))
            yt.append(CLASSES.index(r["class"]))
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])
    seg_pairs = SegDataset(root, manifest, "val").pairs
    with amp, torch.no_grad():
        dl = []
        for ip, mp in seg_pairs:
            ti, tm = seg_tf(Image.open(ip).convert("L"), Image.open(mp).convert("L"))
            pr = torch.sigmoid(svc.seg(ti.unsqueeze(0).to("cuda")))[0, 0].cpu().numpy()
            g = tm[0].numpy()
            binp, bing = pr > 0.5, g > 0.5
            inter = float((binp & bing).sum())
            dl.append((2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6))
    dice = float(np.mean(dl))

    # ---- per-case equivalence sequential vs batched, full val ----
    mism_agree, mism_flag, flags = 0, 0, 0
    with amp:
        for i, r in enumerate(val_rows):
            base = Image.open(root / r["path"]).convert("RGB")
            ra, fa = sequential_consistency(svc, base)
            rb = svc.consistency(base)
            mism_agree += (ra != rb["agreement_fraction"])
            mism_flag += (fa != rb["flagged"])
            flags += rb["flagged"]
            if (i + 1) % 250 == 0:
                print(f"equivalence {i + 1}/1000", flush=True)

    # ---- latency: sequential vs batched + corrected accounting ----
    images = [_synthetic(s) for s in SEEDS]
    with amp:
        for img in images:
            sequential_consistency(svc, img)
            svc.consistency(img)
            svc.analyze(img)
    torch.cuda.synchronize()
    lat: dict[str, list[float]] = {"consistency_seq": [], "consistency_batched": [],
                                   "analyze_core": [], "analyze_full": []}
    with amp:
        for _ in range(REPEATS):
            for img in images:
                t = time.perf_counter()
                sequential_consistency(svc, img)
                torch.cuda.synchronize()
                lat["consistency_seq"].append(time.perf_counter() - t)
                t = time.perf_counter()
                svc.consistency(img)
                torch.cuda.synchronize()
                lat["consistency_batched"].append(time.perf_counter() - t)
                t = time.perf_counter()
                svc.analyze(img)
                torch.cuda.synchronize()
                lat["analyze_core"].append(time.perf_counter() - t)
                t = time.perf_counter()
                svc.analyze(img)
                svc.consistency(img)
                torch.cuda.synchronize()
                lat["analyze_full"].append(time.perf_counter() - t)
    torch.cuda.reset_peak_memory_stats()
    with amp:
        for img in images:
            svc.analyze(img)
            svc.consistency(img)
    torch.cuda.synchronize()
    mem = {"peak_allocated_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
           "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    with amp:
        e1 = (svc.consistency(images[0])["agreement_fraction"],
              svc.consistency(images[0])["flagged"])
        e2 = (svc.consistency(images[0])["agreement_fraction"],
              svc.consistency(images[0])["flagged"])
    payload = {
        "untouched_paths": {"macro_f1": round(f1, 6),
                            "d_f1_vs_fp32ref": round(f1 - ref["fp32"]["macro_f1_val"], 6),
                            "mean_dice": round(dice, 6),
                            "d_dice_vs_fp32ref": round(dice - ref["fp32"]["mean_dice_segval"], 6)},
        "equivalence_val_N1000": {"agreement_mismatches": mism_agree,
                                  "flag_mismatches": mism_flag,
                                  "flagged_total": flags},
        "latency_s": {k: _stats(v) for k, v in lat.items()},
        "definitions": {"analyze_core": "classify+segment (service.analyze)",
                        "analyze_full": "analyze_core+consistency (what /analyze returns)"},
        "peak_gpu_memory": mem,
        "deterministic_repeat": bool(e1 == e2),
    }
    (out / "batchk8_check.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
