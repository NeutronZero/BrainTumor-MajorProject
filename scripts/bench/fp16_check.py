"""FP16 rung — precision conversion only. NOT a model experiment.

Compares FP32 vs FP16 (torch autocast; weights untouched, rule/thresholds/
calibration/contract identical) on: (a) validation accuracy floors —
CLS macro-F1 on project val N=1000, SEG mean Dice on seg-val N=785;
(b) the BENCH-001 deployment latency suite on synthetic inputs.
Accuracy degradation and latency improvement reported as separate quantities.
Floors: Macro-F1 delta >= -0.01, Dice delta >= -0.01.
Writes outputs/SYSINT/fp16_check.json.
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


def _synthetic(seed: int, size: int = 512):
    from PIL import Image

    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size) * 255).astype("uint8"))


def _stats(xs: list[float]) -> dict:
    a = np.array(xs)
    return {
        "n": len(xs),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
    }


def main() -> int:
    from sklearn.metrics import f1_score

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "SYSINT"
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "no CUDA — GPU rung kernel required"
    device = "cuda"

    from brain_tumor.data.brisc import SegDataset, load_manifest
    from brain_tumor.inference.service import InferenceService
    from brain_tumor.preprocessing.pipeline import build_seg_pair_transform

    svc = InferenceService(
        classifier_ckpt=root / "checkpoints" / "CLS-001" / "best.pt",
        segmenter_ckpt=root / "checkpoints" / "SEG-001" / "best.pt",
        device=device,
    )
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    val_rows = [
        r
        for r in manifest
        if r["population"] == "official_train_pool" and r["project_split"] == "val"
    ]
    assert len(val_rows) == 1000, f"val membership changed: {len(val_rows)}"
    seg_pairs = SegDataset(root, manifest, "val").pairs
    assert len(seg_pairs) == 785, f"seg-val membership changed: {len(seg_pairs)}"
    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])
    cls_tf = svc.cls_tf
    T = svc.T

    from PIL import Image

    results: dict = {}

    # ---- (a) validation accuracy, both precisions, same data ----
    for tag, amp in (("fp32", False), ("fp16", True)):
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else torch.no_grad()
        with ctx, torch.no_grad():
            yp, yt = [], []
            for i, r in enumerate(val_rows):
                img = cls_tf(Image.open(root / r["path"]).convert("RGB")).unsqueeze(0).to(device)
                p = torch.softmax(svc.clf(img) / T, dim=1)[0]
                yp.append(int(p.argmax()))
                yt.append(CLASSES.index(r["class"]))
                if (i + 1) % 500 == 0:
                    print(f"cls {tag} {i + 1}/1000", flush=True)
            f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
            dice_l = []
            for i, (ip, mp) in enumerate(seg_pairs):
                ti, tm = seg_tf(Image.open(ip).convert("L"), Image.open(mp).convert("L"))
                pr = torch.sigmoid(svc.seg(ti.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                g = tm[0].numpy()
                binp, bing = pr > 0.5, g > 0.5
                inter = float((binp & bing).sum())
                dice_l.append((2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6))
                if (i + 1) % 400 == 0:
                    print(f"seg {tag} {i + 1}/785", flush=True)
            results[tag] = {
                "macro_f1_val": round(f1, 6),
                "mean_dice_segval": round(float(np.mean(dice_l)), 6),
            }
    results["delta"] = {
        "macro_f1": round(results["fp16"]["macro_f1_val"] - results["fp32"]["macro_f1_val"], 6),
        "mean_dice": round(
            results["fp16"]["mean_dice_segval"] - results["fp32"]["mean_dice_segval"], 6
        ),
    }
    results["floors"] = {
        "macro_f1_pass": results["delta"]["macro_f1"] >= -0.01,
        "mean_dice_pass": results["delta"]["mean_dice"] >= -0.01,
    }

    # ---- (b) latency suite under autocast FP16 ----
    images = [_synthetic(s) for s in SEEDS]
    for img in images:  # warmup
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            svc.analyze(img)
            svc.consistency(img)
    torch.cuda.synchronize()
    lat: dict[str, list[float]] = {
        "classify": [],
        "segment": [],
        "consistency_k8": [],
        "analyze": [],
    }
    for _ in range(REPEATS):
        for img in images:
            for name, fn in (
                ("classify", lambda: svc.classify(img)),
                ("segment", lambda: svc.segment(img)),
                ("consistency_k8", lambda: svc.consistency(img)),
                ("analyze", lambda: svc.analyze(img)),
            ):
                t = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    fn()
                torch.cuda.synchronize()
                lat[name].append(time.perf_counter() - t)
    # Autocast context covers the service forwards executed inside it (no_grad
    # and autocast compose); weights stay FP32 — compute runs in FP16 where
    # kernels support it. No model, rule, or threshold change.
    results["latency_fp16"] = {k: _stats(v) for k, v in lat.items()}

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        d1 = svc.classify(images[0])["probabilities"]
        d2 = svc.classify(images[0])["probabilities"]
    results["fp16_deterministic_repeat"] = bool(all(abs(d1[c] - d2[c]) < 1e-6 for c in CLASSES))

    torch.cuda.reset_peak_memory_stats()
    for img in images:
        svc.analyze(img)
    torch.cuda.synchronize()
    results["peak_gpu_memory_fp32_weights"] = {
        "peak_allocated_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }

    (out / "fp16_check.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
