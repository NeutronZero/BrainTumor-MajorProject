"""analyze_seg002_val.py — GEN-001 validation analysis (ONE seed per invocation).

Clone of analyze_seg_val.py reading checkpoints/GEN-001/seed<SEED>/best.pt.
Seg-val ONLY (never test). Replicates the SEG-001 per-image protocol exactly
(same dice fn, class-from-stem rule, localization invariants, threshold 0.5)
and ADDS: parameter count, §L1 latency probe (runs only where CUDA exists;
records cuda_required=true otherwise — authoritative latency comes from the
T4 kernel), peak-memory probe where measurable.

Usage: python scripts/gen001/analyze_seg002_val.py --seed 42
Writes outputs/GEN-001/seed<SEED>/val_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.data.brisc import SegDataset, load_manifest  # noqa: E402
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.attention_unet import AttentionUNet, count_parameters  # noqa: E402

TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}
L1_SYNTH_SEEDS = (101, 202, 303)
L1_REPEATS = 20


def dice_score(p: np.ndarray, g: np.ndarray) -> float:
    inter = float(((p > 0.5) & (g > 0.5)).sum())
    return (2 * inter + 1e-6) / (float((p > 0.5).sum()) + float((g > 0.5).sum()) + 1e-6)


def _synthetic(seed: int, size: int = 512):
    from PIL import Image
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size) * 255).astype("uint8"))


def l1_latency(model, seg_tf, device: str) -> dict:
    """§L1 protocol replica: eval seg_tf + forward + sigmoid + extract(),
    autocast FP16, cuda synchronize per sample, 3 synthetic seeds x 20 reps."""
    from PIL import Image
    model.eval()
    images = [_synthetic(s) for s in L1_SYNTH_SEEDS]
    with torch.no_grad():
        for img in images:  # warmup
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t = seg_tf(img.convert("L"), img.convert("L"))[0]
                prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                extract(prob, img.size[::-1])
        torch.cuda.synchronize()
        xs = []
        for _ in range(L1_REPEATS):
            for img in images:
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    t = seg_tf(img.convert("L"), img.convert("L"))[0]
                    prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                    extract(prob, img.size[::-1])
                torch.cuda.synchronize()
                xs.append(time.perf_counter() - t0)
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    tag = f"seed{args.seed}"
    outd = root / "outputs" / "GEN-001" / tag
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    run_metrics = json.loads((outd / "metrics.json").read_text())
    assert run_metrics["seed"] == args.seed, "metrics/seed mismatch"
    norm = run_metrics["norm"]
    tf = build_seg_pair_transform(False, image_size=256, mean=norm["mean"], std=norm["std"])

    ckpt = torch.load(root / "checkpoints" / "GEN-001" / tag / "best.pt",
                      map_location="cpu", weights_only=False)  # own file
    model = AttentionUNet()
    model.load_state_dict(ckpt["state"])
    model.eval()

    va = SegDataset(root, manifest, "val", pair_transform=tf)
    assert len(va) == 785, f"seg-val membership changed: {len(va)}"
    print(f"GEN-001 {tag}: seg-val pairs={len(va)}", flush=True)

    per_image = []
    loc_ok = {"empty_null": 0, "nonempty_box": 0, "multi": 0}
    with torch.no_grad():
        for i in range(len(va)):
            img_p, _ = va.pairs[i]
            img, gt = va[i]
            prob = torch.sigmoid(model(img.unsqueeze(0)))[0, 0].numpy()
            g = gt[0].numpy()
            d = dice_score(prob, g)
            cls = TUMOR.get(img_p.stem.split("_")[3], "?")
            from PIL import Image
            ow, oh = Image.open(img_p).size
            loc, w = extract(prob, (oh, ow))
            if loc.area_pixels == 0:
                assert loc.bbox is None and loc.centroid is None, "empty invariant violated"
                loc_ok["empty_null"] += 1
            else:
                x0, y0, x1, y1 = loc.bbox
                assert 0 <= x0 < x1 <= ow and 0 <= y0 < y1 <= oh, "bbox outside frame"
                cx, cy = loc.centroid
                assert 0 <= cx <= ow and 0 <= cy <= oh, "centroid outside frame"
                loc_ok["nonempty_box"] += 1
            if "multiple_components" in w:
                loc_ok["multi"] += 1
            per_image.append({"stem": img_p.stem, "class": cls, "dice": d,
                              "empty_pred": loc.area_pixels == 0,
                              "gt_empty": bool((g > 0.5).sum() == 0)})
            if (i + 1) % 200 == 0:
                print(f"val {i + 1}/{len(va)}", flush=True)

    dices = np.array([p["dice"] for p in per_image])
    by_cls = {}
    for cls in ("glioma", "meningioma", "pituitary"):
        v = np.array([p["dice"] for p in per_image if p["class"] == cls])
        by_cls[cls] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                       "p10": float(np.percentile(v, 10))}

    device = run_metrics.get("device", "cpu")
    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
        lat = l1_latency(model, tf, "cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for img in [_synthetic(s) for s in L1_SYNTH_SEEDS]:
                t = tf(img.convert("L"), img.convert("L"))[0]
                torch.sigmoid(model(t.unsqueeze(0).to("cuda")))
        torch.cuda.synchronize()
        mem = {"peak_allocated_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
               "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
        cuda = {"available": True}
    else:
        lat, mem, cuda = None, None, {"available": False,
                                      "note": "authoritative §L1 runs on T4 kernel"}

    report = {"experiment": "GEN-001", "seed": args.seed, "n_val": len(per_image),
              "params": count_parameters(model),
              "val_global_train_metric": run_metrics["val_dice_global"],
              "val_dice": {"mean": float(dices.mean()), "median": float(np.median(dices)),
                           "p10": float(np.percentile(dices, 10)),
                           "min": float(dices.min()), "max": float(dices.max())},
              "by_class": by_cls, "localization_val": loc_ok,
              "gt_empty_in_val": sum(1 for p in per_image if p["gt_empty"]),
              "l1_latency": lat, "peak_memory": mem, "cuda": cuda}
    (outd / "val_analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
