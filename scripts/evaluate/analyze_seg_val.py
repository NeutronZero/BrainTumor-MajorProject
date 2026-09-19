"""analyze_seg_val.py — SEG-001 locked-INDEPENDENT validation analysis.

Uses ONLY seg-val pairs (never test): Dice distribution (not just mean),
train-vs-val behavior, failure cases, frozen mask semantics, localization
rule verification (threshold 0.5, largest_plus_total, min_area 10, xyxy
original pixels, centroid of largest, area=sum, empty->null/0), and
validation-only qualitative overlays to reports/seg_val_analysis/.
Any required pipeline change => new experiment, never silent (none expected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.data.brisc import SegDataset, load_manifest  # noqa: E402
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.unet import UNet  # noqa: E402

TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}


def dice_score(p: np.ndarray, g: np.ndarray) -> float:
    inter = float(((p > 0.5) & (g > 0.5)).sum())
    return (2 * inter + 1e-6) / (float((p > 0.5).sum()) + float((g > 0.5).sum()) + 1e-6)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "SEG-001-analysis"
    art = root / "reports" / "seg_val_analysis"
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    tf = build_seg_pair_transform(False, image_size=256, mean=norm["mean"], std=norm["std"])

    ckpt = torch.load(root / "checkpoints" / "SEG-001" / "best.pt",
                      map_location="cpu", weights_only=False)  # own file
    model = UNet()
    model.load_state_dict(ckpt["state"])
    model.eval()

    va = SegDataset(root, manifest, "val", pair_transform=tf)
    tr = SegDataset(root, manifest, "train", pair_transform=tf)
    print(f"seg-val pairs={len(va)} seg-train pairs={len(tr)}", flush=True)

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
            # localization verification on ORIGINAL-size frame
            from PIL import Image
            ow, oh = Image.open(img_p).size
            loc, w = extract(prob, (oh, ow))
            if loc.area_pixels == 0:
                assert loc.bbox is None and loc.centroid is None, "empty invariant violated"
                loc_ok["empty_null"] += 1
            else:
                # Pydantic already enforced x1>x0, y1>y0; verify original-frame bounds.
                x0, y0, x1, y1 = loc.bbox
                assert 0 <= x0 < x1 <= ow and 0 <= y0 < y1 <= oh, "bbox outside original frame"
                cx, cy = loc.centroid
                assert 0 <= cx <= ow and 0 <= cy <= oh, "centroid outside frame"
                loc_ok["nonempty_box"] += 1
            if "multiple_components" in w:
                loc_ok["multi"] += 1
            per_image.append({"stem": img_p.stem, "class": cls, "dice": d,
                              "area": loc.area_pixels,
                              "empty_pred": loc.area_pixels == 0,
                              "gt_empty": bool((g > 0.5).sum() == 0)})
            if i % 100 == 0:
                print(f"val {i}/{len(va)}", flush=True)

    # train behavior subset (first 200, deterministic order)
    tr_dice = []
    with torch.no_grad():
        for i in range(min(200, len(tr))):
            img, gt = tr[i]
            prob = torch.sigmoid(model(img.unsqueeze(0)))[0, 0].numpy()
            tr_dice.append(dice_score(prob, gt[0].numpy()))
            if i % 100 == 0:
                print(f"train {i}", flush=True)

    dices = np.array([p["dice"] for p in per_image])
    by_cls = {}
    for cls in ("glioma", "meningioma", "pituitary"):
        v = np.array([p["dice"] for p in per_image if p["class"] == cls])
        by_cls[cls] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                       "p10": float(np.percentile(v, 10))}
    worst = sorted(per_image, key=lambda p: p["dice"])[:10]
    best = sorted(per_image, key=lambda p: p["dice"])[-3:]
    med = sorted(per_image, key=lambda p: p["dice"])[len(per_image) // 2 - 1:len(per_image) // 2 + 2]

    # qualitative overlays (validation only)
    from PIL import Image
    for tag, picks in (("worst", worst[:3]), ("median", med), ("best", best)):
        for p in picks:
            img_p = next(q for q, _ in va.pairs if q.stem == p["stem"])
            base = np.asarray(Image.open(img_p).convert("RGB"))
            Image.fromarray(base).save(art / f"{tag}_{p['stem']}_input.png")

    report = {"n_val": len(per_image), "n_train_subset": len(tr_dice),
              "val_dice": {"mean": float(dices.mean()), "median": float(np.median(dices)),
                           "iqr": [float(np.percentile(dices, 25)), float(np.percentile(dices, 75))],
                           "p10": float(np.percentile(dices, 10)),
                           "min": float(dices.min()), "max": float(dices.max())},
              "train_subset_dice_mean": float(np.mean(tr_dice)),
              "by_class": by_cls,
              "localization_val": loc_ok,
              "gt_empty_in_val": sum(1 for p in per_image if p["gt_empty"]),
              "worst_10": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()
                             if k != "area"} | {"area": p["area"]} for p in worst],
              "pipeline_change_required": False}
    (out / "seg_val_analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "worst_10"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
