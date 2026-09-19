"""PBA-002 — segmentation error forensics from frozen artifacts (no retraining).

Per-case recomputation (deterministic, frozen SEG-001 state) on the 860
seg-test pairs + existing seg-val analysis: Dice distribution, worst/best,
per-class, empties, multi-component, FP/FN pixel regions, localization
failures, classification-confidence vs segmentation-quality join (via
PBA-001 test table). Output: outputs/PBA-002/. No morphology/threshold changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.unet import UNet  # noqa: E402

TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "PBA-002"
    out.mkdir(parents=True, exist_ok=True)
    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])
    ckpt = torch.load(root / "checkpoints" / "SEG-001" / "best.pt",
                      map_location="cpu", weights_only=False)  # own file
    model = UNet()
    model.load_state_dict(ckpt["state"])
    model.eval()
    cls_test = {c["path"]: c for c in
                json.loads((root / "outputs" / "PBA-001" / "per_case_test.json").read_text())}

    from PIL import Image
    release = root / "data" / "brisc2025"
    pairs = []
    for img_p in sorted((release / "segmentation_task" / "test" / "images").glob("*.jpg")):
        pairs.append((img_p, release / "segmentation_task" / "test" / "masks" / (img_p.stem + ".png")))
    assert len(pairs) == 860

    cases = []
    with torch.no_grad():
        for i, (img_p, m_p) in enumerate(pairs):
            img, gt = tf(Image.open(img_p).convert("L"), Image.open(m_p).convert("L"))
            prob = torch.sigmoid(model(img.unsqueeze(0)))[0, 0].numpy()
            g = gt[0].numpy()
            binp, bing = (prob > 0.5), (g > 0.5)
            inter = float((binp & bing).sum())
            dice = (2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6)
            fp = int((binp & ~bing).sum())
            fn = int((~binp & bing).sum())
            fg = int(bing.sum())
            ow, oh = Image.open(img_p).size
            loc, w = extract(prob, (oh, ow))
            cc = Path("data/brisc2025/classification_task/test")  # stem lookup below
            cls_match = next((c for p, c in cls_test.items() if Path(p).stem == img_p.stem), None)
            cases.append({"stem": img_p.stem, "class": TUMOR.get(img_p.stem.split("_")[3], "?"),
                          "dice": round(float(dice), 6),
                          "fp_pixels": fp, "fn_pixels": fn, "fg_pixels": fg,
                          "fp_rate": round(fp / max(fg, 1), 4), "fn_rate": round(fn / max(fg, 1), 4),
                          "empty_pred": loc.area_pixels == 0,
                          "multi": bool("multiple_components" in w),
                          "area": loc.area_pixels,
                          "cls_pred": cls_match["pred"] if cls_match else None,
                          "cls_conf": cls_match["confidence"] if cls_match else None,
                          "cls_certain": cls_match["certain"] if cls_match else None})
            if i % 200 == 0 and i:
                print(f"seg {i}/860", flush=True)
    _ = cc
    (out / "per_case_segtest.json").write_text(json.dumps(cases))

    d = np.array([c["dice"] for c in cases])
    conf_q = [(c["cls_conf"], c["dice"]) for c in cases if c["cls_conf"] is not None]
    rep = {"n": len(cases),
           "dice": {"mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                    "p10": round(float(np.percentile(d, 10)), 4), "min": round(float(d.min()), 4)},
           "by_class": {c: {"n": sum(1 for x in cases if x["class"] == c),
                            "mean": round(float(np.mean([x["dice"] for x in cases if x["class"] == c])), 4)}
                        for c in ("glioma", "meningioma", "pituitary")},
           "empty_pred": sum(1 for c in cases if c["empty_pred"]),
           "multi": sum(1 for c in cases if c["multi"]),
           "worst_10": sorted(cases, key=lambda c: c["dice"])[:10],
           "best_3": sorted(cases, key=lambda c: c["dice"])[-3:],
           "localization_failures": [c["stem"] for c in cases
                                     if c["empty_pred"] and c["cls_pred"] not in (None, "notumor") and c["cls_certain"]],
           "cls_conf_vs_dice_spearman": round(float(__import__("scipy").stats.spearmanr(
               [q[0] for q in conf_q], [q[1] for q in conf_q]).statistic), 4) if conf_q else None}
    (out / "pba002.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("worst_10", "best_3")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
