"""ROB-001 validation-only 2x2 analysis (post-T4 return leg).

Arms (same frozen seg-val population, same frozen pipeline):
  SEG-001 x clean | SEG-001 x sigma=0.05
  ROB-001 x clean | ROB-001 x sigma=0.05

Noise uses the EXACT PBA-003 implementation (apply_perturb, sigma=0.05,
per-case RNG keyed by position in the evaluated population with the same
SEED=7003 scheme). Eval-only: no grad, no tuning, no feedback into ROB-001.
Output: outputs/ROB-001/val_2x2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pba"))
from brain_tumor.data.brisc import SegDataset, load_manifest  # noqa: E402
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.unet import UNet  # noqa: E402
from pba003_robustness import SEED, apply_perturb  # noqa: E402

BATCH = 16
TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}
NOISE = {"id": "gauss_noise_s0p05", "params": {"sigma": 0.05, "seed": SEED}}


def run_arm(model, pairs, tf, perturb: bool):
    from PIL import Image
    dice_l, iou_l, by_cls, empty_n, multi_n = [], [], {}, 0, 0
    worst = []
    model.eval()
    with torch.no_grad():
        ib, gb, meta = [], [], []

        def flush():
            nonlocal empty_n, multi_n
            if not ib:
                return
            prob = torch.sigmoid(model(torch.stack(ib))).numpy()[:, 0]
            for pr, g, (stem, ow, oh) in zip(prob, gb, meta):
                binp, bing = pr > 0.5, g > 0.5
                inter = float((binp & bing).sum())
                dice = (2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6)
                iou = (inter + 1e-6) / (float((binp | bing).sum()) + 1e-6)
                dice_l.append(dice)
                iou_l.append(iou)
                by_cls.setdefault(TUMOR.get(stem.split("_")[3], "?"), []).append(dice)
                loc, w = extract(pr, (oh, ow))
                e = loc.area_pixels == 0
                m = "multiple_components" in w
                empty_n += e
                multi_n += m
                worst.append({"stem": stem, "dice": round(dice, 6), "empty": e, "multi": m})

        for idx, (ip, mp) in enumerate(pairs):
            img = Image.open(ip).convert("L")
            if perturb:
                img = apply_perturb(img, NOISE, idx)
            ti, tm = tf(img, Image.open(mp).convert("L"))
            ow, oh = Image.open(ip).size
            ib.append(ti)
            gb.append(tm[0].numpy() > 0.5)
            meta.append((ip.stem, ow, oh))
            if len(ib) == BATCH or idx == len(pairs) - 1:
                flush()
                ib, gb, meta = [], [], []
            if (idx + 1) % 400 == 0:
                print(f"arm progress {idx + 1}/{len(pairs)}", flush=True)
    d = np.array(dice_l)
    return {"n": len(pairs),
            "mean_dice": round(float(d.mean()), 6),
            "median_dice": round(float(np.median(d)), 6),
            "p10_dice": round(float(np.percentile(d, 10)), 6),
            "min_dice": round(float(d.min()), 6),
            "mean_iou": round(float(np.mean(iou_l)), 6),
            "empty_pred": empty_n, "multi_component": multi_n,
            "by_class_mean_dice": {c: round(float(np.mean(v)), 6) for c, v in by_cls.items()},
            "worst_10": sorted(worst, key=lambda r: r["dice"])[:10]}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    ds = SegDataset(root, manifest, "val")
    pairs = ds.pairs
    print(f"seg-val pairs: {len(pairs)}", flush=True)

    out = root / "outputs" / "ROB-001"
    n_seg = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    n_rob = json.loads((out / "metrics.json").read_text())["norm"]
    assert abs(n_seg["mean"] - n_rob["mean"]) < 1e-9 and abs(n_seg["std"] - n_rob["std"]) < 1e-9, \
        "norm mismatch — populations/pipelines not comparable"

    def load(exp: str):
        m = UNet()
        m.load_state_dict(torch.load(root / "checkpoints" / exp / "best.pt",
                                     map_location="cpu", weights_only=False)["state"])
        m.eval()
        return m

    arms = {}
    for exp in ("SEG-001", "ROB-001"):
        model = load(exp)
        tf = build_seg_pair_transform(False, mean=n_seg["mean"], std=n_seg["std"])
        for tag, pert in (("clean", False), ("noise_s0p05", True)):
            print(f"=== {exp} x {tag} ===", flush=True)
            arms[f"{exp}_{tag}"] = run_arm(model, pairs, tf, pert)
    s_c, s_n, r_c, r_n = (arms["SEG-001_clean"], arms["SEG-001_noise_s0p05"],
                          arms["ROB-001_clean"], arms["ROB-001_noise_s0p05"])
    summary = {
        "population": f"frozen seg-val N={len(pairs)} (SegDataset val rule; identical for both models)",
        "noise_impl": "exact PBA-003 apply_perturb gauss_noise_s0p05 (sigma=0.05, SEED=7003, per-position RNG)",
        "question_A_clean": {"SEG-001": s_c["mean_dice"], "ROB-001": r_c["mean_dice"],
                             "d_mean": round(r_c["mean_dice"] - s_c["mean_dice"], 6),
                             "d_median": round(r_c["median_dice"] - s_c["median_dice"], 6),
                             "d_p10": round(r_c["p10_dice"] - s_c["p10_dice"], 6)},
        "question_B_robustness": {
            "SEG-001_degradation": round(s_n["mean_dice"] - s_c["mean_dice"], 6),
            "ROB-001_degradation": round(r_n["mean_dice"] - r_c["mean_dice"], 6),
            "robustness_gap_closed": round((r_n["mean_dice"] - r_c["mean_dice"]) -
                                           (s_n["mean_dice"] - s_c["mean_dice"]), 6),
            "SEG-001_noise": {"mean": s_n["mean_dice"], "empty": s_n["empty_pred"],
                              "by_class": s_n["by_class_mean_dice"]},
            "ROB-001_noise": {"mean": r_n["mean_dice"], "empty": r_n["empty_pred"],
                              "by_class": r_n["by_class_mean_dice"]}},
        "arms": arms,
        "note": "evidence only; no feedback into ROB-001; no locked-test decision made here",
    }
    (out / "val_2x2.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"A_d_mean": summary["question_A_clean"]["d_mean"],
                      "B_seg_deg": summary["question_B_robustness"]["SEG-001_degradation"],
                      "B_rob_deg": summary["question_B_robustness"]["ROB-001_degradation"],
                      "B_gap_closed": summary["question_B_robustness"]["robustness_gap_closed"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
