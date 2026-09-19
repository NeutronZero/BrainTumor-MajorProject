"""PBA-003 — robustness characterization (evaluation-only, frozen baselines).

Authorized: PI authorization for PBA-003. Strictly descriptive:
  frozen CLS-001 / SEG-001 -> predefined perturbations -> per-perturbation
  reporting -> NO aggregate robustness score.

Suite provenance: plan §68 "Robustness Testing" (C:/Projects/
BrainTumor_MajorProject_Plan.md:3573-3609): Gaussian noise σ=0.05, JPEG q=50,
Downscale 0.75x, Intensity shift (severity "defined" -> fixed here as additive
+0.05 on [0,1], documented). No other perturbations introduced. Report table
columns per plan: Severity | dAccuracy | dDice | Notes. No aggregate score.

Perturbations apply to the raw input image; frozen preprocessing, checkpoints,
T=0.5116, tau1/tau2, mask convention, and localization semantics unchanged.
torch.no_grad + eval mode throughout. Incremental per-perturbation writes allow
resume; completed perturbations are skipped, never recomputed-and-selected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_tumor.classification.models import build_classifier  # noqa: E402
from brain_tumor.data.brisc import load_manifest  # noqa: E402
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import (  # noqa: E402
    build_cls_transform,
    build_seg_pair_transform,
)
from brain_tumor.segmentation.unet import UNet  # noqa: E402
from common import CLASSES  # noqa: E402

SEED = 7003
BATCH_CLS, BATCH_SEG = 32, 16
TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}

SUITE = [
    {"id": "none", "family": "control", "definition": "no perturbation; frozen pipeline as-is",
     "params": {}},
    {"id": "gauss_noise_s0p05", "family": "noise",
     "definition": "plan §68: additive Gaussian sigma=0.05 on [0,1] scale, per-case RNG(seed+idx), clip",
     "params": {"sigma": 0.05, "seed": SEED}},
    {"id": "jpeg_q50", "family": "compression",
     "definition": "plan §68: JPEG encode quality=50 to memory buffer, decode back (Pillow, deterministic)",
     "params": {"quality": 50}},
    {"id": "downscale_0p75x", "family": "resolution/degradation",
     "definition": "plan §68: downsample raw input to 0.75x (w,h) BILINEAR then upsample back",
     "params": {"scale": 0.75}},
    {"id": "intensity_shift_p0p05", "family": "intensity variation",
     "definition": "plan §68 severity 'defined' -> fixed here: additive +0.05 on [0,1] scale, clip",
     "params": {"shift": 0.05}},
]


def apply_perturb(img, spec: dict, idx: int):
    """Deterministic perturbation on a PIL image. Returns new PIL image."""
    import io
    from PIL import Image
    pid = spec["id"]
    if pid == "none":
        return img
    if pid == "intensity_shift_p0p05":
        a = np.asarray(img).astype(np.float32) / 255.0 + spec["params"]["shift"]
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if pid == "gauss_noise_s0p05":
        rng = np.random.RandomState(spec["params"]["seed"] + idx)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = np.clip(a + rng.normal(0, spec["params"]["sigma"], a.shape), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8))
    if pid == "jpeg_q50":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=spec["params"]["quality"])
        buf.seek(0)
        return Image.open(buf).convert(img.mode)
    if pid == "downscale_0p75x":
        from PIL import Image as _I
        w, h = img.size
        s = spec["params"]["scale"]
        small = img.resize((max(int(w * s), 1), max(int(h * s), 1)), _I.BILINEAR)
        return small.resize((w, h), _I.BILINEAR)
    raise ValueError(pid)


def ece_of(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    out = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            pr = probs[m].argmax(1)
            out += m.mean() * abs((pr == labels[m]).mean() - conf[m].mean())
    return float(out)


def main() -> int:
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "PBA-003"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Freeze suite definition BEFORE evaluation.
    (out / "perturbation_suite.json").write_text(json.dumps({
        "provenance": "plan §68 Robustness Testing (BrainTumor_MajorProject_Plan.md:3573-3609): "
                      "Gaussian noise s=0.05, JPEG q=50, Downscale 0.75x, Intensity shift "
                      "(severity 'defined' -> additive +0.05, documented in SUITE)",
        "frozen_model_params": "checkpoints/CLS-001/best.pt + checkpoints/SEG-001/best.pt; "
                               "T/tau1/tau2 from outputs/CLS-001/calibration_frozen.json; "
                               "no refit under perturbation",
        "perturbations": SUITE,
        "populations": {"classification": "locked cls-test N=1000",
                        "segmentation": "seg-test N=860 (masks never perturbed)"},
        "rule": "per-perturbation reporting only; no aggregate robustness score"}, indent=1))

    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    cls_rows = [r for r in manifest if r["population"] == "official_test_locked"]
    assert len(cls_rows) == 1000

    clf = build_classifier("convnext_tiny")
    clf.load_state_dict(torch.load(root / "checkpoints" / "CLS-001" / "best.pt",
                                   map_location="cpu", weights_only=False)["state"])
    clf.eval()
    cls_tf = build_cls_transform(False)

    norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    seg_tf = build_seg_pair_transform(False, mean=norm["mean"], std=norm["std"])
    seg_model = UNet()
    seg_model.load_state_dict(torch.load(root / "checkpoints" / "SEG-001" / "best.pt",
                                         map_location="cpu", weights_only=False)["state"])
    seg_model.eval()
    release = root / "data" / "brisc2025"
    seg_pairs = [(p, release / "segmentation_task" / "test" / "masks" / (p.stem + ".png"))
                 for p in sorted((release / "segmentation_task" / "test" / "images").glob("*.jpg"))]
    assert len(seg_pairs) == 860

    for spec in SUITE:
        dest = out / f"pert_{spec['id']}.json"
        if dest.exists():
            print(f"skip {spec['id']} (exists)", flush=True)
            continue
        # ---- classification arm ----
        P_list, Y_list, C_list, M_list = [], [], [], []
        with torch.no_grad():
            batch = []
            for idx, r in enumerate(cls_rows):
                img = apply_perturb(Image.open(root / r["path"]).convert("RGB"), spec, idx)
                batch.append((cls_tf(img), CLASSES.index(r["class"]), r))
                if len(batch) == BATCH_CLS or idx == len(cls_rows) - 1:
                    bt = torch.stack([b[0] for b in batch])
                    probs = torch.softmax(clf(bt) / T, dim=1)
                    top2 = probs.topk(2).values
                    for ( _, y, rr), pr, t2v in zip(batch, probs, top2):
                        P_list.append(pr.tolist())
                        Y_list.append(y)
                        C_list.append((float(t2v[0]), float(t2v[0] - t2v[1]), rr))
                    batch = []
                if (idx + 1) % 500 == 0:
                    print(f"cls {spec['id']} {idx + 1}/1000", flush=True)
        P, Y = np.array(P_list), np.array(Y_list)
        pred = P.argmax(1)
        ok = pred == Y
        conf_arr = P.max(1)
        cls_rec = {"n": 1000, "evaluable": 1000,
                   "accuracy": round(float((pred == Y).mean()), 6),
                   "macro_f1": round(float(f1_score(Y, pred, average="macro")), 6),
                   "per_class_f1": {c: round(float(f1_score(Y, pred, labels=[i], average="macro", zero_division=0)), 6)
                                    for i, c in enumerate(CLASSES)},
                   "ece_frozen_T": round(ece_of(P, Y), 6),
                   "uncertain_n": int(sum(1 for c, m, _ in C_list if not (c >= t1 and m >= t2))),
                   "errors_n": int((~ok).sum()),
                   "cbw_n": int(sum(1 for (c, m, _), o in zip(C_list, ok) if not o and c >= t1 and m >= t2)),
                   "conf_correct_mean": round(float(conf_arr[ok].mean()), 4),
                   "conf_wrong_mean": round(float(conf_arr[~ok].mean()), 4) if (~ok).any() else None,
                   "error_stems": sorted([Path(rr["path"]).stem for _, _, rr in
                                          [x for x, o in zip(C_list, ok) if not o]])}
        # ---- segmentation arm ----
        dice_l, iou_l, by_cls, empty_n, multi_n, fp_l, fn_l = [], [], {}, 0, 0, [], []
        with torch.no_grad():
            ib, gb, meta = [], [], []
            def flush():
                nonlocal empty_n, multi_n
                if not ib:
                    return
                prob = torch.sigmoid(seg_model(torch.stack(ib))).numpy()[:, 0]
                for pr, g, (stem, ow, oh) in zip(prob, gb, meta):
                    binp, bing = pr > 0.5, g > 0.5
                    inter = float((binp & bing).sum())
                    dice_l.append((2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6))
                    iou_l.append((inter + 1e-6) / (float((binp | bing).sum()) + 1e-6))
                    fg = int(bing.sum())
                    fp_l.append(int((binp & ~bing).sum()) / max(fg, 1))
                    fn_l.append(int((~binp & bing).sum()) / max(fg, 1))
                    loc, w = extract(pr, (oh, ow))
                    if loc.area_pixels == 0:
                        empty_n += 1
                    if "multiple_components" in w:
                        multi_n += 1
                    by_cls.setdefault(segmap_cls[stem], []).append(dice_l[-1])
            segmap_cls = {}
            for idx, (ip, mp) in enumerate(seg_pairs):
                stem = ip.stem
                segmap_cls[stem] = TUMOR.get(stem.split("_")[3], "?")
                img = apply_perturb(Image.open(ip).convert("L"), spec, idx)
                ti, tm = seg_tf(img, Image.open(mp).convert("L"))
                ow, oh = Image.open(ip).size
                ib.append(ti)
                gb.append(tm[0].numpy() > 0.5)
                meta.append((stem, ow, oh))
                if len(ib) == BATCH_SEG or idx == len(seg_pairs) - 1:
                    flush()
                    ib, gb, meta = [], [], []
                if (idx + 1) % 430 == 0:
                    print(f"seg {spec['id']} {idx + 1}/860", flush=True)
        d = np.array(dice_l)
        seg_rec = {"n": 860, "evaluable": 860,
                   "mean_dice": round(float(d.mean()), 6),
                   "median_dice": round(float(np.median(d)), 6),
                   "p10_dice": round(float(np.percentile(d, 10)), 6),
                   "mean_iou": round(float(np.mean(iou_l)), 6),
                   "empty_pred": empty_n, "multi_component": multi_n,
                   "by_class_mean_dice": {c: round(float(np.mean(v)), 6) for c, v in by_cls.items()},
                   "mean_fp_rate": round(float(np.mean(fp_l)), 6),
                   "mean_fn_rate": round(float(np.mean(fn_l)), 6)}
        dest.write_text(json.dumps({"perturbation": spec, "cls": cls_rec, "seg": seg_rec}, indent=1))
        print(f"wrote {spec['id']}: cls_acc={cls_rec['accuracy']} f1={cls_rec['macro_f1']} "
              f"ece={cls_rec['ece_frozen_T']} unc={cls_rec['uncertain_n']} | "
              f"seg_dice={seg_rec['mean_dice']} empty={empty_n} multi={multi_n}", flush=True)

    # 2. Summary: per-perturbation deltas vs control arm. NO aggregate score.
    arms = {s["id"]: json.loads((out / f"pert_{s['id']}.json").read_text()) for s in SUITE}
    base_c, base_s = arms["none"]["cls"], arms["none"]["seg"]
    summ = {"control": "none",
            "note": "descriptive deltas vs control arm; no aggregate robustness score computed "
                    "(forbidden by PBA-003 charter)",
            "per_perturbation": {}}
    for s in SUITE:
        if s["id"] == "none":
            continue
        c, g = arms[s["id"]]["cls"], arms[s["id"]]["seg"]
        new_err = sorted(set(c["error_stems"]) - set(base_c["error_stems"]))
        recov = sorted(set(base_c["error_stems"]) - set(c["error_stems"]))
        summ["per_perturbation"][s["id"]] = {
            "family": s["family"],
            "cls": {"accuracy": c["accuracy"], "d_acc": round(c["accuracy"] - base_c["accuracy"], 6),
                    "macro_f1": c["macro_f1"], "d_f1": round(c["macro_f1"] - base_c["macro_f1"], 6),
                    "ece_frozen_T": c["ece_frozen_T"], "uncertain_n": c["uncertain_n"],
                    "errors_n": c["errors_n"], "cbw_n": c["cbw_n"],
                    "new_errors_vs_control": new_err, "recovered_vs_control": recov},
            "seg": {"mean_dice": g["mean_dice"], "d_dice": round(g["mean_dice"] - base_s["mean_dice"], 6),
                    "median_dice": g["median_dice"], "p10_dice": g["p10_dice"],
                    "mean_iou": g["mean_iou"], "empty_pred": g["empty_pred"],
                    "multi_component": g["multi_component"],
                    "by_class_mean_dice": g["by_class_mean_dice"]}}
    summ["control_values"] = {"cls_acc": base_c["accuracy"], "cls_f1": base_c["macro_f1"],
                              "cls_ece": base_c["ece_frozen_T"], "cls_unc": base_c["uncertain_n"],
                              "seg_dice": base_s["mean_dice"], "seg_empty": base_s["empty_pred"]}
    (out / "pba003.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: {"d_acc": v["cls"]["d_acc"], "d_dice": v["seg"]["d_dice"]}
                      for k, v in summ["per_perturbation"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
