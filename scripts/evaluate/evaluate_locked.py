"""evaluate_locked.py — joint locked-test evaluation (PI-authorized, one-way gate).

Refuses unless configs/deployment/test_lock.yaml: test_set_locked is true.
Evaluates FROZEN models/thresholds/localization exactly once; writes
outputs/test_evaluation_<sha>.json (+ log). Never tunes, never overwrites
without --force --reason. No feedback flows back into any configuration.

Populations: CLS primary N=1000 / sensitivity N=993; SEG primary 860 pairs /
sensitivity 853. Disagreement = P(confident tumor AND empty segmentation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification import calibrate  # noqa: E402
from brain_tumor.classification.models import build_classifier  # noqa: E402
from brain_tumor.data.brisc import load_manifest  # noqa: E402
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import (  # noqa: E402
    build_cls_transform,
    build_seg_pair_transform,
)
from brain_tumor.segmentation.unet import UNet  # noqa: E402

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def _open(p: Path, mode: str):
    from PIL import Image

    return Image.open(p).convert(mode)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    lock = yaml.safe_load(open(root / "configs" / "deployment" / "test_lock.yaml"))
    if not lock.get("test_set_locked", False):
        print("REFUSED: test_set_locked is not true. Unlock is a PI decision.")
        return 2

    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    test_rows = [r for r in manifest if r["population"] == "official_test_locked"]
    assert len(test_rows) == 1000, "test membership changed — aborting"
    excl = json.loads(
        (root / "outputs" / "data_gate_0" / "cross_split_exclusion_list.json").read_text()
    )
    excl_test = {p.replace("\\", "/") for p in excl["excluded_test_paths"]}
    sens_rows = [r for r in test_rows if r["path"] not in excl_test]
    assert len(sens_rows) == 993
    cont_hashes = set(excl["hashes"])

    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]

    device = "cpu"
    clf = build_classifier("convnext_tiny")
    ckpt = torch.load(
        root / "checkpoints" / "CLS-001" / "best.pt", map_location=device, weights_only=False
    )  # own file
    clf.load_state_dict(ckpt["state"])
    clf.eval()
    seg = UNet()
    sckpt = torch.load(
        root / "checkpoints" / "SEG-001" / "best.pt", map_location=device, weights_only=False
    )  # own file
    seg.load_state_dict(sckpt["state"])
    seg.eval()
    seg_norm = json.loads((root / "outputs" / "SEG-001" / "metrics.json").read_text())["norm"]
    cls_tf = build_cls_transform(False)
    seg_tf = build_seg_pair_transform(False, mean=seg_norm["mean"], std=seg_norm["std"])

    # ---- classification over N=1000 ----
    records = []
    with torch.no_grad():
        for i, r in enumerate(test_rows):
            img = cls_tf(_open(root / r["path"], "RGB")).unsqueeze(0)
            probs = torch.softmax(clf(img) / T, dim=1)[0]
            top2 = probs.topk(2).values
            conf, margin = float(top2[0]), float(top2[0] - top2[1])
            pred = CLASSES[int(probs.argmax())]
            certain = conf >= t1 and margin >= t2
            records.append(
                {
                    "path": r["path"],
                    "true": r["class"],
                    "pred": pred,
                    "probs": [round(float(p), 6) for p in probs],
                    "confidence": round(conf, 6),
                    "certain": bool(certain),
                    "contaminated": bool(
                        r["path"] in excl_test or _sha(root / r["path"]) in cont_hashes
                    ),
                }
            )
            if i % 250 == 0:
                print(f"cls {i}/1000", flush=True)

    cls_metrics = _cls_block(records, "N=1000 primary")
    sens_metrics = _cls_block(
        [c for c in records if c["path"] not in excl_test], "N=993 sensitivity"
    )
    cont_cases = [c for c in records if c["contaminated"]]

    # ---- segmentation over 860 official seg-test pairs ----
    release = root / "data" / "brisc2025"
    pairs = []
    for img_p in sorted((release / "segmentation_task" / "test" / "images").glob("*.jpg")):
        m_p = release / "segmentation_task" / "test" / "masks" / (img_p.stem + ".png")
        assert m_p.exists(), f"missing test mask {img_p.stem}"
        pairs.append((img_p, m_p))
    assert len(pairs) == 860
    seg_cases = []
    with torch.no_grad():
        for i, (img_p, m_p) in enumerate(pairs):
            img, gt = seg_tf(_open(img_p, "L"), _open(m_p, "L"))
            prob = torch.sigmoid(seg(img.unsqueeze(0)))[0, 0].numpy()
            g = gt[0].numpy()
            binp, bing = (prob > 0.5), (g > 0.5)
            inter = float((binp & bing).sum())
            dice = (2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6)
            union = float((binp | bing).sum())
            iou = (inter + 1e-6) / (union + 1e-6)
            ow, oh = _open(img_p, "RGB").size
            loc, w = extract(prob, (oh, ow))
            seg_cases.append(
                {
                    "stem": img_p.stem,
                    "dice": dice,
                    "iou": iou,
                    "empty_pred": loc.area_pixels == 0,
                    "area": loc.area_pixels,
                    "bbox": loc.bbox,
                    "centroid": loc.centroid,
                    "warnings": w,
                    "contaminated": bool(_sha(img_p) in cont_hashes),
                }
            )
            if i % 200 == 0:
                print(f"seg {i}/860", flush=True)
    seg_metrics = _seg_block(seg_cases, "seg-test N=860")
    seg_sens = _seg_block([c for c in seg_cases if not c["contaminated"]], "seg-test N=853")

    # ---- joint disagreement on the 1000 cls-test images ----
    # Faithful to integrated mode: every tumor prediction runs the segmenter
    # on the raw MRI (seg-pair cache first; live UNet for images without pairs,
    # i.e. no_tumor-GT false positives). notumor predictions -> no-mask state.
    live_cache: dict[str, bool] = {}

    def _empty_for(path: str, stem: str) -> bool:
        if stem in by_stem_seg:
            return by_stem_seg[stem]["empty_pred"]
        if stem in live_cache:
            return live_cache[stem]
        # No seg pair (no_tumor-GT image with tumor prediction): run the UNet
        # live per integrated mode; dummy mask discarded (output [0] is image).
        pil_gray = _open(root / path, "L")
        ow, oh = _open(root / path, "RGB").size
        with torch.no_grad():
            img_t = seg_tf(pil_gray, pil_gray)[0]
            prob = torch.sigmoid(seg(img_t.unsqueeze(0)))[0, 0].numpy()
        loc, _ = extract(prob, (oh, ow))
        live_cache[stem] = loc.area_pixels == 0
        return live_cache[stem]

    by_stem_seg = {c["stem"]: c for c in seg_cases}
    dis, dis_by_cls, n_conf_tumor = 0, {}, 0
    sys_states = {}
    for c in records:
        stem = Path(c["path"]).stem
        tumor = c["pred"] != "notumor"
        if tumor and c["certain"]:
            n_conf_tumor += 1
            if _empty_for(c["path"], stem):
                dis += 1
                dis_by_cls[c["pred"]] = dis_by_cls.get(c["pred"], 0) + 1
                st = "tumor_unlocalized"
            else:
                st = "tumor_localized"
        elif not tumor:
            st = "healthy" if c["certain"] else "uncertain"
        else:  # tumor predicted but uncertain
            st = "uncertain"
        sys_states[st] = sys_states.get(st, 0) + 1
    disagreement = {
        "rate": dis / n_conf_tumor if n_conf_tumor else 0.0,
        "count": dis,
        "n_confident_tumor": n_conf_tumor,
        "per_class": dis_by_cls,
        "exceeds_5pct": (dis / n_conf_tumor > 0.05) if n_conf_tumor else False,
    }

    # NOTE on seg-test stems vs cls-test stems: seg-test pairs cover tumor
    # cases of the official test release; no_tumor-GT images have no pairs and
    # their tumor predictions run the UNet live above. notumor predictions use
    # the integrated no-mask state without invoking the segmenter.
    payload = {
        "evaluation": "locked-test, frozen models/thresholds/localization",
        "calibration": {"T": T, "tau1": t1, "tau2": t2, "source": "validation-only, frozen"},
        "cls_primary": cls_metrics,
        "cls_sensitivity": sens_metrics,
        "cls_delta": _delta(cls_metrics, sens_metrics),
        "contaminated_test_cases": [
            {
                "path": c["path"],
                "true": c["true"],
                "pred": c["pred"],
                "confidence": c["confidence"],
                "certain": c["certain"],
            }
            for c in cont_cases
        ],
        "system_state_distribution": sys_states,
        "seg_primary": seg_metrics,
        "seg_sensitivity": seg_sens,
        "disagreement_confident_tumor_empty_seg": disagreement,
        "test_lock": "official test evaluated exactly once; no tuning permitted after this point",
    }

    key = sorted((r["path"], r["sha256"]) for r in test_rows)
    sha = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
    out = root / "outputs" / f"test_evaluation_{sha}.json"
    if out.exists() and not (args.force and args.reason):
        print(f"REFUSED: {out.name} exists; use --force --reason to supersede.")
        return 2
    out.write_text(json.dumps(payload, indent=1))
    with open(root / "outputs" / "test_evaluation_log.md", "a", encoding="utf-8") as f:
        f.write(
            f"\n- eval {sha} force={args.force} reason={args.reason!r} "
            f"cls_f1={cls_metrics['macro_f1']:.4f} seg_dice={seg_metrics['mean_dice']:.4f}\n"
        )
    print(
        json.dumps(
            {
                "sha": sha,
                "cls": cls_metrics,
                "cls_sens": sens_metrics,
                "seg": seg_metrics,
                "seg_sens": seg_sens,
                "disagreement": disagreement,
                "system_states": sys_states,
            },
            indent=1,
        )[:3000]
    )
    return 0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _cls_block(records: list[dict], tag: str) -> dict:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = [c["true"] for c in records]
    p = [c["pred"] for c in records]
    # AUC needs integer targets; labels arrive as class strings.
    y_idx = [CLASSES.index(v) for v in y]
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    # Fail loud on AUC failure: a silently-null metric in a frozen artifact
    # is evidence corruption, not a graceful degradation.
    auc = float(
        roc_auc_score(
            y_idx, [c["probs"] for c in records], multi_class="ovr", labels=list(range(4))
        )
    )
    import torch as _t

    ece = calibrate.ece(
        _t.tensor([c["probs"] for c in records]), _t.tensor([CLASSES.index(v) for v in y])
    )
    unc = sum(1 for c in records if not c["certain"])
    return {
        "tag": tag,
        "n": len(records),
        "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(records), 6),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 6),
        "per_class": {
            c: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
            }
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": confusion_matrix(y, p, labels=CLASSES).tolist(),
        },
        "roc_auc_ovr": round(auc, 6) if auc is not None else None,
        "ece_post_T": round(ece, 6),
        "uncertain": unc,
        "uncertain_rate": round(unc / len(records), 6),
    }


def _seg_block(cases: list[dict], tag: str) -> dict:
    import numpy as _np

    d = _np.array([c["dice"] for c in cases])
    return {
        "tag": tag,
        "n": len(cases),
        "mean_dice": round(float(d.mean()), 6),
        "median_dice": round(float(_np.median(d)), 6),
        "p10_dice": round(float(_np.percentile(d, 10)), 6),
        "min_dice": round(float(d.min()), 6),
        "mean_iou": round(float(_np.mean([c["iou"] for c in cases])), 6),
        "empty_pred": sum(1 for c in cases if c["empty_pred"]),
        "multi_component": sum(1 for c in cases if "multiple_components" in c["warnings"]),
    }


def _delta(a: dict, b: dict) -> dict:
    return {k: round(a[k] - b[k], 6) for k in ("accuracy", "macro_f1") if k in a and k in b}


if __name__ == "__main__":
    sys.exit(main())
