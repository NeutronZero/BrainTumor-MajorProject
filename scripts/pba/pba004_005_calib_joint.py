"""PBA-004 (calibration & uncertainty) + PBA-005 (cls x seg joint consistency).

Pure analysis of frozen artifacts: PBA-001 per-case test probabilities
+ PBA-002 per-case segmentation, plus reliability/confidence data.
No threshold or T changes. Output: outputs/PBA-004/ and outputs/PBA-005/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
BINS = 15


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    test = json.loads((root / "outputs" / "PBA-001" / "per_case_test.json").read_text())
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    seg = json.loads((root / "outputs" / "PBA-002" / "per_case_segtest.json").read_text())

    P = np.array([c["probs"] for c in test])
    Y = np.array([CLASSES.index(c["true"]) for c in test])
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == Y).astype(float)

    out4 = root / "outputs" / "PBA-004"
    out4.mkdir(parents=True, exist_ok=True)
    bins, ece = [], 0.0
    for i in range(BINS):
        m = (conf > i / BINS) & (conf <= (i + 1) / BINS)
        if m.any():
            acc, avg = float(correct[m].mean()), float(conf[m].mean())
            bins.append({"bin": f"{i/BINS:.2f}-{(i+1)/BINS:.2f}", "n": int(m.sum()),
                         "accuracy": round(acc, 4), "mean_confidence": round(avg, 4),
                         "error_rate": round(1 - acc, 4)})
            ece += m.mean() * abs(acc - avg)
    cw = [{"path": c["path"], "true": c["true"], "pred": c["pred"],
           "confidence": c["confidence"], "certain": c["certain"]}
          for c in test if c["pred"] != c["true"] and c["certain"]]
    hist = {f"{i/BINS:.2f}-{(i+1)/BINS:.2f}": int(((conf > i / BINS) & (conf <= (i + 1) / BINS)).sum())
            for i in range(BINS)}
    rep4 = {"T_frozen": cal["temperature"], "tau1": cal["tau1"], "tau2": cal["tau2"],
            "ece_post_T": round(float(ece), 6),
            "reliability_bins": bins,
            "confidence_histogram": hist,
            "confident_but_wrong": cw,
            "uncertain_cases": [{"path": c["path"], "true": c["true"], "pred": c["pred"],
                                 "confidence": c["confidence"], "margin": c["margin"]}
                                for c in test if not c["certain"]],
            "class_conditional_ece": json.loads(
                (root / "outputs" / "PBA-001" / "pba001.json").read_text())["test"]["class_ece"]}
    (out4 / "pba004.json").write_text(json.dumps(rep4, indent=1))

    # PBA-005: joint matrix classification state x segmentation state.
    # Denominators are explicit: only the 860 paired seg-test cases have a
    # defined joint outcome. 140 healthy-only classification cases have no
    # segmentation GT -> unmeasured, NOT zero. Link is by image identity.
    seg_by_stem = {c["stem"]: c for c in seg}
    seg_stems = set(seg_by_stem)
    cls_stems = {Path(c["path"]).stem: c for c in test}
    matched = cls_stems.keys() & seg_stems
    unmeasured = {
        "cls_only": sorted(cls_stems.keys() - seg_stems),
        "seg_only": sorted(seg_stems - cls_stems.keys()),
    }
    joint: dict[str, int] = {}
    rows = []
    for stem in sorted(matched):
        c = cls_stems[stem]
        s = seg_by_stem[stem]
        cls_state = "uncertain" if not c["certain"] else ("tumor" if c["pred"] != "notumor" else "notumor_pred")
        seg_state = ("empty" if s["empty_pred"] else "segmented")
        if c["true"] == "notumor":
            cls_state = "true_notumor"
        key = f"{cls_state}|{seg_state}"
        joint[key] = joint.get(key, 0) + 1
        rows.append({"stem": s["stem"], "cls": cls_state, "seg": seg_state,
                     "dice": s["dice"], "cls_conf": c["confidence"]})

    def flag(r):
        if r["cls"] == "uncertain":
            return "uncertainty_pathway"
        if r["cls"] == "true_notumor":
            return "concordant" if r["seg"] == "empty" else "potential_fp_segmentation"
        if r["cls"] == "tumor":
            return "concordant" if r["seg"] == "segmented" else "potential_localization_failure"
        return "concordant"

    for r in rows:
        r["interpretation"] = flag(r)
    summary = {}
    for r in rows:
        summary[r["interpretation"]] = summary.get(r["interpretation"], 0) + 1
    out5 = root / "outputs" / "PBA-005"
    out5.mkdir(parents=True, exist_ok=True)
    rep5 = {"populations": {
                "all_cls_test": len(test),
                "seg_test_paired": len(seg),
                "cls_seg_matched_joint_measurable": len(rows),
                "healthy_cls_no_seg_gt": sum(1 for c in test if c["true"] == "notumor"
                                             and Path(c["path"]).stem not in seg_stems),
                "excluded_unmeasured": {k: len(v) for k, v in unmeasured.items()},
                "note": "unmeasured != zero; joint metrics use only matched population"},
            "unmeasured_lists": unmeasured,
            "n_pairs": len(rows), "joint_counts": joint,
            "interpretation_counts": summary,
            "localization_failure_cases": [r for r in rows if r["interpretation"] == "potential_localization_failure"],
            "potential_fp_segmentation": [r for r in rows if r["interpretation"] == "potential_fp_segmentation"][:20],
            "note": "confident-tumor+empty-seg (1.17% measured in locked eval) is one component here; no new joint score invented"}
    (out5 / "pba005.json").write_text(json.dumps(rep5, indent=1))
    (out5 / "joint_rows.json").write_text(json.dumps(rows))
    print(json.dumps({"ece": rep4["ece_post_T"], "joint": rep5["interpretation_counts"],
                      "conf_wrong": len(cw), "uncertain": len(rep4["uncertain_cases"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
