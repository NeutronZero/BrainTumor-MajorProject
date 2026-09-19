"""PBA-001 — classification error forensics from frozen artifacts (no retraining).

Per-case recomputation (deterministic, frozen ckpt/T/taus) on val + test;
analyzes the 5 test errors, confusion structure, confidence correct-vs-wrong,
uncertain cases, class-wise confidence + calibration, image characteristics,
representative cases, contamination flags. Output: outputs/PBA-001/.
No threshold changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, img_chars, infer_split, load_excl, load_frozen  # noqa: E402


def ece_of(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    out = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            pred = probs[m].argmax(1)
            out += m.mean() * abs((pred == labels[m]).mean() - conf[m].mean())
    return float(out)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "PBA-001"
    out.mkdir(parents=True, exist_ok=True)
    cal, manifest, clf = load_frozen(root)
    excl = load_excl(root)
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]

    val = infer_split(root, clf, manifest, "val", T, t1, t2, excl)
    test = infer_split(root, clf, manifest, "test", T, t1, t2, excl)
    (out / "per_case_val.json").write_text(json.dumps(val))
    (out / "per_case_test.json").write_text(json.dumps(test))

    rep = {"frozen": {"T": T, "tau1": t1, "tau2": t2}, "n_val": len(val), "n_test": len(test)}
    for tag, cases in (("val", val), ("test", test)):
        P = np.array([c["probs"] for c in cases])
        Y = np.array([CLASSES.index(c["true"]) for c in cases])
        pred = np.array([CLASSES.index(c["pred"]) for c in cases])
        ok = pred == Y
        conf = np.array([c["confidence"] for c in cases])
        rep[tag] = {
            "errors": [{**{k: c[k] for k in ("path", "true", "pred", "confidence", "margin", "certain", "contaminated")},
                        **img_chars(root, c["path"])} for c in cases if c["pred"] != c["true"]],
            "n_errors": int((~ok).sum()),
            "conf_correct_mean": round(float(conf[ok].mean()), 4),
            "conf_wrong_mean": round(float(conf[~ok].mean()), 4) if (~ok).any() else None,
            "conf_wrong_values": [round(float(v), 4) for v in conf[~ok]],
            "uncertain": [c["path"] for c in cases if not c["certain"]],
            "class_conf": {c: {"mean": round(float(conf[Y == i].mean()), 4),
                               "std": round(float(conf[Y == i].std()), 4),
                               "n": int((Y == i).sum())} for i, c in enumerate(CLASSES)},
            "class_ece": {c: round(ece_of(P[Y == i], Y[Y == i]), 6) for i, c in enumerate(CLASSES)},
            "overall_ece": round(ece_of(P, Y), 6),
            "contaminated_errors": sum(1 for c in cases if c["pred"] != c["true"] and c["contaminated"]),
        }
        # representative stems per outcome type
        for outcome, sel in (("TP", lambda c: c["pred"] == c["true"] and c["true"] != "notumor"),
                             ("TN", lambda c: c["pred"] == c["true"] == "notumor"),
                             ("FP", lambda c: c["pred"] != "notumor" and c["true"] == "notumor"),
                             ("FN", lambda c: c["pred"] == "notumor" and c["true"] != "notumor"),
                             ("mis_tumor", lambda c: c["pred"] != c["true"] and c["true"] != "notumor" and c["pred"] != "notumor")):
            rep[tag][f"rep_{outcome}"] = [c["path"] for c in cases if sel(c)][:5]
    (out / "pba001.json").write_text(json.dumps(rep, indent=1))
    e = rep["test"]
    print(f"errors={e['n_errors']} conf_ok={e['conf_correct_mean']} conf_bad={e['conf_wrong_mean']}")
    print(f"uncertain={len(e['uncertain'])} contam_errors={e['contaminated_errors']} ece={e['overall_ece']}")
    for er in e["errors"]:
        print(er)
    return 0


if __name__ == "__main__":
    sys.exit(main())
