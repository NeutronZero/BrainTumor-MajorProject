"""UNC-001 locked evaluation — frozen consistency indicator on locked test.

Implements outputs/UNC-001/test_freeze.json exactly. Clean arm reuses frozen
CLS-001 locked records (no clean re-inference); probes + eval-noise run the
frozen CLS-001 once each. Writes outputs/UNC-001/unc001_locked.json.
No re-cut under any outcome.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pba"))
from brain_tumor.classification.models import build_classifier  # noqa: E402
from brain_tumor.data.brisc import load_manifest  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_cls_transform  # noqa: E402
from pba003_robustness import SEED, apply_perturb  # noqa: E402

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
K = 8
BATCH = 32
EVAL_SEED = 9000


def main() -> int:
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    root = Path(__file__).resolve().parents[2]
    lock = yaml.safe_load(open(root / "configs" / "deployment" / "test_lock.yaml"))
    if not lock.get("test_set_locked", False):
        print("REFUSED: test_set_locked is not true.")
        return 2
    out = root / "outputs" / "UNC-001"
    freeze = json.loads((out / "test_freeze.json").read_text())
    assert freeze["frozen"]["K"] == K and freeze["frozen"]["sigma"] == 0.05
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]

    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    rows = [r for r in manifest if r["population"] == "official_test_locked"]
    assert len(rows) == 1000, f"test membership changed: {len(rows)}"

    clean = json.loads((root / "outputs" / "PBA-001" / "per_case_test.json").read_text())
    assert len(clean) == 1000
    assert sum(1 for c in clean if c["pred"] != c["true"]) == 5
    assert sum(1 for c in clean if not c["certain"]) == 4
    # Same order check: freeze pins test_idx to per_case_test.json order.
    assert [r["path"] for r in rows] == [c["path"] for c in clean], "order changed — aborting"
    clean = {c["path"]: c for c in clean}

    clf = build_classifier("convnext_tiny")
    clf.load_state_dict(torch.load(root / "checkpoints" / "CLS-001" / "best.pt",
                                   map_location="cpu", weights_only=False)["state"])
    clf.eval()
    tf = build_cls_transform(False)

    jobs = []
    with torch.no_grad():
        for idx, r in enumerate(rows):
            base = Image.open(root / r["path"]).convert("RGB")
            for k in range(K):
                spec = {"id": "gauss_noise_s0p05",
                        "params": {"sigma": 0.05, "seed": SEED + idx * 16 + k}}
                jobs.append((idx, f"p{k}", tf(apply_perturb(base, spec, idx))))
            espec = {"id": "gauss_noise_s0p05",
                     "params": {"sigma": 0.05, "seed": EVAL_SEED + idx}}
            jobs.append((idx, "eval", tf(apply_perturb(base, espec, idx))))
            if (idx + 1) % 250 == 0:
                print(f"prepared {idx + 1}/1000", flush=True)

        probe_pred = np.zeros((len(rows), K), dtype=int)
        eval_pred = np.zeros(len(rows), dtype=int)
        eval_certain = np.zeros(len(rows), dtype=bool)
        for s in range(0, len(jobs), BATCH):
            chunk = jobs[s:s + BATCH]
            probs = torch.softmax(clf(torch.stack([t for _, _, t in chunk])) / T, dim=1)
            top2 = probs.topk(2).values
            for (idx, kind, _), pr, t2v in zip(chunk, probs, top2):
                c, m = float(t2v[0]), float(t2v[0] - t2v[1])
                if kind == "eval":
                    eval_pred[idx] = int(pr.argmax())
                    eval_certain[idx] = bool(c >= t1 and m >= t2)
                else:
                    probe_pred[idx, int(kind[1:])] = int(pr.argmax())
            if (s // BATCH) % 40 == 0:
                print(f"inferred {min(s + BATCH, len(jobs))}/{len(jobs)}", flush=True)

    per_case = []
    for idx, r in enumerate(rows):
        cc = clean[r["path"]]
        cp = CLASSES.index(cc["pred"])
        agree = float((probe_pred[idx] == cp).mean())
        per_case.append({"path": r["path"], "true": r["class"],
                         "clean_pred": cc["pred"], "clean_certain": cc["certain"],
                         "clean_correct": bool(cc["pred"] == r["class"]),
                         "agreement_fraction": round(agree, 4),
                         "flagged": bool(agree < 1.0),
                         "eval_pred": CLASSES[int(eval_pred[idx])],
                         "eval_correct": bool(eval_pred[idx] == CLASSES.index(r["class"])),
                         "eval_certain": bool(eval_certain[idx])})
    (out / "per_case_test.json").write_text(json.dumps(per_case))

    Y = np.array([CLASSES.index(c["true"]) for c in per_case])
    EP = np.array([CLASSES.index(c["eval_pred"]) for c in per_case])
    FL = np.array([c["flagged"] for c in per_case])
    CC = np.array([c["clean_certain"] for c in per_case])
    clean_ok = np.array([c["clean_correct"] for c in per_case])
    eval_err = EP != Y
    cbw = eval_err & CC

    def rate(m):
        return round(float(FL[m].mean()), 6) if m.any() else None

    val = json.loads((out / "unc001.json").read_text())
    cp_idx = np.array([CLASSES.index(c["clean_pred"]) for c in per_case])
    agree_counts = (probe_pred == cp_idx[:, None]).sum(1)
    rep = {
        "N": len(per_case),
        "clean_locked_baseline": "existing CLS-001 result reused, not recomputed "
                                 "(acc 0.9950, macro-F1 0.9952, 5 errors, 4 uncertain)",
        "detection": {
            "eval_noise_errors": int(eval_err.sum()),
            "flagged_among_eval_errors": int((FL & eval_err).sum()),
            "detection_rate": rate(eval_err),
            "baseline_detection_rate": round(float((~eval_certain[eval_err]).mean()), 6)
            if eval_err.any() else None,
            "cbw_eval_errors": int(cbw.sum()),
            "cbw_detection_rate": rate(cbw),
        },
        "burden": {
            "flagged_total": int(FL.sum()),
            "flag_rate_overall": round(float(FL.mean()), 6),
            "false_alert_rate": rate(clean_ok),
        },
        "agreement_distribution": {f"{i}/{K}": int((agree_counts == i).sum())
                                     for i in range(K + 1)},
        "by_class": {},
        "confusion_eval_noise": confusion_matrix(Y, EP, labels=list(range(4))).tolist(),
        "confusion_labels": CLASSES,
        "validation_comparison": {
            "val_detection": val["detection"]["detection_rate"],
            "val_baseline": val["detection"]["baseline_detection_rate"],
            "val_cbw": val["detection"]["cbw_detection_rate"],
            "val_false_alert": val["burden"]["false_alert_rate"],
        },
        "note": "no re-cut; outcome accepted as observed",
    }
    for i, c in enumerate(CLASSES):
        m = Y == i
        rep["by_class"][c] = {"n": int(m.sum()),
                              "eval_errors": int((m & eval_err).sum()),
                              "detection_rate": rate(m & eval_err),
                              "false_alert_rate": rate(m & clean_ok)}
    (out / "unc001_locked.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"eval_errors": rep["detection"]["eval_noise_errors"],
                      "detect": rep["detection"]["detection_rate"],
                      "baseline": rep["detection"]["baseline_detection_rate"],
                      "cbw": rep["detection"]["cbw_detection_rate"],
                      "false_alert": rep["burden"]["false_alert_rate"],
                      "flagged": rep["burden"]["flagged_total"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
