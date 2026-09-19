"""UNC-001 execution — validation-only perturbation-consistency experiment.

Implements outputs/UNC-001/design_freeze.json exactly. Frozen CLS-001;
K=8 probes + 1 independent eval-noise realization per validation case.
Writes outputs/UNC-001/per_case_val.json and outputs/UNC-001/unc001.json.
No test contact. No tuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def _probe_spec(idx: int, k: int) -> dict:
    return {"id": "gauss_noise_s0p05",
            "params": {"sigma": 0.05, "seed": SEED + idx * 16 + k, "_k": k}}


def _eval_spec(idx: int) -> dict:
    return {"id": "gauss_noise_s0p05",
            "params": {"sigma": 0.05, "seed": EVAL_SEED + idx}}


def main() -> int:
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "UNC-001"
    freeze = json.loads((out / "design_freeze.json").read_text())
    assert freeze["probe"]["K"] == K
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    T, t1, t2 = cal["temperature"], cal["tau1"], cal["tau2"]
    assert (T, t1, t2) == (freeze["frozen_reference"]["T"], freeze["frozen_reference"]["tau1"],
                           freeze["frozen_reference"]["tau2"])

    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    rows = [r for r in manifest if r["population"] == "official_train_pool"
            and r["project_split"] == "val"]
    assert len(rows) == 1000, f"val membership changed: {len(rows)}"
    assert not any(r["population"] == "official_test_locked" for r in rows)

    clean = {c["path"]: c for c in
             json.loads((root / "outputs" / "PBA-001" / "per_case_val.json").read_text())}
    assert len(clean) == 1000

    clf = build_classifier("convnext_tiny")
    clf.load_state_dict(torch.load(root / "checkpoints" / "CLS-001" / "best.pt",
                                   map_location="cpu", weights_only=False)["state"])
    clf.eval()
    tf = build_cls_transform(False)

    # Batch all (case, variant) inputs: 8 probes + 1 eval-noise per case.
    jobs = []  # (idx, kind, tensor, true_idx)
    with torch.no_grad():
        for idx, r in enumerate(rows):
            base = Image.open(root / r["path"]).convert("RGB")
            for k in range(K):
                spec = _probe_spec(idx, k)
                assert spec["params"]["seed"] == SEED + idx * 16 + k
                jobs.append((idx, f"p{k}", tf(apply_perturb(base, spec, idx)),
                             CLASSES.index(r["class"])))
            jobs.append((idx, "eval", tf(apply_perturb(base, _eval_spec(idx), idx)),
                         CLASSES.index(r["class"])))
            if (idx + 1) % 250 == 0:
                print(f"prepared {idx + 1}/1000", flush=True)

        probe_pred = np.zeros((len(rows), K), dtype=int)
        probe_conf = np.zeros((len(rows), K))
        eval_pred = np.zeros(len(rows), dtype=int)
        eval_certain = np.zeros(len(rows), dtype=bool)
        for s in range(0, len(jobs), BATCH):
            chunk = jobs[s:s + BATCH]
            probs = torch.softmax(clf(torch.stack([t for _, _, t, _ in chunk])) / T, dim=1)
            top2 = probs.topk(2).values
            for (idx, kind, _, _), pr, t2v in zip(chunk, probs, top2):
                c, m = float(t2v[0]), float(t2v[0] - t2v[1])
                if kind == "eval":
                    eval_pred[idx] = int(pr.argmax())
                    eval_certain[idx] = bool(c >= t1 and m >= t2)
                else:
                    k = int(kind[1:])
                    probe_pred[idx, k] = int(pr.argmax())
                    probe_conf[idx, k] = c
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
                         "min_probe_conf": round(float(probe_conf[idx].min()), 4),
                         "eval_pred": CLASSES[int(eval_pred[idx])],
                         "eval_correct": bool(eval_pred[idx] == CLASSES.index(r["class"])),
                         "eval_certain": bool(eval_certain[idx])})
    (out / "per_case_val.json").write_text(json.dumps(per_case))

    Y = np.array([CLASSES.index(c["true"]) for c in per_case])
    CP = np.array([CLASSES.index(c["clean_pred"]) for c in per_case])
    EP = np.array([CLASSES.index(c["eval_pred"]) for c in per_case])
    FL = np.array([c["flagged"] for c in per_case])
    CC = np.array([c["clean_certain"] for c in per_case])
    clean_ok = CP == Y
    eval_err = EP != Y
    # CBW w.r.t. evaluated condition: eval-wrong AND clean-certain (flag must catch
    # failures the frozen pathway was confident about on the clean image).
    cbw = eval_err & CC

    def rate(m):
        return round(float(FL[m].mean()), 6) if m.any() else None

    rep = {
        "N": len(per_case),
        "detection": {
            "eval_noise_errors": int(eval_err.sum()),
            "flagged_among_eval_errors": int((FL & eval_err).sum()),
            "detection_rate": rate(eval_err),
            "baseline_uncertain_among_eval_errors": int(((~eval_certain) & eval_err).sum()),
            "baseline_detection_rate": round(float((~eval_certain[eval_err]).mean()), 6)
            if eval_err.any() else None,
            "cbw_eval_errors": int(cbw.sum()),
            "flagged_among_cbw": int((FL & cbw).sum()),
            "cbw_detection_rate": rate(cbw),
        },
        "burden": {
            "flagged_total": int(FL.sum()),
            "flag_rate_overall": round(float(FL.mean()), 6),
            "clean_correct_N": int(clean_ok.sum()),
            "false_alerts_among_clean_correct": int((FL & clean_ok).sum()),
            "false_alert_rate": rate(clean_ok),
        },
        "agreement_distribution": {
            f"{i}/{K}": int((((probe_pred == CP[:, None]).sum(1)) == i).sum())
            for i in range(K + 1)},
        "by_class": {},
        "confusion_clean": confusion_matrix(Y, CP, labels=list(range(4))).tolist(),
        "confusion_eval_noise": confusion_matrix(Y, EP, labels=list(range(4))).tolist(),
        "note": "PI judges H-UNCERT-001; no single uncertainty score constructed",
    }
    for i, c in enumerate(CLASSES):
        m = Y == i
        rep["by_class"][c] = {"n": int(m.sum()),
                              "eval_errors": int((m & eval_err).sum()),
                              "detection_rate": rate(m & eval_err),
                              "false_alert_rate": rate(m & clean_ok)}
    # confusion labels order
    rep["confusion_labels"] = CLASSES
    (out / "unc001.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"eval_errors": rep["detection"]["eval_noise_errors"],
                      "detect": rep["detection"]["detection_rate"],
                      "baseline": rep["detection"]["baseline_detection_rate"],
                      "cbw": rep["detection"]["cbw_detection_rate"],
                      "false_alert": rep["burden"]["false_alert_rate"],
                      "flagged": rep["burden"]["flagged_total"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
