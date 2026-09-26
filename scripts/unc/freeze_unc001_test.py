"""Freeze UNC-001 locked-test procedure — written BEFORE any test contact."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
freeze = {
    "experiment": "UNC-001 locked evaluation (uncertainty-analysis layer on frozen CLS-001)",
    "engineering_note": "engineering verification of a failure-detection mechanism; "
    "not a replacement classifier; no research-novelty claim",
    "frozen": {
        "checkpoint": "checkpoints/CLS-001/best.pt",
        "checkpoint_sha256": sha(root / "checkpoints" / "CLS-001" / "best.pt"),
        "preprocessing": "build_cls_transform(False), image_size 224",
        "T": cal["temperature"],
        "tau1": cal["tau1"],
        "tau2": cal["tau2"],
        "sigma": 0.05,
        "K": 8,
        "flag_rule": "FLAG iff agreement_fraction < 1.0 "
        "(fraction of 8 probes matching the clean predicted class)",
        "probe_seed": "7003 + test_idx*16 + k, k=0..7 "
        "(test_idx = position in locked-test manifest order, "
        "identical order to outputs/PBA-001/per_case_test.json)",
        "eval_noise_seed": "9000 + test_idx (independent realization)",
        "noise_impl": "exact PBA-003 apply_perturb gauss_noise",
    },
    "clean_source": "frozen CLS-001 locked records "
    "(outputs/PBA-001/per_case_test.json: 5 errors / 4 uncertain, "
    "verified identical to outputs/test_evaluation_7b860dca72ea.json); "
    "existing CLS-001 locked baseline is NOT recomputed",
    "locked_rules": [
        "no re-cut of K/cutoff/seeds",
        "no class-specific rule",
        "no calibration refit",
        "no composite score",
        "weaker-than-validation test result is accepted as the result",
    ],
}
(root / "outputs" / "UNC-001" / "test_freeze.json").write_text(json.dumps(freeze, indent=1))
print("UNC-001 test procedure frozen; no test contact yet")
