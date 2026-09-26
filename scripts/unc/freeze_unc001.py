"""UNC-001 design freeze — written BEFORE any execution. No post-hoc changes permitted."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
freeze = {
    "experiment": "UNC-001 perturbation-consistency uncertainty",
    "hypothesis": "H-UNCERT-001: consistency across a fixed predefined σ=0.05 "
    "Gaussian-noise probe identifies a meaningful fraction of the "
    "classifier's noise-induced failures missed by single-image "
    "confidence thresholds. No OOD-detection claim.",
    "frozen_reference": {
        "checkpoint": "checkpoints/CLS-001/best.pt (unchanged, no retraining)",
        "preprocessing": "build_cls_transform(False), image_size 224",
        "T": 0.5115954875946045,
        "tau1": 0.95,
        "tau2": 0.05,
        "calibration": "no refit",
    },
    "probe": {
        "implementation": "exact PBA-003 apply_perturb gauss_noise (additive σ=0.05 "
        "on [0,1], clip), scripts/pba/pba003_robustness.py",
        "K": 8,
        "probe_seed": "7003 + idx*16 + k for case position idx, variant k=0..7 "
        "(extends the PBA-003 seed+idx convention to K variants; "
        "identical procedure for every case)",
        "eval_noise_seed": "9000 + idx (independent realization defining the "
        "evaluated σ=0.05 condition; distinct from probe streams)",
    },
    "flag_rule": "FLAG iff agreement_fraction < 1.0, where agreement_fraction = "
    "fraction of the K=8 probes whose predicted class equals the "
    "clean predicted class. Fixed before execution; the full "
    "agreement distribution is reported descriptively, never re-cut.",
    "populations": "project validation only (official_train_pool/project_split==val, "
    "N=1000); clean predictions reused from frozen "
    "outputs/PBA-001/per_case_val.json; NO test contact",
    "baseline": "existing pathway detects a noisy error iff the noisy-eval "
    "prediction itself is uncertain under frozen T/taus",
    "report": [
        "noisy errors detected (flag rate among eval-noise errors)",
        "confident-but-wrong detected (flag rate among eval-noise errors that were clean-certain)",
        "uncertain cases generated (flag rate overall + clean)",
        "clean-validation false-alert rate (flag rate among clean-correct)",
        "class-wise behavior",
        "agreement/confidence distributions",
        "clean and perturbed confusion matrices",
    ],
    "success_criterion": "substantially better detection of σ=0.05-induced errors "
    "than the confidence-only mechanism, with small explicitly "
    "reported clean false-alert burden. Judged by PI; no "
    "single uncertainty score constructed.",
    "forbidden": [
        "retraining",
        "calibration refit",
        "threshold changes",
        "test contact",
        "post-hoc rule changes",
        "aggregate score",
    ],
}
(root / "outputs" / "UNC-001" / "design_freeze.json").write_text(json.dumps(freeze, indent=1))
print("UNC-001 design frozen: K=8, flag=agreement<1.0, val-only, no test")
