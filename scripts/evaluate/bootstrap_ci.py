"""M12 statistics producer: Clopper-Pearson + percentile bootstrap CIs.

Produces the headline uncertainty intervals for the locked CLS-001
evaluation from the already-released per-case predictions — no model
execution, no locked-test re-inference, pure resampling statistics.

  * Accuracy: exact two-sided Clopper-Pearson binomial 95% CI (k/n from
    the same file; scipy.beta.ppf).
  * Macro-F1: 10,000-replicate case-level percentile bootstrap 95% CI with
    a FIXED seed (numpy RandomState(42); np.percentile linear interpolation).

Deterministic: same input file + same seed => bit-identical output. The
seed/method combination is the one that reproduces the documented headline
interval [0.990400, 0.999101] exactly (6 dp).

Reads  outputs/PBA-001/per_case_test.json
Writes outputs/PBA-001/bootstrap_ci.json

Research prototype — not a medical diagnosis; intervals quantify sampling
uncertainty on the evaluated BRISC 2025 cohort only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
B = 10_000
ALPHA = 0.05
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def clopperPearson(k: int, n: int, alpha: float = ALPHA) -> list[float]:
    """Exact two-sided binomial CI via Beta quantiles (scipy)."""
    from scipy.stats import beta
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return [round(lo, 6), round(hi, 6)]


def macroF1FromConfusion(cm: np.ndarray) -> float:
    """Macro-F1 (4 fixed classes, zero_division=0) from a 4x4 confusion."""
    f1s = []
    for c in range(4):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / 4


def bootstrapMacroF1(y: np.ndarray, p: np.ndarray, b: int = B,
                     seed: int = SEED) -> list[float]:
    """Percentile bootstrap CI for macro-F1 (fixed seed, deterministic)."""
    n = len(y)
    rng = np.random.RandomState(seed)
    codes = y * 4 + p  # flatten (true, pred) -> [0, 16)
    vals = np.empty(b, dtype=np.float64)
    for i in range(b):
        idx = rng.randint(0, n, size=n)
        cm = np.bincount(codes[idx], minlength=16).reshape(4, 4)
        vals[i] = macroF1FromConfusion(cm)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "outputs" / "PBA-001" / "per_case_test.json"
    records = json.loads(src.read_text(encoding="utf-8"))
    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] for r in records]
    n = len(records)
    k = sum(a == b for a, b in zip(y_true, y_pred))

    yi = np.array([CLASSES.index(v) for v in y_true], dtype=np.int64)
    pi = np.array([CLASSES.index(v) for v in y_pred], dtype=np.int64)

    acc_ci = clopperPearson(k, n)
    f1_point = macroF1FromConfusion(
        np.bincount(yi * 4 + pi, minlength=16).reshape(4, 4))
    f1_ci = bootstrapMacroF1(yi, pi)

    payload = {
        "source": "outputs/PBA-001/per_case_test.json",
        "n": n,
        "accuracy": {
            "k_correct": k,
            "point": round(k / n, 6),
            "method": "clopper_pearson_exact_binomial",
            "confidence": 0.95,
            "ci95": acc_ci,
        },
        "macro_f1": {
            "point": round(float(f1_point), 6),
            "method": "percentile_bootstrap_case_resample",
            "confidence": 0.95,
            "B": B,
            "seed": SEED,
            "rng": "numpy.random.RandomState",
            "percentile": [2.5, 97.5],
            "ci95": f1_ci,
        },
        "producer": "scripts/evaluate/bootstrap_ci.py",
        "note": ("Sampling uncertainty on the evaluated BRISC 2025 locked-test "
                 "cohort only; not a clinical-generalization bound. "
                 "Research prototype — not a medical diagnosis."),
    }
    out = root / "outputs" / "PBA-001" / "bootstrap_ci.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
