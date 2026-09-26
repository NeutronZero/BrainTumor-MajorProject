"""Unit tests for validation-only calibration (temperature + tau selection).

Covers the statistically load-bearing module that previously had zero test
coverage (audit finding M2): fit_temperature, ece, select_taus, np_arange.
Synthetic tensors only — no data, no model, no locked-test contact.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification.calibrate import (  # noqa: E402
    ece,
    fit_temperature,
    np_arange,
    select_taus,
)


def test_np_arange_covers_grid_endpoints():
    vals = np_arange(0.5, 0.95, 0.05)
    assert len(vals) == 10
    assert vals[0] == 0.5
    assert abs(vals[-1] - 0.95) < 1e-9
    assert np_arange(0.05, 0.40, 0.05)[0] == 0.05
    assert np_arange(1.0, 0.5, 0.05) == []  # empty when start >= stop


def test_fit_temperature_in_plausible_range_and_deterministic():
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(256, 4, generator=g) * 3.0
    labels = logits.argmax(dim=1)
    t1 = fit_temperature(logits, labels)
    t2 = fit_temperature(logits, labels)
    assert 0.05 < t1 < 20.0
    assert t1 == t2  # no RNG in the LBFGS path


def test_fit_temperature_never_raises_on_degenerate_input():
    # All-identical logits give a flat loss surface; the fail-safe must return
    # a valid temperature (T=1.0 fallback) rather than abort the run.
    logits = torch.zeros(8, 4)
    labels = torch.zeros(8, dtype=torch.long)
    t = fit_temperature(logits, labels)
    assert 0.05 < t < 20.0


def test_ece_is_zero_for_confident_correct():
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([2, 0])
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_overconfident_errors():
    probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    assert ece(probs, labels) > 0.0


def test_select_taus_returns_grid_values_and_criterion():
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(300, 4, generator=g) * 2.0 + 1.0
    # one-hot-ish probs so argmax is unambiguous
    probs = torch.softmax(logits, dim=1)
    labels = probs.argmax(dim=1)
    out = select_taus(probs, labels)
    assert out["criterion"] == "val_macro_f1_minus_half_uncertain_rate"
    assert 0.5 <= out["tau1"] <= 0.95
    assert 0.05 <= out["tau2"] <= 0.40
    assert isinstance(out["score"], float)
