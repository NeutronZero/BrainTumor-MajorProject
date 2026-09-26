"""Regression: reliability engine must never crash on non-numeric confidence.

The pre-fix engine formatted confidence with :.4f unconditionally when
classification_state == "confident", raising TypeError on None. A confident
state paired with missing/non-numeric confidence is an inconsistent input:
it must degrade to "review" with a clear basis entry, not raise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.reliability.engine import reliability_report  # noqa: E402


def _base(conf, state="confident"):
    return dict(
        classification={
            "predicted_class": "glioma",
            "confidence": conf,
            "classification_state": state,
        },
        consistency={"k": 8, "agreement_fraction": 1.0, "flagged": False},
        quality={"verdict": "accept", "failed": [], "facts": {}},
        segmentation_state="nonempty",
        localization={"bbox": (0, 0, 10, 10), "centroid": (5.0, 5.0), "area_pixels": 50},
        system_state="tumor_localized",
    )


def test_none_confidence_with_confident_state_is_review():
    r = reliability_report(**_base(None))
    assert r["reliability"]["summary"] == "review"
    assert any("non-numeric" in b for b in r["reliability"]["basis"])
    assert r["reliability"]["clinical_meaning"] is False
    assert r["classification"]["confidence"] is None


def test_string_confidence_with_confident_state_is_review():
    r = reliability_report(**_base("0.99"))
    assert r["reliability"]["summary"] == "review"
    assert any("non-numeric" in b for b in r["reliability"]["basis"])


def test_bool_confidence_with_confident_state_is_review():
    # bool is an int subclass; True must not format as a calibrated confidence.
    r = reliability_report(**_base(True))
    assert r["reliability"]["summary"] == "review"
    assert any("non-numeric" in b for b in r["reliability"]["basis"])


def test_none_confidence_with_uncertain_state_keeps_original_basis():
    r = reliability_report(**_base(None, state="uncertain"))
    assert r["reliability"]["summary"] == "review"
    assert "classification uncertain — review" in r["reliability"]["basis"]


def test_numeric_confidence_still_stable_branch():
    r = reliability_report(**_base(0.9977))
    assert r["reliability"]["summary"] == "stable"
    assert any("calibrated confidence 0.9977" in b for b in r["reliability"]["basis"])
