"""REL-001 engine rule tests: stable/review branches on fabricated inputs.

Pure-function tests (no images, no models). The stable fixture mirrors the
live meningioma observation; the review fixture mirrors the empty-seg case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.reliability.engine import reliability_report  # noqa: E402


def test_stable_all_good():
    r = reliability_report(
        classification={
            "predicted_class": "meningioma",
            "confidence": 0.9977,
            "classification_state": "confident",
        },
        consistency={"k": 8, "agreement_fraction": 1.0, "flagged": False},
        quality={"verdict": "accept", "failed": [], "facts": {}},
        segmentation_state="nonempty",
        localization={"bbox": (82, 183, 170, 264), "centroid": (123.1, 223.2), "area_pixels": 1835},
        system_state="tumor_localized",
    )
    assert r["reliability"]["summary"] == "stable"
    assert r["reliability"]["clinical_meaning"] is False
    assert len(r["reliability"]["basis"]) == 4
    assert r["system_state_unchanged"] == "tumor_localized"


def test_review_each_failure_mode():
    base = dict(
        classification={
            "predicted_class": "glioma",
            "confidence": 0.99,
            "classification_state": "confident",
        },
        consistency={"k": 8, "agreement_fraction": 1.0, "flagged": False},
        quality={"verdict": "accept", "failed": [], "facts": {}},
        segmentation_state="nonempty",
        localization={"bbox": (0, 0, 10, 10), "centroid": (5.0, 5.0), "area_pixels": 50},
        system_state="tumor_localized",
    )
    import copy

    cases = [
        (
            "classification",
            {"predicted_class": "glioma", "confidence": 0.5, "classification_state": "uncertain"},
        ),
        ("consistency", {"k": 8, "agreement_fraction": 0.875, "flagged": True}),
        ("quality", {"verdict": "reject", "failed": ["Q05_blank_or_uniform"], "facts": {}}),
        ("segmentation_state", "empty"),
    ]
    for key, val in cases:
        kw = copy.deepcopy(base)
        kw[key] = val
        if key == "segmentation_state":
            kw["localization"] = {"bbox": None, "centroid": None, "area_pixels": 0}
        r = reliability_report(**kw)
        assert r["reliability"]["summary"] == "review", key
