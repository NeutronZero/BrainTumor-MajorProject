import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_contract_invariants_degraded():
    import pytest

    from brain_tumor.contracts import BrainTumorResult

    with pytest.raises(Exception):
        BrainTumorResult(
            predicted_class="glioma",
            probabilities={"glioma": 0.7, "meningioma": 0.1, "pituitary": 0.1, "notumor": 0.1},
            confidence=0.7,
            classification_state="confident",
            segmentation_state="unavailable",
            localization={"bbox": None, "centroid": None, "area_pixels": 0},
            warnings=[],
            system_state="degraded",
        )


def test_derive_precedence():
    from brain_tumor.contracts import derive_system_state

    assert (
        derive_system_state("glioma", "confident", "unavailable", ["segmentation_unavailable"])
        == "degraded"
    )
    assert derive_system_state("notumor", "uncertain", "empty", ["low_confidence"]) == "uncertain"


def test_derive_full_precedence_matrix():
    """M3 regression: every derivation result must be constructible as a
    BrainTumorResult (no dead/contradictory branches)."""
    import pytest

    from brain_tumor.contracts import BrainTumorResult

    loc_empty = {"bbox": None, "centroid": None, "area_pixels": 0}

    def _probs(top):
        return {
            c: (0.7 if c == top else 0.1) for c in ("glioma", "meningioma", "pituitary", "notumor")
        }

    def build(pred, cs, seg, warnings, ss, loc=loc_empty):
        return BrainTumorResult(
            predicted_class=pred,
            probabilities=_probs(pred),
            confidence=0.7,
            classification_state=cs,
            segmentation_state=seg,
            localization=loc,
            warnings=warnings,
            system_state=ss,
        )

    # All five derivations on constructible inputs.
    r = build("glioma", "confident", "unavailable", ["segmentation_unavailable"], "degraded")
    assert r.system_state == "degraded"
    r = build("notumor", "uncertain", "empty", ["low_confidence"], "uncertain")
    assert r.system_state == "uncertain"
    r = build("glioma", "confident", "empty", [], "tumor_unlocalized")
    assert r.system_state == "tumor_unlocalized"
    r = build(
        "glioma",
        "confident",
        "nonempty",
        [],
        "tumor_localized",
        loc={"bbox": (0, 0, 5, 5), "centroid": (2.0, 2.0), "area_pixels": 25},
    )
    assert r.system_state == "tumor_localized"
    r = build("notumor", "confident", "empty", [], "healthy")
    assert r.system_state == "healthy"
    # Contradictions must still raise.
    with pytest.raises(Exception):
        build("notumor", "confident", "empty", [], "tumor_localized")
    with pytest.raises(Exception):
        build("glioma", "confident", "empty", [], "healthy")
    with pytest.raises(Exception):
        build("glioma", "confident", "unavailable", ["segmentation_unavailable"], "tumor_localized")


def test_bt_result_argmax_confidence_invariant():
    """M1 regression: BrainTumorResult must enforce the same argmax/confidence
    guarantee as ClassificationResult."""
    import pytest

    from brain_tumor.contracts import BrainTumorResult

    base = {
        "probabilities": {"glioma": 0.7, "meningioma": 0.1, "pituitary": 0.1, "notumor": 0.1},
        "classification_state": "confident",
        "segmentation_state": "empty",
        "localization": {"bbox": None, "centroid": None, "area_pixels": 0},
        "warnings": [],
        "system_state": "tumor_unlocalized",
    }
    # confidence != max(probabilities) -> rejected
    with pytest.raises(Exception):
        BrainTumorResult(predicted_class="glioma", confidence=0.6, **base)
    # predicted_class != argmax -> rejected
    with pytest.raises(Exception):
        BrainTumorResult(predicted_class="meningioma", confidence=0.7, **base)
    # 6-dp wire rounding within tolerance -> accepted
    ok = BrainTumorResult(predicted_class="glioma", confidence=0.7, **base)
    assert ok.tumor_detected is True


def test_uncertain_system_state_requires_uncertain_classification():
    """M3: uncertain system_state cannot co-occur with confident classification."""
    import pytest

    from brain_tumor.contracts import BrainTumorResult

    with pytest.raises(Exception):
        BrainTumorResult(
            predicted_class="notumor",
            probabilities={"glioma": 0.0, "meningioma": 0.0, "pituitary": 0.0, "notumor": 1.0},
            confidence=1.0,
            classification_state="confident",
            segmentation_state="empty",
            localization={"bbox": None, "centroid": None, "area_pixels": 0},
            warnings=["low_confidence"],
            system_state="uncertain",
        )


def test_inference_offline():
    import io

    import numpy as np
    from PIL import Image

    from brain_tumor.inference.service import InferenceService

    rng = np.random.RandomState(3)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    svc = InferenceService.from_registry()
    res = svc.analyze(Image.open(buf))
    assert abs(sum(res.probabilities.values()) - 1.0) < 0.01
