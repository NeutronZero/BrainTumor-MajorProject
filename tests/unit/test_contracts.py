import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_contract_invariants_degraded():
    from brain_tumor.contracts import BrainTumorResult
    import pytest
    with pytest.raises(Exception):
        BrainTumorResult(predicted_class="glioma",
                         probabilities={"glioma": 0.7, "meningioma": 0.1, "pituitary": 0.1, "notumor": 0.1},
                         confidence=0.7, classification_state="confident",
                         segmentation_state="unavailable",
                         localization={"bbox": None, "centroid": None, "area_pixels": 0},
                         warnings=[], system_state="degraded")


def test_derive_precedence():
    from brain_tumor.contracts import derive_system_state
    assert derive_system_state("glioma", "confident", "unavailable", ["segmentation_unavailable"]) == "degraded"
    assert derive_system_state("notumor", "uncertain", "empty", ["low_confidence"]) == "uncertain"


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
