"""EXPL-001 verification: observer contract (identical outputs/state),
determinism, endpoint shape, error behavior. Synthetic images only.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))

from brain_tumor.contracts import BrainTumorResult  # noqa: E402

_rng = np.random.RandomState(31)


def _png(array: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def _img():
    from PIL import Image
    return Image.open(io.BytesIO(_png(_rng.rand(256, 256) * 255)))


def _svc():
    from brain_tumor.inference.service import InferenceService
    return InferenceService.from_registry()


def test_explain_preserves_inference_outputs():
    svc = _svc()
    img = _img()
    a = svc.analyze(img).model_dump()
    e = svc.explain(img)
    for k in ("predicted_class", "probabilities", "confidence",
              "classification_state", "segmentation_state", "localization",
              "warnings", "system_state"):
        assert e[k] == a[k], k
    BrainTumorResult(**{k: v for k, v in e.items()
                        if k in BrainTumorResult.model_fields})


def test_explain_deterministic_and_shaped():
    svc = _svc()
    img = _img()
    e1, e2 = svc.explain(img), svc.explain(img)
    assert e1["gradcam"]["heatmap_png_b64"] == e2["gradcam"]["heatmap_png_b64"]
    assert e1["gradcam"]["overlay_png_b64"] == e2["gradcam"]["overlay_png_b64"]
    assert e1["gradcam"]["target_layer"] == "features.7.2.block.0"
    raw = base64.b64decode(e1["gradcam"]["overlay_png_b64"])
    from PIL import Image
    ov = Image.open(io.BytesIO(raw))
    assert ov.size == img.size and ov.mode == "RGB"
    assert "not a clinical" in e1["report_text"]
    assert e1["segmentation_vis"] is None or "mask_png_b64" in e1["segmentation_vis"]


def test_explain_endpoint():
    import main as api_main
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    body = client.post("/explain",
                       files={"file": ("t.png", _png(_rng.rand(256, 256) * 255),
                                       "image/png")}).json()
    assert body["gradcam"]["target_layer"] == "features.7.2.block.0"
    assert body["consistency"]["k"] == 8
    assert "report_text" in body and "disclaimer" in body
    bad = client.post("/explain", files={"file": ("t.png", b"junk", "image/png")})
    assert bad.status_code == 422
