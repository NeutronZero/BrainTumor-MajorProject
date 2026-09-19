"""Streamlit payload path: bytes reach the service; all display fields present.

Synthetic images only. UI rendering itself is covered by the headless boot
smoke (server starts, models load); here we verify build_payload contracts:
field completeness, empty-seg branch data, error envelope.
"""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "streamlit"))

import app as streamlit_app  # noqa: E402
from brain_tumor.inference.service import InferenceService  # noqa: E402

_rng = np.random.RandomState(21)


def _png(array: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def _svc():
    return InferenceService.from_registry()


def test_payload_complete_fields():
    img = _png(_rng.rand(256, 256) * 255)
    p = streamlit_app.build_payload(_svc(), img)
    assert "error" not in p
    for k in ("predicted_class", "confidence", "classification_state",
              "system_state", "probabilities", "segmentation_state",
              "localization", "warnings", "consistency"):
        assert k in p, k
    assert p["consistency"]["k"] == 8
    if p["segmentation_state"] == "empty":
        assert p["localization"]["area_pixels"] == 0
    else:
        assert p["localization"]["area_pixels"] > 0


def test_payload_error_envelope():
    svc = _svc()
    assert streamlit_app.build_payload(svc, b"x" * (11 * 1024 * 1024))["error"] == "file_too_large"
    assert streamlit_app.build_payload(svc, b"not-an-image")["error"] == "undecodable_image"


def test_render_branches():
    class _St:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def _rec(*a, **k):
                self.calls.append(name)
                class _Ctx:
                    def __enter__(self_inner):
                        return None

                    def __exit__(self_inner, *e):
                        return False
                return _Ctx()
            return _rec

    st = _St()
    streamlit_app.render_result(st, {"error": "inference_failed", "detail": "X"})
    assert "error" in st.calls
    st2 = _St()
    streamlit_app.render_result(st2, {
        "predicted_class": "notumor", "confidence": 0.99,
        "classification_state": "confident", "system_state": "healthy",
        "probabilities": {"glioma": 0.0, "meningioma": 0.0,
                          "pituitary": 0.01, "notumor": 0.99},
        "segmentation_state": "empty",
        "localization": {"bbox": None, "centroid": None, "area_pixels": 0},
        "warnings": [], "consistency": {"agreement_fraction": 1.0,
                                        "flagged": False, "k": 8, "sigma": 0.05}})
    assert "info" in st2.calls  # empty-seg branch renders
