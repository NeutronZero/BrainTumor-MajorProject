"""Regression: /analyze ~3x redundancy fix — threaded precomputed outputs
must be bit-identical to the original compute-everything-every-time paths.

Synthetic images only; no locked-test contact.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "streamlit"))

from brain_tumor.inference.service import InferenceService  # noqa: E402

_rng = np.random.RandomState(77)


def _png(array):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def _img_and_raw():
    raw = _png(_rng.rand(256, 256) * 255)
    from PIL import Image

    return Image.open(io.BytesIO(raw)), raw


@pytest.fixture(scope="module")
def svc():
    return InferenceService.from_registry()


def test_consistency_clean_pred_matches_internal(svc):
    """clean_pred threading skips classify() but yields the identical dict."""
    img, _ = _img_and_raw()
    default = svc.consistency(img)
    clean = svc.classify(img)["predicted_class"]
    threaded = svc.consistency(img, clean_pred=clean)
    assert threaded == default
    assert threaded["clean_pred"] == clean


def test_reliability_threaded_args_match_default(svc):
    """reliability(result=..., con=..., qual=...) == default reliability()."""
    img, raw = _img_and_raw()
    default = svc.reliability(img, raw=raw)
    result = svc.analyze(img)
    con = svc.consistency(img, clean_pred=result.predicted_class)
    qual = svc.quality(raw)
    threaded = svc.reliability(img, raw=raw, result=result, con=con, qual=qual)
    assert threaded == default


def test_api_analyze_payload_bit_identical(svc):
    """API /analyze (threaded) == manual composition of the same stages."""
    import json

    import main as api_main
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    img, raw = _img_and_raw()
    body = client.post("/analyze", files={"file": ("t.png", raw, "image/png")}).json()

    result = svc.analyze(img)
    manual = result.model_dump()
    manual["quality"] = svc.quality(raw)
    manual["consistency"] = svc.consistency(img, clean_pred=result.predicted_class)
    manual["reliability"] = svc.reliability(
        img, raw=raw, result=result, con=manual["consistency"], qual=manual["quality"]
    )
    manual["disclaimer"] = body["disclaimer"]
    # JSON round-trip: HTTP payloads carry lists where model_dump has tuples.
    assert body == json.loads(json.dumps(manual))


def test_api_analyze_without_probes_keeps_reliability_internal():
    """consistency_probes=false: no top-level consistency, reliability still
    computed internally (con=None path) — matches pre-threading behavior."""
    import main as api_main
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    _, raw = _img_and_raw()
    body = client.post(
        "/analyze",
        files={"file": ("t.png", raw, "image/png")},
        params={"consistency_probes": False},
    ).json()
    # Absent entirely (not null): response_model must not change the wire.
    assert "consistency" not in body
    assert body["reliability"]["consistency"]["k"] == 8
    assert body["reliability"]["system_state_unchanged"] == body["system_state"]


def test_streamlit_payload_bit_identical(svc):
    """build_payload (threaded) == manual composition of the same stages."""
    import app as streamlit_app

    img, raw = _img_and_raw()
    payload = streamlit_app.build_payload(svc, raw)

    result = svc.analyze(img)
    manual = result.model_dump()
    manual["quality"] = svc.quality(raw)
    manual["consistency"] = svc.consistency(img, clean_pred=result.predicted_class)
    manual["reliability"] = svc.reliability(
        img, raw=raw, result=result, con=manual["consistency"], qual=manual["quality"]
    )
    assert payload == manual


def test_explain_threaded_result_con_match_default(svc):
    """explain(result=..., con=...) == default explain() on identical input."""
    img, raw = _img_and_raw()
    default = svc.explain(img, raw=raw)
    result = svc.analyze(img)
    con = svc.consistency(img, clean_pred=result.predicted_class)
    threaded = svc.explain(img, raw=raw, result=result, con=con)
    assert threaded == default
