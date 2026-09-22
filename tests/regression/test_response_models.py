"""Regression: all 9 FastAPI endpoints must declare a response_model.

Response models (extra="allow", disclaimer declared) give the OpenAPI schema
real payload shapes without constraining field drift. Error envelopes are
JSONResponse (bypass response_model) — covered by test_error_envelopes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))

EXPECTED = {
    ("GET", "/health"),
    ("POST", "/classify"),
    ("POST", "/segment"),
    ("POST", "/localize"),
    ("POST", "/quality"),
    ("POST", "/consistency"),
    ("POST", "/analyze"),
    ("POST", "/reliability"),
    ("POST", "/explain"),
}


def test_all_nine_endpoints_declare_response_model():
    import main as api_main
    from fastapi.routing import APIRoute
    seen = {}
    for r in api_main.app.routes:
        if isinstance(r, APIRoute):
            for m in r.methods - {"HEAD", "OPTIONS"}:
                seen[(m, r.path)] = r.response_model
    for key in EXPECTED:
        assert key in seen, f"route missing: {key}"
        assert seen[key] is not None, f"no response_model on {key}"
    # Every response model must declare the disclaimer field (H2).
    for key in EXPECTED:
        model = seen[key]
        assert "disclaimer" in model.model_fields, f"{key} missing disclaimer"


def test_analyze_response_model_allows_absent_consistency():
    """consistency_probes=false omits top-level consistency — must be optional
    AND serialize as absent (not null) to preserve the wire shape."""
    import main as api_main
    from fastapi.routing import APIRoute
    analyze = next(r for r in api_main.app.routes
                   if isinstance(r, APIRoute) and r.path == "/analyze")
    model = analyze.response_model
    field = model.model_fields["consistency"]
    assert field.is_required() is False
    # round-trip: dict without consistency must stay without consistency
    dumped = model.model_validate({"predicted_class": "notumor",
                                   "probabilities": {"glioma": 0.0,
                                                     "meningioma": 0.0,
                                                     "pituitary": 0.0,
                                                     "notumor": 1.0},
                                   "confidence": 1.0,
                                   "classification_state": "confident",
                                   "segmentation_state": "empty",
                                   "localization": {"bbox": None,
                                                    "centroid": None,
                                                    "area_pixels": 0},
                                   "warnings": [],
                                   "system_state": "healthy",
                                   "quality": {"verdict": "accept"},
                                   "reliability": {},
                                   "disclaimer": "x"}).model_dump()
    assert "consistency" not in dumped


def test_error_envelope_bypasses_response_model():
    """415 envelope carries no disclaimer and is not validated/filtered."""
    import main as api_main
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    r = client.post("/classify", files={"file": ("t.png", b"x", "text/plain")})
    assert r.status_code == 415
    body = r.json()
    assert body["error"] == "unsupported_type"
    assert "disclaimer" not in body
