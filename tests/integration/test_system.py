"""System-integration tests: endpoints, contracts, states, determinism.

Synthetic images only (no locked-test contact). Covers: /health, /classify,
/segment, /localize, /consistency, /analyze (+ consistency payload),
error envelopes (415/413/422), contract validation by construction,
localization invariants, and call-to-call determinism.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))

from brain_tumor.contracts import BrainTumorResult  # noqa: E402

_rng = np.random.RandomState(11)


def _png(array: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def noise_img():
    return _png(_rng.rand(256, 256) * 255)


@pytest.fixture(scope="module")
def gradient_img():
    g = np.tile(np.linspace(0, 255, 256, dtype=float), (256, 1))
    return _png(g)


@pytest.fixture(scope="module")
def client():
    import main as api_main
    return TestClient(api_main.app)


def _post(client, path, img: bytes, ctype="image/png"):
    return client.post(path, files={"file": ("t.png", img, ctype)})


def test_health_shape(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models"]["classifier"] == "loaded"
    assert body["models"]["segmenter"] == "loaded"
    assert body["status"] == "ok"
    assert body["calibration"]["T"] == pytest.approx(0.5116, abs=1e-3)


def test_classify_contract(client, noise_img):
    body = _post(client, "/classify", noise_img).json()
    assert set(body["probabilities"]) == {"glioma", "meningioma", "pituitary", "notumor"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["classification_state"] in ("confident", "uncertain")


def test_segment_localize_invariants(client, noise_img):
    seg = _post(client, "/segment", noise_img).json()
    assert seg["segmentation_state"] in ("empty", "nonempty")
    loc = {k: v for k, v in _post(client, "/localize", noise_img).json().items()
           if k != "disclaimer"}  # envelope notice is not part of the result
    if seg["segmentation_state"] == "empty":
        assert loc == {"bbox": None, "centroid": None, "area_pixels": 0}
    else:
        assert loc["area_pixels"] > 0
        x0, y0, x1, y1 = loc["bbox"]
        assert x1 > x0 and y1 > y0
        assert seg["localization"] == loc


def test_analyze_valid_result_and_states(client, noise_img, gradient_img):
    for img in (noise_img, gradient_img):
        body = _post(client, "/analyze", img).json()
        BrainTumorResult(**{k: v for k, v in body.items() if k != "consistency"})
        assert body["system_state"] in ("degraded", "uncertain", "tumor_unlocalized",
                                        "tumor_localized", "healthy")
        assert "consistency" in body
        assert body["consistency"]["k"] == 8


def test_analyze_determinism(client, noise_img):
    a = _post(client, "/analyze", noise_img).json()
    b = _post(client, "/analyze", noise_img).json()
    assert a == b


def test_consistency_batched_matches_sequential(noise_img):
    """Batching rung: identical seeds/rule must give identical decisions."""
    import io
    import torch
    import numpy as _np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from brain_tumor.inference import service as svc_mod
    svc = svc_mod.InferenceService.from_registry()
    base = Image.open(io.BytesIO(noise_img)).convert("RGB")
    got = svc.consistency(base)
    # Frozen sequential reference (pre-batching implementation).
    clean_pred = svc.classify(base)["predicted_class"]
    agree = 0
    with torch.no_grad():
        for kk in range(8):
            rng = _np.random.RandomState(7003 + kk)
            a = _np.asarray(base).astype(_np.float32) / 255.0
            a = _np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)
            noisy = Image.fromarray((a * 255).astype(_np.uint8))
            probs = torch.softmax(
                svc.clf(svc.cls_tf(noisy).unsqueeze(0)) / svc.T, dim=1)[0]
            if svc_mod.CLASSES[int(probs.argmax())] == clean_pred:
                agree += 1
    assert got["agreement_fraction"] == round(agree / 8, 4)
    assert got["flagged"] == bool(agree / 8 < 1.0)


def test_error_envelopes(client, noise_img):
    assert _post(client, "/analyze", noise_img, "text/plain").status_code == 415
    big = client.post("/analyze", files={"file": ("t.png", b"x" * (11 * 1024 * 1024),
                                                  "image/png")})
    assert big.status_code == 413
    bad = client.post("/analyze", files={"file": ("t.png", b"not-an-image",
                                                  "image/png")})
    assert bad.status_code == 422


def test_quality_endpoint(client, noise_img):
    ok = _post(client, "/quality", noise_img).json()
    assert ok["verdict"] == "accept" and ok["failed"] == []
    bad = _post(client, "/quality", b"not-an-image").json()
    assert bad["verdict"] == "reject" and "Q01_undecodable" in bad["failed"]


def test_healthy_path_no_validation_error():
    """Regression: notumor predictions must build valid empty localization
    (live-check caught bare LocalizationResult() on this path)."""
    import io
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from brain_tumor.inference import service as svc_mod
    svc = svc_mod.InferenceService.from_registry()
    arr = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)
    assert svc.classify(img)["predicted_class"] == "notumor"
    r = svc.analyze(img)
    assert r.system_state in ("healthy", "uncertain")
    assert r.localization.area_pixels == 0
    e = svc.explain(img)
    assert e["predicted_class"] == "notumor" and e["segmentation_vis"] is None


def test_degraded_paths_without_checkpoints(noise_img):
    """Missing weights: classifier raises, segmenter reports unavailable —
    with the FULL contract shape (localization + warnings), matching
    analyze()'s degraded path and the /segment endpoint contract."""
    import io
    from PIL import Image
    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from brain_tumor.contracts import LocalizationResult
    from brain_tumor.inference import service as svc_mod
    svc = svc_mod.InferenceService(classifier_ckpt=Path("none"), segmenter_ckpt=Path("none"))
    assert svc.segmentation_available is False
    assert svc.segment(Image.open(io.BytesIO(noise_img))) == {
        "segmentation_state": "unavailable",
        "localization": LocalizationResult(bbox=None, centroid=None, area_pixels=0),
        "warnings": ["segmentation_unavailable"]}
    with pytest.raises(RuntimeError, match="classifier_unavailable"):
        svc.classify(Image.open(io.BytesIO(noise_img)))
    with pytest.raises(RuntimeError, match="classifier_unavailable"):
        svc.consistency(Image.open(io.BytesIO(noise_img)))


def test_segment_endpoint_degraded_contract(noise_img):
    """H1 regression: /segment with unavailable segmenter must return 200
    with the frozen contract shape — never KeyError -> 500."""
    from unittest.mock import patch
    import main as api_main
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from brain_tumor.contracts import LocalizationResult
    from brain_tumor.inference import service as svc_mod
    svc = svc_mod.InferenceService(classifier_ckpt=Path("none"), segmenter_ckpt=Path("none"))
    client = TestClient(api_main.app)
    with patch.object(api_main, "_service", svc):
        r = _post(client, "/segment", noise_img)
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["segmentation_state"] == "unavailable"
    assert body["localization"] == {"bbox": None, "centroid": None, "area_pixels": 0}
    assert "segmentation_unavailable" in body["warnings"]
    assert "disclaimer" in body
    # And the payload is a constructible BrainTumorResult localization.
    LocalizationResult(**body["localization"])


def test_disclaimer_on_all_endpoints(noise_img):
    """H2 regression: every API payload must carry the non-diagnosis notice."""
    import main as api_main
    client = TestClient(api_main.app)
    for path in ("/health", "/classify", "/segment", "/localize", "/quality",
                 "/consistency", "/analyze", "/reliability", "/explain"):
        if path == "/health":
            r = client.get(path)
        else:
            r = _post(client, path, noise_img)
        assert r.status_code == 200, (path, r.status_code)
        body = r.json()
        assert "disclaimer" in body, path
        assert "not a medical diagnosis" in body["disclaimer"], path
    # Error envelopes stay machine-readable without the notice (documented).
    assert "disclaimer" not in _post(client, "/analyze", noise_img, "text/plain").json()


def test_truncated_image_is_422():
    """M8 regression: header-valid but truncated uploads must surface as 422
    at decode time, not as a 500 deep inside inference."""
    import main as api_main
    client = TestClient(api_main.app)
    truncated = _png(_rng.rand(256, 256) * 255)[:40]
    r = _post(client, "/classify", truncated)
    assert r.status_code == 422, r.status_code
