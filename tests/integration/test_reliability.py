"""REL-001 verification: observer equivalence + combination matrix.

Synthetic images only. No locked-test contact.
"""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brain_tumor.inference.service import InferenceService  # noqa: E402


def _img(array: np.ndarray):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf), buf.getvalue()


def _svc(**kw):
    root = Path(__file__).resolve().parents[2]
    if not kw:
        return InferenceService.from_registry()
    return InferenceService(
        classifier_ckpt=kw.get("clf", root / "checkpoints" / "CLS-001" / "best.pt"),
        segmenter_ckpt=kw.get("seg", root / "checkpoints" / "SEG-001" / "best.pt"),
    )


def test_observer_equivalence():
    svc = _svc()
    rng = np.random.RandomState(31)
    img, raw = _img(rng.rand(256, 256) * 255)
    before = svc.analyze(img).model_dump()
    rel = svc.reliability(img, raw=raw)
    after = svc.analyze(img).model_dump()
    assert before == after
    assert rel["system_state_unchanged"] == after["system_state"]
    assert rel["reliability"]["clinical_meaning"] is False
    assert rel["reliability"]["summary"] in ("stable", "review")


def test_tumor_path_shape():
    svc = _svc()
    rng = np.random.RandomState(31)
    img, raw = _img(rng.rand(256, 256) * 255)
    rel = svc.reliability(img, raw=raw)
    assert set(rel) >= {
        "classification",
        "consistency",
        "quality",
        "segmentation",
        "localization",
        "reliability",
    }
    assert isinstance(rel["reliability"]["basis"], list) and rel["reliability"]["basis"]


def test_notumor_and_empty_seg_paths():
    svc = _svc()
    checker = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype("uint8")
    img, raw = _img(checker)
    rel = svc.reliability(img, raw=raw)
    assert rel["classification"]["class"] == "notumor"
    dark, draw = _img(np.zeros((256, 256)))
    rel2 = svc.reliability(dark, raw=draw)
    assert rel2["classification"]["class"] == "glioma"
    assert rel2["segmentation"]["nonempty"] is False
    assert rel2["reliability"]["summary"] == "review"


def test_quality_reject_and_missing_capabilities():
    svc = _svc()
    blank, braw = _img(np.zeros((256, 256)))
    rel = svc.reliability(blank, raw=braw)
    assert rel["quality"]["state"] == "rejected"
    assert rel["reliability"]["summary"] == "review"
    noseg = _svc(seg=Path("none"))
    import pytest

    rng = np.random.RandomState(31)
    img, raw = _img(rng.rand(256, 256) * 255)
    rel_ns = noseg.reliability(img, raw=raw)
    assert rel_ns["localization"]["state"] == "unavailable"
    noclf = _svc(clf=Path("none"), seg=Path("none"))
    with pytest.raises(RuntimeError, match="classifier_unavailable"):
        noclf.reliability(img, raw=raw)


def test_determinism():
    svc = _svc()
    rng = np.random.RandomState(31)
    img, raw = _img(rng.rand(256, 256) * 255)
    assert svc.reliability(img, raw=raw) == svc.reliability(img, raw=raw)


def test_reliability_endpoint():
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))
    import main as api_main
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    rng = np.random.RandomState(31)
    body = client.post(
        "/reliability", files={"file": ("t.png", _img(rng.rand(256, 256) * 255)[1], "image/png")}
    ).json()
    assert body["reliability"]["summary"] in ("stable", "review")
    assert body["reliability"]["clinical_meaning"] is False
    body2 = client.post(
        "/analyze", files={"file": ("t.png", _img(rng.rand(128, 128) * 255)[1], "image/png")}
    ).json()
    assert "reliability" in body2
