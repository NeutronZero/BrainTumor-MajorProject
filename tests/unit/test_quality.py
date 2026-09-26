"""Quality-gate unit tests: coded checks on synthetic inputs (fast, no data)."""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.quality.gate import assess  # noqa: E402


def _png(array: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def test_accept_textured():
    rng = np.random.RandomState(4)
    a = assess(_png(rng.rand(256, 256) * 255))
    assert a["verdict"] == "accept" and a["failed"] == []


def test_reject_undecodable():
    a = assess(b"not an image at all")
    assert a["verdict"] == "reject"
    assert "Q01_undecodable" in a["failed"]
    assert "Q10_mime_mismatch" in a["failed"]


def test_reject_blank_uniform():
    assert assess(_png(np.zeros((256, 256))))["verdict"] == "reject"
    assert "Q05_blank_or_uniform" in assess(_png(np.full((256, 256), 128)))["failed"]


def test_reject_too_small():
    rng = np.random.RandomState(0)
    a = assess(_png(rng.rand(32, 32) * 255))
    assert a["verdict"] == "reject" and "Q03_too_small" in a["failed"]


def test_reject_oversize():
    a = assess(b"x" * (11 * 1024 * 1024))
    assert a["verdict"] == "reject" and a["failed"] == ["Q08_size_limit"]
