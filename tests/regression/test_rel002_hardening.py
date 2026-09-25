"""REL-002 regression tests: batch resource bounds + metric accounting.

Resource bounding: /analyze_batch must reject before unbounded memory use —
file-count and aggregate-byte gates fire pre-parse (no full-body read);
aggregate decoded-pixel gate fires during decode. Per-image limits are the
pre-existing _LIMIT/_MAX_PIXELS defenses and stay in force.

Metric accounting rule (REL-002): every inference endpoint reports exactly
ONE INFERENCE_REQUESTS observation per request, at its terminal status
(413/415/422 validation, 200 success, 500 failure).
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))


def _png_bytes(size_px: int = 32) -> bytes:
    img = Image.new("RGB", (size_px, size_px), (40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import main as api_main

    return TestClient(api_main.app)


def _post_batch(client, n, size_px=32, content_type="image/png"):
    data = _png_bytes(size_px)
    files = [("files", ("t.png", data, content_type)) for _ in range(n)]
    return client.post("/analyze_batch", files=files)


# ---- batch resource bounds ---------------------------------------------------


def test_batch_over_file_count_rejected_413_pre_parse(client, monkeypatch):
    """> max files => 413; gate fires before any body read."""
    import main as api_main

    def _boom(file):
        raise AssertionError("body must not be read when file-count gate trips")

    monkeypatch.setattr(api_main, "_read_bounded", _boom)
    r = _post_batch(client, api_main._MAX_BATCH_FILES + 1)
    assert r.status_code == 413
    assert r.json()["error"] == "too_many_files"


def test_batch_at_file_count_limit_passes_validation(client):
    """Exactly max files must NOT trip the count gate (batch proceeds)."""
    import main as api_main

    r = _post_batch(client, api_main._MAX_BATCH_FILES)
    assert r.status_code == 413 or r.status_code == 200
    if r.status_code == 413:
        assert r.json()["error"] != "too_many_files"


def test_batch_aggregate_bytes_rejected_413_pre_parse(client, monkeypatch):
    """Aggregate declared-size gate fires before full-body reads."""
    import main as api_main

    def _boom(file):
        raise AssertionError("body must not be read when aggregate-size gate trips")

    monkeypatch.setattr(api_main, "_read_bounded", _boom)
    limit = api_main._MAX_BATCH_AGGREGATE_BYTES
    n = 2

    async def _fake_size(f):
        # Each part individually declared over half the aggregate cap, so any
        # two together exceed it while each stays under the per-file limit.
        return limit // 2 + 1

    monkeypatch.setattr(api_main, "_upload_size", _fake_size)
    r = _post_batch(client, n)
    assert r.status_code == 413
    assert r.json()["error"] == "aggregate_too_large"


def test_batch_aggregate_pixels_rejected_during_decode(client, monkeypatch):
    """Aggregate decoded-pixel gate trips during decode, not after inference."""

    import main as api_main

    calls = {"analyze_batch": 0}
    original = api_main._service.analyze_batch

    def _spy(*a, **kw):
        calls["analyze_batch"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(api_main._service, "analyze_batch", _spy)
    # Force the per-image decode to yield a large image so the aggregate
    # pixel budget trips while file/byte gates still pass.
    monkeypatch.setattr(
        api_main, "_decode", lambda data: (Image.new("RGB", (4096, 4096)), None)
    )
    r = _post_batch(client, 4)
    assert r.status_code == 413
    assert r.json()["error"] == "aggregate_too_large"
    assert calls["analyze_batch"] == 0  # rejected before any inference work


def test_batch_bounds_are_frozen_contract_values():
    """Bounds are deliberate, non-zero, and tighter than naive per-file sums:
    the aggregate-byte cap is a real resource ceiling, not files x _LIMIT."""
    import main as api_main

    assert 1 <= api_main._MAX_BATCH_FILES <= 16
    assert 0 < api_main._MAX_BATCH_AGGREGATE_BYTES
    assert 0 < api_main._MAX_BATCH_AGGREGATE_PIXELS
    # The aggregate cap must actually bound: smaller than files x per-file limit.
    assert api_main._MAX_BATCH_AGGREGATE_BYTES < api_main._MAX_BATCH_FILES * api_main._LIMIT


# ---- metric accounting rule --------------------------------------------------


_INFERENCE_ENDPOINTS = [
    "/classify",
    "/segment",
    "/localize",
    "/quality",
    "/consistency",
    "/analyze",
    "/reliability",
    "/explain",
    "/analyze_batch",
]


def test_all_inference_endpoints_accounted_once_on_rejection(client):
    """Terminal validation status => exactly ONE INFERENCE_REQUESTS observation."""
    from prometheus_client import REGISTRY

    import main as api_main

    def _samples(endpoint):
        return sum(
            s.value
            for s in api_main.INFERENCE_REQUESTS.collect()[0].samples
            if s.name == "inference_requests_total"
            and s.labels.get("endpoint") == endpoint
        )

    for endpoint in _INFERENCE_ENDPOINTS:
        before = _samples(endpoint)
        if endpoint == "/analyze_batch":
            r = _post_batch(client, api_main._MAX_BATCH_FILES + 1)
            expected_status = 413  # file-count gate is the terminal status
        else:
            r = client.post(
                endpoint, files={"file": ("t.png", b"x", "text/plain")}
            )
            expected_status = 415
        assert r.status_code == expected_status, endpoint
        after = _samples(endpoint)
        assert after - before == 1, f"{endpoint}: expected 1 observation, got {after - before}"


def test_metric_accounting_helper_is_the_only_direct_labeler():
    """Endpoints must not touch INFERENCE_REQUESTS directly (REL-002 rule):
    .labels() may appear only inside _count_request's own body."""
    source = Path(__file__).resolve().parents[2] / "app" / "api" / "main.py"
    lines = source.read_text(encoding="utf-8").splitlines()
    helper_start = next(
        i for i, ln in enumerate(lines) if "def _count_request(" in ln
    )
    helper_end = next(
        i
        for i in range(helper_start + 1, len(lines))
        if lines[i].startswith("def ") or lines[i].startswith("@app.")
    )
    offenders = [
        f"line {i + 1}: {lines[i].strip()}"
        for i, ln in enumerate(lines)
        if "INFERENCE_REQUESTS.labels(" in ln and not (helper_start < i < helper_end)
    ]
    assert offenders == [], (
        "INFERENCE_REQUESTS must only be labeled inside _count_request; "
        f"offending lines: {offenders}"
    )
    assert "_count_request" in "\n".join(lines)
