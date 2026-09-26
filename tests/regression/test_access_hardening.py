"""Round-3 audit regression tests: bearer-token gate + per-client rate limit.

Contract (app/api/main.py AuthRateLimitMiddleware):
  - BT_API_TOKEN set  => inference endpoints require `Authorization: Bearer <t>`;
    /health and /metrics stay open (liveness/scraper paths never need creds).
  - BT_RATE_LIMIT/min per client (default 120) on inference endpoints; the
    limiter records a hit BEFORE the handler runs, so 415/422 validation
    failures count toward the window.
  - 401/429 short-circuit inside the middleware itself (never raised past
    BaseHTTPMiddleware, which would surface as 500), yet still flow through
    RequestIDMiddleware so each is exactly one INFERENCE_REQUESTS observation.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from starlette.requests import Request as _StarletteRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "api"))


def _png_bytes(size_px: int = 32) -> bytes:
    img = Image.new("RGB", (size_px, size_px), (40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    import main as api_main
    from fastapi.testclient import TestClient

    return TestClient(api_main.app)


@pytest.fixture()
def limiter():
    """The live AuthRateLimitMiddleware instance, with state save/restore.

    The app is a module-level singleton shared with the REL-002 tests, so any
    mutation here (shrunk window, seeded hits) MUST be rolled back or the
    limiter leaks into downstream tests (spurious 429s).
    """
    import main as api_main

    mw = _get_auth_middleware(api_main)
    saved_max = mw._MAX_REQUESTS
    saved_hits = {k: list(v) for k, v in mw._hits.items()}
    yield mw
    mw._MAX_REQUESTS = saved_max
    mw._hits.clear()
    mw._hits.update(saved_hits)


@pytest.fixture()
def token_env(monkeypatch):
    """Set a token for the duration of the test (main._API_TOKEN is read at
    request time, so monkeypatching the module attr is equivalent to the env)."""
    monkeypatch.setattr("main._API_TOKEN", "test-token-123")


def _get_auth_middleware(api_main):
    """Return the live AuthRateLimitMiddleware instance from the built stack."""
    node = api_main.app.middleware_stack
    while node is not None:
        if type(node).__name__ == "AuthRateLimitMiddleware":
            return node
        node = getattr(node, "app", None)
    raise RuntimeError("AuthRateLimitMiddleware instance not found in middleware stack")


def _fake_scope(path: str, client_ip: str, headers: list[tuple[bytes, bytes]] | None = None):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "query_string": b"",
        "client": (client_ip, 1234),
    }


def _gate_req(mw, path: str, ip: str, auth: str | None = None):
    """Drive the middleware's gate directly (no ASGI transport needed)."""
    headers = []
    if auth is not None:
        headers = [(b"authorization", auth.encode())]
    request = _StarletteRequest(_fake_scope(path, ip, headers))
    return mw._gate(request)


# ---- bearer token gate -------------------------------------------------------


def test_token_set_missing_header_is_401(client, token_env):
    r = client.post("/classify", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_token_set_wrong_token_is_401(client, token_env):
    r = client.post(
        "/classify",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 401


def test_token_set_correct_token_passes_gate(client, token_env):
    r = client.post(
        "/classify",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"Authorization": "Bearer test-token-123"},
    )
    assert r.status_code == 200  # got past auth; handler ran
    assert r.json()["predicted_class"]


def test_token_set_health_and_metrics_stay_open(client, token_env):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_bearer_scheme_is_case_insensitive(client, token_env):
    """RFC 7235: the auth scheme is case-insensitive."""
    r = client.post(
        "/classify",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"Authorization": "bearer test-token-123"},
    )
    assert r.status_code == 200


def test_wrong_token_attempts_are_throttled(client, limiter, token_env):
    """Rate-limiting runs BEFORE the token check, so bad-token attempts consume
    budget and token guessing is throttled instead of free."""
    limiter._hits.clear()  # isolate this client's window from earlier tests
    limiter._MAX_REQUESTS = 1
    first = client.post(
        "/classify",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"Authorization": "Bearer nope"},
    )
    assert first.status_code == 401
    second = client.post(
        "/classify",
        files={"file": ("t.png", _png_bytes(), "image/png")},
        headers={"Authorization": "Bearer nope"},
    )
    assert second.status_code == 429


def test_token_unset_gate_is_open(client):
    """Default behavior: no BT_API_TOKEN => no auth required (prototype mode)."""
    import main as api_main

    assert api_main._API_TOKEN is None
    r = client.post("/classify", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200


# ---- rate limit --------------------------------------------------------------


def test_rate_limit_trip_is_429(client, limiter):
    import main as api_main

    limiter._MAX_REQUESTS = 3
    for _ in range(3):
        _gate_req(limiter, "/classify", "10.0.0.1")  # first three hits admitted
    with pytest.raises(api_main._AuthError) as ei:
        _gate_req(limiter, "/classify", "10.0.0.1")
    assert ei.value.status == 429
    assert ei.value.payload["error"] == "rate_limited"


def test_rate_limit_counts_validation_failures(client, limiter):
    """Hits are recorded pre-handler: gate slots are consumed even when the
    eventual handler response would be 415/422."""
    import main as api_main

    limiter._MAX_REQUESTS = 2
    _gate_req(limiter, "/classify", "10.0.0.2")
    _gate_req(limiter, "/classify", "10.0.0.2")
    with pytest.raises(api_main._AuthError) as ei:
        _gate_req(limiter, "/classify", "10.0.0.2")
    assert ei.value.status == 429


def test_rate_limit_is_per_client(client, limiter):
    import main as api_main

    limiter._MAX_REQUESTS = 1
    _gate_req(limiter, "/classify", "10.0.1.1")
    with pytest.raises(api_main._AuthError):
        _gate_req(limiter, "/classify", "10.0.1.1")  # same client blocked
    _gate_req(limiter, "/classify", "10.0.1.2")  # different client admitted


def test_x_forwarded_for_takes_precedence(client, limiter, monkeypatch):
    """Behind a TRUSTED reverse proxy (BT_TRUST_PROXY=1), the limiter keys on
    the forwarded client."""
    import main as api_main

    monkeypatch.setenv("BT_TRUST_PROXY", "1")
    limiter._MAX_REQUESTS = 1
    headers = [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.9")]
    req_a = _StarletteRequest(_fake_scope("/classify", "10.0.0.9", headers))
    req_b = _StarletteRequest(_fake_scope("/classify", "10.0.0.9", headers))
    limiter._gate(req_a)
    with pytest.raises(api_main._AuthError):
        limiter._gate(req_b)  # same forwarded client => blocked
    assert limiter._client_key(req_a) == "203.0.113.7"


def test_x_forwarded_for_ignored_without_trusted_proxy(limiter, monkeypatch):
    """Default (BT_TRUST_PROXY unset): a spoofed XFF must NOT be trusted, or a
    direct caller could mint unlimited limiter buckets and bypass the limit."""
    monkeypatch.delenv("BT_TRUST_PROXY", raising=False)
    headers = [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.9")]
    req = _StarletteRequest(_fake_scope("/classify", "10.0.0.9", headers))
    assert limiter._client_key(req) == "10.0.0.9"


# ---- integration with the REL-002 accounting rule -----------------------------


def test_401_still_counted_as_one_observation(client, token_env):
    """401 short-circuits in Auth middleware (outer) but flows through
    RequestIDMiddleware (inner) => exactly one observation at 401."""
    import main as api_main

    def _samples(endpoint):
        return sum(
            s.value
            for s in api_main.INFERENCE_REQUESTS.collect()[0].samples
            if s.name == "inference_requests_total" and s.labels.get("endpoint") == endpoint
        )

    before = _samples("/classify")
    r = client.post("/classify", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 401
    assert r.headers.get("x-request-id")  # RequestIDMiddleware ran
    assert _samples("/classify") - before == 1


def test_auth_middleware_registered_before_requestid():
    """Ordering: RequestIDMiddleware must sit OUTSIDE Auth so 401/429 responses
    still carry a request ID and count as one observation. app.user_middleware
    lists registration order (Auth was added first); the built stack nests in
    reverse, i.e. Auth ends up innermost-but-outside-the-error-handler and
    RequestID outermost."""
    import main as api_main

    names = [m.cls.__name__ for m in api_main.app.user_middleware]
    assert names.index("AuthRateLimitMiddleware") > names.index("RequestIDMiddleware")
    stack_names = []
    node = api_main.app.middleware_stack
    while node is not None:
        stack_names.append(type(node).__name__)
        node = getattr(node, "app", None)
    assert stack_names.index("RequestIDMiddleware") < stack_names.index("AuthRateLimitMiddleware")


def test_health_and_metrics_bypass_the_gate():
    """/health and /metrics are outside the inference-path set, so no auth
    env var can lock out liveness probes or metric scrapers."""
    import main as api_main

    assert "/health" not in api_main.RequestIDMiddleware._INFERENCE_PATHS
    assert "/metrics" not in api_main.RequestIDMiddleware._INFERENCE_PATHS
