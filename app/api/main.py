"""Thin FastAPI wrapper over the single InferenceService (§19). Offline-first.

Every success payload carries a `disclaimer` field: engineering prototype,
not a medical diagnosis (project instruction: API payloads must disclaim
medical diagnosis). Error envelopes (415/413/422/500) are machine-readable
and carry no result payload. Uploads are size-checked BEFORE any full-body
decode, truncated/truncated-claim bodies surface as 422, and PIL's
decompression-bomb guard is set explicitly (M7/M8 hardening).
MIME sniffing via filetype provides defense-in-depth against content-type spoofing.

REL-002 hardening: /analyze_batch is resource-bounded (max files, aggregate
bytes, aggregate decoded pixels — all rejected before full-body parse), and
inference-request metrics flow through one accounting helper so every
inference endpoint reports exactly one observation per request.

Round-3 audit hardening: inference endpoints are gated by an optional
bearer token (env BT_API_TOKEN; unset = open prototype behavior) and a
fixed-window per-client rate limit (BT_RATE_LIMIT/min, default 120),
because /analyze costs ~11 forward passes and the container binds 0.0.0.0.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import filetype
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import ImageFile
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Harden PIL: disable truncated image loading, decompression bomb guard
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ---- Prometheus metrics -------------------------------------------------------
INFERENCE_REQUESTS = Counter(
    "inference_requests_total", "Total inference requests", ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Inference latency in seconds", ["stage"]
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"]
)


def _count_request(endpoint: str, status: int) -> None:
    """Single accounting point for INFERENCE_REQUESTS (REL-002).

    Called ONLY by RequestIDMiddleware at the terminal HTTP status, so the
    "exactly one observation per request" rule holds for every inference
    response — including FastAPI validation rejections (422) that never
    reach a handler, and pre-parse batch rejections (round-2 audit fix;
    the in-handler approach left those uncounted). /health and /metrics
    are not inference endpoints and are deliberately excluded.
    """
    INFERENCE_REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


# ---- access hardening (round-3 audit) -----------------------------------------
# Bearer-token gate: set BT_API_TOKEN to require `Authorization: Bearer <token>`
# on every inference endpoint. Unset (default) keeps the open research-
# prototype behavior; /health and /metrics stay open either way so liveness
# probes and metric scrapers never need credentials. Comparison is a plain
# string compare — this token gates a prototype API surface, it is not a
# secret-keeping credential store; do not reuse it for anything sensitive.
_API_TOKEN = os.environ.get("BT_API_TOKEN") or None


class _AuthError(Exception):
    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate + fixed-window per-client rate limit on inference
    endpoints (round-3 audit: the API shipped unauthenticated and unthrottled
    while /analyze runs classify + segment + 8-probe consistency + quality +
    reliability per request).

    Scope: exactly RequestIDMiddleware._INFERENCE_PATHS — /health, /metrics,
    /docs and /openapi.json pass through untouched. State is in-memory and
    per-process by design (single-container deployment profile; a multi-
    replica deployment should front this with a shared limiter).
    """

    _MAX_REQUESTS = int(os.environ.get("BT_RATE_LIMIT", "120"))
    _WINDOW_SECONDS = 60.0
    # Key-cap: an attacker spoofing unique XFF IPs must not grow _hits
    # unboundedly (memory DoS). Past the cap, expired keys are pruned; if
    # still over, the table resets (worst case: a brief grace window).
    _MAX_CLIENTS = 10_000

    def __init__(self, app: Any):
        super().__init__(app)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _client_key(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _gate(self, request: Request) -> None:
        """Raise _AuthError on missing token or exhausted window; else record hit."""
        if _API_TOKEN is not None:
            header = request.headers.get("authorization", "")
            if header != f"Bearer {_API_TOKEN}":
                raise _AuthError(
                    401,
                    {"error": "unauthorized", "detail": "missing or invalid bearer token"},
                )
        key = self._client_key(request)
        now = time.perf_counter()
        with self._lock:
            if len(self._hits) > self._MAX_CLIENTS:
                self._hits = {
                    k: [t for t in v if now - t < self._WINDOW_SECONDS]
                    for k, v in self._hits.items()
                    if any(now - t < self._WINDOW_SECONDS for t in v)
                }
                if len(self._hits) > self._MAX_CLIENTS:
                    self._hits.clear()
            window = [t for t in self._hits.get(key, []) if now - t < self._WINDOW_SECONDS]
            if len(window) >= self._MAX_REQUESTS:
                raise _AuthError(
                    429,
                    {
                        "error": "rate_limited",
                        "detail": f"max {self._MAX_REQUESTS} inference requests per "
                        f"{int(self._WINDOW_SECONDS)}s window",
                    },
                )
            window.append(now)
            self._hits[key] = window

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in RequestIDMiddleware._INFERENCE_PATHS:
            return await call_next(request)
        try:
            self._gate(request)
        except _AuthError as e:
            # Convert to the envelope HERE (never raise past BaseHTTPMiddleware:
            # it would surface as 500 from ServerErrorMiddleware). The response
            # flows through RequestIDMiddleware, which counts it as exactly one
            # INFERENCE_REQUESTS observation at 401/429.
            return JSONResponse(e.payload, status_code=e.status)
        return await call_next(request)


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.inference.service import (  # noqa: E402
    InferenceService,
    _request_id_var,
    set_request_id,
)

# ---- request ID propagation ---------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Extract or generate request ID, propagate via context var and response header."""

    # Inference endpoints only (REL-002 accounting rule); /health, /metrics,
    # /docs, /openapi.json are excluded.
    _INFERENCE_PATHS = {
        "/classify",
        "/segment",
        "/localize",
        "/quality",
        "/consistency",
        "/analyze",
        "/analyze_batch",
        "/reliability",
        "/explain",
    }

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or set_request_id()
        token = _request_id_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)
        # REL-002 metric accounting (round-2 audit fix): one observation per
        # inference request at its terminal HTTP status, counted at the
        # middleware layer so validation rejections (422) are included and
        # no handler can forget an exit path. Handlers never touch the
        # counter directly.
        if request.url.path in self._INFERENCE_PATHS:
            _count_request(request.url.path, response.status_code)
        response.headers["x-request-id"] = req_id
        return response


# ---- integrity verification --------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTEGRITY_FILES = {
    "clf_checkpoint": "checkpoints/CLS-001/best.pt",
    "seg_checkpoint": "checkpoints/SEG-001/best.pt",
    "clf_calibration": "outputs/CLS-001/calibration_frozen.json",
    "seg_norm": "outputs/SEG-001/metrics.json",
    "cls_config": "configs/experiment/CLS-001.yaml",
    "seg_config": "configs/experiment/SEG-001.yaml",
}


def _sha256_file(path: Path) -> str | None:
    """Return SHA256 hex digest or None if file missing."""
    full = _PROJECT_ROOT / path
    if not full.exists():
        return None
    h = hashlib.sha256()
    with full.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _integrity_report() -> dict[str, str | None]:
    """Compute integrity hashes for all tracked artifacts."""
    return {k: _sha256_file(v) for k, v in _INTEGRITY_FILES.items()}


# /health integrity cache (round-2 audit fix): the tracked artifacts total
# ~600MB; rehashing them on every probe makes /health unusable as a liveness
# check and invites over-polling. Hashes are content-addressed at rest, so a
# short TTL only delays detection of out-of-band tampering — full rehash
# remains available via scripts/release/build_manifest.py.
_INTEGRITY_TTL_SECONDS = 300.0
_INTEGRITY_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
# /health runs in Starlette's sync threadpool — the cache dict is touched from
# multiple threads (round-3 audit). Holding the lock across a rehash also
# collapses concurrent probes into one 600MB rehash instead of N.
_INTEGRITY_CACHE_LOCK = threading.Lock()


def _integrity_report_cached() -> dict[str, str | None]:
    """TTL-cached integrity report for /health (300s)."""
    with _INTEGRITY_CACHE_LOCK:
        now = time.time()
        if (
            _INTEGRITY_CACHE["value"] is None
            or now - _INTEGRITY_CACHE["at"] >= _INTEGRITY_TTL_SECONDS
        ):
            _INTEGRITY_CACHE["value"] = _integrity_report()
            _INTEGRITY_CACHE["at"] = now
        return _INTEGRITY_CACHE["value"]


API_DISCLAIMER = (
    "Engineering prototype output — not a medical diagnosis. "
    "Not a certified medical device; never use for clinical "
    "diagnosis, triage, or therapy planning. Research use only "
    "(retrospective BRISC 2025 data)."
)


# ---- response models (one per endpoint) -----------------------------------
# Flexible by design: extra="allow" preserves every payload field, `disclaimer`
# is declared on all of them (H2: success payloads must carry the notice), and
# fields that vary by branch (e.g. /analyze without consistency_probes) are
# optional. Nested blobs are typed Any so response validation can never turn a
# schema drift into a 500. Error envelopes (415/413/422/500) are returned as
# JSONResponse directly, which bypasses response_model entirely.
class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow")


class HealthPayload(_Payload):
    status: str
    models: dict[str, str]
    calibration: dict[str, Any] | str
    version: str
    disclaimer: str


class ClassifyPayload(_Payload):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    classification_state: str
    disclaimer: str


class SegmentPayload(_Payload):
    segmentation_state: str
    localization: dict[str, Any]
    warnings: list[str]
    disclaimer: str


class LocalizePayload(_Payload):
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    area_pixels: int
    disclaimer: str


class QualityPayload(_Payload):
    verdict: str
    failed: list[str]
    facts: dict[str, Any]
    disclaimer: str


class ConsistencyPayload(_Payload):
    k: int
    sigma: float
    clean_pred: str
    agreement_fraction: float
    flagged: bool
    disclaimer: str


class AnalyzePayload(_Payload):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    classification_state: str
    segmentation_state: str
    localization: dict[str, Any]
    warnings: list[str]
    system_state: str
    quality: dict[str, Any]
    # Absent (not null) when consistency_probes=false — exclude_if preserves
    # the pre-response_model wire shape exactly; OpenAPI still documents it.
    consistency: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)
    reliability: dict[str, Any]
    disclaimer: str


class ReliabilityPayload(_Payload):
    classification: dict[str, Any]
    consistency: dict[str, Any]
    quality: dict[str, Any]
    segmentation: dict[str, Any]
    localization: dict[str, Any]
    reliability: dict[str, Any]
    system_state_unchanged: str
    disclaimer: str


class ExplainPayload(_Payload):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    classification_state: str
    segmentation_state: str
    localization: dict[str, Any]
    warnings: list[str]
    system_state: str
    consistency: dict[str, Any]
    # Present when raw bytes supplied; absent otherwise (service omits the key).
    quality: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)
    gradcam: dict[str, Any]
    segmentation_vis: dict[str, Any] | None
    report_text: str
    disclaimer: str


app = FastAPI(
    title="BrainTumor-MajorProject",
    version="0.1.0",
    description=API_DISCLAIMER + " Every inference payload embeds this notice in `disclaimer`.",
)
# Auth/rate-limit registered FIRST so RequestIDMiddleware sits outermost:
# 401/429 short-circuit responses still get a request ID and still count as
# exactly one INFERENCE_REQUESTS observation at their terminal status.
app.add_middleware(AuthRateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.middleware("http")
async def _request_latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.perf_counter() - start)
    return response


_service = InferenceService.from_registry()

_ALLOWED = ("image/jpeg", "image/png", "image/bmp", "image/tiff")
_LIMIT = 10 * 1024 * 1024
# Decompression-bomb guard: project inputs are ≤1024²; 32 MP leaves a wide
# margin while rejecting pathological declares (PIL default ≈ 89 MP).
_MAX_PIXELS = 32_000_000


def _with_disclaimer(result: dict) -> dict:
    """Attach the non-diagnostic notice to a success payload (H2)."""
    return {**result, "disclaimer": API_DISCLAIMER}


_ALLOWED_MIME = {"jpeg", "png", "bmp", "tiff"}
_ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

# ---- /analyze_batch resource bounds (REL-002) ------------------------------
# Request-level complement to the per-image defenses (_LIMIT, _MAX_PIXELS):
# memory must not scale unboundedly with files × size. Declared-size and
# file-count gates run BEFORE any request body is read into memory.
_MAX_BATCH_FILES = 8
_MAX_BATCH_AGGREGATE_BYTES = 32 * 1024 * 1024  # 8 files × 10MB per-file limit
# Aggregate decoded-pixel budget: legit MRI PNGs are ≤1MP each, so this only
# trips on adversarial declares that survive the per-image 32MP bomb guard.
_MAX_BATCH_AGGREGATE_PIXELS = 64_000_000


def _batch_limit_response(error: str, detail: str) -> JSONResponse:
    """413 envelope for batch resource-bound rejections (no result payload)."""
    return JSONResponse({"error": error, "detail": detail}, status_code=413)


async def _upload_size(file: UploadFile) -> int:
    """Actual uploaded size. Starlette records it for parsed multipart when
    available; otherwise measure the spooled body (cheap seek, no payload
    copy). Starlette spools >1MB bodies to disk, so this never buffers the
    full upload in memory."""
    size = getattr(file, "size", None)
    if size is None:
        pos = file.file.tell()
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(pos)
    return size


async def _read_bounded(file: UploadFile):
    """415 type check + 413 size check BEFORE any full read; then sniff bytes
    via filetype + return the (<=10MB) bytes. Returns (data, None) or
    (None, error_response). Sniff is an additional gate, not a PIL replacement."""
    # 1. Declared content-type check (fast, no body read) - frozen contract
    if file.content_type not in _ALLOWED:
        return None, JSONResponse(
            {
                "error": "unsupported_type",
                "detail": f"content-type {file.content_type} not supported",
            },
            status_code=415,
        )
    # 2. Size check (fast, uses file.size or seek) - frozen contract
    if await _upload_size(file) > _LIMIT:
        return None, JSONResponse(
            {"error": "file_too_large", "detail": "limit 10MB"}, status_code=413
        )
    # 3. Read the data (bounded) then sniff actual signature.
    # Positive-mismatch only: known non-image signatures (exe, pdf, …)
    # => 415 here; unidentifiable bytes fall through to PIL => 422,
    # preserving the frozen 415/413/422 envelope contract.
    data = file.file.read()
    try:
        kind = filetype.guess(data)
    except Exception:  # noqa: BLE001 — sniff failure => let PIL decide (422)
        kind = None
    if kind is not None and (kind.mime not in _ALLOWED or kind.extension not in _ALLOWED_EXTS):
        return None, JSONResponse(
            {
                "error": "unsupported_type",
                "detail": f"sniffed {kind.mime} not supported",
            },
            status_code=415,
        )
    return data, None


def _decode(data: bytes):
    """Decode image bytes fully (img.load) so truncated files surface as 422
    here rather than as a 500 deep inside inference. Applies the
    decompression-bomb guard + verify() for early bomb detection."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Early detection of decompression bombs / malformed headers
        img = Image.open(io.BytesIO(data))  # Re-open after verify()
        img.load()
    except Exception:  # noqa: BLE001 — includes DecompressionBombError, SyntaxError
        return None, JSONResponse(
            {"error": "undecodable_image", "detail": "cannot decode upload"}, status_code=422
        )
    return img, None


async def _image(file: UploadFile):
    data, err = await _read_bounded(file)
    if err is not None:
        return None, err
    return _decode(data)


@app.get("/health", response_model=HealthPayload)
def health():
    seg = "loaded" if _service.segmentation_available else "unavailable"
    clf = "loaded" if _service.clf is not None else "unavailable"
    status = "ok" if (seg == "loaded" and clf == "loaded") else "degraded"
    integrity = _integrity_report_cached()
    integrity_ok = all(v is not None for v in integrity.values())
    return _with_disclaimer(
        {
            "status": status,
            "models": {"classifier": clf, "segmenter": seg},
            "calibration": {"T": _service.T, "tau1": _service.tau1, "tau2": _service.tau2}
            if clf == "loaded"
            else "unavailable",
            "version": "0.1.0",
            "integrity": {
                "artifacts": integrity,
                "all_present": integrity_ok,
            },
        }
    )


@app.post("/classify", response_model=ClassifyPayload)
async def classify(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.classify(img))
        return result
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/segment", response_model=SegmentPayload)
async def segment(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        s = _service.segment(img)
        s["localization"] = s["localization"].model_dump()
        result = _with_disclaimer(s)
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/localize", response_model=LocalizePayload)
async def localize(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.localize(img).model_dump())
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/quality", response_model=QualityPayload)
async def quality(file: UploadFile):
    data, err = await _read_bounded(file)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.quality(data))
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/consistency", response_model=ConsistencyPayload)
async def consistency(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.consistency(img))
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/analyze", response_model=AnalyzePayload)
async def analyze(file: UploadFile, consistency_probes: bool = True):
    data, err = await _read_bounded(file)
    if err is not None:
        return err
    img, err = _decode(data)
    if err is not None:
        return err
    try:
        # Compute each stage once; thread precomputed outputs into the
        # observers below (redundancy fix: previously /analyze re-ran
        # classify/analyze/consistency/quality ~3x per request; outputs are
        # bit-identical because every stage is deterministic).
        result = _service.analyze(img)
        payload = result.model_dump()
        qual = _service.quality(data)
        payload["quality"] = qual
        con = None
        if consistency_probes:
            con = _service.consistency(img, clean_pred=result.predicted_class)
            payload["consistency"] = con
        # consistency_probes=False => con=None => reliability computes its
        # consistency internally (matches pre-threading behavior exactly).
        payload["reliability"] = _service.reliability(
            img, raw=data, result=result, con=con, qual=qual
        )
        return _with_disclaimer(payload)
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/reliability", response_model=ReliabilityPayload)
async def reliability(file: UploadFile):
    """REL-001 descriptive report (observer; SB-1 outputs unchanged)."""
    data, err = await _read_bounded(file)
    if err is not None:
        return err
    img, err = _decode(data)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.reliability(img, raw=data))
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/explain", response_model=ExplainPayload)
async def explain(file: UploadFile):
    """EXPL-001 unified explanation (observer; inference outputs identical)."""
    data, err = await _read_bounded(file)
    if err is not None:
        return err
    img, err = _decode(data)
    if err is not None:
        return err
    try:
        result = _with_disclaimer(_service.explain(img, raw=data))
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )


@app.post("/analyze_batch", response_model=list[AnalyzePayload])
async def analyze_batch(files: list[UploadFile], consistency_probes: bool = True):
    """Batch analyze multiple images with tensor-level batching.

    Accepts multiple files in a single request. Returns list of results
    in the same order as input files. Uses true GPU batching for classification.

    Resource-bounded (REL-002): file count and aggregate declared bytes are
    rejected BEFORE any body is read; aggregate decoded pixels are rejected
    incrementally during decode. Per-image limits (_LIMIT, _MAX_PIXELS)
    still apply to every file individually.
    """
    if not files:
        # Empty batch is a valid terminal 200 — still exactly one observation
        # (round-2 audit fix: this exit previously bypassed accounting).
        return []

    # Gate 1: file count (pre-parse — no bytes read).
    if len(files) > _MAX_BATCH_FILES:
        return _batch_limit_response("too_many_files", f"max {_MAX_BATCH_FILES} files per batch")

    # Gate 2: aggregate declared size (pre-parse — Starlette records each
    # part's size from headers before the body is read).
    declared_total = 0
    for f in files:
        declared_total += await _upload_size(f)
    if declared_total > _MAX_BATCH_AGGREGATE_BYTES:
        return _batch_limit_response(
            "aggregate_too_large",
            f"aggregate limit {_MAX_BATCH_AGGREGATE_BYTES} bytes",
        )

    # Read and decode all images (per-file gates + incremental pixel budget)
    images = []
    raw_bytes_list = []
    decoded_pixels = 0
    for file in files:
        data, err = await _read_bounded(file)
        if err is not None:
            return err
        img, err = _decode(data)
        if err is not None:
            return err
        decoded_pixels += img.width * img.height
        if decoded_pixels > _MAX_BATCH_AGGREGATE_PIXELS:
            return _batch_limit_response(
                "aggregate_too_large",
                f"aggregate decoded-pixel limit {_MAX_BATCH_AGGREGATE_PIXELS}",
            )
        images.append(img)
        raw_bytes_list.append(data)

    try:
        results = _service.analyze_batch(images, consistency_probes)
        payloads = []
        for i, result in enumerate(results):
            payload = result.model_dump()
            # Add quality for each result
            payload["quality"] = _service.quality(raw_bytes_list[i])
            if consistency_probes:
                payload["consistency"] = _service.consistency(
                    images[i], clean_pred=result.predicted_class
                )
            payload["reliability"] = _service.reliability(
                images[i],
                raw=raw_bytes_list[i],
                result=result,
                con=payload.get("consistency"),
                qual=payload["quality"],
            )
            payloads.append(_with_disclaimer(payload))
        return payloads
    except Exception as e:  # noqa: BLE001 — never leak stack/paths
        return JSONResponse(
            {"error": "inference_failed", "detail": str(type(e).__name__)}, status_code=500
        )
