"""Thin FastAPI wrapper over the single InferenceService (§19). Offline-first.

Every success payload carries a `disclaimer` field: engineering prototype,
not a medical diagnosis (project instruction: API payloads must disclaim
medical diagnosis). Error envelopes (415/413/422/500) are machine-readable
and carry no result payload. Uploads are size-checked BEFORE any full-body
decode, truncated/truncated-claim bodies surface as 422, and PIL's
decompression-bomb guard is set explicitly (M7/M8 hardening).
MIME sniffing via filetype provides defense-in-depth against content-type spoofing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import filetype
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.inference.service import InferenceService  # noqa: E402

API_DISCLAIMER = ("Engineering prototype output — not a medical diagnosis. "
                  "Not a certified medical device; never use for clinical "
                  "diagnosis, triage, or therapy planning. Research use only "
                  "(retrospective BRISC 2025 data).")


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
    consistency: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda v: v is None)
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
    quality: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda v: v is None)
    gradcam: dict[str, Any]
    segmentation_vis: dict[str, Any] | None
    report_text: str
    disclaimer: str


app = FastAPI(title="BrainTumor-MajorProject", version="0.1.0",
              description=API_DISCLAIMER
              + " Every inference payload embeds this notice in `disclaimer`.")
_service = InferenceService.from_registry()

_ALLOWED = ("image/jpeg", "image/png", "image/bmp", "image/tiff")
_ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
_LIMIT = 10 * 1024 * 1024
# Decompression-bomb guard: project inputs are ≤1024²; 32 MP leaves a wide
# margin while rejecting pathological declares (PIL default ≈ 89 MP).
_MAX_PIXELS = 32_000_000


def _with_disclaimer(result: dict) -> dict:
    """Attach the non-diagnostic notice to a success payload (H2)."""
    return {**result, "disclaimer": API_DISCLAIMER}


def _check_type(file: UploadFile):
    if file.content_type not in _ALLOWED:
        return JSONResponse({"error": "unsupported_type",
                             "detail": f"content-type {file.content_type} not supported"},
                            status_code=415)
    return None


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
    (None, error_response). Sniff is an additional gate, not a PIL replacement:
    positive mismatches (known non-image signatures) => 415; unidentifiable
    bytes fall through to PIL => 422, preserving the envelope contract."""
    if (r := _check_type(file)) is not None:
        return None, r
    if await _upload_size(file) > _LIMIT:
        return None, JSONResponse({"error": "file_too_large",
                                   "detail": "limit 10MB"}, status_code=413)
    data = file.file.read()
    try:
        kind = filetype.guess(data)
    except Exception:  # noqa: BLE001 — sniff failure => let PIL decide (422)
        kind = None
    if kind is not None and (kind.mime not in _ALLOWED or kind.extension not in _ALLOWED_EXTS):
        return None, JSONResponse({"error": "unsupported_type",
                                   "detail": f"sniffed {kind.mime} not supported"},
                                  status_code=415)
    return data, None


def _decode(data: bytes):
    """Decode image bytes fully (img.load) so truncated files surface as 422
    here rather than as a 500 deep inside inference. Applies the
    decompression-bomb guard."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:  # noqa: BLE001 — includes DecompressionBombError
        return None, JSONResponse({"error": "undecodable_image",
                                   "detail": "cannot decode upload"}, status_code=422)
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
    return _with_disclaimer(
        {"status": status,
         "models": {"classifier": clf, "segmenter": seg},
         "calibration": {"T": _service.T, "tau1": _service.tau1, "tau2": _service.tau2}
         if clf == "loaded" else "unavailable",
         "version": "0.1.0"})


@app.post("/classify", response_model=ClassifyPayload)
async def classify(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _with_disclaimer(_service.classify(img))
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/segment", response_model=SegmentPayload)
async def segment(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        s = _service.segment(img)
        s["localization"] = s["localization"].model_dump()
        return _with_disclaimer(s)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/localize", response_model=LocalizePayload)
async def localize(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _with_disclaimer(_service.localize(img).model_dump())
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/quality", response_model=QualityPayload)
async def quality(file: UploadFile):
    data, err = await _read_bounded(file)
    if err is not None:
        return err
    try:
        return _with_disclaimer(_service.quality(data))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/consistency", response_model=ConsistencyPayload)
async def consistency(file: UploadFile):
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _with_disclaimer(_service.consistency(img))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


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
            img, raw=data, result=result, con=con, qual=qual)
        return _with_disclaimer(payload)
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


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
        return _with_disclaimer(_service.reliability(img, raw=data))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


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
        return _with_disclaimer(_service.explain(img, raw=data))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)
