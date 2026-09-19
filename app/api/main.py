"""Thin FastAPI wrapper over the single InferenceService (§19). Offline-first."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.inference.service import InferenceService  # noqa: E402

app = FastAPI(title="BrainTumor-MajorProject", version="0.1.0")
_service = InferenceService.from_registry()

_ALLOWED = ("image/jpeg", "image/png", "image/bmp", "image/tiff")
_LIMIT = 10 * 1024 * 1024


async def _image(file: UploadFile):
    from PIL import Image
    data = await file.read()
    if len(data) > _LIMIT:
        return None, JSONResponse({"error": "file_too_large",
                                   "detail": "limit 10MB"}, status_code=413)
    try:
        return Image.open(io.BytesIO(data)), None
    except Exception:  # noqa: BLE001
        return None, JSONResponse({"error": "undecodable_image",
                                   "detail": "cannot decode upload"}, status_code=422)


def _check_type(file: UploadFile):
    if file.content_type not in _ALLOWED:
        return JSONResponse({"error": "unsupported_type",
                             "detail": f"content-type {file.content_type} not supported"},
                            status_code=415)
    return None


def _image_bytes(data: bytes):
    from PIL import Image
    try:
        return Image.open(io.BytesIO(data)), None
    except Exception:  # noqa: BLE001
        return None, JSONResponse({"error": "undecodable_image",
                                   "detail": "cannot decode upload"}, status_code=422)


@app.get("/health")
def health():
    seg = "loaded" if _service.segmentation_available else "unavailable"
    clf = "loaded" if _service.clf is not None else "unavailable"
    status = "ok" if (seg == "loaded" and clf == "loaded") else "degraded"
    return {"status": status,
            "models": {"classifier": clf, "segmenter": seg},
            "calibration": {"T": _service.T, "tau1": _service.tau1, "tau2": _service.tau2}
            if clf == "loaded" else "unavailable",
            "version": "0.1.0"}


@app.post("/classify")
async def classify(file: UploadFile):
    if (r := _check_type(file)) is not None:
        return r
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _service.classify(img)
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/segment")
async def segment(file: UploadFile):
    if (r := _check_type(file)) is not None:
        return r
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        s = _service.segment(img)
        s["localization"] = s["localization"].model_dump()
        return s
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/localize")
async def localize(file: UploadFile):
    if (r := _check_type(file)) is not None:
        return r
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _service.localize(img).model_dump()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/quality")
async def quality(file: UploadFile):
    if (r := _check_type(file)) is not None:
        return r
    data = await file.read()
    if len(data) > _LIMIT:
        return {"verdict": "reject", "failed": ["Q08_size_limit"],
                "facts": {"n_bytes": len(data)}}
    try:
        return _service.quality(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/consistency")
async def consistency(file: UploadFile):
    if (r := _check_type(file)) is not None:
        return r
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        return _service.consistency(img)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/analyze")
async def analyze(file: UploadFile, consistency_probes: bool = True):
    if (r := _check_type(file)) is not None:
        return r
    img, err = await _image(file)
    if err is not None:
        return err
    try:
        result = _service.analyze(img)
        payload = result.model_dump()
        if consistency_probes:
            payload["consistency"] = _service.consistency(img)
        return payload
    except Exception as e:  # noqa: BLE001 — never leak stack/paths (§64)
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)


@app.post("/explain")
async def explain(file: UploadFile):
    """EXPL-001 unified explanation (observer; inference outputs identical)."""
    if (r := _check_type(file)) is not None:
        return r
    data = await file.read()
    if len(data) > _LIMIT:
        return JSONResponse({"error": "file_too_large",
                             "detail": "limit 10MB"}, status_code=413)
    img, err = _image_bytes(data)
    if err is not None:
        return err
    try:
        return _service.explain(img, raw=data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "inference_failed", "detail": str(type(e).__name__)},
                            status_code=500)
