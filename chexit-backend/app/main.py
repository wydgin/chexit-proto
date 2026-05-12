from __future__ import annotations

import io
import json
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

# Before TensorFlow loads (via chexit_inference): no GPU on Render — avoids cuInit ERROR spam.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.chexit_inference import predict_chexit_from_pil_rgb
from app.model_loader import download_models_if_needed

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_UPLOADS_ROOT = Path(os.environ.get("CHEXIT_UPLOADS_DIR", "/tmp/chexit_uploads")).resolve()
_UPLOADS_FILES_DIR = _UPLOADS_ROOT / "files"
_UPLOADS_LATEST_JSON = _UPLOADS_ROOT / "latest.json"

_DICOM_EXTS = (".dcm", ".dicom")
_DICOM_CTYPES = {"application/dicom", "application/dicom+json", "application/octet-stream"}

def _looks_like_dicom(upload: UploadFile) -> bool:
    ctype = (upload.content_type or "").lower()
    name = (upload.filename or "").lower()
    return (ctype in _DICOM_CTYPES) or name.endswith(_DICOM_EXTS)


def _ensure_upload_dirs() -> None:
    _UPLOADS_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _safe_uploaded_filename(raw_name: str | None) -> str:
    suffix = Path(raw_name or "").suffix.lower()
    if len(suffix) > 10:
        suffix = ""
    return f"{int(time.time() * 1000)}_{uuid4().hex[:8]}{suffix}"


def _latest_upload_record() -> dict[str, str] | None:
    if not _UPLOADS_LATEST_JSON.is_file():
        return None
    try:
        return json.loads(_UPLOADS_LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        _api_log.warning("Failed to parse latest upload metadata: %s", _UPLOADS_LATEST_JSON)
        return None


def _write_latest_upload_record(record: dict[str, str]) -> None:
    _UPLOADS_LATEST_JSON.write_text(json.dumps(record), encoding="utf-8")

def _dicom_bytes_to_pil_rgb(file_bytes: bytes) -> Image.Image:
    ds = dcmread(io.BytesIO(file_bytes), force=True)
    if "PixelData" not in ds:
        raise ValueError("DICOM has no pixel data.")

    arr = ds.pixel_array
    # Multi-frame: use first frame for now
    if arr.ndim == 3:
        arr = arr[0]

    arr = apply_voi_lut(arr, ds) if hasattr(ds, "WindowCenter") else arr
    arr = arr.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        raise ValueError("DICOM pixel range is invalid.")
    arr = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    # MONOCHROME1 means inverted grayscale
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 255 - arr

    return Image.fromarray(arr).convert("RGB")

def _api_logger() -> logging.Logger:
    log = logging.getLogger("chexit.api")
    if log.handlers:
        return log
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(levelname)s [chexit.api] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


_api_log = _api_logger()


def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "CHEXIT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def _cors_origin_regex() -> str | None:
    """Allow any Vercel preview/production *.vercel.app origin when credentials are used."""
    raw = os.environ.get("CHEXIT_CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app").strip()
    return raw or None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _ensure_upload_dirs()
    _api_log.info("Ensuring U-Net models (gdown from Drive if missing)...")
    t0 = time.perf_counter()
    try:
        download_models_if_needed()
    except Exception:
        _api_log.exception("Model download failed after %.2fs", time.perf_counter() - t0)
        raise
    _api_log.info("Model assets ready in %.2fs", time.perf_counter() - t0)
    yield


app = FastAPI(title="Chexit API", version="0.2.0", lifespan=_lifespan)
# StaticFiles requires the directory to exist at mount time; lifespan runs later.
_ensure_upload_dirs()
app.mount("/uploads/files", StaticFiles(directory=str(_UPLOADS_FILES_DIR)), name="uploads-files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    diagnosis: str
    risk_score: float = Field(..., description="Estimated TB probability (0–100)")
    confidence_label: str
    heatmap: str = Field(..., description="PNG overlay (CAM on CXR) as base64")
    model_contributions: dict[str, float] = Field(
        ...,
        description="Per-image normalized model contribution percentages.",
    )


class UploadResponse(BaseModel):
    download_url: str
    file_name: str
    uploaded_at: str


@app.get("/")
def root() -> RedirectResponse:
    """Browser default; API lives under /docs, /health, /predict."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    is_image = bool(file.content_type and file.content_type.startswith("image/"))
    is_dicom = _looks_like_dicom(file)

    if not (is_image or is_dicom):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image (PNG/JPG) or DICOM (.dcm).",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Max file size is 10MB.",
        )

    _api_log.info(
        "POST /predict filename=%r content_type=%s bytes=%d",
        file.filename,
        file.content_type,
        len(file_bytes),
    )

    try:
        if is_dicom:
            image = _dicom_bytes_to_pil_rgb(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image/DICOM.")

    t0 = time.perf_counter()
    try:
        out = predict_chexit_from_pil_rgb(image)
    except FileNotFoundError as e:
        _api_log.error("Inference missing model/weights: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        _api_log.exception("Inference failed after %.2fs", time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}") from e

    _api_log.info(
        "POST /predict OK in %.2fs diagnosis=%s risk_score=%s",
        time.perf_counter() - t0,
        out["diagnosis"],
        out["risk_score"],
    )
    return PredictResponse(
        diagnosis=str(out["diagnosis"]),
        risk_score=float(out["risk_score"]),
        confidence_label=str(out["confidence_label"]),
        heatmap=str(out["heatmap"]),
        model_contributions=dict(out["model_contributions"]),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    is_image = bool(file.content_type and file.content_type.startswith("image/"))
    is_dicom = _looks_like_dicom(file)
    if not (is_image or is_dicom):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image (PNG/JPG) or DICOM (.dcm).",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Max file size is 10MB.",
        )

    saved_name = _safe_uploaded_filename(file.filename)
    out_path = _UPLOADS_FILES_DIR / saved_name
    out_path.write_bytes(file_bytes)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    download_url = f"/uploads/files/{saved_name}"
    record = {
        "downloadURL": download_url,
        "fileName": file.filename or saved_name,
        "uploadedAt": uploaded_at,
    }
    _write_latest_upload_record(record)
    _api_log.info("POST /upload OK filename=%r bytes=%d", file.filename, len(file_bytes))
    return UploadResponse(
        download_url=download_url,
        file_name=record["fileName"],
        uploaded_at=uploaded_at,
    )


@app.get("/uploads/latest")
def uploads_latest() -> dict[str, str]:
    record = _latest_upload_record()
    if not record:
        raise HTTPException(status_code=404, detail="No uploads yet.")
    return record
