"""Integration tests for /health and /predict (in-process via TestClient)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"
SAMPLE_CXR = ASSETS / "cxrin.png"
UNET_PATH = ASSETS / "models" / "unet_lung_seg_best.keras"
MOBILENET_WEIGHTS = ASSETS / "tb_classifier_output" / "weights" / "fold_0_weights.weights.h5"


def predict_dependencies_present() -> bool:
    return UNET_PATH.is_file() and MOBILENET_WEIGHTS.is_file()


def _png_bytes() -> bytes:
    """Minimal PNG suitable for the U-Net 512 pipeline (grayscale square)."""
    buf = io.BytesIO()
    Image.new("L", (512, 512), color=96).save(buf, format="PNG")
    return buf.getvalue()


def _sample_image_bytes() -> bytes:
    if SAMPLE_CXR.is_file():
        return SAMPLE_CXR.read_bytes()
    return _png_bytes()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(
    not predict_dependencies_present(),
    reason=f"Missing U-Net or MobileNet weights under {ASSETS} (gitignored; copy locally).",
)
def test_predict_returns_contract(client: TestClient) -> None:
    raw = _sample_image_bytes()
    r = client.post(
        "/predict",
        files={"file": ("sample.png", io.BytesIO(raw), "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "diagnosis" in data
    assert "risk_score" in data
    assert "confidence_label" in data
    assert "heatmap" in data
    assert isinstance(data["diagnosis"], str)
    assert isinstance(data["risk_score"], (int, float))
    assert isinstance(data["confidence_label"], str)
    assert isinstance(data["heatmap"], str)
    assert len(data["heatmap"]) > 500


def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("x.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 400
