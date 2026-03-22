"""
Download U-Net artifacts from Google Drive (gdown) when missing under assets/models/.

Drive files must be shared: Anyone with the link → Viewer.
Override IDs with CHEXIT_GDOWN_UNET_BEST_ID, CHEXIT_GDOWN_UNET_WEIGHTS_ID, CHEXIT_GDOWN_UNET_FINAL_ID.
Set CHEXIT_SKIP_GDOWN=1 to skip downloads (local dev when weights are already on disk).

By default only ``unet_lung_seg_best.keras`` is downloaded (what /predict uses). Set
CHEXIT_GDOWN_ALL_UNET=1 to also fetch the weights-only and final Keras files (~310MB extra per cold start).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gdown

_log = logging.getLogger("chexit.model_loader")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(levelname)s [chexit.model_loader] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MIN_BYTES = 1_000_000  # ignore tiny files (likely HTML error pages)


def _assets_models_dir() -> Path:
    env = os.environ.get("CHEXIT_ASSETS_ROOT", "").strip()
    root = Path(env).resolve() if env else (_REPO_ROOT / "assets")
    d = root / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _drive_models() -> dict[str, dict[str, str]]:
    """Map logical name → file id and destination filename under assets/models/."""
    models: dict[str, dict[str, str]] = {
        "unet_best": {
            "id": os.environ.get(
                "CHEXIT_GDOWN_UNET_BEST_ID", "1Lw2yROpyz3-GaYXsrJOdxaEXM7EMnPzQ"
            ).strip(),
            "filename": "unet_lung_seg_best.keras",
        },
    }
    if os.environ.get("CHEXIT_GDOWN_ALL_UNET", "").strip().lower() in ("1", "true", "yes"):
        models["unet_weights"] = {
            "id": os.environ.get(
                "CHEXIT_GDOWN_UNET_WEIGHTS_ID", "1vBhVrAE2dsCtKjZ__Gg-iYQbjlQa06vB"
            ).strip(),
            "filename": "unet_lung_seg_best_weights.weights.h5",
        }
        models["unet_final"] = {
            "id": os.environ.get(
                "CHEXIT_GDOWN_UNET_FINAL_ID", "1Cc2BB9yIUIMuBnXx57_UEoDYPBxPEE9P"
            ).strip(),
            "filename": "unet_lung_seg_final.keras",
        }
    return models


def download_models_if_needed() -> None:
    if os.environ.get("CHEXIT_SKIP_GDOWN", "").strip().lower() in ("1", "true", "yes"):
        _log.info("CHEXIT_SKIP_GDOWN set — skipping Google Drive model download.")
        return

    models_dir = _assets_models_dir()
    for name, spec in _drive_models().items():
        path = models_dir / spec["filename"]
        if path.is_file() and path.stat().st_size >= _MIN_BYTES:
            _log.info("Model present, skip: %s", path.name)
            continue

        file_id = spec["id"]
        if not file_id:
            _log.warning("No Drive id for %s — skipping.", name)
            continue

        _log.info("Downloading %s from Google Drive → %s", name, path)
        gdown.download(
            id=file_id,
            output=str(path),
            quiet=False,
            fuzzy=True,
        )
        if not path.is_file() or path.stat().st_size < _MIN_BYTES:
            raise RuntimeError(
                f"gdown failed or file too small for {name} ({path}). "
                "Check that the Drive file is public (Anyone with the link) and the ID is correct."
            )
