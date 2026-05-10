"""
Chexit inference pipeline (single image):

1. Optional downscale (`CHEXIT_MAX_CXR_EDGE`)
2. Optional global CLAHE on full‑resolution grayscale (`CHEXIT_USE_CLAHE`)
3. U‑Net lung segmentation
4. Per‑model preprocessing (mask → resize → head preprocess), then predictions
5. Per‑model Score‑CAM, then ensemble‑weighted fusion of CAMs

Single module for FastAPI — no mobilenetv2_prog dependency; paths resolve to repo ``assets/``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import joblib
import numpy as np

# If this module is imported before app.main, still avoid CUDA probe on CPU-only hosts.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # less TF stderr noise on CPU servers

import tensorflow as tf
from app.explainability.densenet_fast_scorecam import compute_densenet_fast_scorecam
from app.explainability.fast_scorecam import compute_fast_scorecam

try:
    tf.config.set_visible_devices([], "GPU")
except (ValueError, RuntimeError):
    pass

from tensorflow.keras.applications import DenseNet121, EfficientNetB2, MobileNetV3Large

# --- Repo layout: chexit-backend/app/thisfile.py → parents[2] = monorepo root ---
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _assets_root() -> Path:
    """Override with CHEXIT_ASSETS_ROOT for Docker / production (must contain models/ and mobilenet_tb_output/)."""
    env = os.environ.get("CHEXIT_ASSETS_ROOT", "").strip()
    return Path(env).resolve() if env else (_REPO_ROOT / "assets")


_ASSETS = _assets_root()
_UNET_KERAS = _ASSETS / "models" / "unet_lung_seg_best.keras"
_MOBILENET_WEIGHTS_DIR = _ASSETS / "mobilenet_tb_output" / "weights"
_MOBILENET_OPTUNA_JSON = _ASSETS / "mobilenet_tb_output" / "optuna_best_params.json"
_EFFICIENTNET_WEIGHTS_DIR = _ASSETS / "efficientnet_tb_output" / "weights"
_DENSENET_WEIGHTS_DIR = _ASSETS / "densenet_tb_output" / "weights"
_META_LEARNER_PATH = Path(
    os.environ.get(
        "CHEXIT_META_LEARNER_PATH",
        str(_ASSETS / "ensemble_output" / "meta_learner.joblib"),
    )
).resolve()
_META_THRESHOLD_JSON = Path(
    os.environ.get(
        "CHEXIT_META_THRESHOLD_PATH",
        str(_ASSETS / "ensemble_output" / "ensemble_threshold.json"),
    )
).resolve()

IMG_SIZE = 224
UNET_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

USE_CLAHE = os.environ.get("CHEXIT_USE_CLAHE", "1").strip().lower() in ("1", "true", "yes")
ENSEMBLE_FOLD = int(
    os.environ.get("CHEXIT_ENSEMBLE_FOLD", os.environ.get("CHEXIT_MOBILENET_FOLD", "1"))
)
UNET_MASK_THRESHOLD = float(os.environ.get("CHEXIT_UNET_THRESHOLD", "0.5"))
MIN_LUNG_PIXELS = int(os.environ.get("CHEXIT_MIN_LUNG_PIXELS", "200"))
EFFICIENTNET_DROPOUT = float(os.environ.get("CHEXIT_EFFICIENTNET_DROPOUT", "0.6"))
EFFICIENTNET_LR_HEAD = float(os.environ.get("CHEXIT_EFFICIENTNET_LR_HEAD", "1e-3"))
USE_ENSEMBLE = os.environ.get("CHEXIT_USE_ENSEMBLE", "1").strip().lower() in ("1", "true", "yes")
ENSEMBLE_DYNAMIC_WEIGHTS = os.environ.get("CHEXIT_DYNAMIC_ENSEMBLE_WEIGHTS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _max_cxr_long_edge() -> int | None:
    """CHEXIT_MAX_CXR_EDGE=N caps longest image side; unset or 0 = no cap."""
    raw = os.environ.get("CHEXIT_MAX_CXR_EDGE", "").strip()
    if not raw:
        return None
    v = int(raw)
    return None if v <= 0 else v


CLASS_NAMES = ("TB Negative", "TB Positive")


def _setup_pipeline_logger() -> logging.Logger:
    log = logging.getLogger("chexit.pipeline")
    if log.handlers:
        return log
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(levelname)s [chexit.pipeline] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


_pipeline_log = _setup_pipeline_logger()

_unet_model: Optional[tf.keras.Model] = None
_mobilenet_model: Optional[tf.keras.Model] = None
_efficientnet_model: Optional[tf.keras.Model] = None
_densenet_model: Optional[tf.keras.Model] = None
_meta_learner: Any = None
_meta_threshold: Optional[float] = None


def _params_for_classifier() -> Dict[str, Any]:
    if _MOBILENET_OPTUNA_JSON.is_file():
        with open(_MOBILENET_OPTUNA_JSON) as f:
            return json.load(f)
    return {
        "backbone_name": "MobileNetV3Large",
        "dense_units": 128,
        "dropout_rate": 0.4,
        "l2_strength": 1e-4,
    }


def build_mobilenet_classifier() -> tf.keras.Model:
    p = _params_for_classifier()
    backbone = MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
        pooling=None,
        include_preprocessing=True,
        minimalistic=False,
        alpha=1.0,
    )
    backbone.trainable = False
    reg = tf.keras.regularizers.l2(float(p.get("l2_strength", 1e-4)))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(float(p.get("dropout_rate", 0.4)))(x)
    x = tf.keras.layers.Dense(
        int(p.get("dense_units", 128)),
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(float(p.get("dropout_rate", 0.4)))(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, out)


def build_mobilenet_classifier_with_params(params: Dict[str, Any]) -> tf.keras.Model:
    p = dict(params)
    backbone = MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
        pooling=None,
        include_preprocessing=True,
        minimalistic=False,
        alpha=1.0,
    )
    backbone.trainable = False
    reg = tf.keras.regularizers.l2(float(p.get("l2_strength", 1e-4)))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(float(p.get("dropout_rate", 0.4)))(x)
    x = tf.keras.layers.Dense(
        int(p.get("dense_units", 128)),
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(float(p.get("dropout_rate", 0.4)))(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, out)


def build_efficientnet_classifier() -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal")])
    base = EfficientNetB2(
        input_shape=(260, 260, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(260, 260, 3)),
            data_augmentation,
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(EFFICIENTNET_DROPOUT),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=EFFICIENTNET_LR_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_densenet_classifier() -> tf.keras.Model:
    return build_densenet_classifier_with_params(
        {
            "dense_units": 128,
            "dropout_rate": 0.4,
            "l2_strength": 1e-4,
            "head_depth": 1,
        }
    )


def build_densenet_classifier_with_params(params: Dict[str, Any]) -> tf.keras.Model:
    dense_units = int(params.get("dense_units", 128))
    dropout_rate = float(params.get("dropout_rate", 0.4))
    l2_strength = float(params.get("l2_strength", 1e-4))
    head_depth = int(params.get("head_depth", 1))

    base = DenseNet121(
        input_shape=(256, 256, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(256, 256, 3), name="image_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
        name="head_dense_1",
    )(x)
    if head_depth >= 2:
        x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_2")(x)
        x = tf.keras.layers.Dense(
            max(32, dense_units // 2),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
            name="head_dense_2",
        )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_final")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="tb_output")(x)
    return tf.keras.Model(inputs, outputs, name="densenet121_tb_classifier")


def _resolve_weight_path(
    weights_dir: Path,
    primary_pattern: str,
    *,
    fold: int,
    fallback_pattern: Optional[str] = None,
) -> Path:
    candidates: List[Path] = [weights_dir / primary_pattern.format(fold=fold)]
    if fold >= 1:
        candidates.append(weights_dir / primary_pattern.format(fold=fold - 1))
    if fallback_pattern:
        candidates.append(weights_dir / fallback_pattern.format(fold=fold))
        if fold >= 1:
            candidates.append(weights_dir / fallback_pattern.format(fold=fold - 1))
    for p in candidates:
        if p.is_file():
            return p
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No matching weights found (tried: {tried})")


def get_unet() -> tf.keras.Model:
    global _unet_model
    if _unet_model is None:
        if not _UNET_KERAS.is_file():
            raise FileNotFoundError(f"U-Net model not found: {_UNET_KERAS}")
        _unet_model = tf.keras.models.load_model(str(_UNET_KERAS), compile=False)
    return _unet_model


def get_mobilenet() -> tf.keras.Model:
    global _mobilenet_model
    if _mobilenet_model is None:
        wpath = _resolve_weight_path(
            _MOBILENET_WEIGHTS_DIR,
            "fold_{fold}_weights.weights.h5",
            fold=ENSEMBLE_FOLD,
            fallback_pattern="fold_{fold}_best_val_auc.weights.h5",
        )
        base_params = _params_for_classifier()
        # Some checkpoints were trained with different dense_units than current optuna json.
        dense_candidates: List[int] = []
        configured_dense = int(base_params.get("dense_units", 128))
        for v in (configured_dense, 128, 64, 256):
            if v not in dense_candidates:
                dense_candidates.append(v)

        last_err: Optional[Exception] = None
        for dense_units in dense_candidates:
            try:
                trial_params = dict(base_params)
                trial_params["dense_units"] = dense_units
                model = build_mobilenet_classifier_with_params(trial_params)
                model.load_weights(str(wpath))
                _pipeline_log.info(
                    "Loaded MobileNet weights: %s (dense_units=%d)",
                    wpath.name,
                    dense_units,
                )
                _mobilenet_model = model
                break
            except Exception as e:
                last_err = e
                _pipeline_log.warning(
                    "MobileNet load attempt failed for dense_units=%d: %s",
                    dense_units,
                    e,
                )
        if _mobilenet_model is None:
            assert last_err is not None
            raise last_err
    return _mobilenet_model


def get_efficientnet() -> tf.keras.Model:
    global _efficientnet_model
    if _efficientnet_model is None:
        wpath = _resolve_weight_path(
            _EFFICIENTNET_WEIGHTS_DIR,
            "fold_{fold}.weights.h5",
            fold=ENSEMBLE_FOLD,
        )
        model = build_efficientnet_classifier()
        model.load_weights(str(wpath))
        _pipeline_log.info("Loaded EfficientNet weights: %s", wpath.name)
        _efficientnet_model = model
    return _efficientnet_model


def get_densenet() -> tf.keras.Model:
    global _densenet_model
    if _densenet_model is None:
        wpath = _resolve_weight_path(
            _DENSENET_WEIGHTS_DIR,
            "fold_{fold}_phase2_best.weights.h5",
            fold=ENSEMBLE_FOLD,
            fallback_pattern="fold_{fold}_best.weights.h5",
        )
        candidate_dense = [512, 256, 128, 64]
        candidate_head_depth = [1, 2]
        last_err: Optional[Exception] = None
        for dense_units in candidate_dense:
            for head_depth in candidate_head_depth:
                try:
                    model = build_densenet_classifier_with_params(
                        {
                            "dense_units": dense_units,
                            "dropout_rate": 0.4,
                            "l2_strength": 1e-4,
                            "head_depth": head_depth,
                        }
                    )
                    model.load_weights(str(wpath))
                    _pipeline_log.info(
                        "Loaded DenseNet weights: %s (dense_units=%d, head_depth=%d)",
                        wpath.name,
                        dense_units,
                        head_depth,
                    )
                    _densenet_model = model
                    break
                except Exception as e:
                    last_err = e
                    _pipeline_log.warning(
                        "DenseNet load attempt failed for dense_units=%d head_depth=%d: %s",
                        dense_units,
                        head_depth,
                        e,
                    )
            if _densenet_model is not None:
                break
        if _densenet_model is None:
            assert last_err is not None
            raise last_err
    return _densenet_model


# ----- Preprocessing & mask (from scorecam_mobnet.py) -----


def apply_clahe(
    gray_uint8: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    if gray_uint8.ndim != 2:
        raise ValueError("apply_clahe expects HxW uint8.")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray_uint8)


def apply_global_clahe_bgr(bgr_uint8: np.ndarray) -> np.ndarray:
    """CLAHE on luminance at native (post‑downscale) resolution before U‑Net and classifiers."""
    if not USE_CLAHE:
        return bgr_uint8
    gray = _to_gray_uint8(bgr_uint8)
    return cv2.cvtColor(apply_clahe(gray), cv2.COLOR_GRAY2BGR)


def apply_lung_mask(
    gray_uint8: np.ndarray,
    mask: np.ndarray,
    *,
    mask_is_binary: bool = True,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    if gray_uint8.ndim != 2:
        raise ValueError("apply_lung_mask expects HxW grayscale uint8.")
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError("mask must be HxW or HxW1.")
    if m.shape != gray_uint8.shape:
        m = cv2.resize(
            m.astype(np.float32),
            (gray_uint8.shape[1], gray_uint8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mf = m.astype(np.float32)
    if mask_is_binary:
        mf = (mf > mask_threshold).astype(np.float32)
    else:
        mf = np.clip(mf, 0.0, 1.0)
    out = gray_uint8.astype(np.float32) * mf
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        g = image
    elif image.ndim == 3 and image.shape[-1] == 3:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be HxW or HxWx3 BGR uint8.")
    if g.dtype != np.uint8:
        if np.issubdtype(g.dtype, np.floating):
            g = np.clip(g, 0, 255).astype(np.uint8)
        else:
            g = g.astype(np.uint8)
    return g


def preprocess_cxr_for_mobilenet(
    image_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
    lung_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gray = _to_gray_uint8(image_bgr_or_gray)
    masked_gray = apply_lung_mask(gray, lung_mask) if lung_mask is not None else gray
    resized = cv2.resize(masked_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # CLAHE is applied once globally upstream (predict path / run_scorecam path).
    proc_gray = resized
    overlay_base_bgr = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)
    x255 = np.stack([proc_gray, proc_gray, proc_gray], axis=-1).astype(np.float32)
    x_model = np.expand_dims(x255, axis=0)
    mg_used = masked_gray if lung_mask is not None else None
    return x_model, overlay_base_bgr, mg_used


def preprocess_cxr_for_efficientnet(
    image_bgr_or_gray: np.ndarray,
    *,
    lung_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    gray = _to_gray_uint8(image_bgr_or_gray)
    masked_gray = apply_lung_mask(gray, lung_mask) if lung_mask is not None else gray
    resized = cv2.resize(masked_gray, (260, 260), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(rgb)
    return np.expand_dims(x, axis=0)


def preprocess_cxr_for_densenet(
    image_bgr_or_gray: np.ndarray,
    *,
    lung_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    gray = _to_gray_uint8(image_bgr_or_gray)
    masked_gray = apply_lung_mask(gray, lung_mask) if lung_mask is not None else gray
    resized = cv2.resize(masked_gray, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32)
    x = tf.keras.applications.densenet.preprocess_input(rgb)
    return np.expand_dims(x, axis=0)


def _ensemble_weights() -> Tuple[float, float, float]:
    raw = os.environ.get("CHEXIT_ENSEMBLE_WEIGHTS", "0.34,0.33,0.33").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("CHEXIT_ENSEMBLE_WEIGHTS must have exactly 3 comma-separated numbers.")
    vals = np.asarray([float(x) for x in parts], dtype=np.float32)
    s = float(np.sum(vals))
    if s <= 0:
        raise ValueError("CHEXIT_ENSEMBLE_WEIGHTS must sum to > 0.")
    vals = vals / s
    return float(vals[0]), float(vals[1]), float(vals[2])


def _model_performance_priors() -> Tuple[float, float, float]:
    """
    Optional validation-performance priors (e.g., AUC) for
    MobileNet/EfficientNet/DenseNet.
    """
    raw = os.environ.get("CHEXIT_MODEL_PERFORMANCE_PRIORS", "1.0,1.0,1.0").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("CHEXIT_MODEL_PERFORMANCE_PRIORS must have exactly 3 comma-separated numbers.")
    vals = np.asarray([float(x) for x in parts], dtype=np.float32)
    vals = np.maximum(vals, 0.0)
    s = float(np.sum(vals))
    if s <= 0:
        raise ValueError("CHEXIT_MODEL_PERFORMANCE_PRIORS must sum to > 0.")
    vals = vals / s
    return float(vals[0]), float(vals[1]), float(vals[2])


def _dynamic_ensemble_weights(prob_mob: float, prob_eff: float, prob_den: float) -> Tuple[float, float, float]:
    """
    Per-image model contribution weights.
    - confidence term: farther from 0.5 => higher confidence
    - performance term: optional global prior (e.g., validation AUC)
    """
    pri_m, pri_e, pri_d = _model_performance_priors()
    conf = np.asarray(
        [
            abs(float(prob_mob) - 0.5) * 2.0,
            abs(float(prob_eff) - 0.5) * 2.0,
            abs(float(prob_den) - 0.5) * 2.0,
        ],
        dtype=np.float32,
    )
    conf = np.clip(conf, 0.0, 1.0)
    # Keep small floor so uncertain models still contribute a bit.
    conf = np.maximum(conf, 0.05)
    pri = np.asarray([pri_m, pri_e, pri_d], dtype=np.float32)
    weights = conf * pri
    s = float(np.sum(weights))
    if s <= 0.0:
        return _ensemble_weights()
    weights = weights / s
    return float(weights[0]), float(weights[1]), float(weights[2])


def _meta_threshold_value() -> float:
    global _meta_threshold
    if _meta_threshold is None:
        if _META_THRESHOLD_JSON.is_file():
            with open(_META_THRESHOLD_JSON) as f:
                data = json.load(f)
            _meta_threshold = float(data.get("threshold", 0.5))
        else:
            _meta_threshold = 0.5
    return float(_meta_threshold)


def _get_meta_learner() -> Any:
    global _meta_learner
    if _meta_learner is None:
        if not _META_LEARNER_PATH.is_file():
            raise FileNotFoundError(f"Meta learner file not found: {_META_LEARNER_PATH}")
        _meta_learner = joblib.load(str(_META_LEARNER_PATH))
        _pipeline_log.info("Loaded meta learner: %s", _META_LEARNER_PATH.name)
    return _meta_learner


def _meta_weights_and_prob(prob_mob: float, prob_eff: float, prob_den: float) -> Tuple[float, float, float, float]:
    """
    Use trained meta learner for final probability and per-image contribution weights.
    Input order is [densenet, mobilenet, efficientnet] based on ensemble artifacts.
    Returns (w_m, w_e, w_d, p_meta).
    """
    meta = _get_meta_learner()
    x = np.asarray([[float(prob_den), float(prob_mob), float(prob_eff)]], dtype=np.float32)
    if hasattr(meta, "predict_proba"):
        p_meta = float(meta.predict_proba(x)[0, 1])
    else:
        # Generic fallback for regressors / decision_function models.
        p_raw = float(np.squeeze(meta.predict(x)))
        p_meta = float(np.clip(p_raw, 0.0, 1.0))

    if hasattr(meta, "coef_"):
        coef = np.asarray(meta.coef_, dtype=np.float32).reshape(-1)
        if coef.size >= 3:
            # Match x feature order: [den, mob, eff]
            contrib = np.abs(np.asarray([coef[0] * prob_den, coef[1] * prob_mob, coef[2] * prob_eff]))
            s = float(np.sum(contrib))
            if s > 0:
                contrib = contrib / s
                w_d, w_m, w_e = float(contrib[0]), float(contrib[1]), float(contrib[2])
                return w_m, w_e, w_d, p_meta
    # Fallback if coef_ unavailable
    w_m, w_e, w_d = _dynamic_ensemble_weights(prob_mob, prob_eff, prob_den)
    return w_m, w_e, w_d, p_meta


def preprocess_original_for_overlay_base(
    image_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    gray = _to_gray_uint8(image_bgr_or_gray)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def preprocess_original_for_overlay_base_fullres(image_bgr_or_gray: np.ndarray) -> np.ndarray:
    """Grayscale CXR as BGR at native resolution (for heatmap overlay matching input dimensions)."""
    gray = _to_gray_uint8(image_bgr_or_gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def lung_mask_from_unet(bgr_uint8: np.ndarray, unet: tf.keras.Model) -> np.ndarray:
    """Full-resolution float mask in [0, 1], same HxW as input gray."""
    gray = _to_gray_uint8(bgr_uint8)
    h0, w0 = gray.shape
    g512 = cv2.resize(gray, (UNET_SIZE, UNET_SIZE), interpolation=cv2.INTER_AREA)
    x = (g512.astype(np.float32) / 255.0)[..., np.newaxis]
    x = np.expand_dims(x, 0)
    pred = unet.predict(x, verbose=0)[0, ..., 0]
    mask512 = (pred > UNET_MASK_THRESHOLD).astype(np.float32)
    mask = cv2.resize(mask512, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return mask


# ----- Score-CAM (from scorecam_mobnet.py) -----


def get_target_conv_layer(
    model: tf.keras.Model,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
) -> tf.keras.layers.Layer:
    if penultimate_layer is not None:
        if isinstance(penultimate_layer, str):
            return model.get_layer(penultimate_layer)
        return penultimate_layer
    gap_idx: Optional[int] = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_idx = i
            break
    if gap_idx is None or gap_idx < 1:
        raise ValueError("No GlobalAveragePooling2D found.")
    return model.layers[gap_idx - 1]


def _normalize_minmax_hw(maps_hw: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    out = np.empty_like(maps_hw, dtype=np.float32)
    for i in range(maps_hw.shape[0]):
        a = maps_hw[i].astype(np.float32)
        amin, amax = float(a.min()), float(a.max())
        if amax - amin < epsilon:
            out[i] = 0.0
        else:
            out[i] = (a - amin) / (amax - amin + epsilon)
    return out


def _normalize_cam_to_unit(cam: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    cam = np.maximum(cam.astype(np.float32), 0.0)
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin < epsilon:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cmin) / (cmax - cmin + epsilon)


def _gather_target_score(
    y: np.ndarray,
    target_class: int,
    num_classes: int,
) -> np.ndarray:
    if num_classes == 1:
        p_tb = y.reshape(-1).astype(np.float32)
        p_ntb = 1.0 - p_tb
        return p_tb if target_class == 1 else p_ntb
    return y[:, target_class].astype(np.float32)


def _predict_probs(model: tf.keras.Model, x: np.ndarray) -> Tuple[np.ndarray, int]:
    y = model.predict(x, verbose=0)
    if y.ndim == 2 and y.shape[-1] == 1:
        return y, 1
    if y.ndim == 2 and y.shape[-1] > 1:
        return y, int(y.shape[-1])
    if y.ndim == 1:
        return y.reshape(-1, 1), 1
    raise ValueError(f"Unexpected model output shape: {y.shape}")


def _resolve_binary_target_class(prob_tb: float, mode: str = "predicted") -> int:
    if mode == "tb":
        return 1
    if mode == "non_tb":
        return 0
    if mode == "predicted":
        return 1 if prob_tb >= 0.5 else 0
    raise ValueError(mode)


def compute_scorecam(
    model: tf.keras.Model,
    seed_input: np.ndarray,
    *,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
    target_class: int,
    batch_size: int = 32,
    max_channels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    feat_layer = get_target_conv_layer(model, penultimate_layer)
    t_act0 = time.perf_counter()
    # Keras 3 nested models can expose disconnected graph/input attributes.
    # For nested backbones, run the layer directly on the actual tensor input.
    if isinstance(feat_layer, tf.keras.Model):
        acts = feat_layer(tf.convert_to_tensor(seed_input), training=False).numpy()
    else:
        feat_model = tf.keras.Model(model.input, feat_layer.output, name="scorecam_features")
        acts = feat_model.predict(seed_input, verbose=0)
    timings["extract_activations"] = time.perf_counter() - t_act0
    if acts.ndim != 4:
        raise ValueError(f"Expected 4D activations, got {acts.shape}")
    _, h, w, n_ch = acts.shape
    _, H, W, _ = seed_input.shape
    acts_ch = np.transpose(acts[0], (2, 0, 1))
    t_up0 = time.perf_counter()
    ups = np.empty((n_ch, H, W), dtype=np.float32)
    for c in range(n_ch):
        ups[c] = cv2.resize(acts_ch[c], (W, H), interpolation=cv2.INTER_LINEAR)
    timings["upsample_maps"] = time.perf_counter() - t_up0
    masks = _normalize_minmax_hw(ups)
    selected_masks = masks
    if max_channels is not None and max_channels > 0 and int(max_channels) < n_ch:
        variances = np.var(masks, axis=(1, 2))
        top_idx = np.argsort(-variances)[: int(max_channels)]
        selected_masks = masks[top_idx]
    timings["channels_total"] = float(n_ch)
    timings["channels_used"] = float(selected_masks.shape[0])
    x0 = seed_input.astype(np.float32)
    weights: List[float] = []
    t_mask_fwd0 = time.perf_counter()
    n_sel = int(selected_masks.shape[0])
    for start in range(0, n_sel, batch_size):
        end = min(start + batch_size, n_sel)
        bsz = end - start
        batch = np.empty((bsz, H, W, 3), dtype=np.float32)
        for j in range(bsz):
            batch[j] = x0[0] * selected_masks[start + j][..., np.newaxis]
        yb, n_out = _predict_probs(model, batch)
        scores = _gather_target_score(yb, target_class, n_out)
        weights.extend([float(s) for s in scores])
    timings["masked_forwards"] = time.perf_counter() - t_mask_fwd0
    w_vec = np.asarray(weights, dtype=np.float32).reshape(n_sel)
    cam = np.tensordot(w_vec, selected_masks, axes=([0], [0]))
    cam = np.maximum(cam.astype(np.float32), 0.0)
    norm_cam = _normalize_cam_to_unit(cam)
    timings["total"] = time.perf_counter() - t0
    return cam, norm_cam, timings


def overlay_cam_on_image_masked(
    base_bgr_uint8: np.ndarray,
    cam01_hw: np.ndarray,
    mask_hw: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = base_bgr_uint8.shape[:2]
    m = mask_hw.astype(np.float32)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = np.clip(m, 0.0, 1.0)
    cam_m = np.clip(cam01_hw.astype(np.float32) * m, 0.0, 1.0)
    cam_u8 = np.clip(cam_m * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, colormap)
    heat = (heat.astype(np.float32) * m[..., np.newaxis]).astype(np.uint8)
    a = float(alpha) * m[..., np.newaxis]
    overlay = np.clip(
        base_bgr_uint8.astype(np.float32) * (1.0 - a) + heat.astype(np.float32) * a,
        0,
        255,
    ).astype(np.uint8)
    return heat, overlay


def run_scorecam_with_unet_lung_mask(
    model: tf.keras.Model,
    original_bgr: np.ndarray,
    lung_mask_fullres: np.ndarray,
    *,
    binary_target_mode: str = "predicted",
    batch_size: int = 32,
    overlay_alpha: float = 0.45,
) -> Tuple[int, float, np.ndarray]:
    """
    Returns predicted_label, prob_tb, overlay_bgr at **same H×W as original_bgr**.
    CAM is computed at 224² then upsampled; blend uses native-resolution grayscale base + lung mask.

    Applies the same global CLAHE policy as ``predict_chexit_from_bgr`` when ``USE_CLAHE``.
    """
    original_bgr = apply_global_clahe_bgr(original_bgr)
    gray = _to_gray_uint8(original_bgr)
    if lung_mask_fullres.shape != gray.shape:
        lung_mask_fullres = cv2.resize(
            lung_mask_fullres.astype(np.float32),
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if float(np.sum(lung_mask_fullres > 0.5)) < MIN_LUNG_PIXELS:
        lung_mask_fullres = np.ones_like(lung_mask_fullres, dtype=np.float32)
        _pipeline_log.info("  Lung mask tiny (< %d px); using full image for mask.", MIN_LUNG_PIXELS)

    _pipeline_log.info("  Preprocess resize %s (CLAHE already global upstream=%s)", IMG_SIZE, USE_CLAHE)
    x, _, _ = preprocess_cxr_for_mobilenet(original_bgr, lung_mask=lung_mask_fullres)
    lung_m_224 = cv2.resize(
        lung_mask_fullres.astype(np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    lung_m_224 = np.clip(lung_m_224, 0.0, 1.0)

    _pipeline_log.info("  MobileNet TB classification (224, lung-masked input)…")
    y, n_out = _predict_probs(model, x)
    if n_out != 1:
        raise ValueError("Chexit API expects a binary sigmoid MobileNet head.")

    prob_tb = float(np.squeeze(y))
    pred_label = 1 if prob_tb >= 0.5 else 0
    target_class = _resolve_binary_target_class(prob_tb, binary_target_mode)
    _pipeline_log.info(
        "  TB prob=%.4f → class=%s (idx=%d), CAM target_class=%d",
        prob_tb,
        CLASS_NAMES[pred_label],
        pred_label,
        target_class,
    )

    if _env_truthy("CHEXIT_SKIP_SCORECAM"):
        _pipeline_log.info(
            "  Score-CAM skipped (CHEXIT_SKIP_SCORECAM=1); lung-mask visualization for heatmap."
        )
        norm_cam = lung_m_224.astype(np.float32)
    else:
        _pipeline_log.info("  Score-CAM (channel-wise masked forwards — often several minutes)…")
        t_cam = time.perf_counter()
        _, norm_cam, cam_timings = compute_scorecam(
            model,
            x,
            target_class=target_class,
            batch_size=batch_size,
        )
        _pipeline_log.info(
            "  Score-CAM done in %.1fs (masked_forwards=%.1fs, total_internal=%.1fs)",
            time.perf_counter() - t_cam,
            cam_timings.get("masked_forwards", 0.0),
            cam_timings.get("total", 0.0),
        )

    _pipeline_log.info("  Blending CAM overlay on CXR full resolution (alpha=%.2f)…", overlay_alpha)
    hf, wf = int(gray.shape[0]), int(gray.shape[1])
    norm_cam_full = cv2.resize(
        norm_cam.astype(np.float32),
        (wf, hf),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay_base_full = preprocess_original_for_overlay_base_fullres(original_bgr)
    _, ovl_bgr = overlay_cam_on_image_masked(
        overlay_base_full,
        norm_cam_full,
        lung_mask_fullres,
        alpha=overlay_alpha,
    )
    return pred_label, prob_tb, ovl_bgr


def overlay_to_png_base64(ovl_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", ovl_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed for overlay PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _maybe_downscale_bgr_max_edge(bgr_uint8: np.ndarray) -> np.ndarray:
    lim = _max_cxr_long_edge()
    if lim is None:
        return bgr_uint8
    h0, w0 = bgr_uint8.shape[:2]
    m = max(h0, w0)
    if m <= lim:
        return bgr_uint8
    scale = lim / float(m)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    _pipeline_log.info(
        "Downscaling CXR %dx%d → %dx%d (CHEXIT_MAX_CXR_EDGE=%d)",
        w0,
        h0,
        nw,
        nh,
        lim,
    )
    return cv2.resize(bgr_uint8, (nw, nh), interpolation=cv2.INTER_AREA)


def predict_chexit_from_bgr(bgr_uint8: np.ndarray) -> Dict[str, Any]:
    """
    Full pipeline for API: optional global CLAHE → U-Net mask → ensemble classifiers +
    fused Score-CAM overlay.
    ``bgr_uint8``: OpenCV BGR, uint8.
    """
    t_all = time.perf_counter()
    bgr_uint8 = _maybe_downscale_bgr_max_edge(bgr_uint8)
    bgr_uint8 = apply_global_clahe_bgr(bgr_uint8)
    h0, w0 = bgr_uint8.shape[:2]
    _pipeline_log.info("=== pipeline start image=%dx%d CLAHE_global=%s ===", w0, h0, USE_CLAHE)

    t0 = time.perf_counter()
    unet = get_unet()
    _pipeline_log.info("U-Net model ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    mobilenet = get_mobilenet()
    _pipeline_log.info("MobileNet classifier ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    efficientnet = get_efficientnet()
    _pipeline_log.info("EfficientNet classifier ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    densenet = get_densenet()
    _pipeline_log.info("DenseNet classifier ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    _pipeline_log.info("Step 1/4: lung segmentation after global CLAHE (U-Net @%d)…", UNET_SIZE)
    lung_mask = lung_mask_from_unet(bgr_uint8, unet)
    _pipeline_log.info("Step 1/4 done in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    _pipeline_log.info("Step 2–4/4: preprocess + TB score + ensemble + fused Score-CAM + overlay…")
    x_mob, _, _ = preprocess_cxr_for_mobilenet(bgr_uint8, lung_mask=lung_mask)
    x_eff = preprocess_cxr_for_efficientnet(bgr_uint8, lung_mask=lung_mask)
    x_den = preprocess_cxr_for_densenet(bgr_uint8, lung_mask=lung_mask)
    prob_mob = float(np.squeeze(mobilenet.predict(x_mob, verbose=0)))
    prob_eff = float(np.squeeze(efficientnet.predict(x_eff, verbose=0)))
    prob_den = float(np.squeeze(densenet.predict(x_den, verbose=0)))
    pred_label = 1 if prob_mob >= 0.5 else 0
    prob_tb = prob_mob
    w_m, w_e, w_d = _ensemble_weights()
    if USE_ENSEMBLE:
        try:
            w_m, w_e, w_d, prob_tb = _meta_weights_and_prob(prob_mob, prob_eff, prob_den)
            pred_label = 1 if prob_tb >= _meta_threshold_value() else 0
        except Exception as e:
            _pipeline_log.warning("Meta learner unavailable; falling back to weighted averaging: %s", e)
            if ENSEMBLE_DYNAMIC_WEIGHTS:
                w_m, w_e, w_d = _dynamic_ensemble_weights(prob_mob, prob_eff, prob_den)
            prob_tb = (w_m * prob_mob) + (w_e * prob_eff) + (w_d * prob_den)
            pred_label = 1 if prob_tb >= 0.5 else 0

    target_class = pred_label
    if _env_truthy("CHEXIT_SKIP_SCORECAM"):
        _pipeline_log.info("Score-CAM skipped (CHEXIT_SKIP_SCORECAM=1); using lung-mask heatmap.")
        gray = _to_gray_uint8(bgr_uint8)
        norm_cam_full = np.clip(lung_mask.astype(np.float32), 0.0, 1.0)
        if norm_cam_full.shape != gray.shape:
            norm_cam_full = cv2.resize(
                norm_cam_full,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        _pipeline_log.info("Computing per-model Score-CAM maps for fused explanation…")
        _, cam_mob, t_mob = compute_fast_scorecam(
            mobilenet,
            x_mob,
            target_class=target_class,
            batch_size=int(os.environ.get("CHEXIT_SCORECAM_BATCH_MOBILENET", "32")),
            max_channels=int(os.environ.get("CHEXIT_SCORECAM_MAX_CHANNELS_MOBILENET", "256")),
        )
        _, cam_eff, t_eff = compute_fast_scorecam(
            efficientnet,
            x_eff,
            target_class=target_class,
            penultimate_layer=os.environ.get("CHEXIT_EFFICIENTNET_SCORECAM_LAYER", "top_conv"),
            batch_size=int(os.environ.get("CHEXIT_SCORECAM_BATCH_EFFICIENTNET", "16")),
            max_channels=int(os.environ.get("CHEXIT_SCORECAM_MAX_CHANNELS_EFFICIENTNET", "256")),
        )
        cam_den, t_den = compute_densenet_fast_scorecam(
            densenet,
            x_den,
            target_class=target_class,
            target_layer_name=os.environ.get(
                "CHEXIT_DENSENET_SCORECAM_LAYER",
                "conv5_block16_concat",
            ),
            batch_size=int(os.environ.get("CHEXIT_SCORECAM_BATCH_DENSENET", "16")),
            max_channels=int(os.environ.get("CHEXIT_SCORECAM_MAX_CHANNELS_DENSENET", "256")),
        )
        h_full, w_full = bgr_uint8.shape[:2]
        cam_mob_f = cv2.resize(cam_mob.astype(np.float32), (w_full, h_full), interpolation=cv2.INTER_LINEAR)
        cam_eff_f = cv2.resize(cam_eff.astype(np.float32), (w_full, h_full), interpolation=cv2.INTER_LINEAR)
        cam_den_f = cv2.resize(cam_den.astype(np.float32), (w_full, h_full), interpolation=cv2.INTER_LINEAR)
        cam_w_m, cam_w_e, cam_w_d = (w_m, w_e, w_d) if USE_ENSEMBLE else (1.0, 0.0, 0.0)
        merged_cam = (cam_w_m * cam_mob_f) + (cam_w_e * cam_eff_f) + (cam_w_d * cam_den_f)
        norm_cam_full = _normalize_cam_to_unit(merged_cam)
        _pipeline_log.info(
            "Score-CAM done m=%.1fs e=%.1fs d=%.1fs | channels used m=%d e=%d d=%d",
            t_mob.get("total", 0.0),
            t_eff.get("total", 0.0),
            t_den.get("total", 0.0),
            int(t_mob.get("channels_used", 0)),
            int(t_eff.get("channels_used", 0)),
            int(t_den.get("channels_used", 0)),
        )

    overlay_base_full = preprocess_original_for_overlay_base_fullres(bgr_uint8)
    _, ovl = overlay_cam_on_image_masked(
        overlay_base_full,
        norm_cam_full,
        lung_mask,
        alpha=0.45,
    )
    _pipeline_log.info(
        "Model probs m=%.4f e=%.4f d=%.4f | ensemble=%.4f (enabled=%s, w=[%.2f, %.2f, %.2f])",
        prob_mob,
        prob_eff,
        prob_den,
        prob_tb,
        USE_ENSEMBLE,
        w_m,
        w_e,
        w_d,
    )
    _pipeline_log.info("Step 2–4/4 done in %.2fs", time.perf_counter() - t0)

    diagnosis = CLASS_NAMES[pred_label]
    risk_score = round(prob_tb * 100.0, 2)
    confidence_label = "High risk" if pred_label == 1 else "Low risk"
    model_contributions = {
        "mobilenet-v2": round(w_m * 100.0, 2),
        "efficientnet-b2": round(w_e * 100.0, 2),
        "densenet-121": round(w_d * 100.0, 2),
    }
    _pipeline_log.info("Encoding overlay PNG → base64…")
    heatmap_b64 = overlay_to_png_base64(ovl)
    _pipeline_log.info(
        "=== pipeline complete total=%.2fs diagnosis=%s risk_score=%s heatmap_b64_len=%d ===",
        time.perf_counter() - t_all,
        diagnosis,
        risk_score,
        len(heatmap_b64),
    )
    return {
        "diagnosis": diagnosis,
        "risk_score": risk_score,
        "confidence_label": confidence_label,
        "heatmap": heatmap_b64,
        "model_contributions": model_contributions,
    }


def predict_chexit_from_pil_rgb(pil_image) -> Dict[str, Union[str, float]]:
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return predict_chexit_from_bgr(bgr)
