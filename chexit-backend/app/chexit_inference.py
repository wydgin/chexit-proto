"""
Chexit inference: U-Net lung mask (assets/models) → MobileNetV2 TB classifier
(assets/tb_classifier_output/weights) → Score-CAM heatmap (logic from assets/scorecam_mobnet.py).

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
import numpy as np

# If this module is imported before app.main, still avoid CUDA probe on CPU-only hosts.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # less TF stderr noise on CPU servers

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except (ValueError, RuntimeError):
    pass

from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# --- Repo layout: chexit-backend/app/thisfile.py → parents[2] = monorepo root ---
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _assets_root() -> Path:
    """Override with CHEXIT_ASSETS_ROOT for Docker / production (must contain models/ and tb_classifier_output/)."""
    env = os.environ.get("CHEXIT_ASSETS_ROOT", "").strip()
    return Path(env).resolve() if env else (_REPO_ROOT / "assets")


_ASSETS = _assets_root()
_UNET_KERAS = _ASSETS / "models" / "unet_lung_seg_best.keras"
_WEIGHTS_DIR = _ASSETS / "tb_classifier_output" / "weights"
_OPTUNA_JSON = _ASSETS / "tb_classifier_output" / "optuna_best_params.json"

IMG_SIZE = 224
UNET_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

USE_CLAHE = os.environ.get("CHEXIT_USE_CLAHE", "1").strip().lower() in ("1", "true", "yes")
MOBILENET_FOLD = int(os.environ.get("CHEXIT_MOBILENET_FOLD", "0"))
UNET_MASK_THRESHOLD = float(os.environ.get("CHEXIT_UNET_THRESHOLD", "0.5"))
MIN_LUNG_PIXELS = int(os.environ.get("CHEXIT_MIN_LUNG_PIXELS", "200"))

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


def _params_for_classifier() -> Dict[str, Any]:
    if _OPTUNA_JSON.is_file():
        with open(_OPTUNA_JSON) as f:
            return json.load(f)
    return {
        "dense_units": 128,
        "dropout_rate": 0.4,
        "l2_strength": 1e-4,
    }


def build_mobilenet_classifier() -> tf.keras.Model:
    p = _params_for_classifier()
    base = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(
        int(p.get("dense_units", 128)),
        activation="relu",
        kernel_regularizer=regularizers.l2(float(p.get("l2_strength", 1e-4))),
    )(x)
    x = Dropout(float(p.get("dropout_rate", 0.4)))(x)
    out = Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, out)


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
        wpath = _WEIGHTS_DIR / f"fold_{MOBILENET_FOLD}_weights.weights.h5"
        if not wpath.is_file():
            raise FileNotFoundError(f"MobileNet weights not found: {wpath}")
        model = build_mobilenet_classifier()
        model.load_weights(str(wpath))
        _mobilenet_model = model
    return _mobilenet_model


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
    if USE_CLAHE:
        proc_gray = apply_clahe(resized)
    else:
        proc_gray = resized
    overlay_base_bgr = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)
    x01 = np.stack([proc_gray, proc_gray, proc_gray], axis=-1).astype(np.float32) / 255.0
    x_model = np.expand_dims(x01, axis=0)
    mg_used = masked_gray if lung_mask is not None else None
    return x_model, overlay_base_bgr, mg_used


def preprocess_original_for_overlay_base(
    image_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    gray = _to_gray_uint8(image_bgr_or_gray)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    feat_layer = get_target_conv_layer(model, penultimate_layer)
    feat_model = tf.keras.Model(model.input, feat_layer.output, name="scorecam_features")
    t_act0 = time.perf_counter()
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
    x0 = seed_input.astype(np.float32)
    weights: List[float] = []
    t_mask_fwd0 = time.perf_counter()
    for start in range(0, n_ch, batch_size):
        end = min(start + batch_size, n_ch)
        bsz = end - start
        batch = np.empty((bsz, H, W, 3), dtype=np.float32)
        for j, c in enumerate(range(start, end)):
            batch[j] = x0[0] * masks[c][..., np.newaxis]
        yb, n_out = _predict_probs(model, batch)
        scores = _gather_target_score(yb, target_class, n_out)
        weights.extend([float(s) for s in scores])
    timings["masked_forwards"] = time.perf_counter() - t_mask_fwd0
    w_vec = np.asarray(weights, dtype=np.float32).reshape(n_ch)
    cam = np.tensordot(w_vec, masks, axes=([0], [0]))
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
    Returns predicted_label, prob_tb, overlay_bgr (224²).
    Overlay = original CXR (resized, no CLAHE) + lung-masked Score-CAM.
    """
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

    _pipeline_log.info("  Preprocess resize %s, CLAHE=%s…", IMG_SIZE, USE_CLAHE)
    x, _, _ = preprocess_cxr_for_mobilenet(original_bgr, lung_mask=lung_mask_fullres)
    overlay_base_natural = preprocess_original_for_overlay_base(original_bgr)
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

    _pipeline_log.info("  Blending CAM overlay on CXR (alpha=%.2f)…", overlay_alpha)
    _, ovl_bgr = overlay_cam_on_image_masked(
        overlay_base_natural,
        norm_cam,
        lung_m_224,
        alpha=overlay_alpha,
    )
    return pred_label, prob_tb, ovl_bgr


def overlay_to_png_base64(ovl_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", ovl_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed for overlay PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def predict_chexit_from_bgr(bgr_uint8: np.ndarray) -> Dict[str, Union[str, float]]:
    """
    Full pipeline for API: U-Net mask → MobileNet + Score-CAM overlay.
    ``bgr_uint8``: OpenCV BGR, uint8.
    """
    t_all = time.perf_counter()
    h0, w0 = bgr_uint8.shape[:2]
    _pipeline_log.info("=== pipeline start image=%dx%d ===", w0, h0)

    t0 = time.perf_counter()
    unet = get_unet()
    _pipeline_log.info("U-Net model ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    mobilenet = get_mobilenet()
    _pipeline_log.info("MobileNet classifier ready (load/if-cached %.2fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    _pipeline_log.info("Step 1/4: lung segmentation (U-Net @%d)…", UNET_SIZE)
    lung_mask = lung_mask_from_unet(bgr_uint8, unet)
    _pipeline_log.info("Step 1/4 done in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    _pipeline_log.info("Step 2–4/4: preprocess + TB score + Score-CAM + overlay…")
    pred_label, prob_tb, ovl = run_scorecam_with_unet_lung_mask(
        mobilenet,
        bgr_uint8,
        lung_mask,
    )
    _pipeline_log.info("Step 2–4/4 done in %.2fs", time.perf_counter() - t0)

    diagnosis = CLASS_NAMES[pred_label]
    risk_score = round(prob_tb * 100.0, 2)
    confidence_label = "High risk" if pred_label == 1 else "Low risk"
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
    }


def predict_chexit_from_pil_rgb(pil_image) -> Dict[str, Union[str, float]]:
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return predict_chexit_from_bgr(bgr)
