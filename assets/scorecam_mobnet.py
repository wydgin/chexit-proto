"""
Score-CAM for the Chexit MobileNet TB classifier (TensorFlow/Keras).

Preprocessing matches mobilenetv2_prog.load_image_for_classifier (order matters):
uint8 grayscale → resize to 224 (INTER_AREA) → CLAHE when USE_CLAHE in mobilenetv2_prog →
3-channel stack → float32 / 255.0. The classifier sees [0, 1] (no mobilenet_v2.preprocess_input).

Official inference model (MobileNet branch, thesis / mbnet_test default):
  fold 0 weights: mobilenet_tb_output/weights/fold_0_weights.weights.h5
  See OFFICIAL_INFERENCE_FOLD and load_official_mobilenet().

Inputs: U-Net–segmented CXRs (e.g. unet_export/.../*_unetseg.png), same distribution as training.

Running ``python scorecam_mobnet.py`` (no arguments) processes **DEFAULT_CLI_BATCH_N** training
pairs where the filename’s **third** underscore segment is ``1`` (e.g. ``MCUCXR_0243_1.png``),
not ``…_0.png``. Same rule for Montgomery and Shenzhen. Pairs use ``Training/.../*.png`` +
``unet_export/.../{stem}_unetseg.png``; overlay uses the **original** CXR with lung-masked heatmaps.
Outputs: ``{stem}_scorecam_*.png`` and ``scorecam_output/scorecam_batch_metadata.csv``.

Score-CAM: gradient-free channel weights from forward-pass target scores on masked inputs.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from mobilenetv2_prog import (
    BASE_DIR,
    MONTGOMERY_CXR_DIR,
    OUTPUT_DIR,
    SHENZHEN_CXR_DIR,
    UNET_EXPORT_MONTGOMERY,
    UNET_EXPORT_SHENZHEN,
    USE_CLAHE,
    WEIGHTS_DIR,
    build_model,
    get_default_params,
)

# Official MobileNet inference checkpoint (same default as mbnet_test.py).
OFFICIAL_INFERENCE_FOLD = 0

# Run ``python scorecam_mobnet.py`` with no args → batch this many (original + U-Net) pairs.
DEFAULT_CLI_BATCH_N = 10

# Keep in sync with mobilenetv2_prog.py (classifier spatial size + CLAHE grid when USE_CLAHE).
IMG_SIZE = 224
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

try:
    from tf_keras_vis.scorecam import Scorecam as _TFKVScorecam  # type: ignore

    _TF_KERAS_VIS_SCORECAM = _TFKVScorecam
except ImportError:
    _TF_KERAS_VIS_SCORECAM = None


BinaryTargetMode = Literal["predicted", "tb", "non_tb"]
MultiClassMode = Literal["predicted", "forced"]


@dataclass
class ScoreCamOutputs:
    predicted_label: int
    predicted_score: float
    target_class_used: int
    raw_cam: np.ndarray
    normalized_cam: np.ndarray  # [0,1] full 224² Score-CAM
    heatmap_image: np.ndarray
    overlay_image: np.ndarray
    # Lung-masked CAM at 224² (optional; set for original-base + U-Net mask pipeline).
    normalized_cam_lung_masked: Optional[np.ndarray] = None
    # 224×224 BGR uint8 (base for overlay: segmented+CLAHE or original resized).
    overlay_base_image: Optional[np.ndarray] = None
    masked_lung_image_used: Optional[np.ndarray] = None
    timings_sec: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def metadata_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary (no ndarray payloads). Use for APIs and logs."""
        return {
            "predicted_label": int(self.predicted_label),
            "predicted_score": float(self.predicted_score),
            "target_class_used": int(self.target_class_used),
            "normalized_cam_shape": [int(x) for x in self.normalized_cam.shape],
            "normalized_cam_lung_masked_shape": (
                [int(x) for x in self.normalized_cam_lung_masked.shape]
                if self.normalized_cam_lung_masked is not None
                else None
            ),
            "raw_cam_shape": [int(x) for x in self.raw_cam.shape],
            "timings_sec": {k: float(v) for k, v in self.timings_sec.items()},
            "masked_lung_applied": self.masked_lung_image_used is not None,
            **{k: v for k, v in self.meta.items() if isinstance(v, (str, int, float, bool))},
        }

    def as_dict(self, *, include_arrays: bool = False) -> Dict[str, Any]:
        """
        If include_arrays is False (default), returns metadata_dict() only (clean serialization).
        If True, includes all numpy image/CAM arrays (heavy; notebooks / debugging).
        """
        base = self.metadata_dict()
        if not include_arrays:
            return base
        return {
            **base,
            "raw_cam": self.raw_cam,
            "normalized_cam": self.normalized_cam,
            "normalized_cam_lung_masked": self.normalized_cam_lung_masked,
            "heatmap_image": self.heatmap_image,
            "overlay_image": self.overlay_image,
            "overlay_base_image": self.overlay_base_image,
            "masked_lung_image_used": self.masked_lung_image_used,
        }


def apply_clahe(
    gray_uint8: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """CLAHE on grayscale uint8 (same defaults as mobilenetv2_prog._apply_clahe)."""
    if gray_uint8.ndim != 2:
        raise ValueError("apply_clahe expects a single-channel HxW uint8 image.")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray_uint8)


def apply_lung_mask(
    gray_uint8: np.ndarray,
    mask: np.ndarray,
    *,
    mask_is_binary: bool = True,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Apply lung mask to grayscale uint8; mask is resized to gray shape if needed."""
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
        raise ValueError("image must be HxW or HxWx3 (BGR uint8).")
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
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    lung_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Match mobilenetv2_prog.load_image_for_classifier (in-memory variant).
    Same step order: resize first, then CLAHE on the resized tile iff USE_CLAHE (training flag).

    Returns:
        x_model: (1, img_size, img_size, 3) float32
        overlay_base_bgr: uint8 BGR for visualization (same spatial size)
        masked_gray_before_resize: uint8 HxW after mask only, or None if no mask
    """
    gray = _to_gray_uint8(image_bgr_or_gray)
    masked_gray = apply_lung_mask(gray, lung_mask) if lung_mask is not None else gray

    # load_image_for_classifier: cv2.resize → then CLAHE (not CLAHE on full-res then resize).
    resized = cv2.resize(masked_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if USE_CLAHE:
        proc_gray = apply_clahe(resized, clahe_clip_limit, clahe_tile_grid)
    else:
        proc_gray = resized

    overlay_base_bgr = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)

    # Identical to training: three identical channels, scaled to [0, 1] (not preprocess_input).
    x01 = np.stack([proc_gray, proc_gray, proc_gray], axis=-1).astype(np.float32) / 255.0
    x_model = np.expand_dims(x01, axis=0)

    mg_used = masked_gray if lung_mask is not None else None
    return x_model, overlay_base_bgr, mg_used


def get_target_conv_layer(
    model: tf.keras.Model,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
) -> tf.keras.layers.Layer:
    """Default: layer immediately before GlobalAveragePooling2D (last spatial feature map)."""
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
        raise ValueError("No GlobalAveragePooling2D found; pass penultimate_layer explicitly.")
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


def _resolve_binary_target_class(prob_tb: float, mode: BinaryTargetMode) -> int:
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
    use_tf_keras_vis: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Score-CAM: extract conv activations, upsample (bilinear), min-max per channel,
    mask [0,1] inputs, batched forwards, weight by target-class probability, ReLU sum, normalize CAM.
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    if use_tf_keras_vis:
        if _TF_KERAS_VIS_SCORECAM is None:
            raise ImportError("tf_keras_vis is not installed.")
        layer = get_target_conv_layer(model, penultimate_layer)
        scorecam = _TF_KERAS_VIS_SCORECAM(model, layer)
        cam = scorecam(seed_input)[0]  # type: ignore[call-arg]
        if isinstance(cam, tf.Tensor):
            cam = cam.numpy()
        raw = np.maximum(cam.astype(np.float32), 0.0)
        norm = _normalize_cam_to_unit(raw)
        timings["total"] = time.perf_counter() - t0
        return raw, norm, timings

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

    # Weight vector (n_ch,) so tensordot yields (H, W), not (1, H, W) from (n_ch,1) × (n_ch,H,W).
    w_vec = np.asarray(weights, dtype=np.float32).reshape(n_ch)
    cam = np.tensordot(w_vec, masks, axes=([0], [0]))
    cam = np.maximum(cam.astype(np.float32), 0.0)
    norm_cam = _normalize_cam_to_unit(cam)

    timings["total"] = time.perf_counter() - t0
    return cam, norm_cam, timings


def overlay_cam_on_image(
    base_bgr_uint8: np.ndarray,
    cam01_hw: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    if base_bgr_uint8.shape[:2] != cam01_hw.shape[:2]:
        raise ValueError("base image and CAM must match spatial size.")
    cam_u8 = np.clip(cam01_hw * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, colormap)
    overlay = cv2.addWeighted(base_bgr_uint8, 1.0 - alpha, heat, alpha, 0)
    return heat, overlay


def overlay_cam_on_image_masked(
    base_bgr_uint8: np.ndarray,
    cam01_hw: np.ndarray,
    mask_hw: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend CAM onto base only inside mask (e.g. U-Net lung at 224²). Outside mask = original base.
    """
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


def preprocess_original_for_overlay_base(
    image_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Original training CXR resized to classifier size for display only (no CLAHE — natural contrast).
    Returns BGR uint8 (H, W, 3).
    """
    gray = _to_gray_uint8(image_bgr_or_gray)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def lung_mask_from_unetseg_at_size(
    unetseg_bgr_or_gray: np.ndarray,
    img_size: int = IMG_SIZE,
    *,
    threshold: int = 0,
) -> np.ndarray:
    """Binary lung mask float32 in [0,1] at img_size from a U-Net export (nonzero = lung)."""
    g = _to_gray_uint8(unetseg_bgr_or_gray)
    m = (g > threshold).astype(np.float32)
    if m.shape[0] != img_size or m.shape[1] != img_size:
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return m


def stem_has_third_segment_one(stem: str) -> bool:
    """
    Montgomery / Shenzhen naming: ``{source}_{id}_{view}.png`` where the third segment is
    ``0`` or ``1`` (second view). Only process stems with third segment ``"1"`` (e.g.
    ``MCUCXR_0243_1``, not ``MCUCXR_0001_0``).
    """
    parts = stem.split("_")
    return len(parts) >= 3 and parts[2] == "1"


def training_original_path_for_unetseg_filename(unet_export_path: Path) -> Optional[Path]:
    """
    Map ``unet_export/.../{stem}_unetseg.png`` → ``Training/.../{stem}.png`` if it exists.
    Note: ``Path.stem`` for these files is ``{stem}_unetseg`` — do not use ``.stem`` for the core name.
    """
    name = unet_export_path.name
    if not name.endswith("_unetseg.png"):
        return None
    core_stem = name[: -len("_unetseg.png")]
    for cxr_dir in (MONTGOMERY_CXR_DIR, SHENZHEN_CXR_DIR):
        cand = cxr_dir / f"{core_stem}.png"
        if cand.is_file():
            return cand
    return None


def list_training_originals_with_unetseg(max_count: int) -> List[Tuple[Path, Path]]:
    """
    Pairs (original Training/.../*.png, unet_export/.../*_unetseg.png) sorted by path, up to max_count.
    Only filenames whose third ``_`` segment is ``1`` (e.g. ``*_0243_1.png``); skips ``*_0001_0.png``.
    Montgomery first, then Shenzhen.
    """
    pairs: List[Tuple[Path, Path]] = []
    for cxr_dir, exp_dir in (
        (MONTGOMERY_CXR_DIR, UNET_EXPORT_MONTGOMERY),
        (SHENZHEN_CXR_DIR, UNET_EXPORT_SHENZHEN),
    ):
        if not cxr_dir.is_dir() or not exp_dir.is_dir():
            continue
        for p in sorted(cxr_dir.glob("*.png")):
            if not stem_has_third_segment_one(p.stem):
                continue
            u = exp_dir / f"{p.stem}_unetseg.png"
            if u.is_file():
                pairs.append((p, u))
    pairs.sort(key=lambda t: str(t[0]))
    return pairs[:max_count]


def normalized_cam_to_grayscale_u8(cam_01: np.ndarray) -> np.ndarray:
    """Map normalized CAM [0, 1] to uint8 grayscale H×W for saving."""
    return np.clip(cam_01.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)


def save_visualization_outputs(
    out_dir: Union[str, Path],
    *,
    base_bgr: Optional[np.ndarray] = None,
    cam_normalized_01: Optional[np.ndarray] = None,
    heatmap_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    prefix: str = "scorecam",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    if base_bgr is not None:
        base_path = out_dir / f"{prefix}_base.png"
        cv2.imwrite(str(base_path), base_bgr)
        paths["base"] = base_path

    if cam_normalized_01 is not None:
        cam_gray = normalized_cam_to_grayscale_u8(cam_normalized_01)
        cam_path = out_dir / f"{prefix}_cam_gray.png"
        cv2.imwrite(str(cam_path), cam_gray)
        paths["cam_gray"] = cam_path

    heat_path = out_dir / f"{prefix}_heatmap.png"
    ovl_path = out_dir / f"{prefix}_overlay.png"
    cv2.imwrite(str(heat_path), heatmap_bgr)
    cv2.imwrite(str(ovl_path), overlay_bgr)
    paths["heatmap"] = heat_path
    paths["overlay"] = ovl_path
    return paths


def _params_for_classifier_architecture() -> Dict[str, Any]:
    """
    Match mbnet_test / training: Optuna JSON when present, else get_default_params().
    Ensures layer widths etc. match fold_*.weights.h5.
    """
    optuna_path = OUTPUT_DIR / "optuna_best_params.json"
    if optuna_path.exists():
        with open(optuna_path) as f:
            return json.load(f)
    return get_default_params()


def load_trained_mobilenet(fold: int = OFFICIAL_INFERENCE_FOLD) -> tf.keras.Model:
    """
    Build MobileNet classifier like mobilenetv2_prog and load fold weights from WEIGHTS_DIR.
    Default fold is OFFICIAL_INFERENCE_FOLD (production / thesis default).
    """
    params = _params_for_classifier_architecture()
    model = build_model(
        dense_units=params.get("dense_units", 128),
        dropout_rate=params.get("dropout_rate", 0.4),
        l2_strength=params.get("l2_strength", 1e-4),
    )
    weights_path = WEIGHTS_DIR / f"fold_{fold}_weights.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    model.load_weights(str(weights_path))
    return model


def official_weights_path(fold: int = OFFICIAL_INFERENCE_FOLD) -> Path:
    """Path to the canonical fold weights file under WEIGHTS_DIR."""
    return WEIGHTS_DIR / f"fold_{fold}_weights.weights.h5"


def load_official_mobilenet() -> tf.keras.Model:
    """Load the official inference model (OFFICIAL_INFERENCE_FOLD)."""
    return load_trained_mobilenet(fold=OFFICIAL_INFERENCE_FOLD)


def run_scorecam_from_path(
    image_path: Union[str, Path],
    *,
    fold: int = OFFICIAL_INFERENCE_FOLD,
    out_dir: Union[str, Path] = "scorecam_output",
    prefix: str = "scorecam",
    lung_mask: Optional[np.ndarray] = None,
    original_image_path: Optional[Union[str, Path]] = None,
    prefer_original_training_cxr: bool = True,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
    binary_target_mode: BinaryTargetMode = "predicted",
    batch_size: int = 32,
    overlay_alpha: float = 0.45,
) -> Dict[str, Any]:
    """
    Load a U-Net–segmented CXR from disk, run Score-CAM, save PNGs.

    If ``original_image_path`` is set, or ``prefer_original_training_cxr`` and the file is
    ``*_unetseg.png`` with a matching ``Training/.../{stem}.png``, the **overlay base** is that
    original (224², no CLAHE) and the heatmap is **lung-masked** from the U-Net export; the
    classifier still uses the segmented image (training distribution). Otherwise the overlay
    base is the segmented+CLAHE pipeline (legacy single-image behavior).

    Returns JSON-friendly keys: metadata (no arrays), saved_paths (string paths).
    """
    image_path = Path(image_path)
    if image_path.name.endswith("_unetseg.png"):
        core_stem = image_path.name[: -len("_unetseg.png")]
        if not stem_has_third_segment_one(core_stem):
            raise ValueError(
                "Only CXRs with third filename segment `1` are run (e.g. MCUCXR_0243_1_unetseg.png), "
                f"not `{core_stem}`."
            )

    unet_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if unet_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    model = load_trained_mobilenet(fold=fold)

    orig_path: Optional[Path] = None
    if original_image_path is not None:
        orig_path = Path(original_image_path)
    elif (
        prefer_original_training_cxr
        and lung_mask is None
        and image_path.name.endswith("_unetseg.png")
    ):
        orig_path = training_original_path_for_unetseg_filename(image_path)

    use_original_overlay = (
        orig_path is not None
        and orig_path.is_file()
        and lung_mask is None
    )
    orig_bgr: Optional[np.ndarray] = None
    if use_original_overlay:
        orig_bgr = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        if orig_bgr is None:
            use_original_overlay = False
            warnings.warn(f"Could not read original CXR at {orig_path}; using segmented overlay base.")

    if use_original_overlay and orig_bgr is not None:
        out = run_scorecam_original_base_unet_mask(
            model,
            orig_bgr,
            unet_bgr,
            penultimate_layer=penultimate_layer,
            binary_target_mode=binary_target_mode,
            batch_size=batch_size,
            overlay_alpha=overlay_alpha,
        )
        cam_save = out.normalized_cam_lung_masked
        if cam_save is None:
            cam_save = out.normalized_cam
    else:
        out = run_scorecam_mobilenet(
            model,
            unet_bgr,
            lung_mask=lung_mask,
            penultimate_layer=penultimate_layer,
            binary_target_mode=binary_target_mode,
            batch_size=batch_size,
            overlay_alpha=overlay_alpha,
        )
        cam_save = out.normalized_cam

    saved = save_visualization_outputs(
        out_dir,
        base_bgr=out.overlay_base_image,
        cam_normalized_01=cam_save,
        heatmap_bgr=out.heatmap_image,
        overlay_bgr=out.overlay_image,
        prefix=prefix,
    )

    meta = out.metadata_dict()
    meta["inference_fold"] = int(fold)
    meta["weights_path"] = str(official_weights_path(fold=fold))
    meta["use_clahe"] = bool(USE_CLAHE)
    meta["unetseg_input_path"] = str(image_path.resolve())
    meta["overlay_mode"] = (
        "original_base_lung_masked" if use_original_overlay else "segmented_base"
    )
    if use_original_overlay and orig_path is not None:
        meta["original_image_path"] = str(orig_path.resolve())

    return {
        "metadata": meta,
        "saved_paths": {k: str(v) for k, v in saved.items()},
    }


def run_scorecam_mobilenet(
    model: tf.keras.Model,
    image_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
    lung_mask: Optional[np.ndarray] = None,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
    binary_target_mode: BinaryTargetMode = "predicted",
    multiclass_index: Optional[int] = None,
    multiclass_mode: MultiClassMode = "predicted",
    batch_size: int = 32,
    overlay_alpha: float = 0.45,
    use_tf_keras_vis: bool = False,
) -> ScoreCamOutputs:
    """
    One CXR: training-consistent preprocessing (USE_CLAHE from mobilenetv2_prog) → Score-CAM.
    Binary (1 output sigmoid): binary_target_mode predicted | tb | non_tb.
    """
    t_all = time.perf_counter()

    x, overlay_base, masked_gray = preprocess_cxr_for_mobilenet(
        image_bgr_or_gray,
        img_size=img_size,
        lung_mask=lung_mask,
    )

    y, n_out = _predict_probs(model, x)
    timings: Dict[str, float] = {}

    if n_out == 1:
        prob_tb = float(np.squeeze(y))
        pred_label = 1 if prob_tb >= 0.5 else 0
        pred_score = prob_tb if pred_label == 1 else 1.0 - prob_tb
        target_class = _resolve_binary_target_class(prob_tb, binary_target_mode)
    else:
        probs = y.reshape(-1)
        pred_label = int(np.argmax(probs))
        pred_score = float(probs[pred_label])
        if multiclass_mode == "forced":
            if multiclass_index is None:
                raise ValueError("multiclass_mode='forced' requires multiclass_index.")
            target_class = int(multiclass_index)
        else:
            target_class = pred_label

    raw_cam, norm_cam, t_cam = compute_scorecam(
        model,
        x,
        penultimate_layer=penultimate_layer,
        target_class=target_class,
        batch_size=batch_size,
        use_tf_keras_vis=use_tf_keras_vis,
    )
    timings.update(t_cam)

    heat_bgr, ovl_bgr = overlay_cam_on_image(overlay_base, norm_cam, alpha=overlay_alpha)

    timings["end_to_end"] = time.perf_counter() - t_all

    layer_used = get_target_conv_layer(model, penultimate_layer)
    return ScoreCamOutputs(
        predicted_label=pred_label,
        predicted_score=float(pred_score),
        target_class_used=target_class,
        raw_cam=raw_cam,
        normalized_cam=norm_cam,
        heatmap_image=heat_bgr,
        overlay_image=ovl_bgr,
        normalized_cam_lung_masked=None,
        overlay_base_image=overlay_base,
        masked_lung_image_used=masked_gray,
        timings_sec=timings,
        meta={
            "penultimate_layer_used": layer_used.name,
            "classifier_input_range": "[0, 1] per mobilenetv2_prog (float32 / 255.0)",
            "n_model_outputs": n_out,
            "use_clahe": USE_CLAHE,
        },
    )


def run_scorecam_original_base_unet_mask(
    model: tf.keras.Model,
    original_bgr_or_gray: np.ndarray,
    unetseg_bgr_or_gray: np.ndarray,
    *,
    img_size: int = IMG_SIZE,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
    binary_target_mode: BinaryTargetMode = "predicted",
    multiclass_index: Optional[int] = None,
    multiclass_mode: MultiClassMode = "predicted",
    batch_size: int = 32,
    overlay_alpha: float = 0.45,
    use_tf_keras_vis: bool = False,
) -> ScoreCamOutputs:
    """
    Classifier + Score-CAM on U-Net segmented input (training distribution); overlay on **original**
    CXR (resized, no CLAHE); heatmap confined to lung via mask from unet_export image.
    """
    t_all = time.perf_counter()

    x, _, _ = preprocess_cxr_for_mobilenet(
        unetseg_bgr_or_gray,
        img_size=img_size,
        lung_mask=None,
    )
    lung_m = lung_mask_from_unetseg_at_size(unetseg_bgr_or_gray, img_size=img_size)
    overlay_base_bgr = preprocess_original_for_overlay_base(
        original_bgr_or_gray,
        img_size=img_size,
    )

    y, n_out = _predict_probs(model, x)
    timings: Dict[str, float] = {}

    if n_out == 1:
        prob_tb = float(np.squeeze(y))
        pred_label = 1 if prob_tb >= 0.5 else 0
        pred_score = prob_tb if pred_label == 1 else 1.0 - prob_tb
        target_class = _resolve_binary_target_class(prob_tb, binary_target_mode)
    else:
        probs = y.reshape(-1)
        pred_label = int(np.argmax(probs))
        pred_score = float(probs[pred_label])
        if multiclass_mode == "forced":
            if multiclass_index is None:
                raise ValueError("multiclass_mode='forced' requires multiclass_index.")
            target_class = int(multiclass_index)
        else:
            target_class = pred_label

    raw_cam, norm_cam, t_cam = compute_scorecam(
        model,
        x,
        penultimate_layer=penultimate_layer,
        target_class=target_class,
        batch_size=batch_size,
        use_tf_keras_vis=use_tf_keras_vis,
    )
    timings.update(t_cam)

    norm_cam_lung = norm_cam * lung_m
    heat_bgr, ovl_bgr = overlay_cam_on_image_masked(
        overlay_base_bgr,
        norm_cam,
        lung_m,
        alpha=overlay_alpha,
    )

    timings["end_to_end"] = time.perf_counter() - t_all
    layer_used = get_target_conv_layer(model, penultimate_layer)
    return ScoreCamOutputs(
        predicted_label=pred_label,
        predicted_score=float(pred_score),
        target_class_used=target_class,
        raw_cam=raw_cam,
        normalized_cam=norm_cam,
        heatmap_image=heat_bgr,
        overlay_image=ovl_bgr,
        normalized_cam_lung_masked=norm_cam_lung,
        overlay_base_image=overlay_base_bgr,
        masked_lung_image_used=None,
        timings_sec=timings,
        meta={
            "penultimate_layer_used": layer_used.name,
            "classifier_input_range": "[0, 1] per mobilenetv2_prog (float32 / 255.0)",
            "n_model_outputs": n_out,
            "use_clahe": USE_CLAHE,
            "overlay_base": "original_cxr_resized_no_clahe",
            "classifier_input": "unetseg_training_style",
            "heatmap_masked_by": "unet_export_lung",
        },
    )


def batch_scorecam_training_original_overlay(
    n: int = 10,
    *,
    out_dir: Union[str, Path] = "scorecam_output",
    csv_path: Union[str, Path] = "scorecam_output/scorecam_batch_metadata.csv",
    fold: int = OFFICIAL_INFERENCE_FOLD,
    overlay_alpha: float = 0.45,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    First ``n`` Training CXRs that have a matching ``unet_export`` file; saves
    ``{original_stem}_scorecam_*.png`` and a CSV with TB prediction metadata.
    """
    pairs = list_training_originals_with_unetseg(n)
    if not pairs:
        raise FileNotFoundError(
            "No (original, unetseg) pairs found with third filename segment `1` (e.g. *_0243_1.png). "
            "Ensure Training/*/ and unet_export/*/ exist and names match Montgomery/Shenzhen pattern."
        )
    if len(pairs) < n:
        warnings.warn(
            f"Requested {n} images but only {len(pairs)} (original, unetseg) pairs exist.",
            UserWarning,
            stacklevel=2,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_trained_mobilenet(fold=fold)
    rows: List[Dict[str, Any]] = []
    skipped_read: List[str] = []

    for orig_path, unet_path in pairs:
        stem = orig_path.stem
        prefix = f"{stem}_scorecam"
        orig_bgr = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        unet_bgr = cv2.imread(str(unet_path), cv2.IMREAD_COLOR)
        if orig_bgr is None or unet_bgr is None:
            skipped_read.append(f"{orig_path.name} / {unet_path.name}")
            continue

        out = run_scorecam_original_base_unet_mask(
            model,
            orig_bgr,
            unet_bgr,
            overlay_alpha=overlay_alpha,
            batch_size=batch_size,
        )

        cam_save = out.normalized_cam_lung_masked
        if cam_save is None:
            cam_save = out.normalized_cam

        save_visualization_outputs(
            out_dir,
            base_bgr=out.overlay_base_image,
            cam_normalized_01=cam_save,
            heatmap_bgr=out.heatmap_image,
            overlay_bgr=out.overlay_image,
            prefix=prefix,
        )

        try:
            rel_image = str(orig_path.resolve().relative_to(BASE_DIR.resolve()))
        except ValueError:
            rel_image = str(orig_path.resolve())

        # predicted_score is confidence in predicted class; recover P(TB) from sigmoid head.
        if out.predicted_label == 1:
            prob_tb = float(out.predicted_score)
        else:
            prob_tb = float(1.0 - out.predicted_score)

        rows.append(
            {
                "image_file": rel_image,
                "original_filename": orig_path.name,
                "is_tb": int(out.predicted_label),
                "predicted_class": "tb" if out.predicted_label == 1 else "non_tb",
                "probability_tb": prob_tb,
                "output_prefix": prefix,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    if skipped_read:
        head = ", ".join(skipped_read[:10])
        ell = " …" if len(skipped_read) > 10 else ""
        warnings.warn(
            f"Skipped {len(skipped_read)} pair(s) (cv2.imread failed): {head}{ell}",
            UserWarning,
            stacklevel=2,
        )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f"Score-CAM for Chexit MobileNet. No arguments → batch {DEFAULT_CLI_BATCH_N} training "
            "images (original overlay + U-Net mask). Optional: path to one *_unetseg.png for single run."
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Optional: one U-Net segmented CXR (*_unetseg.png). If omitted, runs batch mode.",
    )
    parser.add_argument(
        "--batch-training",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Batch mode only: override number of images (default with no path: {DEFAULT_CLI_BATCH_N}). "
            "Ignored when a positional image path is given."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Batch mode: metadata CSV path (default: <out-dir>/scorecam_batch_metadata.csv)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=OFFICIAL_INFERENCE_FOLD,
        help=f"Weights fold (default: OFFICIAL_INFERENCE_FOLD={OFFICIAL_INFERENCE_FOLD})",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("scorecam_output"), help="Output directory")
    parser.add_argument("--prefix", type=str, default="tb_case_01", help="Single-image mode: output filename prefix")
    parser.add_argument(
        "--no-original-base",
        action="store_true",
        help="Single-image mode: do not look up Training/*.png for overlay; use segmented image as base",
    )
    args = parser.parse_args()

    if args.image is None:
        n_batch = args.batch_training if args.batch_training is not None else DEFAULT_CLI_BATCH_N
        csv_p = args.csv if args.csv is not None else (args.out_dir / "scorecam_batch_metadata.csv")
        df = batch_scorecam_training_original_overlay(
            n=n_batch,
            out_dir=args.out_dir,
            csv_path=csv_p,
            fold=args.fold,
        )
        print(
            f"Processed {len(df)} image(s) (requested {n_batch}); "
            "overlay uses Training originals + lung-masked heatmaps."
        )
        print(f"CSV: {csv_p.resolve()}")
        print(df.to_string(index=False))
        raise SystemExit(0)

    if args.batch_training is not None:
        warnings.warn(
            "Ignoring --batch-training because an image path was provided (single-image mode).",
            UserWarning,
            stacklevel=1,
        )

    img_path = args.image
    if not img_path.is_file():
        raise SystemExit(
            "Single-image mode: file not found.\n"
            f"Expected a path to unet_export/.../*_unetseg.png\nGot: {img_path}"
        )

    summary = run_scorecam_from_path(
        img_path,
        fold=args.fold,
        out_dir=args.out_dir,
        prefix=args.prefix,
        prefer_original_training_cxr=not args.no_original_base,
    )
    print(json.dumps(summary["metadata"], indent=2))
    print("saved_paths:", json.dumps(summary["saved_paths"], indent=2))
