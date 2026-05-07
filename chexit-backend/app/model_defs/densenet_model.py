"""
DenseNet-121 TB classifier pipeline for segmented chest X-rays.

This script keeps MobileNet-style structure while adding:
- DenseNet preprocess_input-based image preprocessing (IMG_SIZE=256)
- BatchNorm-safe fine-tuning
- Non-destructive U-Net export + manifest
- tf.data map/batch/prefetch pipeline (disk cache off by default; optional in-memory cache)
- Expanded Optuna search
- Optional held-out final test mode
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# DenseNet ImageNet preprocessing (identical to keras.applications.densenet.preprocess_input).
# Resolved via tf.keras to avoid basedpyright "could not resolve submodule" on tensorflow.keras.applications.densenet.
densenet_preprocess_input = tf.keras.applications.densenet.preprocess_input

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from optuna.integration import TFKerasPruningCallback

    OPTUNA_PRUNING_AVAILABLE = True
except Exception:
    OPTUNA_PRUNING_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "Training"
MONTGOMERY_CXR_DIR = TRAINING_DIR / "montgomery_cxr"
SHENZHEN_CXR_DIR = TRAINING_DIR / "shenzhen_cxr"
METADATA_DIR = TRAINING_DIR / "metadata"
SHENZHEN_METADATA = METADATA_DIR / "shenzhen_metadata.csv"
MONTGOMERY_METADATA = METADATA_DIR / "montgomery_metadata.csv"

UNET_MODELS_DIR = BASE_DIR / "models"
UNET_BEST_MODEL = UNET_MODELS_DIR / "unet_lung_seg_best.keras"
UNET_LEGACY_MODEL = UNET_MODELS_DIR / "unet_lung_seg_best.h5"
UNET_EXPORT_DIR = BASE_DIR / "unet_export"
UNET_EXPORT_MONTGOMERY = UNET_EXPORT_DIR / "montgomery"
UNET_EXPORT_SHENZHEN = UNET_EXPORT_DIR / "shenzhen"
UNET_INPUT_SIZE = 512

IMG_SIZE = 256
INPUT_CHANNELS = 3
RANDOM_SEED = 42
N_FOLDS = 5
DEFAULT_EPOCHS_PHASE1 = 12
DEFAULT_EPOCHS_PHASE2 = 8
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
DEFAULT_BATCH_SIZE = 16

OUTPUT_DIR = BASE_DIR / "dense_classifier_output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
# Legacy tf.data disk cache directory (older runs). Not used by default: Optuna creates many
# fold/trial datasets and disk cache can fill limited storage ("No space left on device").
CACHE_DIR = OUTPUT_DIR / "cache"
HISTORY_DIR = OUTPUT_DIR / "history"
OPTUNA_DB = OUTPUT_DIR / "optuna_study.db"
OPTUNA_BEST_PARAMS_PATH = OUTPUT_DIR / "optuna_best_params.json"
OPTUNA_BEST_TRIAL_PATH = OUTPUT_DIR / "optuna_best_trial.json"
METRICS_PATH = OUTPUT_DIR / "fold_metrics.json"
SUMMARY_METRICS_PATH = OUTPUT_DIR / "summary_metrics.json"
SUMMARY_FIGURE_PATH = OUTPUT_DIR / "summary_figure.png"
SEGMENTATION_MANIFEST_PATH = OUTPUT_DIR / "segmentation_manifest.csv"

USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DEFAULT_AUGMENTATION_STRENGTH = "light"
DEFAULT_ENABLE_HFLIP = False

BACKBONE_NAME_DEFAULT = "DenseNet121"
DENSENET121_SCORECAM_LAYER_NAME = "conv5_block16_concat"

# Print tf.data pipeline notice once per process (avoid spam from many make_dataset calls).
_MAKE_DATASET_INFO_PRINTED = False


def clear_tfdata_cache() -> None:
    """Delete ``dense_classifier_output/cache`` if present (legacy disk-cache files from older runs)."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"[cache] Removed legacy disk cache directory: {CACHE_DIR}")


def get_scorecam_layer_name(backbone_name: str = BACKBONE_NAME_DEFAULT) -> str:
    if backbone_name == "DenseNet121":
        return DENSENET121_SCORECAM_LAYER_NAME
    raise ValueError(f"No Score-CAM layer mapping for backbone: {backbone_name}")


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _get_unet_model_path() -> Path:
    if UNET_BEST_MODEL.exists():
        return UNET_BEST_MODEL
    if UNET_LEGACY_MODEL.exists():
        return UNET_LEGACY_MODEL
    raise FileNotFoundError(
        f"No U-Net model found at {UNET_BEST_MODEL} or {UNET_LEGACY_MODEL}. Run unet_segmentation.py first."
    )


def _load_unet_model() -> tf.keras.Model:
    return tf.keras.models.load_model(_get_unet_model_path(), compile=False)


def _source_image_path(source: str, study_id: str) -> Path:
    stem = Path(str(study_id)).stem
    return (SHENZHEN_CXR_DIR if source == "shenzhen" else MONTGOMERY_CXR_DIR) / f"{stem}.png"


def _segmented_image_path(source: str, study_id: str) -> Path:
    stem = Path(str(study_id)).stem
    return UNET_EXPORT_DIR / source / f"{stem}_unetseg.png"


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def _export_single_image(
    unet: tf.keras.Model,
    image_path: Path,
    out_path: Path,
    size: int = UNET_INPUT_SIZE,
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img_rs = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = np.reshape(img_rs.astype(np.float32) / 255.0, (1, size, size, 1))
    pred = unet.predict(x, verbose=0)
    mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)
    segmented = (img_rs.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), segmented)
    return True


def _write_manifest(rows: List[Dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(SEGMENTATION_MANIFEST_PATH, index=False)


def load_metadata_and_labels(include_missing: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for csv_path, source in [(SHENZHEN_METADATA, "shenzhen"), (MONTGOMERY_METADATA, "montgomery")]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "findings" not in df.columns or "study_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            study_id = str(r["study_id"]).strip()
            findings = str(r["findings"]).strip().lower()
            label = 0 if findings == "normal" else 1
            seg_path = _segmented_image_path(source, study_id)
            exists = seg_path.exists()
            if not include_missing and not exists:
                continue
            rows.append(
                {
                    "path": str(seg_path),
                    "label": label,
                    "source": source,
                    "study_id": study_id,
                    "segmented_exists": exists,
                }
            )
    if not rows:
        raise FileNotFoundError("No metadata rows found.")
    out = pd.DataFrame(rows)
    if not include_missing:
        out = out[out["segmented_exists"]].copy()
        if out.empty:
            raise FileNotFoundError(f"No segmented-image rows found in {UNET_EXPORT_DIR}.")
    return out.reset_index(drop=True)


def export_unet_segmentations(
    overwrite: bool = False,
    metadata_df: Optional[pd.DataFrame] = None,
) -> int:
    UNET_EXPORT_MONTGOMERY.mkdir(parents=True, exist_ok=True)
    UNET_EXPORT_SHENZHEN.mkdir(parents=True, exist_ok=True)
    if metadata_df is None:
        metadata_df = load_metadata_and_labels(include_missing=True)

    model: Optional[tf.keras.Model] = None
    exported = 0
    manifest_rows: List[Dict[str, Any]] = []

    for _, row in metadata_df.iterrows():
        study_id = str(row["study_id"])
        source = str(row["source"])
        label = int(row["label"])
        original_path = _source_image_path(source, study_id)
        segmented_path = _segmented_image_path(source, study_id)
        status = "skipped_exists"
        if not original_path.exists():
            status = "missing_original"
        elif segmented_path.exists() and not overwrite:
            status = "skipped_exists"
        else:
            if model is None:
                model = _load_unet_model()
            status = "exported" if _export_single_image(model, original_path, segmented_path) else "read_error"
            if status == "exported":
                exported += 1
        manifest_rows.append(
            {
                "original_path": str(original_path),
                "segmented_path": str(segmented_path),
                "source": source,
                "study_id": study_id,
                "label": label,
                "export_status": status,
            }
        )

    _write_manifest(manifest_rows)
    return exported


def preprocess_for_densenet(gray_or_rgb: np.ndarray, use_clahe: bool = USE_CLAHE, size: int = IMG_SIZE) -> np.ndarray:
    """DenseNet preprocessing contract: keep image-like values then call preprocess_input."""
    if gray_or_rgb.ndim == 2:
        gray = gray_or_rgb
    else:
        gray = cv2.cvtColor(gray_or_rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        gray = _apply_clahe(gray)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    return densenet_preprocess_input(rgb)


def load_image_for_classifier(path: str, size: int = IMG_SIZE, use_clahe: bool = USE_CLAHE) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return preprocess_for_densenet(img, use_clahe=use_clahe, size=size)


def _load_raw_rgb_np(path: bytes, use_clahe: bool, size: int) -> np.ndarray:
    img_path = path.decode("utf-8")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.float32)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        img = _apply_clahe(img)
    return np.stack([img, img, img], axis=-1).astype(np.float32)


def _load_raw_rgb_tf(path: tf.Tensor, use_clahe: bool, size: int) -> tf.Tensor:
    rgb = tf.numpy_function(
        lambda p: _load_raw_rgb_np(p, use_clahe=use_clahe, size=size),
        inp=[path],
        Tout=tf.float32,
    )
    rgb.set_shape((size, size, 3))
    return rgb


def _augmentation_layers(strength: str, enable_hflip: bool) -> tf.keras.Sequential:
    layers: List[tf.keras.layers.Layer] = []
    if strength in {"light", "moderate"}:
        rot = 0.02 if strength == "light" else 0.04
        zoom = 0.05 if strength == "light" else 0.10
        shift = 0.03 if strength == "light" else 0.05
        layers.extend(
            [
                tf.keras.layers.RandomRotation(rot, fill_mode="nearest"),
                tf.keras.layers.RandomZoom(height_factor=(-zoom, 0.0), width_factor=(-zoom, 0.0), fill_mode="nearest"),
                tf.keras.layers.RandomTranslation(height_factor=shift, width_factor=shift, fill_mode="nearest"),
            ]
        )
        if enable_hflip:
            layers.append(tf.keras.layers.RandomFlip(mode="horizontal"))
    return tf.keras.Sequential(layers, name="cxr_safe_augmentation")


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    params: Dict[str, Any],
    cache_name: str,
) -> tf.data.Dataset:
    """tf.data with map + batch + prefetch; CLAHE uses numpy_function for OpenCV interoperability.

    Disk ``Dataset.cache(path)`` is disabled by default. Set ``params['use_memory_cache']=True`` for
    in-memory ``ds.cache()`` only (can increase RAM use).
    """
    global _MAKE_DATASET_INFO_PRINTED
    if not _MAKE_DATASET_INFO_PRINTED:
        mem = bool(params.get("use_memory_cache", False))
        if mem:
            print("[tf.data] In-memory dataset cache enabled (use_memory_cache=True).")
        else:
            print("[tf.data] Disk caching disabled; using parallel map + batch + prefetch (AUTOTUNE).")
        _MAKE_DATASET_INFO_PRINTED = True

    use_clahe = bool(params.get("use_clahe", USE_CLAHE))
    augmentation_strength = str(params.get("augmentation_strength", DEFAULT_AUGMENTATION_STRENGTH))
    enable_hflip = bool(params.get("enable_hflip", DEFAULT_ENABLE_HFLIP))
    _ = cache_name  # reserved for logging/debug; disk cache stems removed

    ds = tf.data.Dataset.from_tensor_slices((paths.astype(str), labels.astype(np.float32)))
    options = tf.data.Options()
    options.experimental_deterministic = not training
    ds = ds.with_options(options)
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    aug_model = _augmentation_layers(augmentation_strength, enable_hflip)

    def _map_fn(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = _load_raw_rgb_tf(path, use_clahe=use_clahe, size=IMG_SIZE)
        if training and augmentation_strength != "none":
            image = aug_model(image, training=True)
            if augmentation_strength == "light":
                image = tf.image.random_brightness(image, max_delta=6.0)
                image = tf.image.random_contrast(image, 0.95, 1.05)
            elif augmentation_strength == "moderate":
                image = tf.image.random_brightness(image, max_delta=10.0)
                image = tf.image.random_contrast(image, 0.90, 1.10)
            image = tf.clip_by_value(image, 0.0, 255.0)
        image = densenet_preprocess_input(image)
        image = tf.ensure_shape(image, [IMG_SIZE, IMG_SIZE, 3])
        return image, tf.cast(label, tf.float32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if bool(params.get("use_memory_cache", False)):
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_dataset_arrays(
    df: pd.DataFrame,
    indices: np.ndarray,
    size: int = IMG_SIZE,
    use_clahe: bool = USE_CLAHE,
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in indices:
        row = df.iloc[i]
        X.append(load_image_for_classifier(row["path"], size=size, use_clahe=use_clahe))
        y.append(row["label"])
    return np.array(X), np.array(y, dtype=np.float32)


def _get_backbone_layer(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "densenet" in layer.name.lower():
            return layer
    raise ValueError("DenseNet backbone layer not found.")


def build_model(
    backbone_name: str = BACKBONE_NAME_DEFAULT,
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS),
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    l2_strength: float = 1e-4,
    head_depth: int = 1,
) -> tf.keras.Model:
    if backbone_name != "DenseNet121":
        raise ValueError(f"Unknown backbone: {backbone_name}.")
    backbone = tf.keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    # Intentional: keep BN inference behavior to avoid moving-stat corruption during fine-tuning.
    x = backbone(inputs, training=False)
    reg = tf.keras.regularizers.l2(l2_strength)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="head_dense_1")(x)
    if head_depth >= 2:
        x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_2")(x)
        x = tf.keras.layers.Dense(max(32, dense_units // 2), activation="relu", kernel_regularizer=reg, name="head_dense_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_final")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="tb_output")(x)
    return tf.keras.Model(inputs, outputs, name="densenet121_tb_classifier")


def unfreeze_top_fraction(model: tf.keras.Model, fraction: float) -> None:
    """Unfreeze only top non-BN backbone layers; all BatchNorm layers stay frozen."""
    backbone = _get_backbone_layer(model)
    backbone.trainable = True
    candidates = [
        layer
        for layer in backbone.layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
        and not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    n_unfreeze = max(1, int(len(candidates) * float(np.clip(fraction, 0.0, 1.0))))
    trainable_set = set(candidates[-n_unfreeze:])
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = layer in trainable_set


def _create_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    lr_schedule: str,
    steps_per_epoch: int,
    epochs: int,
) -> tf.keras.optimizers.Optimizer:
    lr: Any = learning_rate
    if lr_schedule == "cosine_decay":
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(1, steps_per_epoch * max(1, epochs)),
            alpha=0.05,
        )
    name = optimizer_name.lower()
    if name == "adamw":
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _phase_callbacks(
    fold: int,
    phase_name: str,
    lr_schedule: str,
    trial: Optional["optuna.Trial"] = None,
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WEIGHTS_DIR / f"fold_{fold}_{phase_name}_best.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if lr_schedule == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1,
            )
        )
    if trial is not None and OPTUNA_PRUNING_AVAILABLE:
        callbacks.append(TFKerasPruningCallback(trial, "val_auc"))
    return callbacks


def _history_to_files(history: tf.keras.callbacks.History, fold: int, phase_name: str) -> Dict[str, float]:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(HISTORY_DIR / f"fold_{fold}_{phase_name}_history.csv", index=False)
    with open(HISTORY_DIR / f"fold_{fold}_{phase_name}_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    if "val_auc" in history.history and history.history["val_auc"]:
        idx = int(np.argmax(history.history["val_auc"]))
        return {"best_epoch": float(idx + 1), "best_val_auc": float(history.history["val_auc"][idx])}
    return {"best_epoch": -1.0, "best_val_auc": 0.0}


def _predict_from_dataset(model: tf.keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    return model.predict(ds, verbose=0).flatten().astype(np.float32)


def train_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    params: Dict[str, Any],
    fold: int,
    class_weight: Optional[Dict[int, float]] = None,
    trial: Optional["optuna.Trial"] = None,
) -> Tuple[tf.keras.Model, Dict[str, float], np.ndarray, np.ndarray]:
    set_seeds(RANDOM_SEED + fold)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.float32)
    if class_weight is None:
        class_weight = {0: 1.0, 1: 1.0}

    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    train_ds = make_dataset(train_df["path"].to_numpy(), y_train, batch_size, True, params, f"fold_{fold}_train")
    val_ds = make_dataset(val_df["path"].to_numpy(), y_val, batch_size, False, params, f"fold_{fold}_val")

    model = build_model(
        backbone_name=params.get("backbone_name", BACKBONE_NAME_DEFAULT),
        dense_units=int(params.get("dense_units", 128)),
        dropout_rate=float(params.get("dropout_rate", 0.4)),
        l2_strength=float(params.get("l2_strength", 1e-4)),
        head_depth=int(params.get("head_depth", 1)),
    )
    optimizer_name = str(params.get("optimizer", "adam"))
    weight_decay = float(params.get("weight_decay", 1e-5))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    lr_schedule = str(params.get("lr_schedule", "plateau"))
    steps = max(1, int(np.ceil(len(train_df) / batch_size)))
    epochs_phase1 = int(params.get("epochs_phase1", DEFAULT_EPOCHS_PHASE1))
    epochs_phase2 = int(params.get("epochs_phase2", DEFAULT_EPOCHS_PHASE2))

    model.compile(
        optimizer=_create_optimizer(
            optimizer_name,
            learning_rate=float(params.get("head_lr", 1e-3)),
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            steps_per_epoch=steps,
            epochs=epochs_phase1,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weight,
        callbacks=_phase_callbacks(fold, "phase1", lr_schedule, trial),
        verbose=1,
    )
    info1 = _history_to_files(hist1, fold, "phase1")

    unfreeze_top_fraction(model, float(params.get("unfreeze_ratio", 0.2)))
    model.compile(
        optimizer=_create_optimizer(
            optimizer_name,
            learning_rate=float(params.get("ft_lr", 1e-5)),
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            steps_per_epoch=steps,
            epochs=epochs_phase2,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        class_weight=class_weight,
        callbacks=_phase_callbacks(fold, "phase2", lr_schedule, trial),
        verbose=1,
    )
    info2 = _history_to_files(hist2, fold, "phase2")

    ckpt = WEIGHTS_DIR / f"fold_{fold}_phase2_best.weights.h5"
    if not ckpt.exists():
        ckpt = WEIGHTS_DIR / f"fold_{fold}_phase1_best.weights.h5"
    if ckpt.exists():
        model.load_weights(ckpt)
    model.save_weights(WEIGHTS_DIR / f"fold_{fold}_best.weights.h5")

    val_proba = _predict_from_dataset(model, val_ds)
    threshold = find_threshold_for_sensitivity(y_val, val_proba, float(params.get("target_sensitivity", 0.9)))
    val_pred = (val_proba >= threshold).astype(np.int32)
    metrics = compute_metrics(y_val, val_pred, val_proba)
    metrics.update(
        {
            "best_epoch_phase1": info1["best_epoch"],
            "best_val_auc_phase1": info1["best_val_auc"],
            "best_epoch_phase2": info2["best_epoch"],
            "best_val_auc_phase2": info2["best_val_auc"],
            "threshold": float(threshold),
        }
    )
    with open(HISTORY_DIR / f"fold_{fold}_meta.json", "w") as f:
        json.dump({"fold": fold, "params": params, "metrics": metrics, "threshold": float(threshold)}, f, indent=2)
    return model, metrics, y_val, val_proba


def find_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_sensitivity: float = 0.9,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = np.where(tpr >= target_sensitivity)[0]
    if len(idx) == 0:
        return 0.5
    i = idx[np.argmin(fpr[idx])]
    return float(thresholds[i]) if i < len(thresholds) else 0.5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.0
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auc": float(auc),
    }


def run_cv(
    df: pd.DataFrame,
    params: Dict[str, Any],
    use_class_weight: bool = True,
    trial: Optional["optuna.Trial"] = None,
    metrics_output_path: Optional[Path] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y = df["label"].values
    fold_metrics: Dict[int, Dict[str, float]] = {}
    all_metrics: Dict[str, List[float]] = {k: [] for k in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]}
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        class_weight = None
        if use_class_weight:
            cw = compute_class_weight("balanced", classes=np.unique(y[train_idx]), y=y[train_idx])
            class_weight = {int(c): float(w) for c, w in zip(np.unique(y[train_idx]), cw)}
        _, metrics, _, _ = train_fold(
            train_idx=train_idx,
            val_idx=val_idx,
            df=df,
            params=params,
            fold=fold,
            class_weight=class_weight,
            trial=trial,
        )
        fold_metrics[fold] = metrics
        for k in all_metrics:
            all_metrics[k].append(float(metrics[k]))
    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_metrics = {k: float(np.std(v)) for k, v in all_metrics.items()}
    if metrics_output_path is not None:
        with open(metrics_output_path, "w") as f:
            json.dump({"fold": fold_metrics, "mean": mean_metrics, "std": std_metrics}, f, indent=2)
    return fold_metrics, mean_metrics, std_metrics


def get_default_params() -> Dict[str, Any]:
    return {
        "backbone_name": BACKBONE_NAME_DEFAULT,
        "optimizer": "adam",
        "head_lr": 8e-4,
        "ft_lr": 8e-6,
        "weight_decay": 1e-5,
        "dropout_rate": 0.4,
        "dense_units": 256,
        "head_depth": 1,
        "l2_strength": 1e-4,
        "label_smoothing": 0.02,
        "batch_size": DEFAULT_BATCH_SIZE,
        "unfreeze_ratio": 0.2,
        "use_clahe": USE_CLAHE,
        "augmentation_strength": DEFAULT_AUGMENTATION_STRENGTH,
        "enable_hflip": DEFAULT_ENABLE_HFLIP,
        "lr_schedule": "plateau",
        "epochs_phase1": DEFAULT_EPOCHS_PHASE1,
        "epochs_phase2": DEFAULT_EPOCHS_PHASE2,
        "target_sensitivity": 0.9,
        "use_memory_cache": False,
    }


def _trial_to_serializable(trial: "optuna.trial.FrozenTrial") -> Dict[str, Any]:
    return {
        "number": trial.number,
        "value": trial.value,
        "state": str(trial.state),
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }


def _write_optuna_best_artifacts(study: "optuna.Study") -> None:
    """Persist merged best params and a rich best-trial record (call when study.best_trial is valid)."""
    bt = study.best_trial
    merged = get_default_params()
    merged.update(bt.params)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    payload = {
        "trial_number": bt.number,
        "value": bt.value,
        "params_merged_with_defaults": merged,
        "optuna_params": dict(bt.params),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "mean_sensitivity": bt.user_attrs.get("mean_sensitivity"),
        "mean_specificity": bt.user_attrs.get("mean_specificity"),
    }
    with open(OPTUNA_BEST_TRIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _optuna_save_best_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    try:
        best = study.best_trial
    except RuntimeError:
        return
    if best.number != trial.number:
        return
    _write_optuna_best_artifacts(study)
    print(
        f"[optuna] New best trial {best.number} (value={best.value}); "
        f"wrote {OPTUNA_BEST_PARAMS_PATH.name} and {OPTUNA_BEST_TRIAL_PATH.name}."
    )


def run_optuna_study(df: pd.DataFrame, n_trials: int = 25) -> "optuna.Study":
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not available.")

    def objective(trial: optuna.Trial) -> float:
        defaults = get_default_params()
        params = {
            "backbone_name": BACKBONE_NAME_DEFAULT,
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"]),
            "head_lr": trial.suggest_float("head_lr", 1e-5, 3e-3, log=True),
            "ft_lr": trial.suggest_float("ft_lr", 1e-7, 3e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
            "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256, 512]),
            "head_depth": trial.suggest_categorical("head_depth", [1, 2]),
            "l2_strength": trial.suggest_float("l2_strength", 1e-6, 1e-3, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [16]),
            "unfreeze_ratio": trial.suggest_float("unfreeze_ratio", 0.05, 0.40),
            "use_clahe": trial.suggest_categorical("use_clahe", [True, False]),
            "augmentation_strength": trial.suggest_categorical("augmentation_strength", ["none", "light", "moderate"]),
            "enable_hflip": trial.suggest_categorical("enable_hflip", [False, True]),
            "lr_schedule": trial.suggest_categorical("lr_schedule", ["plateau", "cosine_decay", "none"]),
            "epochs_phase1": DEFAULT_EPOCHS_PHASE1,
            "epochs_phase2": DEFAULT_EPOCHS_PHASE2,
            "target_sensitivity": 0.9,
            "use_memory_cache": bool(defaults.get("use_memory_cache", False)),
        }
        _, mean_metrics, _ = run_cv(df, params, use_class_weight=True, trial=trial)
        trial.set_user_attr("mean_sensitivity", float(mean_metrics["sensitivity"]))
        trial.set_user_attr("mean_specificity", float(mean_metrics["specificity"]))
        return float(mean_metrics["auc"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="densenet121_tb_optuna",
        direction="maximize",
        sampler=TPESampler(n_startup_trials=8, seed=RANDOM_SEED),
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
    )
    print("[optuna] Best parameters will be saved whenever a new best trial completes.")
    optimize_kwargs: Dict[str, Any] = {
        "n_trials": n_trials,
        "show_progress_bar": True,
        "callbacks": [_optuna_save_best_callback],
    }
    try:
        study.optimize(objective, catch=(tf.errors.ResourceExhaustedError,), **optimize_kwargs)
    except TypeError:
        study.optimize(objective, **optimize_kwargs)

    trials_data = [_trial_to_serializable(t) for t in study.trials]
    with open(OUTPUT_DIR / "optuna_trials.json", "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)
    pd.DataFrame(trials_data).to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)

    try:
        _write_optuna_best_artifacts(study)
    except RuntimeError:
        print("[optuna] No completed trials; skipped writing best JSON files.")
    return study


def load_best_params_from_json(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load merged hyperparameters saved after Optuna (or hand-edited). Returns None if missing."""
    path = path or OPTUNA_BEST_PARAMS_PATH
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_aggregate_roc_cm(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y = df["label"].values
    all_true: List[float] = []
    all_proba: List[float] = []
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        val_df = df.iloc[val_idx].reset_index(drop=True)
        y_val = val_df["label"].to_numpy(dtype=np.float32)
        model = build_model(
            backbone_name=params.get("backbone_name", BACKBONE_NAME_DEFAULT),
            dense_units=int(params.get("dense_units", 128)),
            dropout_rate=float(params.get("dropout_rate", 0.4)),
            l2_strength=float(params.get("l2_strength", 1e-4)),
            head_depth=int(params.get("head_depth", 1)),
        )
        model.load_weights(WEIGHTS_DIR / f"fold_{fold}_best.weights.h5")
        val_ds = make_dataset(
            val_df["path"].to_numpy(),
            y_val,
            batch_size=int(params.get("batch_size", DEFAULT_BATCH_SIZE)),
            training=False,
            params=params,
            cache_name=f"summary_fold_{fold}",
        )
        p = _predict_from_dataset(model, val_ds)
        all_true.extend(y_val.tolist())
        all_proba.extend(p.tolist())
    all_true_np = np.array(all_true, dtype=np.float32)
    all_proba_np = np.array(all_proba, dtype=np.float32)
    threshold = find_threshold_for_sensitivity(
        all_true_np,
        all_proba_np,
        target_sensitivity=float(params.get("target_sensitivity", 0.9)),
    )
    all_pred = (all_proba_np >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(all_true_np, all_proba_np)
    cm = confusion_matrix(all_true_np, all_pred, labels=[0, 1])
    return fpr, tpr, cm, all_proba_np


def save_summary_figure(
    df: pd.DataFrame,
    params: Dict[str, Any],
    mean_metrics: Dict[str, float],
    out_path: Path = SUMMARY_FIGURE_PATH,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, cm, _ = _compute_aggregate_roc_cm(df, params)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {mean_metrics['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (DenseNet-121)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Non-TB", "TB"])
    axes[1].set_yticklabels(["Non-TB", "TB"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im, ax=axes[1], shrink=0.6)
    names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC"]
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    values = [mean_metrics[k] for k in keys]
    axes[2].barh(names, values, color="steelblue", alpha=0.8)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Score")
    axes[2].set_title("Mean metrics (5-fold CV)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _train_single_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: Dict[str, Any],
    tag: str,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.float32)
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), cw)}
    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    train_ds = make_dataset(train_df["path"].to_numpy(), y_train, batch_size, True, params, f"{tag}_train")
    val_ds = make_dataset(val_df["path"].to_numpy(), y_val, batch_size, False, params, f"{tag}_val")

    model = build_model(
        backbone_name=params.get("backbone_name", BACKBONE_NAME_DEFAULT),
        dense_units=int(params.get("dense_units", 128)),
        dropout_rate=float(params.get("dropout_rate", 0.4)),
        l2_strength=float(params.get("l2_strength", 1e-4)),
        head_depth=int(params.get("head_depth", 1)),
    )
    optimizer_name = str(params.get("optimizer", "adam"))
    weight_decay = float(params.get("weight_decay", 1e-5))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    lr_schedule = str(params.get("lr_schedule", "plateau"))
    steps = max(1, int(np.ceil(len(train_df) / batch_size)))
    e1 = int(params.get("epochs_phase1", DEFAULT_EPOCHS_PHASE1))
    e2 = int(params.get("epochs_phase2", DEFAULT_EPOCHS_PHASE2))

    model.compile(
        optimizer=_create_optimizer(optimizer_name, float(params.get("head_lr", 1e-3)), weight_decay, lr_schedule, steps, e1),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=e1,
        callbacks=_phase_callbacks(0, f"{tag}_phase1", lr_schedule, None),
        class_weight=class_weight,
        verbose=1,
    )
    unfreeze_top_fraction(model, float(params.get("unfreeze_ratio", 0.2)))
    model.compile(
        optimizer=_create_optimizer(optimizer_name, float(params.get("ft_lr", 1e-5)), weight_decay, lr_schedule, steps, e2),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=e2,
        callbacks=_phase_callbacks(0, f"{tag}_phase2", lr_schedule, None),
        class_weight=class_weight,
        verbose=1,
    )
    ckpt = WEIGHTS_DIR / f"fold_0_{tag}_phase2_best.weights.h5"
    if not ckpt.exists():
        ckpt = WEIGHTS_DIR / f"fold_0_{tag}_phase1_best.weights.h5"
    if ckpt.exists():
        model.load_weights(ckpt)
    val_proba = _predict_from_dataset(model, val_ds)
    threshold = find_threshold_for_sensitivity(y_val, val_proba, float(params.get("target_sensitivity", 0.9)))
    history_meta = {
        "phase1_epochs": len(h1.history.get("loss", [])),
        "phase2_epochs": len(h2.history.get("loss", [])),
        "phase1_best_val_auc": float(np.max(h1.history.get("val_auc", [0.0]))),
        "phase2_best_val_auc": float(np.max(h2.history.get("val_auc", [0.0]))),
    }
    return model, y_val, val_proba, threshold, history_meta


def _run_heldout_test_protocol(
    df: pd.DataFrame,
    run_optuna: bool,
    n_optuna_trials: int,
    final_test_size: float,
    use_best_params: bool = True,
) -> None:
    train_pool_df, final_test_df = train_test_split(
        df,
        test_size=final_test_size,
        stratify=df["label"].values,
        random_state=RANDOM_SEED,
    )
    train_pool_df = train_pool_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    if run_optuna and OPTUNA_AVAILABLE:
        study = run_optuna_study(train_pool_df, n_trials=n_optuna_trials)
        params = get_default_params()
        params.update(study.best_params)
    else:
        params = get_default_params()
        if use_best_params:
            loaded = load_best_params_from_json()
            if loaded:
                params.update(loaded)
                print(f"Loaded hyperparameters from {OPTUNA_BEST_PARAMS_PATH}")

    fold_metrics, mean_metrics, std_metrics = run_cv(
        train_pool_df,
        params,
        use_class_weight=True,
        trial=None,
        metrics_output_path=OUTPUT_DIR / "training_pool_cv_metrics.json",
    )
    pd.DataFrame(fold_metrics).T.to_csv(OUTPUT_DIR / "training_pool_cv_metrics.csv", index=True)

    final_train_df, final_val_df = train_test_split(
        train_pool_df,
        test_size=0.15,
        stratify=train_pool_df["label"].values,
        random_state=RANDOM_SEED,
    )
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    model, _, _, threshold, history_meta = _train_single_split(final_train_df, final_val_df, params, "final_model")

    test_y = final_test_df["label"].to_numpy(dtype=np.float32)
    test_ds = make_dataset(
        final_test_df["path"].to_numpy(),
        test_y,
        batch_size=int(params.get("batch_size", DEFAULT_BATCH_SIZE)),
        training=False,
        params=params,
        cache_name="final_test",
    )
    test_proba = _predict_from_dataset(model, test_ds)
    test_auc = float(roc_auc_score(test_y, test_proba)) if len(np.unique(test_y)) > 1 else 0.0
    test_pred = (test_proba >= threshold).astype(np.int32)
    test_metrics = compute_metrics(test_y, test_pred, test_proba)

    fpr, tpr, th = roc_curve(test_y, test_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": th}).to_csv(OUTPUT_DIR / "final_test_roc.csv", index=False)
    cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(OUTPUT_DIR / "final_test_confusion_matrix.csv")
    pred_df = final_test_df[["path", "label", "source", "study_id"]].copy()
    pred_df["proba_tb"] = test_proba
    pred_df["pred_tb"] = test_pred
    pred_df.to_csv(OUTPUT_DIR / "final_test_predictions.csv", index=False)

    with open(OUTPUT_DIR / "final_threshold.json", "w") as f:
        json.dump({"selection_source": "validation_only", "threshold": float(threshold)}, f, indent=2)
    with open(OUTPUT_DIR / "final_test_metrics.json", "w") as f:
        json.dump(
            {
                "label": "Held-out final test performance",
                "threshold_free_test_auc": test_auc,
                "thresholded_metrics": test_metrics,
                "params": params,
                "history_meta": history_meta,
                "training_pool_cv_label": "Optuna internal CV performance on training pool",
                "training_pool_cv_mean": mean_metrics,
                "training_pool_cv_std": std_metrics,
            },
            f,
            indent=2,
        )


def main(
    run_export: bool = True,
    run_optuna: bool = False,
    n_optuna_trials: int = 20,
    use_best_params: bool = True,
    evaluation_mode: str = "cv_only",
    final_test_size: float = 0.10,
    auto_export_if_missing: bool = True,
) -> None:
    _ = use_best_params
    set_seeds(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Raw CXR -> U-Net segmentation export (first stage, MobileNet-style flow)
    metadata_all = load_metadata_and_labels(include_missing=True)
    if run_export:
        n = export_unet_segmentations(overwrite=False, metadata_df=metadata_all)
        print(f"U-Net export: {n} images exported (non-destructive).")
    elif auto_export_if_missing:
        missing_count = int((~metadata_all["segmented_exists"]).sum())
        if missing_count > 0:
            n = export_unet_segmentations(overwrite=False, metadata_df=metadata_all)
            print(f"Auto-exported missing segmentations: {n}")
    else:
        print("Skipping U-Net export (run_export=False).")

    # 2) Load segmented-image paths + labels
    df = load_metadata_and_labels(include_missing=False)
    print(f"Loaded {len(df)} segmented samples with labels.")

    # 3) DenseNet training / Optuna / evaluation
    if evaluation_mode == "heldout_test":
        _run_heldout_test_protocol(
            df, run_optuna, n_optuna_trials, final_test_size, use_best_params=use_best_params
        )
        print("Done (heldout_test mode).")
        return

    params = get_default_params()
    if run_optuna and OPTUNA_AVAILABLE:
        study = run_optuna_study(df, n_trials=n_optuna_trials)
        params.update(study.best_params)
    elif use_best_params:
        loaded = load_best_params_from_json()
        if loaded:
            params.update(loaded)
            print(f"Loaded hyperparameters from {OPTUNA_BEST_PARAMS_PATH}")

    fold_metrics, mean_metrics, std_metrics = run_cv(
        df,
        params,
        use_class_weight=True,
        trial=None,
        metrics_output_path=METRICS_PATH,
    )
    with open(SUMMARY_METRICS_PATH, "w") as f:
        json.dump({"mean": mean_metrics, "std": std_metrics}, f, indent=2)
    pd.DataFrame(fold_metrics).T.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=True)
    save_summary_figure(df, params, mean_metrics)
    print("Done (cv_only mode).")


if __name__ == "__main__":
    main(
        run_export=True,
        run_optuna=True,
        n_optuna_trials=20,
        use_best_params=True,
        evaluation_mode="heldout_test",
        final_test_size=0.10,
        auto_export_if_missing=True,
    )
