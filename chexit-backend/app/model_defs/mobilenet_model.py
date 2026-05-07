"""
MobileNet-based binary TB classifier on U-Net-segmented chest X-rays.

Design summary:
- Backbone: MobileNetV3Large by default (supports MobileNetV2 for comparison).
- MobileNetV3 preprocessing: include_preprocessing=True with classifier inputs in [0, 255] float32.
- Training: Phase 1 frozen backbone + Phase 2 fine-tuning top fraction (BatchNorm frozen).
- Optuna objective balances mean AUC and specificity at sensitivity >= 0.90.
- Evaluation uses out-of-fold (OOF) predictions from 5-fold CV (no redundant re-prediction pass).
- Operating point uses highest-specificity threshold that still meets sensitivity >= 0.90.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Optional Optuna; skip tuning if not installed
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "Training"
MONTGOMERY_CXR_DIR = TRAINING_DIR / "montgomery_cxr"
SHENZHEN_CXR_DIR = TRAINING_DIR / "shenzhen_cxr"
METADATA_DIR = TRAINING_DIR / "metadata"
SHENZHEN_METADATA = METADATA_DIR / "shenzhen_metadata.csv"
MONTGOMERY_METADATA = METADATA_DIR / "montgomery_metadata.csv"

# U-Net export (from unet_segmentation.py)
UNET_MODELS_DIR = BASE_DIR / "models"
UNET_BEST_MODEL = UNET_MODELS_DIR / "unet_lung_seg_best.keras"
UNET_LEGACY_MODEL = UNET_MODELS_DIR / "unet_lung_seg_best.h5"
UNET_EXPORT_DIR = BASE_DIR / "unet_export"
UNET_EXPORT_MONTGOMERY = UNET_EXPORT_DIR / "montgomery"
UNET_EXPORT_SHENZHEN = UNET_EXPORT_DIR / "shenzhen"
UNET_INPUT_SIZE = 512

# Classifier
IMG_SIZE = 224
INPUT_CHANNELS = 3
RANDOM_SEED = 42
N_FOLDS = 5
DEFAULT_EPOCHS_PHASE1 = 15
DEFAULT_EPOCHS_PHASE2 = 10
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
DEFAULT_BATCH_SIZE = 16
OUTPUT_DIR = BASE_DIR / "tb_classifier_output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
OPTUNA_DB = OUTPUT_DIR / "optuna_study.db"
METRICS_PATH = OUTPUT_DIR / "fold_metrics.json"
SUMMARY_METRICS_PATH = OUTPUT_DIR / "summary_metrics.json"
SUMMARY_FIGURE_PATH = OUTPUT_DIR / "summary_figure.png"
BEST_PARAMS_PATH = OUTPUT_DIR / "optuna_best_params.json"
FOLD_METRICS_CSV_PATH = OUTPUT_DIR / "fold_metrics.csv"
THRESHOLD_PER_FOLD_PATH = OUTPUT_DIR / "threshold_per_fold.csv"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "oof_predictions.csv"
OPERATING_POINT_SUMMARY_PATH = OUTPUT_DIR / "operating_point_summary.json"

# CLAHE (applied to grayscale before 3ch stack; used in load_image_for_classifier)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------------------------------------------------------------------------
# Seeds and reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Part 1: U-Net export — apply segmentation to original CXRs
# ---------------------------------------------------------------------------

def _get_unet_model_path() -> Path:
    if UNET_BEST_MODEL.exists():
        return UNET_BEST_MODEL
    if UNET_LEGACY_MODEL.exists():
        return UNET_LEGACY_MODEL
    raise FileNotFoundError(
        f"No U-Net model found at {UNET_BEST_MODEL} or {UNET_LEGACY_MODEL}. Run unet_segmentation.py first."
    )


def _load_unet_model() -> tf.keras.Model:
    path = _get_unet_model_path()
    return tf.keras.models.load_model(path, compile=False)


def _export_single_image(
    unet: tf.keras.Model,
    image_path: Path,
    out_path: Path,
    size: int = UNET_INPUT_SIZE,
) -> None:
    """Load CXR, predict mask with U-Net, save segmented image (lung region only) as grayscale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    h, w = img.shape
    img_rs = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = img_rs.astype(np.float32) / 255.0
    x = np.reshape(x, (1, size, size, 1))
    pred = unet.predict(x, verbose=0)
    mask = (pred[0, :, :, 0] > 0.5).astype(np.float32)
    segmented = (img_rs.astype(np.float32) / 255.0 * mask * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), segmented)


def _clear_export_dir() -> None:
    """Remove all files in unet_export and its subdirs so no duplicates or stale files."""
    if not UNET_EXPORT_DIR.exists():
        return
    for p in UNET_EXPORT_DIR.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def export_unet_segmentations(overwrite: bool = True) -> int:
    """
    Apply the best U-Net from unet_segmentation.py to all images in
    montgomery_cxr and shenzhen_cxr. Exports to clear, non-duplicated paths:

    - unet_export/montgomery/<stem>_unetseg.png
    - unet_export/shenzhen/<stem>_unetseg.png

    The export directory is cleared at the start so only this run's files exist.
    Returns the number of exported images.
    """
    _clear_export_dir()
    UNET_EXPORT_MONTGOMERY.mkdir(parents=True, exist_ok=True)
    UNET_EXPORT_SHENZHEN.mkdir(parents=True, exist_ok=True)
    unet = _load_unet_model()
    count = 0
    for folder, out_subdir in (
        (MONTGOMERY_CXR_DIR, UNET_EXPORT_MONTGOMERY),
        (SHENZHEN_CXR_DIR, UNET_EXPORT_SHENZHEN),
    ):
        if not folder.exists():
            continue
        for path in folder.glob("*.png"):
            stem = path.stem
            out_path = out_subdir / f"{stem}_unetseg.png"
            _export_single_image(unet, path, out_path)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Part 2: Metadata and labels
# ---------------------------------------------------------------------------

def load_metadata_and_labels() -> pd.DataFrame:
    """
    Load both metadata CSVs, merge, and map to (segmented_path, label).
    findings == 'normal' -> 0 (non-TB), else 1 (TB).
    study_id matches image filename; segmented file is unet_export/<source>/<stem>_unetseg.png.
    """
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
            stem = Path(study_id).stem
            seg_path = UNET_EXPORT_DIR / source / f"{stem}_unetseg.png"
            if not seg_path.exists():
                continue
            rows.append({"path": str(seg_path), "label": label, "source": source, "study_id": study_id})
    if not rows:
        raise FileNotFoundError(
            f"No (metadata, segmented image) pairs found. Ensure {UNET_EXPORT_DIR} contains "
            f"*_unetseg.png and metadata CSVs have study_id and findings."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: Data loading for classifier (resize, optional CLAHE, 3ch)
# ---------------------------------------------------------------------------

def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale uint8 image. Returns uint8."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def load_image_for_classifier(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    Load segmented image for classifier.
    Preferred MobileNetV3 path: return float32 pixels in [0, 255] and use include_preprocessing=True.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if USE_CLAHE:
        img = _apply_clahe(img)
    img = np.stack([img, img, img], axis=-1)
    return img.astype(np.float32)


def build_dataset_arrays(
    df: pd.DataFrame,
    indices: np.ndarray,
    size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels for a subset of df given by indices."""
    X = []
    y = []
    for i in indices:
        row = df.iloc[i]
        x = load_image_for_classifier(row["path"], size=size)
        X.append(x)
        y.append(row["label"])
    return np.array(X), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Part 4: Model builder (MobileNetV3Large default; MobileNetV2 optional)
# ---------------------------------------------------------------------------

def build_model(
    backbone_name: str = "MobileNetV3Large",
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS),
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    l2_strength: float = 1e-4,
    use_augmentation: bool = False,
) -> tf.keras.Model:
    """Build classifier with optional light augmentation before ImageNet backbone."""
    if backbone_name == "MobileNetV2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
    elif backbone_name == "MobileNetV3Large":
        backbone = tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling=None,
            include_preprocessing=True,
            minimalistic=False,
            alpha=1.0,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    backbone.trainable = False
    reg = tf.keras.regularizers.l2(l2_strength)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if use_augmentation:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.03),
                tf.keras.layers.RandomZoom(0.05),
                tf.keras.layers.RandomTranslation(0.03, 0.03),
                tf.keras.layers.RandomContrast(0.10),
            ],
            name="train_augmentation",
        )
        x = aug(x)
    if backbone_name == "MobileNetV2":
        # Keep a single input convention ([0,255] float32) and normalize only for MobileNetV2.
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, out)
    return model


def freeze_batchnorm_layers(model: tf.keras.Model) -> None:
    """Keep all BatchNormalization layers frozen during fine-tuning."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        if isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.BatchNormalization):
                    sublayer.trainable = False


def unfreeze_top_fraction(model: tf.keras.Model, fraction: float) -> None:
    """
    Unfreeze the top `fraction` (0–1) of backbone layers.
    Keras 3 applications may use Input + Rescaling + backbone; we unfreeze by
    model layers (backbone = everything before the first GlobalAveragePooling2D).
    """
    freeze_batchnorm_layers(model)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            "mobilenetv3" in layer.name.lower() or "mobilenetv2" in layer.name.lower()
        ):
            backbone = layer
            break
    if backbone is None:
        return
    backbone.trainable = True
    conv_candidates = []
    for layer in backbone.layers:
        lname = layer.__class__.__name__.lower()
        layer.trainable = False
        if "batchnormalization" in lname:
            continue
        if any(k in lname for k in ["conv", "depthwise", "pointwise"]):
            conv_candidates.append(layer)
    if not conv_candidates:
        return
    n_unfreeze = max(1, int(len(conv_candidates) * fraction))
    for layer in conv_candidates[-n_unfreeze:]:
        layer.trainable = True
    freeze_batchnorm_layers(model)


# ---------------------------------------------------------------------------
# Part 5: Training (phase 1 + phase 2)
# ---------------------------------------------------------------------------

def train_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    params: Dict[str, Any],
    fold: int,
    class_weight: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train one fold: phase 1 (frozen backbone) then phase 2 (unfreeze top).
    Returns model, val metrics dict, val_y_true, val_y_pred_proba.
    """
    set_seeds(RANDOM_SEED + fold)
    X_train, y_train = build_dataset_arrays(df, train_idx)
    X_val, y_val = build_dataset_arrays(df, val_idx)
    if class_weight is None:
        class_weight = {0: 1.0, 1: 1.0}

    model = build_model(
        backbone_name=params.get("backbone_name", "MobileNetV3Large"),
        dense_units=params.get("dense_units", 128),
        dropout_rate=params.get("dropout_rate", 0.4),
        l2_strength=params.get("l2_strength", 1e-4),
        use_augmentation=params.get("use_augmentation", False),
    )
    loss_type = params.get("loss_type", "bce")
    if loss_type == "focal":
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=params.get("focal_gamma", 2.0),
            alpha=params.get("focal_alpha", 0.5),
        )
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get("head_lr", 1e-3)),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    fold_ckpt_path = WEIGHTS_DIR / f"fold_{fold}_best_val_auc.weights.h5"
    fold_csv_log = OUTPUT_DIR / f"fold_{fold}_train_log.csv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=fold_ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(fold_csv_log)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    # Phase 1
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params.get("epochs_phase1", DEFAULT_EPOCHS_PHASE1),
        batch_size=params.get("batch_size", DEFAULT_BATCH_SIZE),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    # Phase 2: unfreeze top
    unfreeze_ratio = params.get("unfreeze_ratio", 0.3)
    unfreeze_top_fraction(model, unfreeze_ratio)
    freeze_batchnorm_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get("ft_lr", 1e-5)),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params.get("epochs_phase2", DEFAULT_EPOCHS_PHASE2),
        batch_size=params.get("batch_size", DEFAULT_BATCH_SIZE),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    if fold_ckpt_path.exists():
        model.load_weights(fold_ckpt_path)
    val_proba = model.predict(X_val, verbose=0).flatten()
    target_sens = params.get("target_sensitivity", 0.9)
    operating_point = find_best_threshold_at_sensitivity(y_val, val_proba, target_sensitivity=target_sens)
    threshold = operating_point["threshold"]
    val_pred = (val_proba >= threshold).astype(np.int32)
    metrics = compute_metrics(y_val, val_pred, val_proba)
    metrics["threshold"] = float(threshold)
    metrics["op_sensitivity"] = float(operating_point["sensitivity"])
    metrics["op_specificity"] = float(operating_point["specificity"])
    metrics["op_fpr"] = float(operating_point["fpr"])
    metrics["threshold_warning"] = bool(operating_point["warning"])
    return model, metrics, y_val, val_proba


def find_best_threshold_at_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_sensitivity: float = 0.9,
) -> Dict[str, Any]:
    """Pick threshold with minimum FPR among points where sensitivity >= target."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = np.where(tpr >= target_sensitivity)[0]
    if len(idx) == 0:
        return {
            "threshold": 0.5,
            "sensitivity": float(recall_score(y_true, (y_proba >= 0.5).astype(np.int32), zero_division=0)),
            "specificity": float(np.nan),
            "fpr": float(np.nan),
            "warning": True,
        }
    i = idx[np.argmin(fpr[idx])]
    threshold = float(thresholds[i]) if i < len(thresholds) else 0.5
    return {
        "threshold": threshold,
        "sensitivity": float(tpr[i]),
        "specificity": float(1.0 - fpr[i]),
        "fpr": float(fpr[i]),
        "warning": False,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall (sensitivity), specificity, F1, AUC."""
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


# ---------------------------------------------------------------------------
# Part 6: 5-fold CV and Optuna
# ---------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    params: Dict[str, Any],
    use_class_weight: bool = True,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Run 5-fold stratified CV. Returns fold_metrics, mean_metrics, std_metrics.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y = df["label"].values
    fold_metrics: Dict[int, Dict[str, float]] = {}
    all_metrics: Dict[str, List[float]] = {k: [] for k in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]}
    threshold_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        class_weight = None
        if use_class_weight:
            cw = compute_class_weight(
                "balanced",
                classes=np.unique(y[train_idx]),
                y=y[train_idx],
            )
            class_weight = {int(c): float(w) for c, w in zip(np.unique(y[train_idx]), cw)}
            class_weight[1] = class_weight.get(1, 1.0) * float(params.get("tb_class_weight_multiplier", 1.0))
        model, metrics, y_val, val_proba = train_fold(
            train_idx,
            val_idx,
            df,
            params,
            fold=fold,
            class_weight=class_weight,
        )
        fold_metrics[fold] = metrics
        for k in all_metrics:
            all_metrics[k].append(metrics[k])
        model.save_weights(WEIGHTS_DIR / f"fold_{fold}_weights.weights.h5")
        for loc, row_idx in enumerate(val_idx):
            row = df.iloc[row_idx]
            oof_rows.append(
                {
                    "study_id": row["study_id"],
                    "source": row["source"],
                    "true_label": int(y_val[loc]),
                    "predicted_probability": float(val_proba[loc]),
                    "fold": int(fold),
                }
            )
        threshold_rows.append(
            {
                "fold": int(fold),
                "threshold": float(metrics.get("threshold", 0.5)),
                "sensitivity": float(metrics.get("op_sensitivity", metrics["sensitivity"])),
                "specificity": float(metrics.get("op_specificity", metrics["specificity"])),
                "fpr": float(metrics.get("op_fpr", 1.0 - metrics["specificity"])),
                "warning": bool(metrics.get("threshold_warning", False)),
            }
        )
    mean_metrics = {k: float(np.mean(all_metrics[k])) for k in all_metrics}
    std_metrics = {k: float(np.std(all_metrics[k])) for k in all_metrics}
    threshold_df = pd.DataFrame(threshold_rows)
    oof_df = pd.DataFrame(oof_rows)
    return fold_metrics, mean_metrics, std_metrics, threshold_df, oof_df


def get_default_params() -> Dict[str, Any]:
    return {
        "backbone_name": "MobileNetV3Large",
        "head_lr": 1e-3,
        "ft_lr": 1e-5,
        "dropout_rate": 0.4,
        "dense_units": 128,
        "l2_strength": 1e-4,
        "batch_size": DEFAULT_BATCH_SIZE,
        "unfreeze_ratio": 0.3,
        "loss_type": "bce",
        "focal_gamma": 2.0,
        "focal_alpha": 0.5,
        "tb_class_weight_multiplier": 1.0,
        "use_augmentation": True,
        "epochs_phase1": DEFAULT_EPOCHS_PHASE1,
        "epochs_phase2": DEFAULT_EPOCHS_PHASE2,
        "target_sensitivity": 0.9,  # threshold chosen per fold to achieve this recall (TB sensitivity)
    }


def run_optuna_study(df: pd.DataFrame, n_trials: int = 25) -> optuna.Study:
    """Optuna study with AUC + specificity@90%sensitivity objective."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "backbone_name": "MobileNetV3Large",
            "head_lr": trial.suggest_float("head_lr", 1e-4, 3e-3, log=True),
            "ft_lr": trial.suggest_float("ft_lr", 1e-6, 3e-5, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.25, 0.55),
            "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
            "l2_strength": trial.suggest_float("l2_strength", 1e-5, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "unfreeze_ratio": trial.suggest_float("unfreeze_ratio", 0.10, 0.35),
            "loss_type": trial.suggest_categorical("loss_type", ["bce", "focal"]),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
            "focal_alpha": trial.suggest_float("focal_alpha", 0.25, 0.75),
            "tb_class_weight_multiplier": trial.suggest_float("tb_class_weight_multiplier", 0.8, 1.4),
            "use_augmentation": True,
            "epochs_phase1": DEFAULT_EPOCHS_PHASE1,
            "epochs_phase2": DEFAULT_EPOCHS_PHASE2,
            "target_sensitivity": 0.9,
        }
        _, mean_metrics, _, threshold_df, _ = run_cv(df, params, use_class_weight=True)
        mean_auc = float(mean_metrics["auc"])
        mean_sensitivity = float(threshold_df["sensitivity"].mean()) if not threshold_df.empty else 0.0
        mean_specificity = float(threshold_df["specificity"].mean()) if not threshold_df.empty else 0.0
        score = mean_auc + 0.25 * mean_specificity
        if mean_auc < 0.90:
            score -= 1.0
        if mean_sensitivity < 0.90:
            score -= 0.5
        if mean_specificity < 0.75:
            score -= 0.2
        return float(score)
    sampler = TPESampler(n_startup_trials=5, seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ---------------------------------------------------------------------------
# Part 7: Summary figure and outputs
# ---------------------------------------------------------------------------

def save_summary_figure(
    oof_df: pd.DataFrame,
    operating_point: Dict[str, Any],
    mean_metrics: Dict[str, float],
    out_path: Path = SUMMARY_FIGURE_PATH,
) -> None:
    """Generate thesis-ready figure: ROC curve, confusion matrix, metrics bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y_true = oof_df["true_label"].to_numpy(dtype=np.int32)
    y_proba = oof_df["predicted_probability"].to_numpy(dtype=np.float32)
    threshold = float(operating_point["threshold"])
    y_pred = (y_proba >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # ROC
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {mean_metrics['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    # CM
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
    # Metrics bar
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
    print(f"Saved summary figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_operating_point_summary(
    oof_df: pd.DataFrame,
    target_sensitivity: float = 0.90,
) -> Dict[str, Any]:
    y_true = oof_df["true_label"].to_numpy(dtype=np.int32)
    y_proba = oof_df["predicted_probability"].to_numpy(dtype=np.float32)
    op = find_best_threshold_at_sensitivity(y_true, y_proba, target_sensitivity=target_sensitivity)
    threshold = float(op["threshold"])
    y_pred = (y_proba >= threshold).astype(np.int32)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": threshold,
        "sensitivity": float(op["sensitivity"]),
        "specificity": float(op["specificity"]) if not np.isnan(op["specificity"]) else None,
        "fpr": float(op["fpr"]) if not np.isnan(op["fpr"]) else None,
        "auc": float(metrics["auc"]),
        "precision": float(metrics["precision"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "confusion_matrix": cm,
        "warning": bool(op["warning"]),
    }

def main(
    run_export: bool = True,
    run_optuna: bool = False,
    n_optuna_trials: int = 20,
    use_best_params: bool = True,
) -> None:
    set_seeds(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) U-Net export
    if run_export:
        n = export_unet_segmentations(overwrite=False)
        print(f"U-Net export: {n} images in {UNET_EXPORT_DIR}")
    # 2) Metadata + labels
    df = load_metadata_and_labels()
    print(f"Loaded {len(df)} samples with labels from metadata.")
    # 3) Params: Optuna or default
    if run_optuna and OPTUNA_AVAILABLE:
        study = run_optuna_study(df, n_trials=n_optuna_trials)
        best = study.best_params
        params = get_default_params()
        params.update(best)
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2)
        trials_summary = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ]
        with open(OUTPUT_DIR / "optuna_trials.json", "w") as f:
            json.dump(trials_summary, f, indent=2)
        print("Optuna best params:", params)
    else:
        params = get_default_params()
        if use_best_params and (not run_optuna) and BEST_PARAMS_PATH.exists():
            with open(BEST_PARAMS_PATH, "r") as f:
                loaded = json.load(f)
            params.update(loaded)
            print(f"Loaded best params from {BEST_PARAMS_PATH}")
    # 4) 5-fold CV
    fold_metrics, mean_metrics, std_metrics, threshold_df, oof_df = run_cv(df, params, use_class_weight=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"fold": fold_metrics, "mean": mean_metrics, "std": std_metrics}, f, indent=2)
    with open(SUMMARY_METRICS_PATH, "w") as f:
        json.dump({"mean": mean_metrics, "std": std_metrics}, f, indent=2)
    fold_df = pd.DataFrame(fold_metrics).T
    fold_df.to_csv(FOLD_METRICS_CSV_PATH, index=True)
    threshold_df.to_csv(THRESHOLD_PER_FOLD_PATH, index=False)
    oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
    op_summary = compute_operating_point_summary(oof_df, target_sensitivity=params.get("target_sensitivity", 0.90))
    with open(OPERATING_POINT_SUMMARY_PATH, "w") as f:
        json.dump(op_summary, f, indent=2)
    print("Mean metrics (5-fold):", mean_metrics)
    print("Std:", std_metrics)
    # 5) Summary figure
    save_summary_figure(oof_df, op_summary, mean_metrics)
    print("Done.")


if __name__ == "__main__":
    main(
        run_export=True,
        run_optuna=True,
        n_optuna_trials=20,
        use_best_params=True,
    )
