from __future__ import annotations

import json
import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    auc, accuracy_score, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

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

UNET_EXPORT_DIR = BASE_DIR / "unet_export"
IMG_SIZE = 260
RANDOM_SEED = 42
N_FOLDS = 5
TOTAL_EPOCHS = 50 
INITIAL_EPOCHS = 5  # <-- Added: Epochs for frozen backbone training
DEFAULT_BATCH_SIZE = 16

OUTPUT_DIR = BASE_DIR / "efficientnet_tb_output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
SUMMARY_FIGURE_PATH = OUTPUT_DIR / "summary_figure.png"

def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

# ---------------------------------------------------------------------------
# Part 2: Metadata and Labels
# ---------------------------------------------------------------------------
def load_metadata_and_labels() -> pd.DataFrame:
    rows = []
    for csv_path, source in [(SHENZHEN_METADATA, "shenzhen"), (MONTGOMERY_METADATA, "montgomery")]:
        if not csv_path.exists(): continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            label = 0 if str(r["findings"]).strip().lower() == "normal" else 1
            stem = Path(str(r["study_id"])).stem
            seg_path = UNET_EXPORT_DIR / source / f"{stem}_unetseg.png"
            if seg_path.exists():
                rows.append({"path": str(seg_path), "label": label})
    return pd.DataFrame(rows)

def get_data_arrays(df_subset):
    X, y = [], []
    for _, row in df_subset.iterrows():
        img = cv2.imread(row['path'])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(tf.keras.applications.efficientnet.preprocess_input(img))
        y.append(row['label'])
    return np.array(X), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Part 3: EfficientNetB2 Model Builder
# ---------------------------------------------------------------------------
def build_efficientnet_model(lr, dropout) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
    ])

    # Removed the custom 'name' argument so Keras downloads the correct .h5 file
    base_model = tf.keras.applications.EfficientNetB2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        include_top=False, 
        weights="imagenet"
    )
    
    # Start with frozen backbone
    base_model.trainable = False 

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ---------------------------------------------------------------------------
# Part 4: CV Training and Metrics
# ---------------------------------------------------------------------------
def run_cv(df, params):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results_records = []
    cv_history = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['path'], df['label']), 1):
        print(f"\n" + "="*30)
        print(f"         FOLD {fold} ")
        print("="*30)
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        X_train, y_train = get_data_arrays(df_train)
        X_val, y_val = get_data_arrays(df_val)

        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), cw)}

        model = build_efficientnet_model(lr=params['lr_head'], dropout=params['dropout'])
        
        # --- PHASE 1: Train Top Layers Only (Frozen Backbone) ---
        print(f"\n--- Phase 1: Training Head (Frozen Backbone) ---")
        
        # Use the higher learning rate for the head
        model = build_efficientnet_model(lr=params['lr_head'], dropout=params['dropout'])
        
        callbacks_p1 = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
        ]
        
        history_head = model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=INITIAL_EPOCHS, 
            batch_size=params['batch_size'], 
            class_weight=class_weights, 
            callbacks=callbacks_p1, 
            verbose=1
        )

        # --- PHASE 2: Fine-Tuning Entire Network ---
        print(f"\n--- Phase 2: Fine-Tuning (Unfrozen Backbone) ---")
        
        base_model = model.get_layer("efficientnetb2") 
        base_model.trainable = True
        
        # CRITICAL FIX: Re-freeze all BatchNormalization layers
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        # Recompile with the much lower fine-tuning learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr_tune']),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        
        # Recreate callbacks to reset early stopping's internal tracking
        callbacks_p2 = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        ]
        
        last_epoch = history_head.epoch[-1] + 1
        
        if last_epoch < TOTAL_EPOCHS:
            history_ft = model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=TOTAL_EPOCHS, 
                initial_epoch=last_epoch, # Starts the epoch counter where Phase 1 left off
                batch_size=params['batch_size'], 
                class_weight=class_weights, 
                callbacks=callbacks_p2, 
                verbose=1
            )
            
            # Safely combine the histories of both phases and fix float types for JSON
            combined_history = {}
            for k in history_head.history.keys():
                p1_vals = [float(i) for i in history_head.history[k]]
                p2_vals = [float(i) for i in history_ft.history[k]] if k in history_ft.history else []
                combined_history[k] = p1_vals + p2_vals
        else:
            # Fallback if Phase 1 hit total epochs (unlikely)
            combined_history = {k: [float(i) for i in v] for k, v in history_head.history.items()}

        cv_history[f"fold_{fold}"] = combined_history

        probs = model.predict(X_val).flatten()
        
        for idx_in_val, (local_idx, prob) in enumerate(zip(val_idx, probs)):
            results_records.append({
                "path": df.iloc[local_idx]['path'],
                "true_label": int(y_val[idx_in_val]),
                "predicted_probability": float(prob),
                "fold": fold
            })

        model.save_weights(WEIGHTS_DIR / f"fold_{fold}.weights.h5")

    return pd.DataFrame(results_records), cv_history

def calculate_who_metrics(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    # 1. Standardized pAUC (FPR 0.4 to 0.6)
    fpr_start, fpr_end = 0.4, 0.6
    tpr_at_start = np.interp(fpr_start, fpr, tpr)
    tpr_at_end = np.interp(fpr_end, fpr, tpr)
    mask = (fpr > fpr_start) & (fpr < fpr_end)
    
    fpr_slice = np.concatenate([[fpr_start], fpr[mask], [fpr_end]])
    tpr_slice = np.concatenate([[tpr_at_start], tpr[mask], [tpr_at_end]])
    
    # Dividing by 0.2 standardizes the area to a 0.0 - 1.0 scale
    who_pauc = auc(fpr_slice, tpr_slice) / 0.2
    
    # 2. Specificity at 90% Sensitivity
    idx_90 = np.where(tpr >= 0.90)[0][0]
    who_spec = (1 - fpr[idx_90]) # Keep as decimal for main script consistency
    
    return who_pauc, who_spec, tpr[idx_90]

def main():
    start_time = time.time()
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata_and_labels()
    if df.empty: return

    params = {
    "lr_head": 1e-3,     # Higher LR for Phase 1
    "lr_tune": 1e-5,     # Lower LR for Phase 2
    "dropout": 0.6, 
    "batch_size": DEFAULT_BATCH_SIZE
    }
    results_df, cv_history = run_cv(df, params)

    y_true = results_df['true_label'].values
    y_proba = results_df['predicted_probability'].values

    #WHO TPP pAUC
    who_pauc, who_spec, at_sens = calculate_who_metrics(y_true, y_proba)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Final Target Sensitivity Thresholding
    target_sensitivity = 0.90
    idx = np.where(tpr >= target_sensitivity)[0][0]
    best_threshold = thresholds[idx]
    
    results_df['predicted_label'] = (results_df['predicted_probability'] >= best_threshold).astype(int)
    results_df['is_misclassified'] = results_df['true_label'] != results_df['predicted_label']

    # Saving metrics and reporting
    accuracy = accuracy_score(y_true, results_df['predicted_label'])
    precision = precision_score(y_true, results_df['predicted_label'])
    recall = recall_score(y_true, results_df['predicted_label'])
    f1 = f1_score(y_true, results_df['predicted_label'])
    auc_val = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, results_df['predicted_label'])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    total_min = (time.time() - start_time) / 60

    final_metrics = {
        "auc": float(auc_val),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(recall),
        "specificity": float(spec),
        "f1_score": float(f1),
        "sensitivity_target_threshold": float(best_threshold),
        "total_runtime_minutes": float(total_min)
    }
    
    # Save JSON data
    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump({"cv_history": cv_history, "summary_metrics": final_metrics}, f, indent=4)

    # Save CSV reports
    results_df.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    misclassified = results_df[results_df['is_misclassified'] == True]
    misclassified.to_csv(OUTPUT_DIR / "misclassified_report.csv", index=False)

    # Re-generate and save the plot
    plt.figure(figsize=(18, 5))
    
    # --- 1. Confusion Matrix ---
    plt.subplot(1, 3, 1)
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix\n(Thr: {best_threshold:.2f})")
    # Added back: Text overlay for confusion matrix numbers
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", 
                    color="black" if cm[i,j] < cm.max()/2 else "white")
    plt.xticks([0, 1], ["Non-TB", "TB"])
    plt.yticks([0, 1], ["Non-TB", "TB"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # --- 2. ROC Curve ---
    plt.subplot(1, 3, 2)
    spec_pct = (1 - fpr) * 100
    tpr_pct = tpr * 100

    plt.plot(spec_pct, tpr_pct, color='#1f77b4', lw=3, label=f'Model (AUC: {auc_val:.3f})')

    fpr_slice = np.concatenate([[0.4], fpr[(fpr > 0.4) & (fpr < 0.6)], [0.6]])
    tpr_slice = np.interp(fpr_slice, fpr, tpr)
    plt.fill_between((1 - fpr_slice) * 100, 0, tpr_slice * 100, color='orange', alpha=0.3, 
                    label=f'WHO pAUC: {who_pauc:.3f}')

    plt.scatter(who_spec * 100, at_sens * 100, color='red', s=100, edgecolors='black', zorder=5,
                label=f'Target (Spec: {who_spec*100:.1f}%)')

    plt.axvline(x=40, color='red', linestyle='--', alpha=0.4)
    plt.axvline(x=60, color='red', linestyle='--', alpha=0.4)

    plt.xlim(100, 0) 
    plt.ylim(0, 105)
    plt.title('WHO TPP Compliance View')
    plt.xlabel('Specificity (%)')
    plt.ylabel('Sensitivity (%)')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # --- 3. Metrics Summary ---
    plt.subplot(1, 3, 3)
    metric_names = ['AUC', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1']
    metric_values = [auc_val, accuracy, precision, recall, spec, f1]
    
    #Assign to 'bars' and loop to add values
    bars = plt.barh(metric_names, metric_values, color='#779ecb')
    plt.xlim(0, 1.1)
    plt.title("Performance Metrics Summary")
    plt.gca().invert_yaxis()  # Put AUC at the top
    
    #Text overlay for bar chart values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center')

    plt.tight_layout()
    plt.savefig(SUMMARY_FIGURE_PATH)

    print("\n" + "="*45)
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"WHO pAUC (40-60%): {who_pauc:.4f} (Goal: >0.9)")
    print(f"WHO Spec @ 90% Sens: {who_spec:.4f} (Goal: >0.4)")
    print(f"ROC AUC:           {auc_val:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Sensitivity:       {recall:.4f}")
    print(f"Specificity:       {spec:.4f}")
    print(f"Total Runtime:     {total_min:.2f} minutes")
    print("=" * 45)
    print(f"\nReport saved! {len(misclassified)} images misclassified.")


if __name__ == "__main__":
    main()