"""
EfficientNet Score-CAM for single-image TB diagnosis workflow.

Web-app friendly:
- importable predictor object that can stay in memory across requests
- single-image CLI entrypoint with input/output paths
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "efficientnet_tb_output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
IMG_SIZE = 260

PathLike = Union[str, Path]
_PREDICTOR_CACHE: Dict[str, "EfficientNetScoreCamPredictor"] = {}


def build_efficientnet_model() -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal")])
    base_model = tf.keras.applications.EfficientNetB2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            data_augmentation,
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


class ScoreCAM:
    def __init__(self, model: tf.keras.Model, target_layer_name: str = "top_conv"):
        self.model = model
        self.base_model = model.layers[1]
        self.target_layer = self.base_model.get_layer(target_layer_name)
        self.activation_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=[self.target_layer.output, self.base_model.output],
        )

    def __call__(
        self,
        score_fn,
        x: np.ndarray,
        batch_size: int = 16,
        max_channels: int = 256,
    ) -> np.ndarray:
        activations, _ = self.activation_model(x, training=False)
        input_shape = tf.shape(x)[1:3]
        act_resized = tf.image.resize(activations, input_shape)[0]

        mins = tf.reduce_min(act_resized, axis=[0, 1])
        maxs = tf.reduce_max(act_resized, axis=[0, 1])
        act_normalized = (act_resized - mins) / (maxs - mins + 1e-10)

        variances = tf.math.reduce_variance(act_normalized, axis=[0, 1])
        n_channels = int(act_normalized.shape[-1])
        max_channels = min(max_channels, n_channels)
        top_indices = tf.argsort(variances, direction="DESCENDING")[:max_channels]

        scores = np.zeros(max_channels, dtype=np.float32)
        for i in tqdm(
            range(0, len(top_indices), batch_size),
            desc="Batch ScoreCAM",
            leave=False,
        ):
            end_idx = min(i + batch_size, len(top_indices))
            current_indices = top_indices[i:end_idx]

            maps = tf.gather(act_normalized, current_indices, axis=-1)
            maps = tf.transpose(maps, [2, 0, 1])
            maps = tf.expand_dims(maps, axis=-1)
            masked_batch = x * maps

            preds = self.model(masked_batch, training=False)
            scores[i:end_idx] = score_fn(preds).numpy()

        cam = np.zeros(input_shape.numpy(), dtype=np.float32)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        for idx, weight in enumerate(scores):
            channel_idx = top_indices[idx]
            cam += weight * act_normalized[..., channel_idx].numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam


@dataclass
class EfficientNetScoreCamResult:
    output_path: str
    predicted_probability: float
    predicted_label: int


class EfficientNetScoreCamPredictor:
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer_name: str = "top_conv",
        batch_size: int = 16,
        max_channels: int = 256,
    ):
        self.model = model
        self.scorecam = ScoreCAM(model, target_layer_name=target_layer_name)
        self.batch_size = batch_size
        self.max_channels = max_channels

    @classmethod
    def from_weights(
        cls,
        weights_path: PathLike,
        target_layer_name: str = "top_conv",
        batch_size: int = 16,
        max_channels: int = 256,
    ) -> "EfficientNetScoreCamPredictor":
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        model = build_efficientnet_model()
        model.load_weights(weights_path)
        model.layers[0].trainable = False
        return cls(
            model=model,
            target_layer_name=target_layer_name,
            batch_size=batch_size,
            max_channels=max_channels,
        )

    def diagnose_and_save(
        self,
        input_image_path: PathLike,
        output_image_path: PathLike,
        overlay_image_path: Optional[PathLike] = None,
    ) -> EfficientNetScoreCamResult:
        input_image_path = Path(input_image_path)
        output_image_path = Path(output_image_path)

        seg_bgr = cv2.imread(str(input_image_path))
        if seg_bgr is None:
            raise FileNotFoundError(f"Unable to read input image: {input_image_path}")
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        seg_resized = cv2.resize(seg_rgb, (IMG_SIZE, IMG_SIZE))

        if overlay_image_path:
            overlay_bgr = cv2.imread(str(overlay_image_path))
            if overlay_bgr is not None:
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                overlay_resized = cv2.resize(overlay_rgb, (IMG_SIZE, IMG_SIZE))
            else:
                overlay_resized = seg_resized
        else:
            overlay_resized = seg_resized

        x_img = np.expand_dims(seg_resized, axis=0).astype(np.float32)
        x_img = tf.keras.applications.efficientnet.preprocess_input(x_img)

        pred_prob = float(self.model(x_img, training=False).numpy()[0][0])
        pred_label = int(pred_prob >= 0.5)

        def score_fn(output, p_lab=pred_label):
            return output[:, 0] if p_lab == 1 else -output[:, 0]

        heatmap = self.scorecam(
            score_fn,
            x_img,
            batch_size=self.batch_size,
            max_channels=self.max_channels,
        )

        heat_rgb = (cm.jet(heatmap)[..., :3] * 255).astype(np.float32)
        lung_mask = (np.max(seg_resized, axis=-1, keepdims=True) > 10).astype(np.float32)
        heatmap_expanded = np.expand_dims(heatmap, axis=-1)
        alpha_map = np.clip((heatmap_expanded - 0.25) / 0.75, 0.0, 1.0) * 0.65
        final_alpha = alpha_map * lung_mask

        base_float = overlay_resized.astype(np.float32)
        blended = (heat_rgb * final_alpha + base_float * (1.0 - final_alpha)).astype(
            np.uint8
        )

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return EfficientNetScoreCamResult(
            output_path=str(output_image_path),
            predicted_probability=pred_prob,
            predicted_label=pred_label,
        )


def get_cached_predictor(
    weights_path: PathLike,
    target_layer_name: str = "top_conv",
    batch_size: int = 16,
    max_channels: int = 256,
) -> EfficientNetScoreCamPredictor:
    key = f"{Path(weights_path).resolve()}::{target_layer_name}::{batch_size}::{max_channels}"
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = EfficientNetScoreCamPredictor.from_weights(
            weights_path=weights_path,
            target_layer_name=target_layer_name,
            batch_size=batch_size,
            max_channels=max_channels,
        )
        _PREDICTOR_CACHE[key] = predictor
    return predictor


def generate_scorecam_heatmap(
    input_image_path: PathLike,
    output_image_path: PathLike,
    predictor: Optional[EfficientNetScoreCamPredictor] = None,
    weights_path: Optional[PathLike] = None,
    overlay_image_path: Optional[PathLike] = None,
    use_cache: bool = True,
    target_layer_name: str = "top_conv",
    batch_size: int = 16,
    max_channels: int = 256,
) -> EfficientNetScoreCamResult:
    if predictor is None:
        resolved_weights = (
            Path(weights_path) if weights_path else (WEIGHTS_DIR / "fold_0.weights.h5")
        )
        if use_cache:
            predictor = get_cached_predictor(
                weights_path=resolved_weights,
                target_layer_name=target_layer_name,
                batch_size=batch_size,
                max_channels=max_channels,
            )
        else:
            predictor = EfficientNetScoreCamPredictor.from_weights(
                weights_path=resolved_weights,
                target_layer_name=target_layer_name,
                batch_size=batch_size,
                max_channels=max_channels,
            )

    return predictor.diagnose_and_save(
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        overlay_image_path=overlay_image_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EfficientNet Score-CAM heatmap for one image."
    )
    parser.add_argument("--input", required=True, help="Input segmented image path.")
    parser.add_argument("--output", required=True, help="Output heatmap image path.")
    parser.add_argument(
        "--weights",
        default=str(WEIGHTS_DIR / "fold_0.weights.h5"),
        help="Model weights path.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional original image path used as the visualization base.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for Score-CAM masking."
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=256,
        help="Top activation channels used for fast Score-CAM.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable predictor cache (loads fresh model each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_scorecam_heatmap(
        input_image_path=args.input,
        output_image_path=args.output,
        weights_path=args.weights,
        overlay_image_path=args.overlay,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
        max_channels=args.max_channels,
    )
    print(
        f"Saved heatmap: {result.output_path} | "
        f"pred={result.predicted_label} prob={result.predicted_probability:.4f}"
    )


if __name__ == "__main__":
    main()