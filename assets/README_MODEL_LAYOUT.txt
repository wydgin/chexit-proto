Chexit model asset layout
=========================

Drop your model files under these folders:

- models/
  - U-Net segmentation models (.keras/.h5)
  - Current backend default: models/unet_lung_seg_best.keras

- mobilenet_tb_output/
  - weights/
    - MobileNet fold weights (e.g. fold_0_weights.weights.h5)
  - optuna_best_params.json (optional hyperparameter file)

- efficientnet_tb_output/
  - weights/
    - EfficientNet fold weights (e.g. fold_0.weights.h5)

- densenet_tb_output/
  - weights/
    - DenseNet fold weights (e.g. fold_0.weights.h5)

You can keep each architecture in its own folder; this is the intended setup for an ensemble.
