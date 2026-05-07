# Classifier model definitions

Put your exact architecture code here for each classifier so backend inference can
rebuild the network graph before loading `.weights.h5` files.

## Files

- `mobilenet_model.py`
- `efficientnet_model.py`
- `densenet_model.py`

## Expected builder function names

- `build_mobilenet_classifier()`
- `build_efficientnet_classifier()`
- `build_densenet_classifier()`

Each function should return a compiled or uncompiled `tf.keras.Model` with the
exact layer structure used during training (input size, preprocessing assumptions,
head layers, output activation, etc.).

