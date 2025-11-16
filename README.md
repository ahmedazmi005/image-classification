# Skin Condition Classifier (MobileNetV3)

This project trains a MobileNetV3 model to classify dermatology images into 24 skin-condition classes using PyTorch and exports the model to TorchScript and ONNX for deployment.

## Project Layout

- `train/` – dataset root with one subfolder per class (e.g. `Eczema Photos/`, `Psoriasis pictures Lichen Planus and related diseases/`, `clear_skin/`, etc.)
- `test.py` – main training script (PyTorch, TensorBoard logging, TorchScript + ONNX export).
- `export_onnx.py` – script to export the latest trained weights to ONNX without retraining.
- `infer_pt.py` – simple PyTorch inference script for running the model on a single image.
- `export/` – exported models (`.pth`, `.torchscript`, `.onnx`).
- `runs/` – TensorBoard logs.

## Training

From the project root:

```bash
python -u test.py
```

This will:

- Load `train/` with a train/val split.
- Train MobileNetV3 (large) for `EPOCHS` from `test.py`.
- Log metrics to TensorBoard (`runs/`).
- Save weights to `export/mobilenetv3_skin.pth`.
- Export TorchScript and ONNX to `export/`.

To view training curves:

```bash
tensorboard --logdir runs
```

## ONNX Export

To re-export ONNX after retraining:

```bash
python export_onnx.py
```

The ONNX model is written to `export/mobilenetv3_skin.onnx`.

## Inference (PyTorch)

Edit `infer_pt.py` to point at an image path, then:

```bash
python infer_pt.py
```

The script will print the predicted class and confidence.

## Inference (ONNX Runtime)

You can run the exported ONNX model on any device with `onnxruntime` installed by loading `export/mobilenetv3_skin.onnx` and applying the same preprocessing used during training.


