import torch
import torch.nn as nn
from torchvision import models, datasets


MODEL_NAME = "mobilenetv3_skin"  # must match test.py
MODEL_TYPE = "mobilenet_v3_large"  # change to "mobilenet_v3_small" if you trained that
DATA_DIR = "train"
DEVICE = "cpu"


def build_model(num_classes: int) -> torch.nn.Module:
    """Recreate the same MobileNetV3 architecture used in training."""
    if MODEL_TYPE == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_large(weights=None)

    # Replace classifier to match number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def main():
    # Infer number of classes from the dataset folders
    base_ds = datasets.ImageFolder(DATA_DIR)
    num_classes = len(base_ds.classes)
    print("Detected classes:", base_ds.classes)
    print("Number of classes:", num_classes)

    # Build model and load trained weights
    model = build_model(num_classes)
    state = torch.load(f"export/{MODEL_NAME}.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # Dummy input matching training input size
    example_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    # Export ONNX
    onnx_path = f"export/{MODEL_NAME}.onnx"
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print("Saved ONNX model:", onnx_path)


if __name__ == "__main__":
    main()


