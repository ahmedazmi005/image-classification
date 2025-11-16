import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# =============================
# SETTINGS
# =============================
# This expects your current layout: "train/<class-folders>/*.jpg"
DATA_DIR = "train"         # dataset root with all class folders
MODEL_NAME = "mobilenetv3_skin"
BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-4
VAL_SPLIT = 0.2            # 20% of data for validation
MODEL_TYPE = "mobilenet_v3_large"  # or "mobilenet_v3_small"

# Use Apple Silicon MPS if available
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

# =============================
# DATA TRANSFORMS
# =============================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset and applies a given transform to the images.
    This lets us share the same underlying ImageFolder but use
    different transforms for train/val.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =============================
# LOAD DATASET (auto train/val split)
# =============================
base_ds = datasets.ImageFolder(DATA_DIR)  # uses folder names as classes

num_classes = len(base_ds.classes)
print("Detected classes:", base_ds.classes)

val_size = int(len(base_ds) * VAL_SPLIT)
train_size = len(base_ds) - val_size

train_subset, val_subset = random_split(base_ds, [train_size, val_size])

train_ds = TransformSubset(train_subset, transform_train)
val_ds = TransformSubset(val_subset, transform_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# LOAD MOBILE NET V3
# =============================
if MODEL_TYPE == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
else:
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")

# Replace final classifier layer
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

model.to(DEVICE)

# =============================
# LOAD EXISTING WEIGHTS (CONTINUE TRAINING IF AVAILABLE)
# =============================
weights_path = Path("export") / f"{MODEL_NAME}.pth"
if weights_path.exists():
    print(f"Loading existing weights from {weights_path} for continued training...")
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
else:
    print("No existing weights found, training from scratch.")

# =============================
# OPTIMIZER + LOSS
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)

# =============================
# TENSORBOARD WRITER
# =============================
writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}")

# =============================
# TRAINING LOOP
# =============================
print("Starting training...")

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0.0

    total_batches = len(train_loader)
    print(f"\nEpoch {epoch+1}/{EPOCHS} (batches: {total_batches})")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Print every 20 batches (tweak this if you want more/less noise)
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx+1}/{total_batches}")

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    # Log to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)

    print(f"Epoch {epoch+1}/{EPOCHS} DONE | Loss: {train_loss:.4f} | Val Acc: {acc:.2f}% | Time: {time.time() - start:.1f}s")

# Close TensorBoard writer
writer.close()

# =============================
# SAVE MODEL WEIGHTS (.pth)
# =============================
Path("export").mkdir(exist_ok=True)
pth_path = f"export/{MODEL_NAME}.pth"
torch.save(model.state_dict(), pth_path)
print("Saved PyTorch weights:", pth_path)

# =============================
# EXPORT TORCHSCRIPT
# =============================
example_input = torch.randn(1, 3, 224, 224).to(DEVICE)
model.eval()
traced = torch.jit.trace(model, example_input)
ts_path = f"export/{MODEL_NAME}.torchscript"
traced.save(ts_path)
print("Saved TorchScript model:", ts_path)

# =============================
# EXPORT ONNX
# =============================
onnx_path = f"export/{MODEL_NAME}.onnx"
torch.onnx.export(
    model,
    example_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)
print("Saved ONNX model:", onnx_path)
