import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from PIL import Image

MODEL_NAME = "mobilenetv3_skin"
DATA_DIR = "train"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Same transforms as training (val)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Get class names from your folders
base_ds = datasets.ImageFolder(DATA_DIR)
classes = base_ds.classes
print("Classes:", classes)

# Rebuild model
num_classes = len(classes)
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

# Load trained weights
state = torch.load(f"export/{MODEL_NAME}.pth", map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

def predict_image(path: str):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)  # shape (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    print(f"Image: {path}")
    print(f"Predicted class: {classes[top_idx]}  (index {top_idx.item()})")
    print(f"Confidence: {top_prob.item():.3f}")

if __name__ == "__main__":
    # Change this to any image you want to test
    predict_image("/Users/ahmedazmi/Documents/GitHub/HackNYU/test2.jpg")