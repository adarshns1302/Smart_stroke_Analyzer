import os
import torch
import json
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from utils import load_classifier_model

# ========== CONFIGURATION ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "data", "final_dataset")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "classification_report.json")
BATCH_SIZE = 32
IMAGE_SIZE = 224

# ========== DEVICE SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== SETUP TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Standard for ResNet
                         [0.229, 0.224, 0.225])
])

# ========== LOAD DATASET ==========
dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes
print(f"[📁] Found {len(dataset)} images across {len(class_names)} classes.")

# ========== LOAD MODEL ==========
model = load_classifier_model()
model.to(device)
model.eval()

# ========== PREDICT ==========
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

# ========== GENERATE CLASSIFICATION REPORT ==========
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

# ========== SAVE TO JSON ==========
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print(f"[✅] classification_report.json saved at {OUTPUT_JSON_PATH}")
