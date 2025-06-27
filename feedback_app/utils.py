import torch
import json
import os
from PIL import Image
from fpdf import FPDF
import torchvision.transforms as transforms
from torchvision import models

# ========== PATHS ==========
CLASSIFIER_MODEL_PATH = "models/resnet18_stroke_classifier.pth"
CLASS_MAPPING_PATH = "utils/class_mapping.json"

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== LOAD MODEL ==========
def load_classifier_model():
    with open(CLASS_MAPPING_PATH, "r") as f:
        class_mapping = json.load(f)

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_mapping))
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ========== PREDICT ==========
def predict_stroke_type(image_file, model):
    image = Image.open(image_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    with open(CLASS_MAPPING_PATH, "r") as f:
        class_mapping = json.load(f)

    idx_to_class = {int(k): v for k, v in class_mapping.items()}
    label = idx_to_class[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

    return label, confidence

# ========== PDF REPORT ==========
def generate_pdf_report(report_path, stroke_type, suggestions, contact_pose_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "🏏 Smart Stroke Analyzer Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Stroke Type: {stroke_type.replace('_', ' ').title()}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Suggestions:", ln=True)
    pdf.set_font("Arial", "", 12)
    for s in suggestions:
        pdf.multi_cell(0, 10, f"• {s}")

    if os.path.exists(contact_pose_path):
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Pose at Contact Frame:", ln=True)
        pdf.image(contact_pose_path, w=150)

    pdf.output(report_path)

# ========== VIDEO LISTING ==========
def list_previous_videos(output_dir):
    videos = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_streamlit.mp4"):
                relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                videos.append(relative_path)
    return sorted(videos, reverse=True)
