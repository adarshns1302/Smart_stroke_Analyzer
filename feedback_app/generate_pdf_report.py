from fpdf import FPDF
import os
import json

# === CONFIGURATION ===
classification_json_path = "feedback_app/classification_report.json"
contact_pose_img_path = "pose_estimation/keypoints/contact_frame_pose.jpg"
video_thumb_img_path = "feedback_app/video_thumbnail.jpg"  # Optional
report_path = "feedback_app/SmartStrokeReport.pdf"

# === LOAD METRICS ===
with open(classification_json_path, "r") as f:
    report = json.load(f)

# === CREATE PDF ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Smart Stroke Analyzer Report", ln=True, align='C')

# === Add Summary ===
pdf.set_font("Arial", '', 12)
pdf.ln(10)
pdf.cell(0, 10, f"Overall Accuracy: {report['accuracy']*100:.2f}%", ln=True)

pdf.ln(5)
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Per-Class Metrics", ln=True)
pdf.set_font("Arial", '', 12)

for cls, metrics in report.items():
    if cls not in ["accuracy", "macro avg", "weighted avg"]:
        pdf.cell(0, 10, f"{cls} - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1-score']:.2f}", ln=True)

# === Add Pose Image ===
if os.path.exists(contact_pose_img_path):
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Pose at Contact Frame", ln=True)
    pdf.image(contact_pose_img_path, x=10, y=30, w=180)

# === Add Video Snapshot (optional) ===
if os.path.exists(video_thumb_img_path):
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Stroke Video Snapshot", ln=True)
    pdf.image(video_thumb_img_path, x=10, y=30, w=180)

# === Save PDF ===
pdf.output(report_path)
print(f"[✅] PDF saved at: {report_path}")
