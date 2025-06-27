# src/video_pipeline.py

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import subprocess
from datetime import datetime
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "feedback_app"))

from pose_estimation.utils import run_pose_estimation_from_array
from feedback_app.utils import predict_stroke_type, load_classifier_model
from pose_estimation.ai_feedback import get_ai_suggestions

# === CLI ===
if len(sys.argv) > 1:
    INPUT_VIDEO = sys.argv[1]
    if not os.path.exists(INPUT_VIDEO):
        print(f"[❌] Input video not found: {INPUT_VIDEO}")
        sys.exit(1)
else:
    print("[❌] Please provide input video path.")
    sys.exit(1)

# === Auto Setup ===
VIDEO_NAME = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = ROOT_DIR / f"data/outputs/{VIDEO_NAME}_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(ROOT_DIR / "models/yolov8_ball.pt"))
classifier_model = load_classifier_model()

# === Video IO ===
cap = cv2.VideoCapture(INPUT_VIDEO)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(str(OUTPUT_DIR / "annotated_video.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

ball_centers = []
contact_frame_idx = -1
frame_idx = 0
all_frames = []
batsman_box_at_contact = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    all_frames.append(frame.copy())
    results = model(frame)[0]
    ball_detected = False
    batsman_box = None

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        cls = int(cls)
        label = model.names[cls]
        color = (0, 255, 0) if label == 'ball' else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if label == "ball":
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ball_centers.append((frame_idx, cx, cy))
            ball_detected = True
        if label == "batsman":
            batsman_box = (int(x1), int(y1), int(x2), int(y2))

    if contact_frame_idx == -1 and ball_detected and batsman_box:
        bx1, by1, bx2, by2 = batsman_box
        cx, cy = ball_centers[-1][1], ball_centers[-1][2]
        if bx1 < cx < bx2 and by1 < cy < by2:
            contact_frame_idx = frame_idx
            batsman_box_at_contact = batsman_box
            cv2.putText(frame, "🎯 Contact Point", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    for i in range(1, len(ball_centers)):
        _, x1, y1 = ball_centers[i - 1]
        _, x2, y2 = ball_centers[i]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# === Contact frame processing ===
stroke_type = "unknown"
suggestions = []

if contact_frame_idx != -1 and batsman_box_at_contact:
    contact_img_path = OUTPUT_DIR / "contact_frame.jpg"
    contact_frame = all_frames[contact_frame_idx]
    cv2.imwrite(str(contact_img_path), contact_frame)

    bx1, by1, bx2, by2 = batsman_box_at_contact
    cropped_batsman = contact_frame[by1:by2, bx1:bx2]

    # run pose estimation on cropped
    annotated_cropped, pose_arr = run_pose_estimation_from_array(cropped_batsman)

    # paste pose overlay back on original
    contact_frame[by1:by2, bx1:bx2] = annotated_cropped
    cv2.imwrite(str(OUTPUT_DIR / "contact_frame_pose.jpg"), contact_frame)
    np.save(str(OUTPUT_DIR / "contact_pose.npy"), pose_arr)

    # predict stroke type on cropped
    stroke_type, confidence = predict_stroke_type(str(contact_img_path), classifier_model)
    print(f"[✅] Stroke Type: {stroke_type} ({confidence*100:.1f}% confidence)")

    # reference pose suggestions
    ref_pose_path = ROOT_DIR / "pose_estimation" / "reference_poses" / f"{stroke_type}.npy"
    if ref_pose_path.exists():
        suggestions = get_ai_suggestions(str(ref_pose_path), str(OUTPUT_DIR / "contact_pose.npy"), stroke_type)
    else:
        suggestions = ["⚠️ No reference pose found for this stroke type."]

    # feedback
    with open(OUTPUT_DIR / "feedback.json", "w") as f:
        json.dump({
            "stroke_type": stroke_type,
            "confidence": confidence,
            "suggestions": suggestions
        }, f)

else:
    print("[⚠️] No contact detected or batsman missing.")

# === Streamlit re-encode
subprocess.run([
    "ffmpeg", "-y", "-i", str(OUTPUT_DIR / "annotated_video.mp4"),
    "-vcodec", "libx264", "-acodec", "aac",
    str(OUTPUT_DIR / "annotated_video_streamlit.mp4")
])

print(f"[✅] Outputs saved to: {OUTPUT_DIR}")
