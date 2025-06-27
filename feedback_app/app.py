import sys
import os
import json
import shutil
import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np

from utils import (
    load_classifier_model,
    predict_stroke_type,
    generate_pdf_report,
    list_previous_videos
)

st.set_page_config(page_title="Smart Stroke Analyzer — AI Feedback", layout="centered")

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🏏 Analyze New Stroke", "📊 Dashboard & Stats", "📁 Replay Videos", "📄 Export Report"])

# ========== PATH SETUP ==========
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
REPORT_PATH = BASE_DIR / "feedback_app" / "stroke_report.pdf"

# ========== PAGE 1: ANALYZE NEW STROKE ==========
if page == "🏏 Analyze New Stroke":
    st.title("🏏 Smart Stroke Analyzer — AI Feedback")
    st.write("Upload a cricket stroke video for automatic analysis.")

    uploaded_file = st.file_uploader("📁 Choose a video file (.mp4)", type="mp4")
    if uploaded_file:
        raw_video_path = BASE_DIR / "data" / "raw_videos" / uploaded_file.name
        with open(raw_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Running analysis pipeline...")
        import subprocess
        subprocess.run(
            [sys.executable, "src/video_pipeline.py", str(raw_video_path)],
            check=True
        )
        st.success("✅ Video analyzed successfully!")

# ========== PAGE 2: DASHBOARD & STATS ==========
elif page == "📊 Dashboard & Stats":
    st.title("📊 Stroke Analytics Dashboard")

    outputs = sorted(OUTPUT_DIR.glob("*"), key=os.path.getmtime, reverse=True)
    if outputs:
        latest_output = outputs[0]
        annotated_video = latest_output / "annotated_video_streamlit.mp4"
        contact_frame = latest_output / "contact_frame_pose.jpg"
        feedback_file = latest_output / "feedback.json"

        st.subheader("🎥 Annotated Stroke Video")
        if annotated_video.exists():
            with open(annotated_video, "rb") as f:
                st.video(f.read())
        else:
            st.warning("⚠️ Annotated video not found.")

        st.subheader("🧍 Pose at Contact Frame")
        if contact_frame.exists():
            st.image(str(contact_frame), caption="Pose overlay on contact frame")
        else:
            st.warning("⚠️ Contact frame pose not found.")

        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedback = json.load(f)
            st.subheader(f"📝 Stroke Type: {feedback.get('stroke_type', 'unknown')}")
            st.write("### AI Suggestions:")
            for s in feedback.get("suggestions", []):
                st.write(f"- {s}")
        else:
            st.warning("⚠️ No feedback available.")
    else:
        st.warning("⚠️ No outputs found yet. Run an analysis first.")

# ========== PAGE 3: REPLAY PREVIOUS VIDEOS ==========
elif page == "📁 Replay Videos":
    st.title("🎞️ Replay Previous Videos")

    previous = list_previous_videos(OUTPUT_DIR)
    if previous:
        selected = st.selectbox("Select a previous video", previous)
        if selected:
            video_path = OUTPUT_DIR / selected
            with open(video_path, "rb") as f:
                st.video(f.read())
    else:
        st.warning("No past videos found.")

# ========== PAGE 4: EXPORT PDF REPORT ==========
elif page == "📄 Export Report":
    st.title("📄 Export Analysis Report")

    outputs = sorted(OUTPUT_DIR.glob("*"), key=os.path.getmtime, reverse=True)
    if outputs:
        latest_output = outputs[0]
        feedback_file = latest_output / "feedback.json"
        contact_frame = latest_output / "contact_frame_pose.jpg"

        stroke_type = "N/A"
        suggestions = []

        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedback = json.load(f)
                stroke_type = feedback.get("stroke_type", "N/A")
                suggestions = feedback.get("suggestions", [])

        if st.button("📥 Generate & Download Report"):
            generate_pdf_report(
                report_path=REPORT_PATH,
                stroke_type=stroke_type,
                suggestions=suggestions,
                contact_pose_path=str(contact_frame) if contact_frame.exists() else ""
            )
            if REPORT_PATH.exists():
                with open(REPORT_PATH, "rb") as f:
                    st.download_button("📄 Download Report", data=f, file_name="stroke_report.pdf")
            else:
                st.error("Failed to generate report.")
    else:
        st.warning("⚠️ No analysis to generate report from.")
