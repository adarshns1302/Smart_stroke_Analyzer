# src/extract_frames.py
import cv2
import os

input_dir = "data/raw_videos"
output_dir = "data/frames"

os.makedirs(output_dir, exist_ok=True)

for video_file in os.listdir(input_dir):
    if not video_file.endswith(".mp4"): continue
    vid_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(vid_path)

    name = os.path.splitext(video_file)[0]
    folder_path = os.path.join(output_dir, name)
    os.makedirs(folder_path, exist_ok=True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f"{folder_path}/frame_{frame_idx:04d}.jpg", frame)
        frame_idx += 1
    cap.release()
