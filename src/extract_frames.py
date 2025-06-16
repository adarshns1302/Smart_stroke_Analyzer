import cv2
import os

# Paths
INPUT_DIR = "data/raw_videos"
OUTPUT_DIR = "data/frames"

# Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames_from_video(video_path, output_folder, frame_rate=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    print(f"🔄 Extracting from {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save one frame every 'frame_rate' frames
        if count % frame_rate == 0:
            frame_filename = os.path.join(video_output_path, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"✅ {saved_count} frames saved from {video_name}")

def extract_all_videos():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith((".mp4", ".avi", ".mov")):
            full_path = os.path.join(INPUT_DIR, filename)
            extract_frames_from_video(full_path, OUTPUT_DIR)

if __name__ == "__main__":
    extract_all_videos()
