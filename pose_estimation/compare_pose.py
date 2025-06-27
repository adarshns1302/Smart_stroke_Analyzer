import numpy as np
import os

print("Available pose files:")
for f in os.listdir("pose_estimation/keypoints"):
    if f.endswith(".npy"):
        print(f"- {f}")

# Load reference and test pose
ref_pose = np.load("pose_estimation/reference_poses/cover_drive.npy")
test_pose = np.load("pose_estimation/keypoints/cover_drive_01_frame_0030_pose.npy")

# Pose landmarks (optional: you can use specific indices)
keypoint_names = {
    0: "Nose", 11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow", 14: "Right Elbow",
    15: "Left Wrist", 16: "Right Wrist",
    23: "Left Hip", 24: "Right Hip",
    25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle"
}

tolerance = 0.07  # 7% of frame size distance

# Compare keypoints
for idx, name in keypoint_names.items():
    ref_point = ref_pose[idx][:2]  # (x, y)
    test_point = test_pose[idx][:2]

    dist = np.linalg.norm(np.array(ref_point) - np.array(test_point))

    if dist > tolerance:
        print(f"⚠️ {name} deviates too much ({dist:.3f}) — adjust your position.")
    else:
        print(f"✅ {name} is well aligned.")
