# src/extract_pose.py

import sys
import os
from pathlib import Path
import numpy as np
from pose_estimation.utils import run_pose_estimation
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_pose(contact_frame_path):
    """
    Runs pose estimation on the given contact frame image.
    Saves both overlay image and .npy file in same directory.
    """
    contact_frame_path = Path(contact_frame_path)
    output_dir = contact_frame_path.parent

    pose_img, pose_arr = run_pose_estimation(str(contact_frame_path))

    # Save results
    np.save(str(output_dir / "contact_pose.npy"), pose_arr)
    cv2.imwrite(str(output_dir / "contact_frame_pose.jpg"), pose_img)

    print(f"[✅] Pose extraction complete. Saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/extract_pose.py <contact_frame_image>")
    else:
        extract_pose(sys.argv[1])
