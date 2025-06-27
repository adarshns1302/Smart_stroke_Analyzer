# pose_estimation/ai_feedback.py

import numpy as np

# Helper to check distance between 2 points
def is_deviation(pt1, pt2, threshold=0.07):
    return np.linalg.norm(np.array(pt1) - np.array(pt2)) > threshold

def get_ai_suggestions(ref_pose_path, test_pose_path, stroke_type):
    suggestions = []
    
    ref_pose = np.load(ref_pose_path)
    test_pose = np.load(test_pose_path)

    # Nose, Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle
    keypoints = {
        "head": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28
    }

    # Example rules for cover drive
    if stroke_type == "cover_drive":
        if is_deviation(test_pose[keypoints["left_ankle"]][:2], ref_pose[keypoints["left_ankle"]][:2]):
            suggestions.append("⚠️ Front foot not placed forward enough.")
        
        if is_deviation(test_pose[keypoints["right_elbow"]][:2], ref_pose[keypoints["right_elbow"]][:2]):
            suggestions.append("⚠️ Elbow should be higher for proper bat lift.")
        
        if is_deviation(test_pose[keypoints["head"]][:2], ref_pose[keypoints["head"]][:2]):
            suggestions.append("⚠️ Keep your head steady and aligned over the ball.")

    # Add more strokes: pull_shot, leg_glance, etc

    if not suggestions:
        suggestions.append("✅ Excellent pose! No major issues detected.")

    return suggestions
