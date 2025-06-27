# pose_estimation/utils.py

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def run_pose_estimation_from_array(image_array):
    """
    Runs pose estimation on a given numpy array image.
    Returns keypoints as numpy array, and annotated image.
    """
    pose = mp_pose.Pose(static_image_mode=True)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    keypoints = []
    if results.pose_landmarks:
        h, w, _ = image_array.shape
        for lm in results.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y, lm.z, lm.visibility))

        annotated = image_array.copy()
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    else:
        annotated = image_array.copy()
        keypoints = []

    pose.close()
    return annotated, np.array(keypoints)
