import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# -------------------- CONFIG --------------------
MODEL_PATH = "/Users/devanshkedia/Desktop/AI YOGA/pose_landmarker_full.task"
EXEMPLAR_IMAGE = "/Users/devanshkedia/Desktop/AI YOGA/yoga data/train/Boat_Pose_or_Paripurna_Navasana_/Boat_Pose_or_Paripurna_Navasana__image_4.jpg"
USER_IMAGE = "/Users/devanshkedia/Desktop/AI YOGA/yoga data/train/Boat_Pose_or_Paripurna_Navasana_/Boat_Pose_or_Paripurna_Navasana__image_111.jpg"
# ------------------------------------------------

LANDMARK_MAP = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Initialize Pose Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1)
detector = vision.PoseLandmarker.create_from_options(options)

def extract_12_landmarks_xy(detection_result):
    if not detection_result.pose_landmarks:
        return None

    pose_landmarks = detection_result.pose_landmarks[0]
    landmarks = {}

    for name, idx in LANDMARK_MAP.items():
        lm = pose_landmarks[idx]
        landmarks[name] = (lm.x, lm.y)

    return landmarks

def compute_centroid(landmarks):
    """
    Computes centroid (mean x, mean y) of landmarks
    """
    xs = [pt[0] for pt in landmarks.values()]
    ys = [pt[1] for pt in landmarks.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def center_landmarks(landmarks):
    """
    Subtracts centroid from each landmark so pose is centered at origin
    """
    cx, cy = compute_centroid(landmarks)
    centered = {}

    for name, (x, y) in landmarks.items():
        centered[name] = (x - cx, y - cy)

    return centered, (cx, cy)

def extract_landmarks_from_image(detector, image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = detector.detect(mp_image)
    landmarks = extract_12_landmarks_xy(detection_result)

    if landmarks is None:
        raise ValueError(f"No pose detected in image: {image_path}")

    return landmarks

def landmarks_dict_to_array(landmarks_dict):
    """
    Converts landmark dict to (N, 2) numpy array
    using consistent LANDMARK_MAP order
    """
    return np.array([landmarks_dict[name] for name in LANDMARK_MAP.keys()])

def scale_normalize_poses(
    user_centered_arr,
    user_centroid,
    exemplar_centered_arr,
    exemplar_centroid,
    eps=1e-8
):
    """
    Scale-normalizes user and exemplar poses independently
    for Procrustes analysis.

    Parameters:
    - user_centered: (N, 2) numpy array, centroid already subtracted
    - user_centroid: (2,) numpy array (not used here, included for completeness)
    - exemplar_centered: (N, 2) numpy array, centroid already subtracted
    - exemplar_centroid: (2,) numpy array (not used here)
    - eps: small value to avoid division by zero

    Returns:
    - user_scaled: scale-normalized user pose
    - exemplar_scaled: scale-normalized exemplar pose
    - user_scale: scalar scale of user pose
    - exemplar_scale: scalar scale of exemplar pose
    """

    # --- Compute scale as RMS distance from origin ---
    user_scale = np.sqrt(np.mean(np.sum(user_centered_arr**2, axis=1))) + eps
    exemplar_scale = np.sqrt(np.mean(np.sum(exemplar_centered_arr**2, axis=1))) + eps

    # --- Scale normalize ---
    user_scaled = user_centered_arr / user_scale
    exemplar_scaled = exemplar_centered_arr / exemplar_scale

    return user_scaled, exemplar_scaled, user_scale, exemplar_scale

def procrustes_rotation(exemplar_scaled, user_scaled):
    """
    Computes optimal rotation matrix using SVD
    to align exemplar pose to user pose.

    Parameters:
    - exemplar_scaled: (N, 2) numpy array
    - user_scaled: (N, 2) numpy array

    Returns:
    - exemplar_rotated: (N, 2) rotated exemplar pose
    - R: (2, 2) rotation matrix
    """

    # Cross-covariance matrix
    H = exemplar_scaled.T @ user_scaled  # (2xN) @ (Nx2) → (2x2)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Reflection check (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Rotate exemplar
    exemplar_rotated = exemplar_scaled @ R

    return exemplar_rotated, R

# Extract landmarks
exemplar_landmarks = extract_landmarks_from_image(detector, EXEMPLAR_IMAGE)
user_landmarks = extract_landmarks_from_image(detector, USER_IMAGE)

# Center landmarks
exemplar_centered, exemplar_centroid = center_landmarks(exemplar_landmarks)
user_centered, user_centroid = center_landmarks(user_landmarks)

user_centered_arr = landmarks_dict_to_array(user_centered)
exemplar_centered_arr = landmarks_dict_to_array(exemplar_centered)

# Scale normalize
user_scaled, exemplar_scaled, user_scale, exemplar_scale = scale_normalize_poses(
    user_centered_arr,
    user_centroid,
    exemplar_centered_arr,
    exemplar_centroid
)

# Procrustes rotation (align exemplar to user)
exemplar_aligned, rotation_matrix = procrustes_rotation(
    exemplar_scaled,
    user_scaled
)

print("User (scaled):\n", user_scaled)
print("\nExemplar (scaled):\n", exemplar_scaled)
print("\nRotation matrix R:\n", rotation_matrix)
print("\nExemplar (aligned to user):\n", exemplar_aligned)