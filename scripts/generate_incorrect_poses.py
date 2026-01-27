from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from landmarks_to_angles import calculate_angle


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


JOINT_TRIPLES = {
    # name: (a, b, c) indices for angle at b
    "left_elbow": (0, 2, 4),
    "right_elbow": (1, 3, 5),
    "left_knee": (6, 8, 10),
    "right_knee": (7, 9, 11),
    "left_shoulder": (6, 0, 2),
    "right_shoulder": (7, 1, 3),
    "left_hip": (0, 6, 8),
    "right_hip": (1, 7, 9),
}

ANGLE_LIMITS = {
    "shoulder": (0.0, 180.0),
    "hip": (0.0, 120.0),
    "elbow": (30.0, 180.0),
    "knee": (30.0, 180.0),
}

JOINT_TYPE = {
    "left_elbow": "elbow",
    "right_elbow": "elbow",
    "left_knee": "knee",
    "right_knee": "knee",
    "left_shoulder": "shoulder",
    "right_shoulder": "shoulder",
    "left_hip": "hip",
    "right_hip": "hip",
}

SEGMENT_FOR_JOINT = {
    "left_elbow": "la",
    "right_elbow": "ra",
    "left_knee": "ll",
    "right_knee": "rl",
    "left_shoulder": "t",
    "right_shoulder": "t",
    "left_hip": "t",
    "right_hip": "t",
}


def row_to_points(row: pd.Series) -> np.ndarray:
    """
    Convert a landmarks CSV row into (12,2) numpy array.
    """
    joint_names = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    coords: List[float] = []
    for j in joint_names:
        coords.append(row[f"{j}_x"])
        coords.append(row[f"{j}_y"])
    arr = np.array(coords, dtype=np.float32).reshape(12, 2)
    return arr


def compute_joint_angle(points: np.ndarray, triple: Tuple[int, int, int]) -> float:
    a_idx, b_idx, c_idx = triple
    a = points[a_idx]
    b = points[b_idx]
    c = points[c_idx]
    return float(calculate_angle(a, b, c))


def rotate_child_around_joint(
    points: np.ndarray,
    triple: Tuple[int, int, int],
    delta_deg: float,
) -> np.ndarray:
    """
    Rotate the distal segment (b->c) around joint b by delta_deg.
    """
    a_idx, b_idx, c_idx = triple
    out = points.copy()
    b = out[b_idx]
    c = out[c_idx]
    v = c - b
    theta = np.deg2rad(delta_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    v_new = rot @ v
    out[c_idx] = b + v_new
    return out


def generate_single_incorrect(points: np.ndarray, joint_name: str) -> np.ndarray | None:
    """
    Generate a single incorrect pose by perturbing one joint angle within limits.
    """
    triple = JOINT_TRIPLES[joint_name]
    jtype = JOINT_TYPE[joint_name]
    lo, hi = ANGLE_LIMITS[jtype]

    current_angle = compute_joint_angle(points, triple)

    for _ in range(20):  # try up to 20 random perturbations
        perturb = np.random.uniform(20.0, 40.0)
        sign = np.random.choice([-1.0, 1.0])
        target_angle = current_angle + sign * perturb
        target_angle = float(np.clip(target_angle, lo, hi))
        delta = target_angle - current_angle
        if abs(delta) < 5.0:
            continue

        candidate = rotate_child_around_joint(points, triple, delta)
        new_angle = compute_joint_angle(candidate, triple)
        if lo <= new_angle <= hi:
            return candidate

    return None


def main():
    src_csv = DATA_DIR / "yoga_pose_landmarks_train.csv"
    if not src_csv.is_file():
        raise FileNotFoundError(f"Cannot find landmarks CSV at {src_csv}. Run generate_landmarks.py first.")

    df = pd.read_csv(src_csv)

    incorrect_rows = []

    for idx, row in df.iterrows():
        points = row_to_points(row)
        pose_name = row["pose_name"]

        # choose a random joint to perturb
        joint_name = np.random.choice(list(JOINT_TRIPLES.keys()))
        incorrect = generate_single_incorrect(points, joint_name)
        if incorrect is None:
            continue

        out = {
            "pose_name": pose_name,
            "segment": SEGMENT_FOR_JOINT[joint_name],
            "joint_name": joint_name,
        }

        # correct landmarks
        for j in range(12):
            out[f"correct_{j}_x"] = float(points[j, 0])
            out[f"correct_{j}_y"] = float(points[j, 1])

        # incorrect landmarks
        for j in range(12):
            out[f"incorrect_{j}_x"] = float(incorrect[j, 0])
            out[f"incorrect_{j}_y"] = float(incorrect[j, 1])

        incorrect_rows.append(out)

    out_csv = DATA_DIR / "incorrect_pose_landmarks.csv"
    out_df = pd.DataFrame(incorrect_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Generated {len(incorrect_rows)} incorrect poses at {out_csv}")


if __name__ == "__main__":
    main()

