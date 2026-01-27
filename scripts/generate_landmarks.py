import os
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT_DIR / "pose_landmarker_full.task")
DATASET_ROOT = ROOT_DIR / "yoga data"
OUTPUT_DIR = ROOT_DIR / "data"

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


def extract_12_landmarks_xy(detection_result):
    if not detection_result.pose_landmarks:
        return None
    pose_landmarks = detection_result.pose_landmarks[0]
    data = {}
    for name, idx in LANDMARK_MAP.items():
        lm = pose_landmarks[idx]
        data[f"{name}_x"] = lm.x
        data[f"{name}_y"] = lm.y
    return data


def process_split(detector, split: str):
    split_root = DATASET_ROOT / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f"yoga_pose_landmarks_{split}.csv"

    header = ["image_name", "pose_name"]
    for name in LANDMARK_MAP:
        header.extend([f"{name}_x", f"{name}_y"])

    rows = []

    for pose_name in sorted(os.listdir(split_root)):
        pose_folder = split_root / pose_name
        if not pose_folder.is_dir():
            continue

        print(f"\nProcessing {split} pose: {pose_name}")

        for image_name in sorted(os.listdir(pose_folder)):
            image_path = pose_folder / image_name
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Could not read image: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            detection_result = detector.detect(mp_image)
            landmark_data = extract_12_landmarks_xy(detection_result)

            if landmark_data is None:
                print(f"No pose detected: {image_path}")
                continue

            row = {"image_name": image_name, "pose_name": pose_name}
            row.update(landmark_data)
            rows.append(row)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} samples to {output_csv}")


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)

    for split in ("train", "val", "test"):
        try:
            process_split(detector, split)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()

