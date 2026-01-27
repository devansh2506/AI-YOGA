import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------- CONFIG --------------------
MODEL_PATH = "/Users/devanshkedia/Desktop/AI YOGA/pose_landmarker_full.task"
DATASET_ROOT = "/Users/devanshkedia/Desktop/AI YOGA/yoga data/train"
OUTPUT_CSV = "/Users/devanshkedia/Desktop/AI YOGA/yoga_pose_landmarks.csv"
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

def main():
    # Pose Landmarker setup
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)

    # CSV header
    header = ["image_name", "pose_name"]
    for name in LANDMARK_MAP:
        header.extend([f"{name}_x", f"{name}_y"])

    rows = []

    for pose_name in sorted(os.listdir(DATASET_ROOT)):
        pose_folder = os.path.join(DATASET_ROOT, pose_name)
        if not os.path.isdir(pose_folder):
            continue

        print(f"\nProcessing pose: {pose_name}")

        for image_name in sorted(os.listdir(pose_folder)):
            image_path = os.path.join(pose_folder, image_name)
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                # Load and convert to RGB
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    print(f"❌ Could not read image: {image_name}")
                    os.remove(image_path)  # delete unreadable image
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                detection_result = detector.detect(mp_image)
                landmark_data = extract_12_landmarks_xy(detection_result)

                if landmark_data is None:
                    print(f"❌ No pose detected: {pose_name}/{image_name}")
                    os.remove(image_path)  # delete images where pose not detected
                    continue

                row = {"image_name": image_name, "pose_name": pose_name}
                row.update(landmark_data)
                rows.append(row)

            except Exception as e:
                print(f"❌ Error processing {pose_name}/{image_name}: {e}")
                # Optionally delete images that raise errors
                try:
                    os.remove(image_path)
                except:
                    pass

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ Dataset cleaned and CSV created successfully")
    print(f"📄 Output CSV: {OUTPUT_CSV}")
    print(f"📊 Total valid samples: {len(rows)}")

if __name__ == "__main__":
    main()