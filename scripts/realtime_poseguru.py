from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.mcls import load_model_and_labels
from poseguru_core.losses import PoseGuruLoss, PoseGuruLossConfig
from poseguru_core.recourse import run_recourse
from poseguru_core.refinement import refine_pose


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT_DIR / "pose_landmarker_full.task")
DATA_DIR = ROOT_DIR / "data"


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

JOINT_ORDER = [
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

SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 2),  # left shoulder -> left elbow
    (2, 4),  # left elbow -> left wrist
    (1, 3),  # right shoulder -> right elbow
    (3, 5),  # right elbow -> right wrist
    (6, 8),  # left hip -> left knee
    (8, 10),  # left knee -> left ankle
    (7, 9),  # right hip -> right knee
    (9, 11),  # right knee -> right ankle
    (0, 1),  # shoulders
    (6, 7),  # hips
    (0, 6),  # left torso side
    (1, 7),  # right torso side
]


def extract_12_landmarks_xy(detection_result):
    if not detection_result.pose_landmarks:
        return None
    pose_landmarks = detection_result.pose_landmarks[0]
    pts = []
    for name in JOINT_ORDER:
        idx = LANDMARK_MAP[name]
        lm = pose_landmarks[idx]
        pts.append([lm.x, lm.y])
    return np.array(pts, dtype=np.float32)  # (12,2) normalized


def draw_skeleton(
    frame,
    points_norm: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    h, w, _ = frame.shape
    pts = points_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(int)

    for i, j in SKELETON_EDGES:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def main():
    # load classifier and exemplars
    model, label_mapping = load_model_and_labels()
    model.eval()

    ex_data = np.load(DATA_DIR / "exemplars_yoga.npz", allow_pickle=True)
    ex_landmarks = ex_data["landmarks"]  # (C,12,2)
    ex_labels = ex_data["labels"]  # (C,)
    ex_image_names = ex_data["image_names"]
    exemplar_by_class = {int(c): ex_landmarks[i] for i, c in enumerate(ex_labels)}
    exemplar_img_name_by_class = {int(c): ex_image_names[i] for i, c in enumerate(ex_labels)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_module = PoseGuruLoss(model, PoseGuruLossConfig()).to(device)

    # mediapipe pose landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")
    print("Press 'n' for next pose, 'p' for previous pose.")

    # Manual selection init
    sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1]) # List of (name, id) sorted by id
    num_classes = len(sorted_labels)
    current_class_idx = 0 
    
    while True:
        target_pose_name, target_class_id = sorted_labels[current_class_idx]

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        landmarks = extract_12_landmarks_xy(detection_result)
        
        # UI: Draw Target Name
        cv2.putText(
            frame,
            f"Target: {target_pose_name} ({current_class_idx+1}/{num_classes})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press 'n'/'p' to switch",
            (10, 60), 
            cv2.FONT_HERSHEY_TRIPLEX, 
            0.5, 
            (200, 200, 200), 
            1, 
            cv2.LINE_AA
        )

        if landmarks is None:
            # Still show the target name even if no person is detected
            cv2.imshow("PoseGuru Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                current_class_idx = (current_class_idx + 1) % num_classes
            elif key == ord("p"):
                current_class_idx = (current_class_idx - 1 + num_classes) % num_classes
            continue

        # build tensors
        x_prime_np = landmarks[None, :, :]  # (1,12,2)
        x_prime = torch.tensor(x_prime_np, dtype=torch.float32, device=device)

        # MANUAL SELECTION: No classification
        pred_class = target_class_id
        
        # exemplar for predicted class
        x_ex_np = exemplar_by_class[pred_class][None, :, :]
        x_ex = torch.tensor(x_ex_np, dtype=torch.float32, device=device)
        target = torch.tensor([pred_class], dtype=torch.long, device=device)

        # run a small number of optimization steps for realtime
        x_star, _ = run_recourse(
            x_prime=x_prime.cpu(),
            x_ex=x_ex.cpu(),
            target_class=target.cpu(),
            loss_module=loss_module.cpu(),
            num_steps=40,
            lr=5e-2,
            device=torch.device("cpu"),
        )

        x_final, deltas = refine_pose(x_prime.cpu(), x_star)
        x_final_np = x_final.numpy()[0]  # (12,2)

        # draw original (red)
        draw_skeleton(frame, landmarks, (0, 0, 255), thickness=2)
        # HIDING BLUE AND GREEN LINES AS REQUESTED
        # draw_skeleton(frame, x_ex_np[0], (255, 0, 0), thickness=2) # Draw exemplar in Blue
        # draw_skeleton(frame, x_final_np, (0, 255, 0), thickness=2)

        # Overlay Exemplar Image
        try:
            ex_img_name = exemplar_img_name_by_class[pred_class]
            # Use the pose name from our manual selection list
            pred_pose_name = target_pose_name
            
            # Construct path: yoga data/train/<pose_name>/<image_name>
            img_path = ROOT_DIR / "yoga data" / "train" / pred_pose_name / ex_img_name
            
            if img_path.exists():
                ex_img = cv2.imread(str(img_path))
                if ex_img is not None:
                    # Resize to a small fixed size, e.g., 200px height
                    target_h = 200
                    scale = target_h / ex_img.shape[0]
                    target_w = int(ex_img.shape[1] * scale)
                    ex_img_resized = cv2.resize(ex_img, (target_w, target_h))
                    
                    # Overlay on top-right corner
                    h, w, _ = frame.shape
                    # Ensure it fits
                    if target_w < w and target_h < h:
                        frame[0:target_h, w-target_w:w] = ex_img_resized
                        
                        # Draw a border
                        cv2.rectangle(frame, (w-target_w, 0), (w, target_h), (255, 255, 255), 2)
                        
                        # Label it
                        cv2.putText(frame, "Ideal Pose", (w-target_w+5, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Error displaying exemplar image: {e}")


        from poseguru_core.xai_feedback import generate_feedback
        feedback = generate_feedback(landmarks, x_final_np)
        
        # Display Correct/Incorrect status
        status_text = "INCORRECT POSE"
        status_color = (0, 0, 255) # Red
        if not feedback or feedback[0] == "Great pose! Hold it.":
             status_text = "CORRECT POSE"
             status_color = (0, 255, 0) # Green

        cv2.putText(
            frame,
            status_text,
            (10, 90), # Moved down slightly
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3,
            cv2.LINE_AA,
        )

        y_offset = 130
        for msg in feedback:
            if msg == "Great pose! Hold it.": continue # Don't show this if we have big Green text already
            cv2.putText(
                frame,
                msg,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # Yellow color for feedback
                2,
                cv2.LINE_AA
            )
            y_offset += 40


        cv2.imshow("PoseGuru Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            current_class_idx = (current_class_idx + 1) % num_classes
        elif key == ord("p"):
            current_class_idx = (current_class_idx - 1 + num_classes) % num_classes


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

