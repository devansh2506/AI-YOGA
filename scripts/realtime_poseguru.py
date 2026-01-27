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
    exemplar_by_class = {int(c): ex_landmarks[i] for i, c in enumerate(ex_labels)}

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        landmarks = extract_12_landmarks_xy(detection_result)
        if landmarks is None:
            cv2.imshow("PoseGuru Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # build tensors
        x_prime_np = landmarks[None, :, :]  # (1,12,2)
        x_prime = torch.tensor(x_prime_np, dtype=torch.float32, device=device)

        # classify current pose
        x_flat = x_prime.view(1, -1)
        with torch.no_grad():
            logits = model(x_flat)
            probs = torch.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1)[0].item())

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

        # draw original (red) and corrected (green)
        draw_skeleton(frame, landmarks, (0, 0, 255), thickness=2)
        draw_skeleton(frame, x_final_np, (0, 255, 0), thickness=2)

        cv2.putText(
            frame,
            f"Pred: {list(label_mapping.keys())[list(label_mapping.values()).index(pred_class)]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("PoseGuru Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

