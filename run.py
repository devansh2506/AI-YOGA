import sys
import difflib
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import from existing modules
from models.mcls import load_model_and_labels
from poseguru_core.losses import PoseGuruLoss, PoseGuruLossConfig
from poseguru_core.recourse import run_recourse
from poseguru_core.refinement import refine_pose
from scripts.realtime_poseguru import extract_12_landmarks_xy, draw_skeleton

# Fix path to allow imports if running from root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

MODEL_PATH = str(ROOT_DIR / "pose_landmarker_full.task")
DATA_DIR = ROOT_DIR / "data"

def get_target_pose(user_input, label_mapping):
    """
    Fuzzy match user input to available pose names.
    Returns (pose_name, class_id) or None.
    """
    valid_poses = list(label_mapping.keys())
    
    # 1. Exact case-insensitive match on normalized name
    # e.g. "tree pose" matches "Tree_Pose_or_Vrksasana_" if we treat underscores as spaces
    input_norm = user_input.lower().replace(" ", "")
    
    matches = []
    
    for pose in valid_poses:
        pose_norm = pose.lower().replace("_", "").replace("-", "")
        if input_norm in pose_norm:
            matches.append(pose)
            
    if matches:
        # Return the shortest match (likely the most direct one)
        best_match = min(matches, key=len)
        return best_match, label_mapping[best_match]
        
    # 2. Difflib close match
    close_matches = difflib.get_close_matches(user_input, valid_poses, n=1, cutoff=0.4)
    if close_matches:
        return close_matches[0], label_mapping[close_matches[0]]
        
    return None, None

def main():
    # 1. Get User Input
    print("Welcome to PoseGuru AI-Yoga!")
    print("Available Poses detected in system.")
    
    model, label_mapping = load_model_and_labels()
    
    while True:
        user_input = input("\nEnter the yoga pose you want to practice (e.g., 'tree pose', 'cobra'): ").strip()
        if not user_input:
            continue
            
        pose_name, class_id = get_target_pose(user_input, label_mapping)
        
        if pose_name:
            print(f"Found match: {pose_name}")
            confirm = input("Is this correct? (y/n): ").lower()
            if confirm == 'y':
                break
        else:
            print("No matching pose found. Please try again.")

    print(f"\nStarting PoseGuru for target: {pose_name}")
    print("Press 'q' to quit the video feed.")

    # 2. Setup System (Copied/Adapted from realtime_poseguru.py)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_module = PoseGuruLoss(model, PoseGuruLossConfig()).to(device)

    # Load exemplars
    ex_data = np.load(DATA_DIR / "exemplars_yoga.npz", allow_pickle=True)
    ex_landmarks = ex_data["landmarks"]
    ex_labels = ex_data["labels"]
    ex_image_names = ex_data["image_names"]
    
    exemplar_by_class = {int(c): ex_landmarks[i] for i, c in enumerate(ex_labels)}
    exemplar_img_name_by_class = {int(c): ex_image_names[i] for i, c in enumerate(ex_labels)}

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)

    # State for smoothing
    smoothed_landmarks = None
    alpha = 0.5  # Smoothing factor (0.0=no smoothing, 1.0=instant)
    
    # State for feedback stabilizing
    feedback_frame_count = 0
    feedback_interval = 15  # Update feedback every 15 frames
    current_feedback = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    target_class_id = int(class_id)
    target_pose_name = pose_name

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        landmarks_raw = extract_12_landmarks_xy(detection_result)
        
        if landmarks_raw is None:
            # If tracking lost, reset smoothing
            smoothed_landmarks = None
            landmarks = None
        else:
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks_raw
            else:
                # EMA: new_smoothed = alpha * current + (1 - alpha) * old_smoothed
                smoothed_landmarks = alpha * landmarks_raw + (1 - alpha) * smoothed_landmarks
            landmarks = smoothed_landmarks
        
        # Display Target
        cv2.putText(frame, f"Target: {target_pose_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if landmarks is None:
            cv2.imshow("PoseGuru Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Pipeline logic
        x_prime = torch.tensor(landmarks[None, :, :], dtype=torch.float32, device=device)
        x_ex = torch.tensor(exemplar_by_class[target_class_id][None, :, :], dtype=torch.float32, device=device)
        target_tensor = torch.tensor([target_class_id], dtype=torch.long, device=device)

        x_star, _ = run_recourse(
            x_prime=x_prime.cpu(),
            x_ex=x_ex.cpu(),
            target_class=target_tensor.cpu(),
            loss_module=loss_module.cpu(),
            num_steps=10, # Fewer steps for lower latency if needed
            lr=5e-2,
            device=torch.device("cpu"),
        )
        x_final, _ = refine_pose(x_prime.cpu(), x_star)
        x_final_np = x_final.numpy()[0]

        # Draw red (User) skeleton
        draw_skeleton(frame, landmarks, (0, 0, 255), thickness=2)

        # Draw Exemplar Image
        try:
            ex_img_name = exemplar_img_name_by_class[target_class_id]
            img_path = ROOT_DIR / "yoga data" / "train" / target_pose_name / ex_img_name
            
            if img_path.exists():
                ex_img = cv2.imread(str(img_path))
                if ex_img is not None:
                    target_h = 200
                    scale = target_h / ex_img.shape[0]
                    target_w = int(ex_img.shape[1] * scale)
                    ex_img_resized = cv2.resize(ex_img, (target_w, target_h))
                    h, w, _ = frame.shape
                    if target_w < w and target_h < h:
                        frame[0:target_h, w-target_w:w] = ex_img_resized
                        cv2.rectangle(frame, (w-target_w, 0), (w, target_h), (255, 255, 255), 2)
                        cv2.putText(frame, "Ideal Pose", (w-target_w+5, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        # Feedback
        from poseguru_core.xai_feedback import generate_feedback
        
        feedback_frame_count += 1
        if feedback_frame_count % feedback_interval == 0:
             current_feedback = generate_feedback(landmarks, x_final_np)
             
        # Initial case
        if not current_feedback and feedback_frame_count < feedback_interval:
             current_feedback = generate_feedback(landmarks, x_final_np)
             
        feedback = current_feedback
        
        status_text = "INCORRECT POSE"
        status_color = (0, 0, 255)
        if not feedback or feedback[0] == "Great pose! Hold it.":
             status_text = "CORRECT POSE"
             status_color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA)

        y_offset = 130
        for msg in feedback:
            if msg == "Great pose! Hold it.": continue
            cv2.putText(frame, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 40

        cv2.imshow("PoseGuru Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()