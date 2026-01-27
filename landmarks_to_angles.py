import pandas as pd
import numpy as np

# ---------- Helper function to calculate angle ----------
def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three points a, b, c
    Angle between vectors BA and BC
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# ---------- Load CSV ----------
input_csv = "/Users/devanshkedia/Desktop/AI YOGA/yoga_pose_landmarks.csv"   # change path if needed
output_csv = "/Users/devanshkedia/Desktop/AI YOGA/pose_angles.csv"

df = pd.read_csv(input_csv)

# ---------- Compute Angles ----------
angles_data = []

for _, row in df.iterrows():
    data = {
        "pose_name": row["pose_name"],

        # Elbow Angles
        "left_elbow_angle": calculate_angle(
            (row["left_shoulder_x"], row["left_shoulder_y"]),
            (row["left_elbow_x"], row["left_elbow_y"]),
            (row["left_wrist_x"], row["left_wrist_y"])
        ),
        "right_elbow_angle": calculate_angle(
            (row["right_shoulder_x"], row["right_shoulder_y"]),
            (row["right_elbow_x"], row["right_elbow_y"]),
            (row["right_wrist_x"], row["right_wrist_y"])
        ),

        # Knee Angles
        "left_knee_angle": calculate_angle(
            (row["left_hip_x"], row["left_hip_y"]),
            (row["left_knee_x"], row["left_knee_y"]),
            (row["left_ankle_x"], row["left_ankle_y"])
        ),
        "right_knee_angle": calculate_angle(
            (row["right_hip_x"], row["right_hip_y"]),
            (row["right_knee_x"], row["right_knee_y"]),
            (row["right_ankle_x"], row["right_ankle_y"])
        ),

        # Shoulder Angles
        "left_shoulder_angle": calculate_angle(
            (row["left_hip_x"], row["left_hip_y"]),
            (row["left_shoulder_x"], row["left_shoulder_y"]),
            (row["left_elbow_x"], row["left_elbow_y"])
        ),
        "right_shoulder_angle": calculate_angle(
            (row["right_hip_x"], row["right_hip_y"]),
            (row["right_shoulder_x"], row["right_shoulder_y"]),
            (row["right_elbow_x"], row["right_elbow_y"])
        ),

        # Hip Angles
        "left_hip_angle": calculate_angle(
            (row["left_shoulder_x"], row["left_shoulder_y"]),
            (row["left_hip_x"], row["left_hip_y"]),
            (row["left_knee_x"], row["left_knee_y"])
        ),
        "right_hip_angle": calculate_angle(
            (row["right_shoulder_x"], row["right_shoulder_y"]),
            (row["right_hip_x"], row["right_hip_y"]),
            (row["right_knee_x"], row["right_knee_y"])
        )
    }

    angles_data.append(data)

# ---------- Save New CSV ----------
angles_df = pd.DataFrame(angles_data)
angles_df.to_csv(output_csv, index=False)

print("Angle CSV generated successfully:", output_csv)
