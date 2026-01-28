import pandas as pd
from pathlib import Path

csv_path = Path("data/yoga_pose_landmarks_train.csv")
df = pd.read_csv(csv_path)
print("Data types:")
print(df.dtypes)

landmark_cols = []
joint_names = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
for j in joint_names:
    landmark_cols.append(f"{j}_x")
    landmark_cols.append(f"{j}_y")

print("\nChecking first row types:")
row = df.iloc[0]
print("Row dtype:", row.dtype)
sub_row = row[landmark_cols]
print("Sub-row dtype:", sub_row.dtype)
print("Sub-row values type:", type(sub_row.values))
print("Sub-row values:", sub_row.values)
import torch
try:
    coords = torch.tensor(sub_row.values, dtype=torch.float32)
    print("Tensor creation successful")
except Exception as e:
    print(f"Tensor creation failed: {e}")
