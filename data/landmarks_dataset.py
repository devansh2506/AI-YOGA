from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


class YogaLandmarksDataset(Dataset):
    """
    Loads 12-keypoint landmark CSVs produced by scripts/generate_landmarks.py
    and returns (landmarks_tensor, label_index).
    """

    def __init__(self, split: str, label_mapping: Dict[str, int] = None):
        """
        Args:
            split: "train", "val", or "test"
            label_mapping: optional dict pose_name -> int. If None, built from CSV.
        """
        csv_path = DATA_DIR / f"yoga_pose_landmarks_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV for split '{split}' not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        if label_mapping is None:
            pose_names: List[str] = sorted(self.df["pose_name"].unique().tolist())
            self.label_mapping = {name: idx for idx, name in enumerate(pose_names)}
        else:
            self.label_mapping = label_mapping

        self.split = split

        # Precompute landmark columns order
        self.landmark_cols: List[str] = []
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
        for j in joint_names:
            self.landmark_cols.append(f"{j}_x")
            self.landmark_cols.append(f"{j}_y")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        pose_name = row["pose_name"]
        label = self.label_mapping[pose_name]

        coords = torch.tensor(row[self.landmark_cols].values, dtype=torch.float32)  # (24,)
        return coords, label

