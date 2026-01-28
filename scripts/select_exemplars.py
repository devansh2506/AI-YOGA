from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.landmarks_dataset import YogaLandmarksDataset
from models.mcls import load_model_and_labels


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def main():
    model, label_mapping = load_model_and_labels()
    model.eval()

    # Build inverse label mapping
    inv_mapping: Dict[int, str] = {v: k for k, v in label_mapping.items()}

    train_ds = YogaLandmarksDataset(split="train", label_mapping=label_mapping)
    loader = DataLoader(train_ds, batch_size=256, shuffle=False)

    all_landmarks: List[np.ndarray] = []
    all_labels: List[int] = []
    all_pose_names: List[str] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for coords, labels in loader:
            logits = model(coords)
            probs = torch.softmax(logits, dim=1)
            # probability assigned to the true class
            p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            all_landmarks.append(coords.numpy())
            all_labels.append(labels.numpy())
            all_probs.append(p_true.numpy())

    landmarks_arr = np.concatenate(all_landmarks, axis=0)  # (N,24)
    labels_arr = np.concatenate(all_labels, axis=0)  # (N,)
    probs_arr = np.concatenate(all_probs, axis=0)  # (N,)

    # recover pose_name for each row from CSV
    import pandas as pd

    train_csv = DATA_DIR / "yoga_pose_landmarks_train.csv"
    df = pd.read_csv(train_csv)
    pose_names = df["pose_name"].values
    image_names = df["image_name"].values

    num_classes = len(label_mapping)
    exemplar_landmarks = []
    exemplar_labels = []
    exemplar_pose_names = []
    exemplar_image_names = []

    for class_id in range(num_classes):
        mask = labels_arr == class_id
        if not mask.any():
            continue

        class_idxs = np.where(mask)[0]
        class_probs = probs_arr[class_idxs]
        # pick index with highest probability
        best_idx_in_class = class_idxs[class_probs.argmax()]

        exemplar_landmarks.append(landmarks_arr[best_idx_in_class])
        exemplar_labels.append(int(class_id))
        exemplar_pose_names.append(str(pose_names[best_idx_in_class]))
        exemplar_image_names.append(str(image_names[best_idx_in_class]))

    exemplar_landmarks = np.stack(exemplar_landmarks, axis=0)  # (C,24)
    exemplar_landmarks = exemplar_landmarks.reshape(exemplar_landmarks.shape[0], 12, 2)
    exemplar_labels = np.array(exemplar_labels, dtype=np.int64)
    exemplar_pose_names = np.array(exemplar_pose_names, dtype=object)
    exemplar_image_names = np.array(exemplar_image_names, dtype=object)

    out_path = DATA_DIR / "exemplars_yoga.npz"
    np.savez(
        out_path,
        landmarks=exemplar_landmarks,
        labels=exemplar_labels,
        pose_names=exemplar_pose_names,
        image_names=exemplar_image_names,
    )

    print(f"Saved {len(exemplar_labels)} exemplars to {out_path}")


if __name__ == "__main__":
    main()

