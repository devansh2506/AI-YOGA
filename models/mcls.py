from pathlib import Path
from typing import Dict, Tuple

import json
import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class LandmarkMLP(nn.Module):
    """
    Simple MLP classifier for 12-keypoint (24-dim) landmark vectors.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model_and_labels(
    model: nn.Module,
    label_mapping: Dict[str, int],
    model_name: str = "mcls_yoga.pt",
    labels_name: str = "label_mapping.json",
) -> Tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    labels_path = MODELS_DIR / labels_name

    torch.save(model.state_dict(), model_path)
    with open(labels_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    return model_path, labels_path


def load_model_and_labels(
    model_name: str = "mcls_yoga.pt",
    labels_name: str = "label_mapping.json",
) -> Tuple[nn.Module, Dict[str, int]]:
    model_path = MODELS_DIR / model_name
    labels_path = MODELS_DIR / labels_name

    with open(labels_path, "r") as f:
        label_mapping = json.load(f)

    num_classes = len(label_mapping)
    model = LandmarkMLP(input_dim=24, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, label_mapping

