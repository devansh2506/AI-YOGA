from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.landmarks_dataset import YogaLandmarksDataset
from models.mcls import LandmarkMLP, save_model_and_labels


ROOT_DIR = Path(__file__).resolve().parents[1]

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = YogaLandmarksDataset(split="train")
    # Share label mapping across splits
    label_mapping = train_ds.label_mapping
    val_ds = YogaLandmarksDataset(split="val", label_mapping=label_mapping)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    num_classes = len(label_mapping)
    model = LandmarkMLP(input_dim=24, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-2)

    best_val_acc = 0.0
    epochs = 30

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_and_labels(model, label_mapping)
            print(f"Saved best model (val_acc={best_val_acc:.3f})")


if __name__ == "__main__":
    main()

