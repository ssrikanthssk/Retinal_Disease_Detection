
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.model import OCTCNN
from src.dataset import create_dataloaders, CLASS_NAMES


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 4
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# CLASS WEIGHTS
# ==========================================

def calculate_class_weights(train_dataset):
    # Read only labels, not image pixels
    all_labels = np.asarray(
        train_dataset.dataset["label"],
        dtype=np.int64
    )

    train_labels = all_labels[train_dataset.indices]

    counts = np.bincount(
        train_labels,
        minlength=len(CLASS_NAMES)
    )

    if np.any(counts == 0):
        raise ValueError(
            "At least one class has no training images."
        )

    weights = len(train_labels) / (
        len(CLASS_NAMES) * counts
    )

    print("\nTraining class distribution:")

    for name, count, weight in zip(
        CLASS_NAMES, counts, weights
    ):
        print(
            f"{name:8s} "
            f"Count: {count:6d} | "
            f"Weight: {weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    use_amp
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ==========================================
# VALIDATION
# ==========================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    use_amp
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    for images, labels in progress:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ==========================================
# SAVE TRAINING CURVES
# ==========================================

def save_training_curves(history):
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )

    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("f1", "Macro F1")
    ]

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(
            df["epoch"],
            df[f"train_{key}"],
            label="Training"
        )

        ax.plot(
            df["epoch"],
            df[f"val_{key}"],
            label="Validation"
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "training_curves.png",
        dpi=300
    )

    plt.close(fig)


# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    set_seed(SEED)

    print("=" * 60)
    print("RETINAL DISEASE DETECTION - CNN TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    train_loader, val_loader, _ = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0
        )
    )

    model = OCTCNN(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    use_amp = DEVICE.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    history = []

    best_f1 = -1.0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                use_amp
            )
        )

        val_loss, val_acc, val_f1 = (
            validate(
                model,
                val_loader,
                criterion,
                use_amp
            )
        )

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        })

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Accuracy: {train_acc:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        print(
            "Learning rate:",
            optimizer.param_groups[0]["lr"]
        )

        # Save best model using validation macro F1
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "best_val_f1": best_f1,
                "image_size": 224
            }, MODEL_DIR / "best_cnn.pth")

            print("Best model saved!")

        else:
            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        # Save metrics after each completed epoch
        pd.DataFrame(history).to_csv(
            METRICS_DIR / "training_history.csv",
            index=False
        )

        save_training_curves(history)

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("Best validation macro F1:", round(best_f1, 4))
    print("Training time (minutes):", round(elapsed, 2))

    print(
        "Best model:",
        MODEL_DIR / "best_cnn.pth"
    )

    print(
        "Training history:",
        METRICS_DIR / "training_history.csv"
    )

    print(
        "Training curves:",
        FIGURES_DIR / "training_curves.png"
    )


if __name__ == "__main__":
    main()
