
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

from sklearn.metrics import (
    confusion_matrix,
    recall_score
)

from src.dataset import (
    create_dataloaders,
    CLASS_NAMES
)

from src.efficientnet_model import build_efficientnet

from src.train import (
    train_one_epoch,
    validate,
    calculate_class_weights
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"

for directory in [
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 3
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
# VALIDATION RECALL BY CLASS
# ==========================================

@torch.inference_mode()
def get_class_recall(model, loader):
    model.eval()

    all_true = []
    all_pred = []

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(
            predictions.cpu().tolist()
        )

    recalls = recall_score(
        all_true,
        all_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0
    )

    return {
        name: float(value)
        for name, value in zip(
            CLASS_NAMES,
            recalls
        )
    }


# ==========================================
# SAVE TRAINING GRAPHS
# ==========================================

def save_curves(history):
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

    for ax, (key, title) in zip(
        axes,
        metrics
    ):
        ax.plot(
            df["epoch"],
            df[f"train_{key}"],
            marker="o",
            label="Training"
        )

        ax.plot(
            df["epoch"],
            df[f"val_{key}"],
            marker="o",
            label="Validation"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "efficientnet_training_curves.png",
        dpi=300
    )

    plt.close(fig)


# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    set_seed(SEED)

    print("=" * 60)
    print("EFFICIENTNET-B0 TRANSFER LEARNING")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # Load the same training and validation split
    train_loader, val_loader, _ = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0
        )
    )

    # Load ImageNet pretrained EfficientNet
    model = build_efficientnet(
        num_classes=len(CLASS_NAMES),
        pretrained=True
    ).to(DEVICE)

    print("\nPretrained EfficientNet-B0 loaded.")

    # Calculate weights from training labels only
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

    best_f1 = -1.0
    patience_counter = 0

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 60)
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

        print(
            f"\nTrain Loss: {train_loss:.4f} | "
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

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        })

        # Save best checkpoint
        if val_f1 > best_f1 + 1e-4:

            best_f1 = val_f1
            patience_counter = 0

            print("\nCalculating per-class recall...")

            class_recall = get_class_recall(
                model,
                val_loader
            )

            for name, recall in class_recall.items():
                print(
                    f"{name:8s} recall: {recall:.4f}"
                )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_f1": best_f1,
                "class_names": CLASS_NAMES,
                "image_size": 224,
                "architecture": "efficientnet_b0",
                "validation_class_recall": class_recall
            }, MODEL_DIR / "best_efficientnet.pth")

            print("\nBest EfficientNet model saved!")

        else:
            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        # Save training history
        pd.DataFrame(history).to_csv(
            METRICS_DIR / "efficientnet_training_history.csv",
            index=False
        )

        save_curves(history)

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered!")
            break

    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 60)
    print("EFFICIENTNET TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best validation macro F1:",
        round(best_f1, 4)
    )

    print(
        "Training time:",
        round(elapsed, 2),
        "minutes"
    )

    print(
        "Best model:",
        MODEL_DIR / "best_efficientnet.pth"
    )

    print(
        "History:",
        METRICS_DIR / "efficientnet_training_history.csv"
    )

    print(
        "Curves:",
        FIGURES_DIR / "efficientnet_training_curves.png"
    )


if __name__ == "__main__":
    main()
