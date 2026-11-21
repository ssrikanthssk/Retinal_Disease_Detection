
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.model import OCTCNN
from src.dataset import create_dataloaders, CLASS_NAMES
from src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_cnn.pth"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"

for folder in [
    METRICS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    saved_classes = checkpoint.get("class_names")

    if saved_classes != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match "
            "the current dataset class order."
        )

    model = OCTCNN(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation macro F1:",
        checkpoint["best_val_f1"]
    )

    return model


@torch.inference_mode()
def evaluate_model(model, test_loader):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities)
    )


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES))
    )

    np.savetxt(
        METRICS_DIR / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=","
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("CNN Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "test_confusion_matrix.png",
        dpi=300
    )

    plt.close()


def save_misclassified_images(
    test_dataset,
    y_true,
    y_pred,
    max_images=12
):
    wrong_indices = np.flatnonzero(
        y_true != y_pred
    )

    if len(wrong_indices) == 0:
        print("No misclassified images found.")
        return

    selected = wrong_indices[:max_images]

    mean = torch.tensor(
        IMAGENET_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD
    ).view(3, 1, 1)

    cols = 4
    rows = int(np.ceil(len(selected) / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, index in zip(
        axes.flat,
        selected
    ):
        image, _ = test_dataset[int(index)]

        image = image.cpu() * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()

        ax.imshow(image)

        actual = CLASS_NAMES[int(y_true[index])]
        predicted = CLASS_NAMES[int(y_pred[index])]

        ax.set_title(
            f"Actual: {actual}\nPredicted: {predicted}",
            color="red",
            fontsize=10
        )

    plt.suptitle(
        "Misclassified OCT Test Images",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "misclassified_images.png",
        dpi=300
    )

    plt.close()


def main():
    print("=" * 60)
    print("RETINAL DISEASE DETECTION - TEST EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)

    model = load_model()

    _, _, test_loader = create_dataloaders(
        batch_size=8,
        num_workers=0
    )

    print("\nEvaluating on original test dataset...")

    y_true, y_pred, probabilities = evaluate_model(
        model,
        test_loader
    )

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=range(len(CLASS_NAMES)),
            zero_division=0
        )
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        METRICS_DIR / "classification_report.csv"
    )

    metrics = {
        "test_images": len(y_true),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

    with open(
        METRICS_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save individual predictions
    predictions_df = pd.DataFrame({
        "actual_label": [
            CLASS_NAMES[int(x)] for x in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[int(x)] for x in y_pred
        ],
        "correct": y_true == y_pred,
        "confidence": probabilities.max(axis=1)
    })

    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = (
            probabilities[:, class_id]
        )

    predictions_df.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False
    )

    save_confusion_matrix(y_true, y_pred)

    save_misclassified_images(
        test_loader.dataset,
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(
        "Correct predictions:",
        int(np.sum(y_true == y_pred))
    )

    print(
        "Incorrect predictions:",
        int(np.sum(y_true != y_pred))
    )

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
