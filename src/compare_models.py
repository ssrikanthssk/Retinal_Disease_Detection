
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.dataset import create_dataloaders, CLASS_NAMES
from src.model import OCTCNN
from src.efficientnet_model import build_efficientnet


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
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

BATCH_SIZE = 8


# ==========================================
# LOAD SAVED MODEL
# ==========================================

def load_model(model_name, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch for {model_name}"
        )

    if model_name == "Custom_CNN":
        model = OCTCNN(num_classes=len(CLASS_NAMES))

    elif model_name == "EfficientNet_B0":
        model = build_efficientnet(
            num_classes=len(CLASS_NAMES),
            pretrained=False
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"{model_name} loaded | "
        f"Best epoch: {checkpoint['epoch']}"
    )

    return model


# ==========================================
# PREDICT ON VALIDATION DATA
# ==========================================

@torch.inference_mode()
def evaluate_model(model, loader, model_name):

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in tqdm(
        loader,
        desc=f"Evaluating {model_name}"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ==========================================
# CALCULATE METRICS
# ==========================================

def calculate_metrics(
    model_name,
    y_true,
    y_pred
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "Balanced Accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1": float(f1)
    }


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(
    model_name,
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        METRICS_DIR /
        f"{model_name}_validation_confusion_matrix.csv"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title(
        f"{model_name} — Validation Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        f"{model_name}_validation_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

def save_classification_report(
    model_name,
    y_true,
    y_pred
):

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR /
        f"{model_name}_validation_classification_report.csv"
    )


# ==========================================
# SAVE PREDICTIONS
# ==========================================

def save_predictions(
    model_name,
    y_true,
    y_pred,
    probabilities
):

    data = {
        "true_label": [
            CLASS_NAMES[i] for i in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[i] for i in y_pred
        ],
        "correct": y_true == y_pred
    }

    for i, class_name in enumerate(CLASS_NAMES):
        data[f"prob_{class_name}"] = (
            probabilities[:, i]
        )

    pd.DataFrame(data).to_csv(
        PREDICTIONS_DIR /
        f"{model_name}_validation_predictions.csv",
        index=False
    )


# ==========================================
# SAVE COMPARISON GRAPH
# ==========================================

def save_comparison_chart(comparison_df):

    metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ]

    plot_df = comparison_df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model"
    )

    plt.ylim(0, 1.05)

    plt.title(
        "Custom CNN vs EfficientNet-B0 — Validation"
    )

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print("Device:", DEVICE)

    print("\nLoading validation dataset...")

    _, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    models = {
        "Custom_CNN": (
            MODEL_DIR / "best_cnn.pth"
        ),
        "EfficientNet_B0": (
            MODEL_DIR / "best_efficientnet.pth"
        )
    }

    results = []

    for model_name, checkpoint_path in models.items():

        print("\n" + "=" * 60)
        print("MODEL:", model_name)
        print("=" * 60)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        model = load_model(
            model_name,
            checkpoint_path
        )

        y_true, y_pred, probabilities = (
            evaluate_model(
                model,
                val_loader,
                model_name
            )
        )

        metrics = calculate_metrics(
            model_name,
            y_true,
            y_pred
        )

        results.append(metrics)

        save_confusion_matrix(
            model_name,
            y_true,
            y_pred
        )

        save_classification_report(
            model_name,
            y_true,
            y_pred
        )

        save_predictions(
            model_name,
            y_true,
            y_pred,
            probabilities
        )

        print("\nValidation results:")

        for key, value in metrics.items():

            if key == "Model":
                continue

            print(f"{key}: {value:.4f}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(results)

    comparison_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False
    )

    save_comparison_chart(comparison_df)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    best_model = comparison_df.loc[
        comparison_df["Macro F1"].idxmax(),
        "Model"
    ]

    print(
        "\nBest model by validation macro F1:",
        best_model
    )

    print(
        "\nSaved comparison:",
        METRICS_DIR / "model_comparison.csv"
    )

    print(
        "Saved chart:",
        FIGURES_DIR / "model_comparison.png"
    )


if __name__ == "__main__":
    main()

