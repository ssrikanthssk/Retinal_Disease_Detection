
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from src.dataset import (
    create_dataloaders,
    CLASS_NAMES
)
from src.efficientnet_model import build_efficientnet


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_efficientnet.pth"

OUTPUT_DIR = (
    ROOT / "outputs" / "figures" / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# These values should match the normalization
# used by your dataset preprocessing.
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match dataset."
        )

    model = build_efficientnet(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print(
        "Validation macro F1:",
        checkpoint["best_val_f1"]
    )

    return model


# ==========================================
# RESTORE IMAGE FOR DISPLAY
# ==========================================

def denormalize_image(image_tensor):

    image = image_tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = image * STD + MEAN

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image.astype(np.float32)


# ==========================================
# CREATE GRAD-CAM VISUALIZATION
# ==========================================

def generate_gradcam(
    model,
    cam,
    image_tensor,
    true_label,
    sample_number
):

    input_tensor = (
        image_tensor.unsqueeze(0).to(DEVICE)
    )

    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_id = int(
            probabilities.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_id].item()
        )

    # Explain the predicted class
    targets = [
        ClassifierOutputTarget(predicted_id)
    ]

    heatmap = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    original_image = denormalize_image(
        image_tensor
    )

    overlay = show_cam_on_image(
        original_image,
        heatmap,
        use_rgb=True,
        image_weight=0.5
    )

    actual_name = CLASS_NAMES[int(true_label)]
    predicted_name = CLASS_NAMES[predicted_id]

    correct = predicted_id == int(true_label)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5)
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original OCT Image")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    result = "Correct" if correct else "Incorrect"

    fig.suptitle(
        f"Actual: {actual_name} | "
        f"Predicted: {predicted_name}\n"
        f"Confidence: {confidence:.2%} | "
        f"{result}",
        fontsize=13
    )

    plt.tight_layout()

    filename = (
        f"gradcam_{sample_number:02d}_"
        f"{actual_name}_to_{predicted_name}.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {filename} | "
        f"Confidence: {confidence:.2%}"
    )

    return predicted_id


# ==========================================
# MAIN
# ==========================================


def main():

    print("=" * 60)
    print("EFFICIENTNET-B0 GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    print("Device:", DEVICE)

    model = load_trained_model()

    print("\nLoading validation images...")

    _, val_loader, _ = create_dataloaders(
        batch_size=8,
        num_workers=0
    )

    val_dataset = val_loader.dataset

    # Retrieve labels without decoding OCT images
    all_labels = np.asarray(
        val_dataset.dataset["label"],
        dtype=np.int64
    )

    val_labels = all_labels[val_dataset.indices]

    rng = np.random.default_rng(42)

    selected = []

    # Select two images from each disease class
    for class_id, class_name in enumerate(CLASS_NAMES):

        class_indices = np.flatnonzero(
            val_labels == class_id
        )

        if len(class_indices) < 2:
            raise ValueError(
                f"Not enough validation images for {class_name}"
            )

        chosen_indices = rng.choice(
            class_indices,
            size=2,
            replace=False
        )

        for index in chosen_indices:
            image, true_label = val_dataset[
                int(index)
            ]

            selected.append(
                (image, int(true_label))
            )

        print(
            f"{class_name}: 2 validation images selected"
        )

    print(
        f"\nGenerating Grad-CAM for "
        f"{len(selected)} images..."
    )

    target_layers = [
        model.features[-1]
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        for index, (image, true_id) in enumerate(
            selected,
            start=1
        ):

            generate_gradcam(
                model=model,
                cam=cam,
                image_tensor=image,
                true_label=true_id,
                sample_number=index
            )

    print("\n" + "=" * 60)
    print("GRAD-CAM GENERATION COMPLETED")
    print("=" * 60)

    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
