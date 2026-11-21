
import torch
import matplotlib.pyplot as plt

from src.dataset import (
    create_dataloaders,
    CLASS_NAMES
)

from src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD
)


def main():
    print("=" * 55)
    print("RETINAL DISEASE DETECTION - DATALOADER TEST")
    print("=" * 55)

    train_loader, val_loader, test_loader = (
        create_dataloaders(batch_size=8)
    )

    images, labels = next(iter(train_loader))

    print("\nBatch shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Labels:", labels.tolist())

    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)
    assert torch.isfinite(images).all()
    assert labels.min() >= 0
    assert labels.max() < len(CLASS_NAMES)

    print("\nPreprocessing test: PASSED")

    # Reverse normalization for visualization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        image = images[i].cpu() * std + mean
        image = image.clamp(0, 1)

        image = image.permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.set_title(CLASS_NAMES[labels[i].item()])
        ax.axis("off")

    plt.suptitle("Preprocessed OCT Training Images")
    plt.tight_layout()

    from pathlib import Path

    output_path = Path("outputs/figures")
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path / "preprocessed_samples.png",
        dpi=300
    )

    plt.show()

    print("\nVisualization saved successfully!")


if __name__ == "__main__":
    main()
