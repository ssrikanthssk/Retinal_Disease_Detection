
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def build_efficientnet(num_classes=4, pretrained=True):

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    # Replace the original ImageNet classifier
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_efficientnet(
        num_classes=4,
        pretrained=True
    ).to(device)

    # Test with a small batch
    test_input = torch.randn(
        2, 3, 224, 224,
        device=device
    )

    model.eval()

    with torch.no_grad():
        output = model(test_input)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 55)
    print("EFFICIENTNET-B0 MODEL TEST")
    print("=" * 55)

    print("Device:", device)
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)

    print("Total parameters:", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    assert output.shape == (2, 4)

    print("\nEfficientNet-B0 test: PASSED")
