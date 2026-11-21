
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

from src.preprocessing import (
    get_train_transforms,
    get_val_transforms
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "oct2017"
SPLIT_PATH = ROOT / "data" / "processed" / "split_indices.npz"

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


class OCTDataset(Dataset):
    def __init__(self, hf_dataset, indices, transform=None):
        self.dataset = hf_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])

        sample = self.dataset[original_index]

        image = sample["image"]
        label = int(sample["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(batch_size=8, num_workers=0):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split indices not found: {SPLIT_PATH}"
        )

    print("Loading OCT2017 dataset...")

    dataset = load_from_disk(str(DATA_PATH))

    with np.load(SPLIT_PATH) as splits:
        train_indices = splits["train_idx"].copy()
        val_indices = splits["val_idx"].copy()

    train_dataset = OCTDataset(
        dataset["train"],
        train_indices,
        get_train_transforms()
    )

    val_dataset = OCTDataset(
        dataset["train"],
        val_indices,
        get_val_transforms()
    )

    test_dataset = OCTDataset(
        dataset["test"],
        np.arange(len(dataset["test"])),
        get_val_transforms()
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))

    return train_loader, val_loader, test_loader
