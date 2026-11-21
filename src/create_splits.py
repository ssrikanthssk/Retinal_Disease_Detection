
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "outputs/metrics/image_metadata.csv"
SPLIT_PATH = ROOT / "data/processed/split_indices.npz"
REPORT_PATH = ROOT / "outputs/metrics/split_summary.csv"

SEED = 42
TARGET_VAL_FRACTION = 0.20
N_CANDIDATES = 100

def main():
    df = pd.read_csv(
        METADATA_PATH,
        dtype={"identifier": "string"}
    )

    train_df = (
        df[df["split"] == "train"]
        .reset_index(drop=True)
    )

    test_df = df[df["split"] == "test"]

    if train_df["identifier"].isna().any():
        raise ValueError("Missing grouping identifiers")

    groups = train_df["identifier"].astype(str).to_numpy()
    labels = train_df["class"].to_numpy()

    classes, encoded = np.unique(
        labels,
        return_inverse=True
    )

    original_counts = np.bincount(
        encoded,
        minlength=len(classes)
    )

    original_props = original_counts / len(encoded)

    splitter = GroupShuffleSplit(
        n_splits=N_CANDIDATES,
        test_size=TARGET_VAL_FRACTION,
        random_state=SEED
    )

    best_score = float("inf")
    best_indices = None

    for train_idx, val_idx in splitter.split(
        encoded,
        groups=groups
    ):
        val_counts = np.bincount(
            encoded[val_idx],
            minlength=len(classes)
        )

        val_props = val_counts / len(val_idx)

        # Penalize class-distribution differences
        class_error = np.abs(
            val_props - original_props
        ).sum()

        # Penalize deviation from target image fraction
        size_error = abs(
            len(val_idx) / len(encoded)
            - TARGET_VAL_FRACTION
        )

        score = class_error + size_error

        if score < best_score:
            best_score = score
            best_indices = (
                train_idx.copy(),
                val_idx.copy()
            )

    train_idx, val_idx = best_indices

    train_ids = set(groups[train_idx])
    val_ids = set(groups[val_idx])
    test_ids = set(
        test_df["identifier"].dropna().astype(str)
    )

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    SPLIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        SPLIT_PATH,
        train_idx=train_idx,
        val_idx=val_idx
    )

    print("=" * 55)
    print("OPTIMIZED GROUP-AWARE SPLIT")
    print("=" * 55)

    print("Training images:", len(train_idx))
    print("Validation images:", len(val_idx))
    print("Testing images:", len(test_df))
    print("Identifier overlap: 0")

    print("\nClass distribution:")

    for class_id, class_name in enumerate(classes):
        train_pct = (
            np.mean(encoded[train_idx] == class_id) * 100
        )

        val_pct = (
            np.mean(encoded[val_idx] == class_id) * 100
        )

        print(
            f"{class_name:8s} "
            f"Train: {train_pct:6.2f}% | "
            f"Val: {val_pct:6.2f}%"
        )

    summary = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "images": [
            len(train_idx),
            len(val_idx),
            len(test_df)
        ],
        "unique_identifiers": [
            len(train_ids),
            len(val_ids),
            len(test_ids)
        ]
    })

    summary.to_csv(REPORT_PATH, index=False)

    print("\nBest split score:", round(best_score, 5))
    print("Saved:", SPLIT_PATH)


if __name__ == "__main__":
    main()
