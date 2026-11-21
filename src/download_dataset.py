
from pathlib import Path
from datasets import load_dataset, load_from_disk

DATASET_ID = "zacharielegault/Kermany2017-OCT"
SAVE_PATH = Path("data/raw/oct2017")

def main():
    if SAVE_PATH.exists():
        print("Dataset folder already exists.")
        print("Loading saved dataset...")
        ds = load_from_disk(str(SAVE_PATH))
    else:
        print("Downloading OCT2017 dataset...")
        print("This may take some time.")

        ds = load_dataset(DATASET_ID)

        SAVE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Saving dataset locally...")
        ds.save_to_disk(str(SAVE_PATH))

    print("\nDataset information:")
    print(ds)

    print("\nClass names:")
    print(ds["train"].features["label"].names)

    print("\nDownload and verification completed!")

if __name__ == "__main__":
    main()
 