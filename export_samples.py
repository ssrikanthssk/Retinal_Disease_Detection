
from pathlib import Path
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent

dataset = load_from_disk(
    str(ROOT / "data" / "raw" / "oct2017")
)

output_dir = ROOT / "sample_images"
output_dir.mkdir(exist_ok=True)

classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

for class_id, class_name in enumerate(classes):
    for sample in dataset["test"]:
        if sample["label"] == class_id:
            sample["image"].convert("RGB").save(
                output_dir / f"{class_name}.jpg"
            )
            print(f"Saved: {class_name}.jpg")
            break

print("All sample images exported successfully!")
