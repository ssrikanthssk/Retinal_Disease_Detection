
import importlib
import torch

libraries = {
    "NumPy": "numpy",
    "Pandas": "pandas",
    "Matplotlib": "matplotlib",
    "Seaborn": "seaborn",
    "Scikit-learn": "sklearn",
    "Pillow": "PIL",
    "OpenCV": "cv2",
    "TQDM": "tqdm",
    "Streamlit": "streamlit",
    "Grad-CAM": "pytorch_grad_cam",
    "Hugging Face Datasets": "datasets",
    "Hugging Face Hub": "huggingface_hub",
    "Jupyter": "jupyter",
}

print("=" * 50)
print("RETINAL DISEASE DETECTION - LIBRARY CHECK")
print("=" * 50)

failed = []

for name, module in libraries.items():
    try:
        importlib.import_module(module)
        print(f"[OK] {name}")
    except Exception as error:
        print(f"[FAILED] {name}: {error}")
        failed.append(name)

print("-" * 50)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if failed:
    print("Missing or broken libraries:", ", ".join(failed))
else:
    print("ALL LIBRARIES VERIFIED SUCCESSFULLY")

