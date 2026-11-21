# Retinal Disease Detection Using Deep Learning

An end-to-end deep learning project for classifying retinal Optical Coherence Tomography (OCT) images using a Custom CNN and EfficientNet-B0, with Grad-CAM explainability and an interactive Streamlit dashboard.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Project Overview

This project explores deep learning approaches for classifying retinal OCT images into four categories:

- **CNV:** Choroidal Neovascularization
- **DME:** Diabetic Macular Edema
- **DRUSEN:** Drusen
- **NORMAL:** Normal retinal OCT appearance

The project includes dataset exploration, image preprocessing, model training, performance evaluation, explainable AI, and interactive deployment.

## Technology Stack

- **Programming:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Explainable AI:** Grad-CAM
- **Web Application:** Streamlit
- **Dataset Access:** Hugging Face Datasets
- **Hardware Used:** NVIDIA GeForce RTX 4050 Laptop GPU

## Dataset

**Dataset:** Kermany2017-OCT

**Source:** [Hugging Face — Kermany2017-OCT](https://huggingface.co/datasets/zacharielegault/Kermany2017-OCT)

The dataset contains 109,309 retinal OCT images.

| Dataset split | Images |
|---|---:|
| Training | 86,697 |
| Validation | 21,612 |
| Test | 1,000 |
| **Total** | **109,309** |

The training and validation sets were created using a group-aware split based on provisional identifiers extracted from image filenames. These identifiers have not been independently verified as patient IDs.

The original test split was preserved during initial training, although it was subsequently inspected during model development.

## Image Preprocessing

The preprocessing pipeline includes:

1. Conversion to RGB.
2. Aspect-ratio-preserving resizing.
3. Padding to 224 × 224 pixels.
4. ImageNet normalization.
5. Limited brightness and contrast augmentation during training.

The same deterministic preprocessing is applied during validation and application inference.

## Model Architectures

### Custom CNN

A baseline convolutional neural network trained from scratch using five convolutional blocks, batch normalization, ReLU activation, pooling, and a fully connected classification head.

**Parameters:** Approximately 1.01 million.

### EfficientNet-B0

An ImageNet-pretrained EfficientNet-B0 model fine-tuned for four-class retinal OCT classification.

**Parameters:** Approximately 4.01 million.

The model was trained using PyTorch with GPU acceleration, class-weighted cross-entropy loss, and mixed-precision training.

## Model Performance

Both saved checkpoints were evaluated on the same 21,612 validation images.

| Metric | Custom CNN | EfficientNet-B0 |
|---|---:|---:|
| Accuracy | 95.06% | **96.05%** |
| Balanced Accuracy | 93.34% | **93.82%** |
| Macro Precision | 91.21% | **93.17%** |
| Macro Recall | 93.34% | **93.82%** |
| Macro F1 | 92.19% | **93.47%** |

**Selected model:** EfficientNet-B0, based on validation macro F1.

These are validation results, not independent test-set performance estimates. The validation split was used to select the best checkpoints.

The baseline Custom CNN also achieved 89.00% accuracy on the original 1,000-image test split. This test set was subsequently examined during development, so additional test-driven comparisons should be considered exploratory.

## Grad-CAM Explainability

Grad-CAM generates activation heatmaps showing image regions contributing to the model's predicted class.

The project provides:

- Grad-CAM activation maps.
- Heatmap overlays on OCT images.
- Visual explanations for all four classes.

Grad-CAM does not establish clinical relevance or diagnostic validity.

## Streamlit Application

The interactive dashboard contains three sections.

### Prediction Dashboard

- Upload JPG, JPEG, or PNG OCT images.
- View predicted class and confidence score.
- Examine four-class model scores.
- View Grad-CAM heatmaps and overlays.
- Download a JSON research prediction summary.

### Model Performance

- Compare Custom CNN and EfficientNet-B0.
- View validation metrics.
- Explore model comparison visualizations.
- Download the model comparison CSV.

### About Project

- Dataset information.
- Classification categories.
- Model descriptions.
- Technology stack.
- Research limitations.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Retinal_Disease_Detection.git
cd Retinal_Disease_Detection
```

Replace `YOUR-USERNAME` with your GitHub username after creating the repository.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Use a Python and PyTorch version compatible with your hardware. CUDA installation requirements may vary.

### 4. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local address shown in your terminal, usually:

`http://localhost:8501`

The dashboard requires the trained checkpoint at `models/best_efficientnet.pth`.

## Reproducing the Project

Download the OCT dataset using:

```bash
python -m src.download_dataset
```

The training and validation split indices must be generated before training.

To compare the existing trained models:

```bash
python -m src.compare_models
```

To generate Grad-CAM visualizations:

```bash
python -m src.gradcam
```

To export example OCT images:

```bash
python export_samples.py
```

Some scripts require the downloaded dataset and saved split indices, which are excluded from GitHub because of their size.

## Generated Outputs

The project saves:

- Training history CSV files.
- Training curves.
- Classification reports.
- Confusion matrices.
- Model comparison charts.
- Validation predictions.
- Grad-CAM images.

These outputs are located primarily in `outputs/metrics/`, `outputs/figures/`, and `outputs/predictions/`.

## Limitations

- The model has not been validated for clinical deployment.
- Dataset performance may not generalize to other hospitals, devices, or populations.
- Softmax confidence scores are not calibrated disease probabilities.
- The extracted grouping identifiers are provisional.
- The validation set was used for model selection.
- The original test set was inspected during development.
- Grad-CAM visualizations are not proof of medical correctness.

## Future Improvements

- Evaluation on an independent external dataset.
- Additional analysis of DRUSEN classification errors.
- Model confidence calibration.
- Cross-device and cross-institution validation.
- Improved explainability analysis.
- More extensive robustness testing.

## License and Dataset Attribution

The OCT dataset is provided by its respective dataset owners. Review the dataset's license and usage terms before redistribution or commercial use.

The source-code license should be selected separately before publishing the repository.

---

**Retinal Disease Detection — Deep Learning, Computer Vision, and Explainable AI Research Project**