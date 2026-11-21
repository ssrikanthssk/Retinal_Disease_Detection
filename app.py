
from pathlib import Path
from io import BytesIO
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, UnidentifiedImageError

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.preprocessing import get_val_transforms
from src.efficientnet_model import build_efficientnet


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Retinal Disease Detection | AI Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "best_efficientnet.pth"

COMPARISON_PATH = (
    ROOT / "outputs" / "metrics" / "model_comparison.csv"
)

COMPARISON_IMAGE = (
    ROOT / "outputs" / "figures" / "model_comparison.png"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

DESCRIPTIONS = {
    "CNV": "Choroidal Neovascularization",
    "DME": "Diabetic Macular Edema",
    "DRUSEN": "Drusen",
    "NORMAL": "Normal retinal OCT appearance"
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    padding: 28px 32px;
    background: linear-gradient(
        120deg,
        #101827,
        #162b40,
        #103f4a
    );
    border: 1px solid #294454;
    border-radius: 18px;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero-subtitle {
    color: #c0d5df;
    margin-top: 10px;
    font-size: 1rem;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 15px 0;
}

.result-card {
    padding: 22px;
    border: 1px solid #37536a;
    background: #12202e;
    border-radius: 15px;
    margin-bottom: 12px;
}

.result-label {
    color: #b4c4d4;
    font-size: 0.85rem;
}

.result-value {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 800;
}

.result-detail {
    color: #cbd5e1;
    font-size: 0.95rem;
}

.disclaimer {
    padding: 15px;
    border-radius: 10px;
    background: #352b15;
    color: #ffdc88;
    border: 1px solid #665020;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 650;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    if list(checkpoint["class_names"]) != CLASSES:
        raise ValueError("Checkpoint class order mismatch.")

    model = build_efficientnet(
        num_classes=4,
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess(image):

    transform = get_val_transforms()

    tensor = transform(
        image.convert("RGB")
    )

    return tensor.unsqueeze(0).to(DEVICE)


def denormalize(tensor):

    image = tensor[0].detach().cpu().numpy()

    image = np.transpose(image, (1, 2, 0))
    image = image * STD + MEAN

    return np.clip(image, 0, 1).astype(np.float32)


# =====================================================
# PREDICTION
# =====================================================

@torch.inference_mode()
def predict(model, tensor):

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    return predicted_index, probabilities


# =====================================================
# GRAD-CAM
# =====================================================

def generate_gradcam(model, tensor, class_index):

    target_layers = [model.features[-1]]

    targets = [
        ClassifierOutputTarget(class_index)
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )[0]

    image = denormalize(tensor)

    overlay = show_cam_on_image(
        image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5
    )

    return grayscale_cam, overlay


# =====================================================
# REPORT GENERATION
# =====================================================

def create_report(filename, prediction, confidence, probabilities):

    report = {
        "report_type": "Research demonstration",
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "image_filename": filename,
        "model": "EfficientNet-B0",
        "predicted_class": prediction,
        "predicted_description": DESCRIPTIONS[prediction],
        "model_confidence": round(
            float(confidence), 6
        ),
        "class_scores": {
            name: round(float(score), 6)
            for name, score in zip(
                CLASSES, probabilities
            )
        },
        "disclaimer": (
            "Research and educational use only. "
            "Not a medical diagnosis. "
            "Model scores are not calibrated "
            "disease probabilities."
        )
    }

    return json.dumps(report, indent=2)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        👁️ Retinal Disease Detection
    </div>
    <div class="hero-subtitle">
        AI-Powered OCT Image Classification |
        EfficientNet-B0 |
        Explainable AI with Grad-CAM
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ Research and educational demonstration only.
    This application is not a medical device and must
    not be used to diagnose, rule out, or manage disease.
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("👁️ OCT AI")

    st.caption("Retinal Image Analysis Dashboard")

    st.divider()

    st.subheader("Model Details")

    st.write("**Architecture:** EfficientNet-B0")
    st.write("**Input:** 224 × 224")
    st.write("**Classes:** 4")
    st.write(f"**Device:** {DEVICE}")

    st.divider()

    st.subheader("Supported Classes")

    for name, description in DESCRIPTIONS.items():
        st.write(f"**{name}**")
        st.caption(description)

    st.divider()

    st.caption(
        "For research and education only. "
        "Not intended for clinical decision-making."
    )


# =====================================================
# LOAD MODEL SAFELY
# =====================================================

try:

    model, checkpoint = load_model()

except Exception as error:

    st.error(f"Unable to load EfficientNet: {error}")
    st.stop()


# =====================================================
# MAIN TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "🔬 Prediction Dashboard",
    "📊 Model Performance",
    "📘 About Project"
])


# =====================================================
# TAB 1 — PREDICTION DASHBOARD
# =====================================================

with tab1:

    st.subheader("OCT Image Classification")

    st.write(
        "Upload a retinal OCT image to generate "
        "a research classification and Grad-CAM explanation."
    )

    uploaded_file = st.file_uploader(
        "Upload OCT Image",
        type=["jpg", "jpeg", "png"],
        key="oct_upload"
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                BytesIO(uploaded_file.getvalue())
            )

            image.load()
            image = image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):

            st.error("Please upload a valid image.")
            st.stop()

        tensor = preprocess(image)

        predicted_index, probabilities = predict(
            model,
            tensor
        )

        predicted_class = CLASSES[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        col1, col2 = st.columns(
            [1.2, 1],
            gap="large"
        )

        with col1:

            st.markdown(
                '<div class="section-title">'
                'Uploaded OCT Scan'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="section-title">'
                'AI Classification Result'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Category
                    </div>
                    <div class="result-value">
                        {predicted_class}
                    </div>
                    <div class="result-detail">
                        {DESCRIPTIONS[predicted_class]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Confidence is a softmax model score, "
                "not diagnostic certainty."
            )

        st.divider()

        st.subheader("Classification Probability Scores")

        probability_df = pd.DataFrame({
            "Disease": CLASSES,
            "Score (%)": probabilities * 100
        })

        chart_col, table_col = st.columns(
            [1.5, 1],
            gap="large"
        )

        with chart_col:

            st.bar_chart(
                probability_df.set_index("Disease"),
                y="Score (%)",
                horizontal=True
            )

        with table_col:

            st.dataframe(
                probability_df.style.format({
                    "Score (%)": "{:.2f}%"
                }),
                hide_index=True,
                use_container_width=True
            )

        st.divider()

        st.subheader("Grad-CAM Explainability")

        with st.spinner("Generating explanation..."):

            try:

                heatmap, overlay = generate_gradcam(
                    model,
                    tensor,
                    predicted_index
                )

            except Exception as error:

                st.error(
                    f"Grad-CAM generation failed: {error}"
                )
                st.stop()

        cam1, cam2 = st.columns(2)

        with cam1:

            st.image(
                heatmap,
                caption="Grad-CAM Activation Map",
                clamp=True,
                use_container_width=True
            )

        with cam2:

            st.image(
                overlay,
                caption="Grad-CAM Overlay",
                use_container_width=True
            )

        st.info(
            "Highlighted regions indicate relative "
            "model attribution. Grad-CAM does not "
            "establish clinical relevance."
        )

        st.divider()

        st.subheader("Download Prediction Summary")

        report_json = create_report(
            uploaded_file.name,
            predicted_class,
            confidence,
            probabilities
        )

        st.download_button(
            label="⬇️ Download Research Report (JSON)",
            data=report_json,
            file_name="oct_prediction_report.json",
            mime="application/json"
        )

    else:

        st.info(
            "Upload an OCT image to begin analysis."
        )


# =====================================================
# TAB 2 — MODEL PERFORMANCE
# =====================================================

with tab2:

    st.subheader("Model Evaluation")

    st.write(
        "Comparison of the custom CNN and "
        "EfficientNet-B0 on the same validation split."
    )

    if COMPARISON_PATH.exists():

        comparison_df = pd.read_csv(
            COMPARISON_PATH
        )

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.2%}",
                "Balanced Accuracy": "{:.2%}",
                "Macro Precision": "{:.2%}",
                "Macro Recall": "{:.2%}",
                "Macro F1": "{:.2%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        efficientnet_row = comparison_df[
            comparison_df["Model"] == "EfficientNet_B0"
        ]

        if not efficientnet_row.empty:

            row = efficientnet_row.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Validation Accuracy",
                    f"{row['Accuracy']:.2%}"
                )

            with m2:
                st.metric(
                    "Validation Macro F1",
                    f"{row['Macro F1']:.2%}"
                )

            with m3:
                st.metric(
                    "Validation Macro Recall",
                    f"{row['Macro Recall']:.2%}"
                )

        if COMPARISON_IMAGE.exists():

            st.image(
                str(COMPARISON_IMAGE),
                caption="Validation Model Comparison",
                use_container_width=True
            )

        st.download_button(
            "⬇️ Download Model Comparison CSV",
            data=COMPARISON_PATH.read_bytes(),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Comparison results not found. "
            "Run: python -m src.compare_models"
        )

    st.info(
        "These are validation metrics. The same "
        "validation split was used to select model "
        "checkpoints, so these results are not "
        "an independent test-set evaluation."
    )


# =====================================================
# TAB 3 — ABOUT PROJECT
# =====================================================

with tab3:

    st.subheader("About the Project")

    st.write(
        "Retinal Disease Detection is a deep-learning "
        "research project that classifies retinal "
        "Optical Coherence Tomography (OCT) scans "
        "into four categories."
    )

    st.markdown("### Project Objective")

    st.write(
        "Develop and compare deep-learning models "
        "for OCT image classification and visualize "
        "the regions influencing predictions."
    )

    st.markdown("### Dataset")

    st.write(
        "Kermany2017-OCT dataset, accessed through "
        "Hugging Face."
    )

    st.write(
        "The project uses a group-aware training "
        "and validation split based on provisional "
        "identifiers extracted from filenames."
    )

    st.markdown("### Classification Categories")

    for name, description in DESCRIPTIONS.items():
        st.write(f"**{name}:** {description}")

    st.markdown("### Model Architectures")

    st.write(
        "**Custom CNN:** Baseline convolutional "
        "neural network trained from scratch."
    )

    st.write(
        "**EfficientNet-B0:** ImageNet-pretrained "
        "architecture fine-tuned for four-class "
        "OCT image classification."
    )

    st.markdown("### Technology Stack")

    st.write(
        "Python, PyTorch, Torchvision, "
        "Scikit-learn, Pandas, NumPy, "
        "Streamlit, Matplotlib, and Grad-CAM."
    )

    st.markdown("### Limitations")

    st.write(
        "The model was trained on a particular "
        "research dataset and has not been "
        "validated for clinical use or for "
        "images from other institutions."
    )

    st.write(
        "Predicted confidence scores are not "
        "calibrated disease probabilities."
    )

    st.write(
        "Grad-CAM maps are explanatory "
        "visualizations, not medical evidence."
    )

    st.write(
        "The original test set has already "
        "been inspected during development, "
        "so later analyses based on it "
        "should be considered exploratory."
    )

    st.warning(
        "Research and educational use only. "
        "Do not use this application for medical "
        "diagnosis or treatment decisions."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Retinal Disease Detection | "
    "EfficientNet-B0 + Grad-CAM | "
    "AI/ML Research Project"
)
