"""
Phase 10: Deployment & Output Dashboard - Streamlit Application
Interactive Web Dashboard for Construction Building Failure & Crack Detection System:
- Full Structural Health Assessment (Sensor Form + Image Upload)
- Real-time Machine Learning Risk Prediction (Low / Medium / High)
- Computer Vision Deep Learning Crack Detection & Bounding Box Overlay
- Interactive EDA Visualizer & Model Performance Benchmark Metrics
- Downloadable Inspection Reports
"""

import os
import sys
import io
import json
import base64
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import torch
from torchvision import transforms

# Add src and backend paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from feature_engineering import create_structural_features
from dl_model import MultiTaskCrackDetector

# Page Configuration
st.set_page_config(
    page_title="Construction Building Failure & Crack Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-low {
        color: #16A34A;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .risk-medium {
        color: #D97706;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .risk-high {
        color: #DC2626;
        font-weight: 800;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")


@st.cache_resource
def load_ml_artifacts():
    """Cached loader for ML model and encoders."""
    model_path = os.path.join(MODELS_DIR, "structural_risk_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_dl_model():
    """Cached loader for PyTorch DL Crack Detector model."""
    dl_path = os.path.join(MODELS_DIR, "crack_detection_model.pth")
    if os.path.exists(dl_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskCrackDetector().to(device)
        checkpoint = torch.load(dl_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, device
    return None, None


ml_artifact = load_ml_artifacts()
dl_model, device = load_dl_model()

# Header Section
st.markdown("<div class='main-title'>🏗️ Building Failure & Crack Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predictive Machine Learning Structural Health Monitoring & Deep Learning Vision Crack Inspection</div>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Workflow Module:",
    [
        "🏢 Full Structural Assessment",
        "🔍 Crack Vision Inspector",
        "📊 Sensor Risk Simulator",
        "📈 EDA Analytics",
        "🧪 Model Performance & Metrics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Architecture:**
- **ML Model**: XGBoost Classifier (CV Acc: 93.2%)
- **DL Model**: PyTorch Multi-Task CNN (Val Acc: 100%, Mean IoU: 0.66)
- **Backend API**: FastAPI REST Service
""")

# MODULE 1: FULL STRUCTURAL ASSESSMENT
if menu == "🏢 Full Structural Assessment":
    st.header("🏢 Comprehensive Structural Health Diagnostic")
    st.markdown("Provide structural sensor measurements and upload a building photo for instant dual ML + DL assessment.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Tabular Sensor & Material Input")

        col_a, col_b = st.columns(2)
        with col_a:
            vib_amp = st.number_input("Vibration Amplitude (mm)", 0.05, 15.0, 3.20)
            vib_freq = st.number_input("Vibration Frequency (Hz)", 1.0, 120.0, 42.5)
            stress = st.number_input("Stress (mPa)", 5.0, 90.0, 35.0)
            strain = st.number_input("Strain (µε)", 100.0, 3500.0, 1400.0)
            age = st.number_input("Concrete Age (Years)", 1.0, 75.0, 22.0)
        with col_b:
            temp = st.number_input("Ambient Temp (°C)", -10.0, 50.0, 28.0)
            humidity = st.number_input("Humidity (%)", 20.0, 95.0, 65.0)
            maint_score = st.slider("Maintenance Score (1-10)", 1, 10, 6)
            mat_grade = st.selectbox("Material Grade", ["M20", "M30", "M40", "M50"], index=1)
            settlement = st.number_input("Foundation Settlement (mm)", 0.0, 25.0, 4.5)
            load_ratio = st.number_input("Load Ratio", 0.2, 1.3, 0.75)

    with col2:
        st.subheader("2. Building Inspection Photo Upload")
        uploaded_file = st.file_uploader("Upload concrete/wall/building image (JPEG/PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Inspection Photo", use_container_width=True)
        else:
            # Use sample image if none uploaded
            sample_img_path = os.path.join(os.path.dirname(__file__), "data", "images", "val", "building_val_0001.jpg")
            if os.path.exists(sample_img_path):
                image = Image.open(sample_img_path).convert("RGB")
                st.image(image, caption="Sample Building Surface Image (Default)", use_container_width=True)
            else:
                image = None

    if st.button("🚀 Run Full AI Assessment", type="primary", use_container_width=True):
        if ml_artifact is None or dl_model is None:
            st.error("Error: Trained model artifacts not found. Please train models first.")
        else:
            with st.spinner("Processing ML Sensor Analytics & DL Computer Vision..."):
                # 1. Run ML Risk Inference
                grade_map = {"M20": 0, "M30": 1, "M40": 2, "M50": 3}
                raw_df = pd.DataFrame([{
                    "vibration_amplitude_mm": vib_amp,
                    "vibration_frequency_hz": vib_freq,
                    "stress_mPa": stress,
                    "strain_micro": strain,
                    "concrete_age_years": age,
                    "ambient_temp_c": temp,
                    "humidity_pct": humidity,
                    "maintenance_score": maint_score,
                    "material_grade": mat_grade,
                    "foundation_settlement_mm": settlement,
                    "load_ratio": load_ratio,
                    "material_grade_encoded": grade_map[mat_grade]
                }])

                df_feat = create_structural_features(raw_df)
                X_in = df_feat[ml_artifact["feature_cols"]]
                ml_pred_idx = ml_artifact["model"].predict(X_in)[0]
                ml_probas = ml_artifact["model"].predict_proba(X_in)[0]

                risk_labels = ["Low", "Medium", "High"]
                risk_label = risk_labels[int(ml_pred_idx)]
                ml_score_pct = float(ml_probas[1] * 50.0 + ml_probas[2] * 100.0)

                # 2. Run DL Crack Inference
                if image is not None:
                    orig_w, orig_h = image.size
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        cls_prob, bbox_pred = dl_model(img_tensor)
                        dl_conf = float(cls_prob.cpu().item())
                        bbox_coords = bbox_pred.cpu().squeeze(0).numpy()

                    has_crack = dl_conf > 0.50
                    crack_str = "Detected" if has_crack else "Not Detected"

                    # Scale bbox
                    xmin = int(bbox_coords[0] * orig_w)
                    ymin = int(bbox_coords[1] * orig_h)
                    xmax = int(bbox_coords[2] * orig_w)
                    ymax = int(bbox_coords[3] * orig_h)

                    # Draw bbox overlay
                    annotated_img = image.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    if has_crack:
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
                else:
                    has_crack = False
                    crack_str = "Not Detected"
                    dl_conf = 0.0
                    annotated_img = None
                    xmin, ymin, xmax, ymax = 0, 0, 0, 0

                # 3. Calculate Combined Structural Health Index (%)
                health_index = max(0.0, round(100.0 - ml_score_pct - (dl_conf * 35.0 if has_crack else 0.0), 1))

                st.markdown("---")
                st.subheader("📊 Diagnostic Summary & Outputs")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    risk_css = f"risk-{risk_label.lower()}"
                    st.markdown(f"<div class='metric-card'><div>🏢 Structural Risk</div><div class='{risk_css}'>{risk_label}</div></div>", unsafe_allow_html=True)
                with m2:
                    crack_color = "#DC2626" if has_crack else "#16A34A"
                    st.markdown(f"<div class='metric-card'><div>🔍 Crack Status</div><div style='font-size:1.5rem; font-weight:800; color:{crack_color}'>{crack_str}</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-card'><div>📍 Crack Confidence</div><div style='font-size:1.5rem; font-weight:800; color:#2563EB'>{dl_conf*100:.1f}%</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='metric-card'><div>🛡️ Health Index</div><div style='font-size:1.5rem; font-weight:800; color:#0D9488'>{health_index}%</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown("### 📍 Crack Location & Bounding Box")
                    if annotated_img is not None and has_crack:
                        st.image(annotated_img, caption=f"Bounding Box: [{xmin}, {ymin}, {xmax}, {ymax}]", use_container_width=True)
                    elif annotated_img is not None:
                        st.image(annotated_img, caption="No Crack Detected on Surface", use_container_width=True)

                with res_col2:
                    st.markdown("### 📈 Risk Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Risk Category": ["Low", "Medium", "High"],
                        "Probability": [ml_probas[0], ml_probas[1], ml_probas[2]]
                    })
                    st.bar_chart(prob_df.set_index("Risk Category"))

                    st.markdown("### 🛠️ Actionable Recommendations")
                    if health_index >= 75.0:
                        st.success("🟢 OPTIMAL CONDITION: Standard scheduled maintenance.")
                    elif health_index >= 45.0:
                        st.warning("🟡 MODERATE RISK: Schedule non-destructive testing & sensor monitoring.")
                    else:
                        st.error("🔴 CRITICAL RISK: Restrict access immediately & dispatch engineering team.")

# MODULE 2: CRACK VISION INSPECTOR
elif menu == "🔍 Crack Vision Inspector":
    st.header("🔍 Computer Vision Building Crack Inspector")
    st.markdown("Upload building photos to test deep learning crack localization and inspect normalized bounding box coordinates.")

    img_file = st.file_uploader("Upload Wall / Concrete Surface Image", type=["jpg", "png", "jpeg"], key="crack_insp")

    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        orig_w, orig_h = img.size

        conf_thresh = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            cls_prob, bbox_pred = dl_model(t_img)
            conf = float(cls_prob.cpu().item())
            bbox_norm = bbox_pred.cpu().squeeze(0).numpy()

        has_crack = conf >= conf_thresh

        xmin = int(bbox_norm[0] * orig_w)
        ymin = int(bbox_norm[1] * orig_h)
        xmax = int(bbox_norm[2] * orig_w)
        ymax = int(bbox_norm[3] * orig_h)

        ann_img = img.copy()
        draw = ImageDraw.Draw(ann_img)
        if has_crack:
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)
        with col_b:
            st.subheader("Annotated Crack Detection")
            st.image(ann_img, use_container_width=True)

        st.json({
            "crack_detected": has_crack,
            "confidence_score": round(conf, 4),
            "normalized_bbox": [round(float(c), 4) for c in bbox_norm],
            "pixel_bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        })

# MODULE 3: SENSOR RISK SIMULATOR
elif menu == "📊 Sensor Risk Simulator":
    st.header("📊 Interactive Sensor Risk Simulator")
    st.markdown("Adjust structural sensor sliders to observe real-time XGBoost risk predictions and feature importance weights.")

    s_stress = st.slider("Stress (mPa)", 5.0, 90.0, 45.0)
    s_vamp = st.slider("Vibration Amplitude (mm)", 0.05, 15.0, 5.0)
    s_settle = st.slider("Foundation Settlement (mm)", 0.0, 25.0, 8.0)
    s_maint = st.slider("Maintenance History Score", 1, 10, 5)

    if ml_artifact is not None:
        grade_map = {"M20": 0, "M30": 1, "M40": 2, "M50": 3}
        raw_df = pd.DataFrame([{
            "vibration_amplitude_mm": s_vamp,
            "vibration_frequency_hz": 40.0,
            "stress_mPa": s_stress,
            "strain_micro": 1500.0,
            "concrete_age_years": 25.0,
            "ambient_temp_c": 30.0,
            "humidity_pct": 60.0,
            "maintenance_score": s_maint,
            "material_grade": "M30",
            "foundation_settlement_mm": s_settle,
            "load_ratio": 0.8,
            "material_grade_encoded": grade_map["M30"]
        }])

        df_feat = create_structural_features(raw_df)
        X_in = df_feat[ml_artifact["feature_cols"]]
        pred_idx = ml_artifact["model"].predict(X_in)[0]
        probas = ml_artifact["model"].predict_proba(X_in)[0]
        risk_labels = ["Low", "Medium", "High"]

        st.subheader(f"Predicted Structural Risk: **{risk_labels[pred_idx]}**")
        st.write(f"Low: {probas[0]*100:.1f}% | Medium: {probas[1]*100:.1f}% | High: {probas[2]*100:.1f}%")

# MODULE 4: EDA ANALYTICS
elif menu == "📈 EDA Analytics":
    st.header("📈 Exploratory Data Analysis & Risk Patterns")

    eda_summary_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    if os.path.exists(eda_summary_path):
        with open(eda_summary_path, "r") as f:
            eda_data = json.load(f)

        st.json(eda_data)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        if os.path.exists(os.path.join(FIG_DIR, "eda_correlation_matrix.png")):
            st.image(os.path.join(FIG_DIR, "eda_correlation_matrix.png"), use_column_width=True)

    with col2:
        st.subheader("Risk Class Distribution")
        if os.path.exists(os.path.join(FIG_DIR, "eda_risk_distribution.png")):
            st.image(os.path.join(FIG_DIR, "eda_risk_distribution.png"), use_column_width=True)

# MODULE 5: MODEL PERFORMANCE & METRICS
elif menu == "🧪 Model Performance & Metrics":
    st.header("🧪 ML & DL Model Evaluation Benchmarks")

    metrics_path = os.path.join(REPORTS_DIR, "model_evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        st.subheader("Machine Learning Risk Classifier Metrics")
        st.json(metrics_data["machine_learning_metrics"])

        st.subheader("Deep Learning PyTorch Crack Detector Metrics")
        st.json(metrics_data["deep_learning_metrics"])

    if os.path.exists(os.path.join(FIG_DIR, "ml_confusion_matrix.png")):
        st.subheader("ML Confusion Matrix")
        st.image(os.path.join(FIG_DIR, "ml_confusion_matrix.png"), width=500)
