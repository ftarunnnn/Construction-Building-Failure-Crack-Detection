"""
Phase 9: Integration & Backend - FastAPI Application
Provides RESTful APIs for Structural Failure Risk ML Prediction & Computer Vision Crack Detection:
- GET  /health
- POST /api/predict_ml
- POST /api/detect_crack
- POST /api/full_assessment
"""

import os
import sys
import io
import json
import base64
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Ensure src path is accessible
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from feature_engineering import create_structural_features
from dl_model import MultiTaskCrackDetector
from backend.schemas import (
    SensorInput, StructuralRiskResponse, CrackDetectionResponse,
    FullAssessmentResponse, HealthCheckResponse, BoundingBox
)

app = FastAPI(
    title="🏗️ Construction — Building Failure & Crack Detection API",
    description="Integrated AI service combining ML Tabular Sensor Risk Assessment and Deep Learning Building Crack Detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Global model state
ML_MODEL_ARTIFACT = None
LABEL_ENCODER = None
SCALER = None
DL_MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_models():
    """Loads saved trained ML & DL model artifacts upon server startup."""
    global ML_MODEL_ARTIFACT, LABEL_ENCODER, SCALER, DL_MODEL

    # Load ML Model
    ml_path = os.path.join(MODELS_DIR, "structural_risk_model.joblib")
    if os.path.exists(ml_path):
        ML_MODEL_ARTIFACT = joblib.load(ml_path)
        print("✅ FastAPI: Loaded ML Structural Risk Model.")

    # Load Encoders
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    if os.path.exists(le_path):
        LABEL_ENCODER = joblib.load(le_path)

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)

    # Load DL Model
    dl_path = os.path.join(MODELS_DIR, "crack_detection_model.pth")
    if os.path.exists(dl_path):
        DL_MODEL = MultiTaskCrackDetector().to(DEVICE)
        checkpoint = torch.load(dl_path, map_location=DEVICE, weights_only=False)
        DL_MODEL.load_state_dict(checkpoint["model_state_dict"])
        DL_MODEL.eval()
        print("✅ FastAPI: Loaded PyTorch DL Crack Detector Model.")


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """System health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        ml_model_loaded=ML_MODEL_ARTIFACT is not None,
        dl_model_loaded=DL_MODEL is not None
    )


def process_sensor_input(sensor: SensorInput):
    """Calculates engineered features and runs ML inference."""
    if ML_MODEL_ARTIFACT is None:
        raise HTTPException(status_code=500, detail="ML model is not loaded on server.")

    raw_dict = sensor.model_dump()
    df_raw = pd.DataFrame([raw_dict])
    
    # Preprocess categorical map
    grade_map = {"M20": 0, "M30": 1, "M40": 2, "M50": 3}
    df_raw["material_grade_encoded"] = df_raw["material_grade"].map(grade_map).fillna(1)

    # Feature Engineering
    df_feat = create_structural_features(df_raw)

    feature_cols = ML_MODEL_ARTIFACT["feature_cols"]
    X_input = df_feat[feature_cols]

    # Model Prediction
    model = ML_MODEL_ARTIFACT["model"]
    pred_class_idx = model.predict(X_input)[0]
    pred_probas = model.predict_proba(X_input)[0]

    risk_labels = ["Low", "Medium", "High"]
    risk_label = risk_labels[int(pred_class_idx)]

    prob_dict = {
        "Low": round(float(pred_probas[0]), 4),
        "Medium": round(float(pred_probas[1]), 4),
        "High": round(float(pred_probas[2]), 4)
    }

    # Calculate overall risk score % (weighted sum)
    risk_score_pct = round(float(pred_probas[1] * 50.0 + pred_probas[2] * 100.0), 2)
    confidence = round(float(np.max(pred_probas)), 4)

    # Top risk factors from model importances
    top_factors = ML_MODEL_ARTIFACT["feature_importances"][:3]

    return StructuralRiskResponse(
        status="success",
        structural_risk=risk_label,
        risk_score_pct=risk_score_pct,
        confidence_score=confidence,
        risk_probabilities=prob_dict,
        top_risk_factors=top_factors
    )


@app.post("/api/predict_ml", response_model=StructuralRiskResponse)
def predict_structural_risk(sensor: SensorInput):
    """Predicts Structural Failure Risk (Low / Medium / High) from tabular sensor data."""
    return process_sensor_input(sensor)


def process_image_crack_detection(image_bytes: bytes) -> tuple[bool, str, float, BoundingBox, str]:
    """Runs PyTorch CNN inference on image bytes and generates annotated image Base64."""
    if DL_MODEL is None:
        raise HTTPException(status_code=500, detail="DL Crack Detector model is not loaded.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    orig_width, orig_height = pil_img.size

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_prob, bbox_pred = DL_MODEL(img_tensor)
        confidence = float(cls_prob.cpu().item())
        bbox_norm_coords = bbox_pred.cpu().squeeze(0).numpy()  # [xmin, ymin, xmax, ymax]

    has_crack = confidence > 0.50
    label = "Crack Detected" if has_crack else "No Crack Detected"

    # Scale normalized bbox coordinates back to original image size
    xmin = int(bbox_norm_coords[0] * orig_width)
    ymin = int(bbox_norm_coords[1] * orig_height)
    xmax = int(bbox_norm_coords[2] * orig_width)
    ymax = int(bbox_norm_coords[3] * orig_height)

    bbox_obj = BoundingBox(
        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
        xmin_norm=round(float(bbox_norm_coords[0]), 4),
        ymin_norm=round(float(bbox_norm_coords[1]), 4),
        xmax_norm=round(float(bbox_norm_coords[2]), 4),
        ymax_norm=round(float(bbox_norm_coords[3]), 4)
    )

    # Draw bounding box annotation on image copy
    annotated_img = pil_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    if has_crack:
        box_color = "red"
        line_width = max(3, int(min(orig_width, orig_height) / 80))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=line_width)
        text_str = f"Crack Detected ({confidence*100:.1f}%)"
        draw.text((xmin, max(5, ymin - 15)), text_str, fill="red")

    buffered = io.BytesIO()
    annotated_img.save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return has_crack, label, confidence, bbox_obj, img_b64


@app.post("/api/detect_crack", response_model=CrackDetectionResponse)
async def detect_building_crack(file: UploadFile = File(...)):
    """Detects concrete cracks and predicts bounding box coordinates on uploaded building photo."""
    contents = await file.read()
    has_crack, label, conf, bbox, b64_img = process_image_crack_detection(contents)

    return CrackDetectionResponse(
        status="success",
        crack_detected=has_crack,
        crack_label=label,
        confidence_score=round(conf, 4),
        bounding_box=bbox,
        annotated_image_base64=b64_img
    )


@app.post("/api/full_assessment", response_model=FullAssessmentResponse)
async def full_structural_assessment(
    sensor_json: str = Form(...),
    file: UploadFile = File(...)
):
    """Combines Tabular Sensor ML Risk Prediction + Image DL Crack Detection into a Unified Assessment Report."""
    try:
        sensor_data = json.loads(sensor_json)
        sensor_input = SensorInput(**sensor_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid sensor JSON payload: {str(e)}")

    ml_res = process_sensor_input(sensor_input)
    image_bytes = await file.read()
    has_crack, crack_label, crack_conf, bbox, b64_img = process_image_crack_detection(image_bytes)

    # Combined Structural Health Index Calculation (0% = Imminent Failure, 100% = Perfect Health)
    ml_health_pct = 100.0 - ml_res.risk_score_pct
    dl_penalty = (crack_conf * 35.0) if has_crack else 0.0

    combined_health_index = max(0.0, round(ml_health_pct - dl_penalty, 1))

    if combined_health_index >= 75.0:
        overall_category = "OPTIMAL STRUCTURAL INTEGRITY"
        recs = [
            "Maintain standard scheduled monitoring intervals.",
            "No immediate structural intervention required.",
            "Record baseline sensor telemetry."
        ]
    elif combined_health_index >= 45.0:
        overall_category = "MODERATE DEGRADATION - ATTENTION REQUIRED"
        recs = [
            "Increase vibration & strain sensor logging frequency.",
            "Schedule non-destructive ultrasonic testing on concrete columns.",
            "Apply surface sealant if micro-cracks expand."
        ]
    else:
        overall_category = "CRITICAL STRUCTURAL RISK - IMMEDIATE ACTION"
        recs = [
            "EVACUATE / RESTRICT HEAVY LOAD ACCESS IMMEDIATELY.",
            "Dispatch certified structural engineering assessment team.",
            "Install temporary shoring & support jacks near high-stress zones."
        ]

    return FullAssessmentResponse(
        status="success",
        structural_risk=ml_res.structural_risk,
        crack_label=crack_label,
        crack_detected=has_crack,
        combined_health_index_pct=combined_health_index,
        risk_level_category=overall_category,
        actionable_recommendations=recs,
        annotated_image_base64=b64_img
    )
