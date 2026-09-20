"""
Phase 3: Data Preprocessing
Preprocesses tabular sensor data and image data:
1. Missing value handling & outlier clipping
2. Categorical encoding & numerical feature scaling
3. PyTorch Dataset & DataLoader construction for crack images
4. Saves data/processed_structural_data.csv
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NUMERICAL_COLS = [
    "vibration_amplitude_mm",
    "vibration_frequency_hz",
    "stress_mPa",
    "strain_micro",
    "concrete_age_years",
    "ambient_temp_c",
    "humidity_pct",
    "maintenance_score",
    "foundation_settlement_mm",
    "load_ratio"
]

CATEGORICAL_COLS = ["material_grade"]
TARGET_COL = "failure_risk"


def preprocess_tabular_data(csv_filename: str = "raw_structural_data.csv"):
    """Loads raw sensor data, cleans outliers, scales features, and encodes target."""
    raw_path = os.path.join(DATA_DIR, csv_filename)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run Phase 2 first.")

    df = pd.read_csv(raw_path)

    # 1. Check & handle missing values
    if df.isnull().sum().sum() > 0:
        print("Cleaning missing values...")
        df[NUMERICAL_COLS] = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())
        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna(df[CATEGORICAL_COLS].mode().iloc[0])

    # 2. Outlier clipping (3-sigma bounds)
    for col in NUMERICAL_COLS:
        q_low = df[col].quantile(0.005)
        q_high = df[col].quantile(0.995)
        df[col] = df[col].clip(q_low, q_high)

    # 3. Categorical encoding (Material Grade: M20=0, M30=1, M40=2, M50=3)
    grade_map = {"M20": 0, "M30": 1, "M40": 2, "M50": 3}
    df["material_grade_encoded"] = df["material_grade"].map(grade_map)

    # 4. Target Label Encoding
    label_encoder = LabelEncoder()
    # Ensure standard order: Low: 0, Medium: 1, High: 2
    label_encoder.fit(["Low", "Medium", "High"])
    df["failure_risk_encoded"] = label_encoder.transform(df["failure_risk"])

    # Save encoders
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Save processed dataframe
    processed_path = os.path.join(DATA_DIR, "processed_structural_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"✅ Saved preprocessed tabular data: {processed_path} ({len(df)} rows)")

    return df


def prepare_ml_train_test_split(df: pd.DataFrame = None):
    """Splits preprocessed data into Train and Test sets, fitting StandardScaler on Train."""
    if df is None:
        processed_path = os.path.join(DATA_DIR, "processed_structural_data.csv")
        df = pd.read_csv(processed_path)

    feature_cols = NUMERICAL_COLS + ["material_grade_encoded"]
    X = df[feature_cols]
    y = df["failure_risk_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler fitted and saved to: {scaler_path}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols
    }


def load_image_annotations():
    """Loads image annotation dictionary created in Phase 2."""
    ann_path = os.path.join(DATA_DIR, "images", "annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found at {ann_path}. Run Phase 2 first.")

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    train_anns = [a for a in annotations if a["split"] == "train"]
    val_anns = [a for a in annotations if a["split"] == "val"]

    print(f"✅ Loaded Image Dataset: {len(train_anns)} Train, {len(val_anns)} Validation samples")
    return train_anns, val_anns


if __name__ == "__main__":
    df = preprocess_tabular_data()
    data_split = prepare_ml_train_test_split(df)
    train_anns, val_anns = load_image_annotations()
