"""
Phase 6: ML Model Development
Trains ML Classifiers (Random Forest, XGBoost) to predict Structural Failure Risk (Low/Medium/High):
1. Loads feature-engineered tabular dataset
2. Performs 5-Fold Stratified Cross-Validation & hyperparameter tuning
3. Compares Random Forest vs XGBoost accuracy & macro F1-score
4. Saves best model artifact (models/structural_risk_model.joblib)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from xgboost import XGBClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_engineered_data():
    """Loads feature-engineered dataset and returns X, y, and feature column names."""
    filepath = os.path.join(DATA_DIR, "feature_engineered_data.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError("Engineered dataset not found. Run Phase 5 first.")

    df = pd.read_csv(filepath)

    feature_cols = [
        "vibration_amplitude_mm",
        "vibration_frequency_hz",
        "stress_mPa",
        "strain_micro",
        "concrete_age_years",
        "ambient_temp_c",
        "humidity_pct",
        "maintenance_score",
        "foundation_settlement_mm",
        "load_ratio",
        "material_grade_encoded",
        "stress_capacity_ratio",
        "vibration_severity_index",
        "maintenance_decay_index",
        "settlement_load_interaction",
        "environmental_exposure_index",
        "age_group_encoded"
    ]

    X = df[feature_cols]
    y = df["failure_risk_encoded"]

    return X, y, feature_cols


def train_and_evaluate_ml_models():
    """Trains Random Forest & XGBoost, compares metrics, and exports best model."""
    print("🤖 Starting Phase 6 ML Model Training...")
    X, y, feature_cols = load_engineered_data()

    # Load fitted scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    # Note: Tree models can train on raw X or scaled X. We fit scaler on base numeric cols.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    rf_cv = cross_validate(rf_model, X, y, cv=skf, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])

    rf_acc = np.mean(rf_cv["test_accuracy"])
    rf_f1 = np.mean(rf_cv["test_f1_macro"])
    print(f"🌲 Random Forest 5-Fold CV Accuracy: {rf_acc*100:.2f}% | F1-Score: {rf_f1:.4f}")

    # 2. XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )

    xgb_cv = cross_validate(xgb_model, X, y, cv=skf, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])

    xgb_acc = np.mean(xgb_cv["test_accuracy"])
    xgb_f1 = np.mean(xgb_cv["test_f1_macro"])
    print(f"⚡ XGBoost 5-Fold CV Accuracy: {xgb_acc*100:.2f}% | F1-Score: {xgb_f1:.4f}")

    # Select Best Model
    if xgb_f1 >= rf_f1:
        best_model_name = "XGBoost"
        best_model = xgb_model
        best_acc = xgb_acc
        best_f1 = xgb_f1
    else:
        best_model_name = "RandomForest"
        best_model = rf_model
        best_acc = rf_acc
        best_f1 = rf_f1

    print(f"🏆 Best Model Selected: {best_model_name} (CV Accuracy: {best_acc*100:.2f}%)")

    # Train best model on full dataset
    best_model.fit(X, y)

    # Compute Feature Importances
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n⭐ Top 5 Most Important Structural Failure Predictors:")
    for idx, row in feat_imp.head(5).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Save model artifact
    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_cols": feature_cols,
        "cv_accuracy": best_acc,
        "cv_f1_score": best_f1,
        "feature_importances": feat_imp.to_dict(orient="records")
    }

    save_path = os.path.join(MODELS_DIR, "structural_risk_model.joblib")
    joblib.dump(model_artifact, save_path)
    print(f"✅ Saved trained ML model artifact to: {save_path}")

    return model_artifact


if __name__ == "__main__":
    train_and_evaluate_ml_models()
