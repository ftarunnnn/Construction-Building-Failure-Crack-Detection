"""
Phase 8: Model Evaluation
Evaluates ML Structural Risk Classifier and DL Crack Detector:
- ML Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- DL Metrics: Precision, Recall, F1-Score, Mean IoU, mAP@0.5
- Saves evaluation report to reports/model_evaluation_metrics.json and visual plots to reports/figures/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)

from ml_model import load_engineered_data
from dl_model import MultiTaskCrackDetector, BuildingCrackDataset, compute_iou
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def evaluate_ml_model():
    """Evaluates trained ML XGBoost/RandomForest model on test set."""
    print("📊 Evaluating ML Model (Structural Risk Prediction)...")
    X, y, feature_cols = load_engineered_data()

    model_path = os.path.join(MODELS_DIR, "structural_risk_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run Phase 6 first.")

    artifact = joblib.load(model_path)
    model = artifact["model"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro")
    roc_auc = roc_auc_score(y, y_proba, multi_class="ovr")

    # Confusion Matrix Plot
    cm = confusion_matrix(y, y_pred)
    classes = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"ML Risk Prediction Confusion Matrix ({artifact['model_name']})", fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    cm_path = os.path.join(FIG_DIR, "ml_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    ml_metrics = {
        "model_name": artifact["model_name"],
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "roc_auc_ovr": round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist()
    }

    print(f"  ✅ ML Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    return ml_metrics


def evaluate_dl_model():
    """Evaluates trained PyTorch Crack Detector model on validation set."""
    print("👁️ Evaluating DL PyTorch Model (Crack Detector & Bounding Box)...")

    ann_path = os.path.join(DATA_DIR, "images", "annotations.json")
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    val_anns = [a for a in annotations if a["split"] == "val"]
    val_dataset = BuildingCrackDataset(val_anns)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model_path = os.path.join(MODELS_DIR, "crack_detection_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DL Model weights not found at {model_path}. Run Phase 7 first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskCrackDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []
    ious = []

    with torch.no_grad():
        for images, labels, bboxes in val_loader:
            images = images.to(device)
            cls_preds, bbox_preds = model(images)

            preds_binary = (cls_preds > 0.5).float().cpu().numpy()
            true_labels = labels.numpy()

            all_preds.extend(preds_binary)
            all_labels.extend(true_labels)

            for b_pred, b_true, lbl in zip(bbox_preds, bboxes, labels):
                if lbl > 0.5:
                    iou = compute_iou(b_pred.cpu().numpy(), b_true.numpy())
                    ious.append(iou)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # mAP estimate based on IoU threshold >= 0.50
    map50 = float(np.mean([1.0 if iou >= 0.50 else 0.0 for iou in ious])) if ious else 0.0

    dl_metrics = {
        "model_architecture": "PyTorch Multi-Task ResNet/CNN",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "mean_iou": round(mean_iou, 4),
        "mAP_50": round(map50, 4)
    }

    print(f"  ✅ DL Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | Mean IoU: {mean_iou:.4f} | mAP@0.5: {map50:.4f}")
    return dl_metrics


def run_full_evaluation():
    """Runs complete Phase 8 evaluation suite and exports JSON metrics report."""
    ml_res = evaluate_ml_model()
    dl_res = evaluate_dl_model()

    report = {
        "phase": "Phase 8 - Model Evaluation",
        "machine_learning_metrics": ml_res,
        "deep_learning_metrics": dl_res
    }

    report_path = os.path.join(REPORTS_DIR, "model_evaluation_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Full Phase 8 Evaluation Report exported to: {report_path}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
