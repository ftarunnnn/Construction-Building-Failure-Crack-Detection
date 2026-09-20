"""
Phase 4: EDA & Data Analysis
Performs Exploratory Data Analysis on tabular sensor metrics & crack image datasets:
1. Class distribution analysis (Low, Medium, High risk)
2. Correlation matrix plot & top risk-correlated features
3. Feature distribution across structural failure risk levels
4. Image dataset crack class balance & bounding box annotations sample preview
5. Saves visual figures to reports/figures/ and summary stats to reports/eda_summary.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure clean aesthetic plot style
plt.style.use("ggplot")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "figure.autolayout": True
})

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def run_tabular_eda():
    """Generates statistical summaries and visual figures for tabular sensor data."""
    processed_path = os.path.join(DATA_DIR, "processed_structural_data.csv")
    df = pd.read_csv(processed_path)

    # 1. Class Distribution
    risk_counts = df["failure_risk"].value_counts().to_dict()
    print(f"📊 Failure Risk Class Distribution: {risk_counts}")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    df["failure_risk"].value_counts().loc[["Low", "Medium", "High"]].plot(
        kind="bar", color=colors, ax=ax, edgecolor="black"
    )
    ax.set_title("Structural Failure Risk Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Failure Risk Category")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    risk_fig_path = os.path.join(FIG_DIR, "eda_risk_distribution.png")
    fig.savefig(risk_fig_path, dpi=300)
    plt.close(fig)

    # 2. Correlation Matrix Plot
    num_cols = [
        "vibration_amplitude_mm", "vibration_frequency_hz", "stress_mPa",
        "strain_micro", "concrete_age_years", "ambient_temp_c", "humidity_pct",
        "maintenance_score", "foundation_settlement_mm", "load_ratio", "failure_risk_encoded"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=True, square=True)
    ax.set_title("Structural Features Correlation Matrix", fontsize=14, fontweight="bold")
    corr_fig_path = os.path.join(FIG_DIR, "eda_correlation_matrix.png")
    fig.savefig(corr_fig_path, dpi=300)
    plt.close(fig)

    # 3. Key Feature Boxplots by Risk Category
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(data=df, x="failure_risk", y="stress_mPa", order=["Low", "Medium", "High"], palette=colors, ax=axes[0, 0])
    axes[0, 0].set_title("Stress (mPa) by Risk Level", fontweight="bold")

    sns.boxplot(data=df, x="failure_risk", y="vibration_amplitude_mm", order=["Low", "Medium", "High"], palette=colors, ax=axes[0, 1])
    axes[0, 1].set_title("Vibration Amplitude (mm) by Risk Level", fontweight="bold")

    sns.boxplot(data=df, x="failure_risk", y="foundation_settlement_mm", order=["Low", "Medium", "High"], palette=colors, ax=axes[1, 0])
    axes[1, 0].set_title("Foundation Settlement (mm) by Risk Level", fontweight="bold")

    sns.boxplot(data=df, x="failure_risk", y="maintenance_score", order=["Low", "Medium", "High"], palette=colors, ax=axes[1, 1])
    axes[1, 1].set_title("Maintenance Score by Risk Level", fontweight="bold")

    boxplots_fig_path = os.path.join(FIG_DIR, "eda_feature_boxplots.png")
    fig.savefig(boxplots_fig_path, dpi=300)
    plt.close(fig)

    return {
        "total_samples": len(df),
        "class_distribution": risk_counts,
        "figures": [risk_fig_path, corr_fig_path, boxplots_fig_path]
    }


def run_image_eda():
    """Analyzes image annotations and creates sample crack visualization figure."""
    ann_path = os.path.join(DATA_DIR, "images", "annotations.json")
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    df_ann = pd.DataFrame(annotations)
    label_counts = df_ann["label"].value_counts().to_dict()
    crack_type_counts = df_ann["crack_type"].value_counts().to_dict()

    print(f"🖼️ Image Crack Class Counts: {label_counts}")
    print(f"🔍 Crack Type Sub-counts: {crack_type_counts}")

    # Generate sample overlay grid figure
    cracked_samples = [a for a in annotations if a["has_crack"] == 1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ann in enumerate(cracked_samples):
        r, c = divmod(i, 2)
        img_path = ann["filepath"]
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        bbox = ann["bbox"] # [xmin, ymin, xmax, ymax]
        draw.rectangle(bbox, outline="red", width=3)
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"Sample {i+1}: {ann['crack_type']}", fontsize=11, fontweight="bold")
        axes[r, c].axis("off")

    samples_fig_path = os.path.join(FIG_DIR, "eda_crack_samples.png")
    fig.savefig(samples_fig_path, dpi=300)
    plt.close(fig)

    return {
        "total_images": len(annotations),
        "label_counts": label_counts,
        "crack_type_counts": crack_type_counts,
        "sample_fig": samples_fig_path
    }


def generate_eda_report():
    """Runs complete EDA and saves summary JSON report."""
    print("🚀 Running Phase 4 Exploratory Data Analysis...")
    tab_summary = run_tabular_eda()
    img_summary = run_image_eda()

    summary_report = {
        "phase": "Phase 4 - Exploratory Data Analysis",
        "tabular_eda": tab_summary,
        "image_eda": img_summary
    }

    report_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"✅ EDA Analysis complete! Report saved to: {report_path}")
    return summary_report


if __name__ == "__main__":
    generate_eda_report()
