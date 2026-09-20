"""
Phase 5: ML Feature Engineering
Creates domain-specific structural health features:
1. stress_capacity_ratio
2. vibration_severity_index
3. maintenance_decay_index
4. settlement_load_interaction
5. environmental_exposure_index
6. age_group binning
Selects top structural features and saves data/feature_engineered_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def create_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes civil engineering domain features from raw sensor data."""
    df_feat = df.copy()

    # Capacity mapping by grade (mPa)
    grade_capacity = {"M20": 20.0, "M30": 30.0, "M40": 40.0, "M50": 50.0}
    capacity = df_feat["material_grade"].map(grade_capacity).fillna(25.0)

    # 1. Stress Capacity Ratio
    df_feat["stress_capacity_ratio"] = np.round(df_feat["stress_mPa"] / capacity, 4)

    # 2. Vibration Severity Index
    df_feat["vibration_severity_index"] = np.round(
        df_feat["vibration_amplitude_mm"] * df_feat["vibration_frequency_hz"], 2
    )

    # 3. Maintenance Decay Index (Age weighted by inverse maintenance score)
    df_feat["maintenance_decay_index"] = np.round(
        df_feat["concrete_age_years"] / (df_feat["maintenance_score"] + 0.1), 3
    )

    # 4. Settlement Load Interaction
    df_feat["settlement_load_interaction"] = np.round(
        df_feat["foundation_settlement_mm"] * df_feat["load_ratio"], 3
    )

    # 5. Environmental Exposure Index
    df_feat["environmental_exposure_index"] = np.round(
        (df_feat["ambient_temp_c"] + 20.0) * (df_feat["humidity_pct"] / 100.0), 2
    )

    # 6. Age Group Binning (0: Young < 15 yrs, 1: Mature 15-40 yrs, 2: Legacy > 40 yrs)
    df_feat["age_group_encoded"] = pd.cut(
        df_feat["concrete_age_years"],
        bins=[-np.inf, 15, 40, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    return df_feat


def run_feature_engineering():
    """Loads preprocessed data, applies feature engineering, and saves output dataset."""
    processed_path = os.path.join(DATA_DIR, "processed_structural_data.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed dataset not found. Run Phase 3 first.")

    df = pd.read_csv(processed_path)
    df_engineered = create_structural_features(df)

    output_path = os.path.join(DATA_DIR, "feature_engineered_data.csv")
    df_engineered.to_csv(output_path, index=False)
    print(f"✅ Generated engineered structural features! Saved to: {output_path} ({len(df_engineered.columns)} columns)")

    return df_engineered


if __name__ == "__main__":
    run_feature_engineering()
