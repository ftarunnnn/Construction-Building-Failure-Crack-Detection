"""
Phase 2: Data Collection & Synthetic Data Generation
Generates:
1. Tabular sensor & structural maintenance dataset (data/raw_structural_data.csv)
2. Synthetic high-contrast concrete wall building images (cracked vs uncracked)
3. Object detection bounding box annotations (data/images/annotations.json)
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "val")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)


def generate_tabular_data(n_samples: int = 2500) -> pd.DataFrame:
    """Generates synthetic structural health sensor readings and maintenance data."""
    vibration_amp = np.random.uniform(0.05, 15.0, n_samples)
    vibration_freq = np.random.uniform(1.0, 120.0, n_samples)
    stress_mPa = np.random.uniform(5.0, 90.0, n_samples)
    strain_micro = np.random.uniform(100.0, 3500.0, n_samples)
    concrete_age = np.random.uniform(1.0, 75.0, n_samples)
    ambient_temp = np.random.uniform(-10.0, 50.0, n_samples)
    humidity = np.random.uniform(20.0, 95.0, n_samples)
    maintenance_score = np.random.randint(1, 11, n_samples)
    material_grades = ["M20", "M30", "M40", "M50"]
    material_grade = np.random.choice(material_grades, n_samples, p=[0.3, 0.35, 0.25, 0.1])
    settlement_mm = np.random.uniform(0.0, 25.0, n_samples)
    load_ratio = np.random.uniform(0.2, 1.3, n_samples)

    # Material strength mapping (approx ultimate capacity in mPa)
    grade_capacity = {"M20": 20.0, "M30": 30.0, "M40": 40.0, "M50": 50.0}
    capacity = np.array([grade_capacity[g] for g in material_grade])

    # Calculate structural risk score based on physics-inspired formula
    risk_score = (
        (stress_mPa / capacity) * 35.0
        + (vibration_amp / 15.0) * 20.0
        + (settlement_mm / 25.0) * 20.0
        + (strain_micro / 3500.0) * 15.0
        + (concrete_age / 75.0) * 10.0
        + (load_ratio / 1.3) * 15.0
        - (maintenance_score / 10.0) * 15.0
        + np.random.normal(0, 3.0, n_samples)
    )

    # Risk classification thresholds
    q33, q66 = np.percentile(risk_score, [40, 75])
    
    risk_class = []
    for score in risk_score:
        if score < q33:
            risk_class.append("Low")
        elif score < q66:
            risk_class.append("Medium")
        else:
            risk_class.append("High")

    df = pd.DataFrame({
        "vibration_amplitude_mm": np.round(vibration_amp, 2),
        "vibration_frequency_hz": np.round(vibration_freq, 2),
        "stress_mPa": np.round(stress_mPa, 2),
        "strain_micro": np.round(strain_micro, 2),
        "concrete_age_years": np.round(concrete_age, 1),
        "ambient_temp_c": np.round(ambient_temp, 1),
        "humidity_pct": np.round(humidity, 1),
        "maintenance_score": maintenance_score,
        "material_grade": material_grade,
        "foundation_settlement_mm": np.round(settlement_mm, 2),
        "load_ratio": np.round(load_ratio, 2),
        "failure_risk": risk_class
    })

    filepath = os.path.join(DATA_DIR, "raw_structural_data.csv")
    df.to_csv(filepath, index=False)
    print(f"✅ Generated tabular sensor dataset: {filepath} ({len(df)} records)")
    return df


def draw_synthetic_crack(img: Image.Image, img_size: int = 224) -> tuple[Image.Image, list[int], str]:
    """Draws realistic random crack patterns (jagged lines) on top of concrete texture."""
    draw = ImageDraw.Draw(img)
    
    # Choose crack type
    crack_types = ["hairline", "diagonal_shear", "settlement_vertical"]
    crack_type = random.choice(crack_types)

    if crack_type == "hairline":
        n_points = random.randint(4, 7)
        start_x = random.randint(30, img_size - 60)
        start_y = random.randint(30, img_size - 60)
        points = [(start_x, start_y)]
        curr_x, curr_y = start_x, start_y
        for _ in range(n_points):
            curr_x += random.randint(-15, 25)
            curr_y += random.randint(10, 30)
            curr_x = max(10, min(img_size - 10, curr_x))
            curr_y = max(10, min(img_size - 10, curr_y))
            points.append((curr_x, curr_y))
        width = random.randint(2, 3)

    elif crack_type == "diagonal_shear":
        n_points = random.randint(5, 9)
        start_x = random.randint(20, 70)
        start_y = random.randint(20, 70)
        points = [(start_x, start_y)]
        curr_x, curr_y = start_x, start_y
        for _ in range(n_points):
            curr_x += random.randint(15, 30)
            curr_y += random.randint(15, 30)
            curr_x = max(10, min(img_size - 10, curr_x))
            curr_y = max(10, min(img_size - 10, curr_y))
            points.append((curr_x, curr_y))
        width = random.randint(3, 5)

    else:  # settlement_vertical
        n_points = random.randint(6, 10)
        start_x = random.randint(60, img_size - 60)
        start_y = random.randint(15, 40)
        points = [(start_x, start_y)]
        curr_x, curr_y = start_x, start_y
        for _ in range(n_points):
            curr_x += random.randint(-8, 8)
            curr_y += random.randint(15, 25)
            curr_x = max(10, min(img_size - 10, curr_x))
            curr_y = max(10, min(img_size - 10, curr_y))
            points.append((curr_x, curr_y))
        width = random.randint(3, 6)

    # Draw crack line with dark shadow and core
    crack_color = (random.randint(15, 45), random.randint(15, 45), random.randint(15, 45))
    draw.line(points, fill=crack_color, width=width, joint="curve")

    # Compute bounding box [xmin, ymin, xmax, ymax]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    padding = random.randint(4, 10)
    xmin = max(0, min(xs) - padding)
    ymin = max(0, min(ys) - padding)
    xmax = min(img_size, max(xs) + padding)
    ymax = min(img_size, max(ys) + padding)

    return img, [xmin, ymin, xmax, ymax], crack_type


def create_concrete_base_texture(img_size: int = 224) -> Image.Image:
    """Generates a realistic noisy grey concrete wall texture."""
    base_color = random.randint(140, 195)
    noise = np.random.normal(base_color, random.uniform(10, 25), (img_size, img_size, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noise, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def generate_image_dataset(n_train: int = 400, n_val: int = 100, img_size: int = 224) -> dict:
    """Generates synthetic building crack detection dataset with annotations."""
    annotations = []

    for split, n_imgs, folder in [("train", n_train, TRAIN_IMG_DIR), ("val", n_val, VAL_IMG_DIR)]:
        print(f"🖼️ Generating {split} dataset ({n_imgs} images)...")
        for i in range(n_imgs):
            filename = f"building_{split}_{i+1:04d}.jpg"
            filepath = os.path.join(folder, filename)

            img = create_concrete_base_texture(img_size=img_size)
            has_crack = (i % 2 == 0)  # 50% cracked, 50% clean

            if has_crack:
                img, bbox, crack_type = draw_synthetic_crack(img, img_size=img_size)
                label = "Crack Detected"
            else:
                bbox = [0, 0, 0, 0]
                crack_type = "none"
                label = "No Crack Detected"

            img.save(filepath, quality=92)

            annotations.append({
                "filename": filename,
                "split": split,
                "filepath": filepath,
                "has_crack": int(has_crack),
                "label": label,
                "crack_type": crack_type,
                "bbox": bbox,  # [xmin, ymin, xmax, ymax]
                "bbox_norm": [
                    round(bbox[0] / img_size, 4),
                    round(bbox[1] / img_size, 4),
                    round(bbox[2] / img_size, 4),
                    round(bbox[3] / img_size, 4)
                ],
                "width": img_size,
                "height": img_size
            })

    ann_filepath = os.path.join(IMAGE_DIR, "annotations.json")
    with open(ann_filepath, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Saved image dataset annotations to: {ann_filepath}")
    return {"total_images": n_train + n_val, "annotations": ann_filepath}


if __name__ == "__main__":
    generate_tabular_data(n_samples=2500)
    generate_image_dataset(n_train=400, n_val=100)
