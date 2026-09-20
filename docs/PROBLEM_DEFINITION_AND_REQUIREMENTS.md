# Phase 1: Problem Definition & Requirement Analysis

## 1. Executive Summary
Structural failure in civil infrastructure (buildings, bridges, dams, retaining walls) poses severe safety risks and financial losses. Early detection of structural distress using combined Sensor Analytics (Machine Learning) and Visual Computer Vision (Deep Learning) enables predictive maintenance and catastrophe prevention.

This project delivers an end-to-end AI-powered **Construction Structural Health Monitoring System** integrating:
1. **Machine Learning (ML)**: Predicts **Structural Failure Risk** (`Low`, `Medium`, `High`) based on real-time sensor parameters, building age, material properties, and maintenance history.
2. **Deep Learning (DL)**: Detects, classifies (`Crack Detected` / `No Crack`), and localizes structural cracks using bounding box coordinates and confidence scores on building photos.
3. **Integrated REST API**: FastAPI backend providing automated structural health diagnostics.
4. **Interactive Dashboard**: Modern Streamlit Web Application displaying failure risk gauges, bounding-box overlays, EDA charts, and downloadable PDF/CSV inspection reports.

---

## 2. Machine Learning Objective: Structural Failure Risk Prediction
### Primary Goal
Predict the risk class of structural failure for a given building or asset segment:
- 🟢 **Low Risk**: Structure is safe, nominal operating conditions, standard maintenance interval.
- 🟡 **Medium Risk**: Early signs of degradation, elevated vibration/stress, increased inspection frequency required.
- 🔴 **High Risk**: Severe risk of structural compromise or collapse, immediate engineering intervention required.

### Required Tabular & Sensor Input Features
| Feature Name | Type | Unit | Range / Values | Description |
|---|---|---|---|---|
| `vibration_amplitude_mm` | Float | mm | 0.05 - 15.0 | Peak displacement under dynamic load |
| `vibration_frequency_hz` | Float | Hz | 1.0 - 120.0 | Dominant natural frequency of vibration |
| `stress_mPa` | Float | mPa | 5.0 - 90.0 | Applied mechanical stress |
| `strain_micro` | Float | µε | 100 - 3500 | Structural deformation strain |
| `concrete_age_years` | Float | Years | 1.0 - 75.0 | Age of the structural element |
| `ambient_temp_c` | Float | °C | -10.0 - 50.0 | Thermal exposure environment |
| `humidity_pct` | Float | % | 20.0 - 95.0 | Relative humidity |
| `maintenance_score` | Integer | Score (1-10)| 1 (Poor) - 10 (Excellent)| History of maintenance care |
| `material_grade` | Categorical| String | M20, M30, M40, M50 | Concrete strength class |
| `foundation_settlement_mm`| Float | mm | 0.0 - 25.0 | Differential ground movement |
| `load_ratio` | Float | Ratio | 0.2 - 1.3 | Applied load / Design capacity |

---

## 3. Deep Learning Objective: Crack Detection & Bounding Box Localization
### Primary Goal
Analyze visual images of building walls, columns, beams, or concrete surfaces to:
1. **Classify Presence**: Binary (`Crack Detected` vs `No Crack Detected`).
2. **Classify Severity**: Micro-crack, Structural Shear Crack, Settlement Fracture.
3. **Localize Crack**: Regression of bounding box coordinates `[xmin, ymin, xmax, ymax]` normalized between `[0, 1]`.
4. **Confidence Score**: Output probability score (`0.0` to `1.0`).

### Image Requirements
- Formats: JPEG, PNG, WEBP.
- Input Resolution: Standardized to `224x224` or `256x256` RGB.
- Annotations: Bounding box format `[xmin, ymin, xmax, ymax]` with corresponding label.

---

## 4. Evaluation Criteria & Metrics
- **ML Risk Classifier**:
  - Accuracy >= 90%
  - Macro F1-Score >= 0.88
  - Multi-class ROC-AUC >= 0.92
- **DL Crack Detector**:
  - Classification Accuracy >= 92%
  - Bounding Box Intersection over Union (IoU) >= 0.70
  - mean Average Precision (mAP@0.5) >= 0.85

---

## 5. System Deliverables Roadmap (10 Phases)
1. **Problem Definition & Requirement Analysis** (Current Phase)
2. **Data Collection & Synthetic Data Generation**
3. **Data Preprocessing Pipeline**
4. **Exploratory Data Analysis (EDA)**
5. **Feature Engineering**
6. **ML Model Training & Hyperparameter Tuning**
7. **DL PyTorch Crack Detector Training**
8. **Comprehensive Model Evaluation**
9. **FastAPI Backend Service**
10. **Streamlit Deployment Dashboard**
