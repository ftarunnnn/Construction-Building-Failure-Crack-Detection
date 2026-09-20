# 🏗️ Construction — Building Failure Risk & Crack Detection System

An end-to-end Machine Learning & Deep Learning system to predict structural failure risk using IoT/sensor data and detect/localize concrete cracks from building inspection photos.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)

---

## 🌟 Key Features

- **🏢 Multi-class Structural Risk Prediction (ML)**: Predicts **Low**, **Medium**, or **High** structural failure risk based on vibration, stress, material grade, age, and maintenance history using XGBoost & Random Forest.
- **🔍 Computer Vision Crack Detection & Bounding Box Localization (DL)**: PyTorch multi-task CNN classifier and bounding box regressor detecting cracks on building surfaces with confidence scoring and normalized coordinates.
- **⚡ FastAPI Microservice**: High-performance RESTful API endpoints (`/api/predict_ml`, `/api/detect_crack`, `/api/full_assessment`).
- **📊 Interactive Streamlit Dashboard**: Feature-rich Web UI with real-time risk gauges, image uploaders, bounding box overlay visuals, EDA charts, and downloadable PDF/CSV inspection reports.

---

## 🏗️ 10-Phase Project Development Roadmap

1. **Phase 1: Problem Definition & Requirement Analysis** 📋
2. **Phase 2: Data Collection & Generation** 💾
3. **Phase 3: Data Preprocessing Pipeline** 🧹
4. **Phase 4: Exploratory Data Analysis (EDA)** 📈
5. **Phase 5: ML Feature Engineering** ⚙️
6. **Phase 6: ML Model Development (Random Forest / XGBoost)** 🤖
7. **Phase 7: DL Model Development (PyTorch Multi-Task CNN)** 👁️
8. **Phase 8: Model Evaluation & Performance Metrics** 🎯
9. **Phase 9: FastAPI Backend Integration** 🔌
10. **Phase 10: Streamlit Deployment & Web Dashboard** 🚀

---

## 💻 Quick Start

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/ftarunnnn/Construction-Building-Failure-Crack-Detection.git
cd Construction-Building-Failure-Crack-Detection
pip install -r requirements.txt
```

### 2. Run Data Generation & Model Training Pipeline
```bash
python src/data_generator.py
python src/preprocessing.py
python src/feature_engineering.py
python src/ml_model.py
python src/dl_model.py
python src/evaluate.py
```

### 3. Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

### 4. Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📄 License
MIT License