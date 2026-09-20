"""
Phase 9: Integration & Backend - Pydantic Schemas
Defines request and response schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class SensorInput(BaseModel):
    vibration_amplitude_mm: float = Field(..., example=2.45, description="Vibration amplitude in mm")
    vibration_frequency_hz: float = Field(..., example=45.0, description="Vibration frequency in Hz")
    stress_mPa: float = Field(..., example=32.5, description="Applied stress in mPa")
    strain_micro: float = Field(..., example=1200.0, description="Structural deformation strain in micro-strains")
    concrete_age_years: float = Field(..., example=15.0, description="Age of concrete structure in years")
    ambient_temp_c: float = Field(..., example=28.5, description="Ambient temperature in °C")
    humidity_pct: float = Field(..., example=65.0, description="Relative humidity percentage")
    maintenance_score: int = Field(..., ge=1, le=10, example=7, description="Maintenance history score (1-10)")
    material_grade: str = Field(..., example="M30", description="Concrete material grade (M20, M30, M40, M50)")
    foundation_settlement_mm: float = Field(..., example=3.2, description="Foundation settlement in mm")
    load_ratio: float = Field(..., example=0.75, description="Applied load / Design capacity ratio")


class StructuralRiskResponse(BaseModel):
    status: str
    structural_risk: str  # "Low", "Medium", "High"
    risk_score_pct: float
    confidence_score: float
    risk_probabilities: Dict[str, float]
    top_risk_factors: List[Dict[str, float]]


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    xmin_norm: float
    ymin_norm: float
    xmax_norm: float
    ymax_norm: float


class CrackDetectionResponse(BaseModel):
    status: str
    crack_detected: bool
    crack_label: str  # "Crack Detected" / "No Crack Detected"
    confidence_score: float
    bounding_box: BoundingBox
    annotated_image_base64: str


class FullAssessmentResponse(BaseModel):
    status: str
    structural_risk: str
    crack_label: str
    crack_detected: bool
    combined_health_index_pct: float
    risk_level_category: str
    actionable_recommendations: List[str]
    annotated_image_base64: str


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    ml_model_loaded: bool
    dl_model_loaded: bool
