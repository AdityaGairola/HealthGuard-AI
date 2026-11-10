"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


# â”€â”€ Auth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class UserCreate(BaseModel):
    name:     str = Field(..., min_length=2, max_length=100)
    email:    EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email:    EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_name:    str
    user_id:      int


class UserOut(BaseModel):
    id:         int
    name:       str
    email:      str
    created_at: datetime

    class Config:
        from_attributes = True


# â”€â”€ Health Profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ProfileCreate(BaseModel):
    age:               Optional[int]   = Field(None, ge=1, le=120)
    gender:            Optional[str]   = None
    height_cm:         Optional[float] = Field(None, ge=50, le=300)
    weight_kg:         Optional[float] = Field(None, ge=10, le=500)
    smoking:           Optional[bool]  = False
    physical_activity: Optional[str]   = "moderate"
    family_history:    Optional[Dict]  = {}
    medical_history:   Optional[str]   = ""


class ProfileOut(ProfileCreate):
    bmi:        Optional[float] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# â”€â”€ Predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class DiabetesInput(BaseModel):
    pregnancies:                 int   = Field(..., ge=0, le=20)
    glucose:                     float = Field(..., ge=0, le=500)
    blood_pressure:              float = Field(..., ge=0, le=200)
    skin_thickness:              float = Field(..., ge=0, le=100)
    insulin:                     float = Field(..., ge=0, le=900)
    bmi:                         float = Field(..., ge=0, le=70)
    diabetes_pedigree_function:  float = Field(..., ge=0.0, le=3.0)
    age:                         int   = Field(..., ge=1, le=120)


class HeartInput(BaseModel):
    age:      int   = Field(..., ge=1, le=120)
    sex:      int   = Field(..., ge=0, le=1)   # 0=female, 1=male
    cp:       int   = Field(..., ge=0, le=3)   # chest pain type
    trestbps: float = Field(..., ge=0, le=250)
    chol:     float = Field(..., ge=0, le=600)
    fbs:      int   = Field(..., ge=0, le=1)
    restecg:  int   = Field(..., ge=0, le=2)
    thalach:  float = Field(..., ge=0, le=250)
    exang:    int   = Field(..., ge=0, le=1)
    oldpeak:  float = Field(..., ge=0.0, le=10.0)
    slope:    int   = Field(..., ge=0, le=2)
    ca:       int   = Field(..., ge=0, le=4)
    thal:     int   = Field(..., ge=0, le=7)


class HypertensionInput(BaseModel):
    age:               int   = Field(..., ge=1, le=120)
    bmi:               float = Field(..., ge=10, le=70)
    smoking:           int   = Field(..., ge=0, le=1)
    diabetes_history:  int   = Field(..., ge=0, le=1)
    family_history:    int   = Field(..., ge=0, le=1)
    physical_activity: int   = Field(..., ge=0, le=2)  # 0=low,1=mod,2=high
    salt_intake:       int   = Field(..., ge=0, le=2)
    alcohol_intake:    int   = Field(..., ge=0, le=2)
    stress_level:      int   = Field(..., ge=0, le=2)


class PredictionOut(BaseModel):
    disease_type:    str
    predicted_class: int
    risk_category:   str
    model_score:     Optional[float]
    best_model_name: str
    feature_importance: Optional[Dict[str, float]]
    recommendations: List[str]
    prediction_date: datetime
    prediction_id:   int

    class Config:
        from_attributes = True


# â”€â”€ Dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PredictionHistoryOut(BaseModel):
    id:              int
    disease_type:    str
    risk_category:   Optional[str]
    model_score:     Optional[float]
    prediction_date: datetime
    input_parameters: Optional[str]

    class Config:
        from_attributes = True


class DashboardSummary(BaseModel):
    total_assessments:    int
    latest_predictions:   Dict[str, Any]
    risk_overview:        Dict[str, Any]
    bmi_history:          List[Any]
    prediction_history:   List[Any]

