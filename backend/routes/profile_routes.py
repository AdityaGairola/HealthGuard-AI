"""Health profile routes."""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database import models
from backend.authentication.auth import get_current_user
from backend.schemas.schemas import ProfileCreate, ProfileOut

router = APIRouter()


def _calc_bmi(height_cm, weight_kg):
    if height_cm and weight_kg and height_cm > 0:
        return round(weight_kg / ((height_cm / 100) ** 2), 2)
    return None


@router.get("/profile", response_model=ProfileOut)
def get_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.query(models.HealthProfile).filter(
        models.HealthProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found. Please create one.")
    return profile


@router.put("/profile", response_model=ProfileOut)
def upsert_profile(
    profile_in: ProfileCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    bmi = _calc_bmi(profile_in.height_cm, profile_in.weight_kg)
    profile = db.query(models.HealthProfile).filter(
        models.HealthProfile.user_id == current_user.id).first()

    family_history_json = json.dumps(profile_in.family_history or {})

    if profile:
        profile.age               = profile_in.age
        profile.gender            = profile_in.gender
        profile.height_cm         = profile_in.height_cm
        profile.weight_kg         = profile_in.weight_kg
        profile.bmi               = bmi
        profile.smoking           = profile_in.smoking
        profile.physical_activity = profile_in.physical_activity
        profile.family_history    = family_history_json
        profile.medical_history   = profile_in.medical_history
    else:
        profile = models.HealthProfile(
            user_id=current_user.id,
            age=profile_in.age,
            gender=profile_in.gender,
            height_cm=profile_in.height_cm,
            weight_kg=profile_in.weight_kg,
            bmi=bmi,
            smoking=profile_in.smoking,
            physical_activity=profile_in.physical_activity,
            family_history=family_history_json,
            medical_history=profile_in.medical_history,
        )
        db.add(profile)

    db.commit()
    db.refresh(profile)
    return {
    "age": profile.age,
    "gender": profile.gender,
    "height_cm": profile.height_cm,
    "weight_kg": profile.weight_kg,
    "smoking": profile.smoking,
    "physical_activity": profile.physical_activity,
    "family_history": json.loads(profile.family_history) if isinstance(profile.family_history, str) else (profile.family_history or {}),
    "medical_history": profile.medical_history,
    "bmi": profile.bmi,
    "updated_at": profile.updated_at
}

