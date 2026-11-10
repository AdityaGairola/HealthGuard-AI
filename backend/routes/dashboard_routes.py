"""Dashboard summary and history endpoints."""
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database import models
from backend.authentication.auth import get_current_user

router = APIRouter()


@router.get("/dashboard/summary")
def dashboard_summary(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    preds = (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.user_id == current_user.id)
        .order_by(models.PredictionHistory.prediction_date.desc())
        .all()
    )

    total = len(preds)

    # Latest per disease
    latest = {}
    for p in preds:
        if p.disease_type not in latest:
            latest[p.disease_type] = {
                "risk_category": p.risk_category,
                "model_score": p.model_score,
                "date": p.prediction_date.isoformat(),
            }

    # Risk overview
    risk_counts = defaultdict(lambda: {"Low": 0, "Moderate": 0, "High": 0})
    for p in preds:
        if p.risk_category:
            risk_counts[p.disease_type][p.risk_category] += 1

    # Prediction history (last 30, all diseases)
    history = [
        {
            "id": p.id,
            "disease_type": p.disease_type,
            "risk_category": p.risk_category,
            "model_score": p.model_score,
            "date": p.prediction_date.isoformat(),
        }
        for p in preds[:30]
    ]

    # Profile for BMI
    profile = db.query(models.HealthProfile).filter(
        models.HealthProfile.user_id == current_user.id).first()

    return {
        "total_assessments": total,
        "latest_predictions": latest,
        "risk_overview": dict(risk_counts),
        "bmi": profile.bmi if profile else None,
        "prediction_history": history,
        "user_name": current_user.name,
    }

