"""Report download endpoint."""
import io
import json
import requests as rq
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database import models
from backend.authentication.auth import get_current_user
from backend.services.report_service import generate_health_report

router = APIRouter()


@router.get("/report/download")
def download_report(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    preds = (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.user_id == current_user.id)
        .order_by(models.PredictionHistory.prediction_date.desc())
        .all()
    )
    profile = db.query(models.HealthProfile).filter(
        models.HealthProfile.user_id == current_user.id).first()

    pred_dicts = [{
        "disease_type":    p.disease_type,
        "risk_category":   p.risk_category,
        "model_score":     p.model_score,
        "best_model_name": p.best_model_name,
        "prediction_date": p.prediction_date.isoformat() if p.prediction_date else "",
    } for p in preds]

    profile_dict = {}
    if profile:
        profile_dict = {
            "age": profile.age, "gender": profile.gender,
            "bmi": profile.bmi, "smoking": profile.smoking,
            "physical_activity": profile.physical_activity,
        }

    try:
        pdf_bytes = generate_health_report(
            user_name=current_user.name,
            user_email=current_user.email,
            predictions=pred_dicts,
            profile=profile_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=HealthGuard_Report_{current_user.name.replace(' ','_')}.pdf"},
    )

