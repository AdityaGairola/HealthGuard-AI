"""Prediction endpoints for all three diseases."""
import json
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database import models
from backend.authentication.auth import get_current_user
from backend.schemas.schemas import (
    DiabetesInput, HeartInput, HypertensionInput, PredictionOut
)
from backend.services.prediction_service import (
    predict_diabetes, predict_heart, predict_hypertension, get_recommendations
)

router = APIRouter()


def _save_prediction(db, user_id, disease, input_data, pred_class, risk_cat, score, model_name):
    record = models.PredictionHistory(
        user_id=user_id,
        disease_type=disease,
        input_parameters=json.dumps(input_data),
        predicted_class=pred_class,
        risk_category=risk_cat,
        model_score=score,
        best_model_name=model_name,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/predict/diabetes", response_model=PredictionOut)
def predict_diabetes_endpoint(
    data: DiabetesInput,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        pred, score, risk_cat, fi, model_name = predict_diabetes(data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    recs   = get_recommendations("diabetes", risk_cat)
    record = _save_prediction(db, current_user.id, "diabetes",
                               data.model_dump(), pred, risk_cat, score, model_name)
    return PredictionOut(
        disease_type="diabetes",
        predicted_class=pred,
        risk_category=risk_cat,
        model_score=round(score, 4),
        best_model_name=model_name,
        feature_importance=fi,
        recommendations=recs,
        prediction_date=record.prediction_date,
        prediction_id=record.id,
    )


@router.post("/predict/heart", response_model=PredictionOut)
def predict_heart_endpoint(
    data: HeartInput,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        pred, score, risk_cat, fi, model_name = predict_heart(data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    recs   = get_recommendations("heart", risk_cat)
    record = _save_prediction(db, current_user.id, "heart",
                               data.model_dump(), pred, risk_cat, score, model_name)
    return PredictionOut(
        disease_type="heart",
        predicted_class=pred,
        risk_category=risk_cat,
        model_score=round(score, 4),
        best_model_name=model_name,
        feature_importance=fi,
        recommendations=recs,
        prediction_date=record.prediction_date,
        prediction_id=record.id,
    )


@router.post("/predict/hypertension", response_model=PredictionOut)
def predict_hypertension_endpoint(
    data: HypertensionInput,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        pred, score, risk_cat, fi, model_name = predict_hypertension(data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    recs   = get_recommendations("hypertension", risk_cat)
    record = _save_prediction(db, current_user.id, "hypertension",
                               data.model_dump(), pred, risk_cat, score, model_name)
    return PredictionOut(
        disease_type="hypertension",
        predicted_class=pred,
        risk_category=risk_cat,
        model_score=round(score, 4),
        best_model_name=model_name,
        feature_importance=fi,
        recommendations=recs,
        prediction_date=record.prediction_date,
        prediction_id=record.id,
    )


@router.get("/predictions/history")
def get_history(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    records = (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.user_id == current_user.id)
        .order_by(models.PredictionHistory.prediction_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "disease_type": r.disease_type,
            "predicted_class": r.predicted_class,
            "risk_category": r.risk_category,
            "model_score": r.model_score,
            "best_model_name": r.best_model_name,
            "prediction_date": r.prediction_date.isoformat(),
            "input_parameters": r.input_parameters,
        }
        for r in records
    ]

