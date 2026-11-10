"""SQLAlchemy ORM models."""
import datetime
import json
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.database.db import Base


class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(100), nullable=False)
    email          = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password= Column(String(256), nullable=False)
    created_at     = Column(DateTime, default=datetime.datetime.utcnow)
    is_active      = Column(Boolean, default=True)

    profile        = relationship("HealthProfile", back_populates="user", uselist=False)
    predictions    = relationship("PredictionHistory", back_populates="user")


class HealthProfile(Base):
    __tablename__ = "health_profiles"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), unique=True)
    age              = Column(Integer, nullable=True)
    gender           = Column(String(20), nullable=True)
    height_cm        = Column(Float, nullable=True)
    weight_kg        = Column(Float, nullable=True)
    bmi              = Column(Float, nullable=True)
    smoking          = Column(Boolean, default=False)
    physical_activity= Column(String(20), nullable=True)  # low/moderate/high
    family_history   = Column(Text, nullable=True)        # JSON string
    medical_history  = Column(Text, nullable=True)
    updated_at       = Column(DateTime, default=datetime.datetime.utcnow,
                              onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="profile")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    disease_type    = Column(String(50), nullable=False)   # diabetes/heart/hypertension
    input_parameters= Column(Text, nullable=False)         # JSON string
    predicted_class = Column(Integer, nullable=False)      # 0 or 1
    risk_category   = Column(String(20), nullable=True)    # Low/Moderate/High
    model_score     = Column(Float, nullable=True)
    best_model_name = Column(String(60), nullable=True)
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)
    notes           = Column(Text, nullable=True)

    user = relationship("User", back_populates="predictions")

