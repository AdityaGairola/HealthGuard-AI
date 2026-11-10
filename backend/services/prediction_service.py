"""ML model loading and prediction service."""
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
EVAL_DIR   = os.path.join(BASE_DIR, "ml", "evaluation")

_MODELS: Dict[str, Any] = {}
_EVAL:   Dict[str, Any] = {}


def _load_model(disease: str):
    if disease not in _MODELS:
        path = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}. Run: cd ml && python training/train_models.py")
        _MODELS[disease] = joblib.load(path)
    return _MODELS[disease]


def _load_eval(disease: str) -> dict:
    if disease not in _EVAL:
        path = os.path.join(EVAL_DIR, f"{disease}_eval.json")
        if os.path.exists(path):
            with open(path) as f:
                _EVAL[disease] = json.load(f)
        else:
            _EVAL[disease] = {}
    return _EVAL[disease]


def _risk_category(score: float) -> str:
    if score < 0.35:
        return "Low"
    elif score < 0.65:
        return "Moderate"
    else:
        return "High"


def _get_feature_importance(disease: str, feature_names: List[str],
                             input_values: List[float]) -> Dict[str, float]:
    """Return per-feature importance from the saved eval JSON."""
    eval_data = _load_eval(disease)
    fi = eval_data.get("feature_importance", {})
    if fi:
        return {k: round(v, 4) for k, v in sorted(fi.items(), key=lambda x: -x[1])}
    return {}


# â”€â”€ Diabetes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DIABETES_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']

def predict_diabetes(data: dict) -> Tuple[int, float, str, Dict, str]:
    model = _load_model("diabetes")
    vals  = [data["pregnancies"], data["glucose"], data["blood_pressure"],
             data["skin_thickness"], data["insulin"], data["bmi"],
             data["diabetes_pedigree_function"], data["age"]]
    X = pd.DataFrame([vals], columns=DIABETES_COLS)
    pred  = int(model.predict(X)[0])
    score = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    fi    = _get_feature_importance("diabetes", DIABETES_COLS, vals)
    eval_d = _load_eval("diabetes")
    best   = eval_d.get("best_model", "ML Model")
    return pred, score, _risk_category(score), fi, best


# â”€â”€ Heart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HEART_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
              'thalach','exang','oldpeak','slope','ca','thal']

def predict_heart(data: dict) -> Tuple[int, float, str, Dict, str]:
    model = _load_model("heart")
    vals  = [data["age"], data["sex"], data["cp"], data["trestbps"],
             data["chol"], data["fbs"], data["restecg"], data["thalach"],
             data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]]
    X = pd.DataFrame([vals], columns=HEART_COLS)
    pred  = int(model.predict(X)[0])
    score = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    fi    = _get_feature_importance("heart", HEART_COLS, vals)
    eval_d = _load_eval("heart")
    best   = eval_d.get("best_model", "ML Model")
    return pred, score, _risk_category(score), fi, best


# â”€â”€ Hypertension â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HYPER_COLS = ['age','bmi','smoking','diabetes_history','family_history',
              'physical_activity','salt_intake','alcohol_intake','stress_level']

def predict_hypertension(data: dict) -> Tuple[int, float, str, Dict, str]:
    model = _load_model("hypertension")
    vals  = [data["age"], data["bmi"], data["smoking"], data["diabetes_history"],
             data["family_history"], data["physical_activity"], data["salt_intake"],
             data["alcohol_intake"], data["stress_level"]]
    X = pd.DataFrame([vals], columns=HYPER_COLS)
    pred  = int(model.predict(X)[0])
    score = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    fi    = _get_feature_importance("hypertension", HYPER_COLS, vals)
    eval_d = _load_eval("hypertension")
    best   = eval_d.get("best_model", "ML Model")
    return pred, score, _risk_category(score), fi, best


# â”€â”€ Recommendations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RECOMMENDATIONS = {
    "diabetes": {
        "Low":      ["Maintain a balanced diet rich in whole grains and vegetables.",
                     "Stay physically active with at least 150 minutes/week of moderate exercise.",
                     "Monitor your weight and maintain a healthy BMI.",
                     "Get regular blood glucose screenings annually."],
        "Moderate": ["Reduce refined sugar and processed food intake significantly.",
                     "Increase physical activity to 30 minutes daily.",
                     "Monitor fasting blood glucose regularly.",
                     "Consult a healthcare professional about your glucose levels.",
                     "Consider a dietary consultation with a registered dietitian."],
        "High":     ["Seek consultation with a healthcare professional as soon as possible.",
                     "Strictly limit sugary beverages and high-glycemic foods.",
                     "Begin a structured exercise program under medical guidance.",
                     "Monitor blood glucose levels frequently as advised by a doctor.",
                     "Discuss family history and risk factors with your physician."],
    },
    "heart": {
        "Low":      ["Maintain a heart-healthy diet (Mediterranean or DASH style).",
                     "Exercise regularly ” at least 150 minutes of aerobic activity per week.",
                     "Avoid smoking and limit alcohol consumption.",
                     "Get annual cardiovascular checkups."],
        "Moderate": ["Reduce dietary sodium and saturated fats.",
                     "Have your cholesterol and blood pressure checked regularly.",
                     "If you smoke, seek cessation support immediately.",
                     "Manage stress through relaxation techniques.",
                     "Discuss your cardiovascular risk with a healthcare provider."],
        "High":     ["Consult a cardiologist or healthcare professional promptly.",
                     "Follow a strict heart-healthy diet as prescribed.",
                     "Monitor blood pressure and cholesterol closely.",
                     "Avoid strenuous unmonitored exercise until medically cleared.",
                     "Discuss medication or further testing with your doctor."],
    },
    "hypertension": {
        "Low":      ["Maintain a low-sodium diet (less than 2300 mg/day).",
                     "Stay physically active ” regular aerobic exercise lowers blood pressure.",
                     "Maintain a healthy weight.",
                     "Limit alcohol consumption."],
        "Moderate": ["Monitor your blood pressure at home regularly.",
                     "Significantly reduce salt intake in your diet.",
                     "If you smoke, seek cessation support immediately.",
                     "Manage stress through mindfulness or yoga.",
                     "Schedule a check-up with your healthcare provider."],
        "High":     ["Consult a healthcare professional about your blood pressure soon.",
                     "Follow the DASH diet (Dietary Approaches to Stop Hypertension).",
                     "Monitor blood pressure daily and keep a log.",
                     "Avoid high-sodium foods, alcohol, and caffeine excess.",
                     "Ask your doctor about appropriate monitoring or treatment options."],
    },
}

DISCLAIMER = ("WARNING: This tool is for educational and informational purposes only. "
               "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
               "Always consult a qualified healthcare professional.")


def get_recommendations(disease: str, risk_category: str) -> List[str]:
    recs = RECOMMENDATIONS.get(disease, {}).get(risk_category, [])
    return recs + [DISCLAIMER]

