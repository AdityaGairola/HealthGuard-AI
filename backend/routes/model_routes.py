"""Model performance endpoint ” returns actual training evaluation results."""
import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(BASE_DIR, "ml", "evaluation")


@router.get("/model/performance")
def model_performance():
    results = {}
    for disease in ["diabetes", "heart", "hypertension"]:
        path = os.path.join(EVAL_DIR, f"{disease}_eval.json")
        if os.path.exists(path):
            with open(path) as f:
                results[disease] = json.load(f)
        else:
            results[disease] = {"error": "Model not yet trained. Run train_models.py first."}
    return results

