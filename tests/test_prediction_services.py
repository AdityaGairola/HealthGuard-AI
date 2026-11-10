
import pytest

from backend.services.prediction_service import (
    predict_diabetes,
    predict_heart,
    predict_hypertension,
)


TEST_CASES = [
    (
        "Diabetes",
        predict_diabetes,
        {
            "pregnancies": 2,
            "glucose": 120,
            "blood_pressure": 70,
            "skin_thickness": 25,
            "insulin": 80,
            "bmi": 28.0,
            "diabetes_pedigree_function": 0.5,
            "age": 35,
        },
    ),
    (
        "Heart Disease",
        predict_heart,
        {
            "age": 55,
            "sex": 1,
            "cp": 2,
            "trestbps": 140,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        },
    ),
    (
        "Hypertension",
        predict_hypertension,
        {
            "age": 45,
            "bmi": 27.0,
            "smoking": 0,
            "diabetes_history": 0,
            "family_history": 1,
            "physical_activity": 1,
            "salt_intake": 0,
            "alcohol_intake": 0,
            "stress_level": 2,
        },
    ),
]


@pytest.mark.parametrize(
    "disease, prediction_function, input_data",
    TEST_CASES,
)
def test_prediction_service(
    disease, prediction_function, input_data
):
    result = prediction_function(input_data)

    assert isinstance(result, tuple), disease
    assert len(result) == 5, disease

    prediction, probability, risk, importance, model_name = result

    assert prediction in (0, 1), disease
    assert 0.0 <= probability <= 1.0, disease
    assert risk in ("Low", "Moderate", "High"), disease
    assert isinstance(importance, dict), disease
    assert len(importance) > 0, disease
    assert isinstance(model_name, str) and model_name.strip(), disease
