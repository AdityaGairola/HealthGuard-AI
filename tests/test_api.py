
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "HealthGuard AI API is running",
        "version": "1.0.0",
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_routes_registered():
    response = client.get("/openapi.json")
    schema = response.json()

    routes = schema["paths"]

    assert any("/predict" in route for route in routes), (
        "Prediction routes were not registered"
    )


def test_openapi_documentation():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "HealthGuard AI API"
