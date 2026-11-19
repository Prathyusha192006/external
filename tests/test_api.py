import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_endpoint():
    """Test /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data

def test_metrics_endpoint():
    """Test /metrics returns 200 OK and model evaluation metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "roc_auc" in data
    assert data["model_name"] == "TelcoChurnSVM"

def test_predict_valid_loyal_customer():
    """Test /predict with valid payload for a loyal customer profile."""
    valid_payload = {
        "customerID": "LOYAL-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 3312.0
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customerID"] == "LOYAL-001"
    assert data["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert isinstance(data["risk_factors"], list)
    assert "Avg_Monthly_Charge_Ratio" in data["engineered_features"]

def test_predict_valid_at_risk_customer():
    """Test /predict with valid payload for an at-risk churn profile."""
    valid_payload = {
        "customerID": "RISK-002",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.90,
        "TotalCharges": 179.80
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customerID"] == "RISK-002"
    assert data["churn_probability"] > 0.0
    assert "risk_level" in data

def test_predict_invalid_negative_tenure():
    """Test /predict returns 422 for negative tenure."""
    invalid_payload = {
        "customerID": "INVALID-01",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": -5,  # Invalid negative tenure
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("tenure" in err["loc"] for err in errors)

def test_predict_invalid_excessive_tenure():
    """Test /predict returns 422 for tenure exceeding allowable maximum (>120 months)."""
    invalid_payload = {
        "customerID": "INVALID-02",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 300,  # Invalid tenure > 120
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_invalid_negative_monthly_charges():
    """Test /predict returns 422 for negative MonthlyCharges."""
    invalid_payload = {
        "customerID": "INVALID-03",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": -25.5,  # Invalid negative charge
        "TotalCharges": 100.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("MonthlyCharges" in err["loc"] for err in errors)

def test_predict_invalid_categorical_contract():
    """Test /predict returns 422 for unsupported Contract category."""
    invalid_payload = {
        "customerID": "INVALID-04",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Lifetime Subscription",  # Invalid category
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Contract" in err["loc"] for err in errors)

def test_predict_missing_required_fields():
    """Test /predict returns 422 when required fields are missing."""
    incomplete_payload = {
        "customerID": "MISSING-01",
        "tenure": 12
        # Missing all other required fields
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

def test_predict_invalid_negative_total_charges():
    """Test /predict returns 422 for negative TotalCharges."""
    invalid_payload = {
        "customerID": "INVALID-05",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": -500.0  # Invalid negative TotalCharges
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
