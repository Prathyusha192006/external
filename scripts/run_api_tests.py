import os
import sys
import json
from pydantic import ValidationError

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import CustomerInput
from api.app import predict_churn, get_model_metrics, health_check

def test_api_pipeline():
    print("=" * 75)
    print("AUTOMATED FASTAPI ENDPOINT & PYDANTIC VALIDATION TEST SUITE")
    print("Testing /predict, /metrics, /health with VALID and INVALID Payloads")
    print("=" * 75)

    passed = 0
    failed = 0

    # -------------------------------------------------------------
    # 1. GET /health
    # -------------------------------------------------------------
    print("\n[TEST 1] GET /health Endpoint Check...")
    try:
        health = health_check()
        assert health["status"] == "healthy"
        assert health["artifacts_loaded"] is True
        print(f"  --> PASS [200 OK]: Service Status: {health['status']} | Model Type: {health['model_type']}")
        passed += 1
    except Exception as e:
        print(f"  --> FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------
    # 2. GET /metrics
    # -------------------------------------------------------------
    print("\n[TEST 2] GET /metrics Endpoint Check...")
    try:
        m = get_model_metrics()
        assert m.accuracy > 0.70
        assert m.roc_auc > 0.80
        assert m.model_stage == "Production"
        print(f"  --> PASS [200 OK]: Accuracy={m.accuracy:.4f} | ROC-AUC={m.roc_auc:.4f} | F1={m.f1_score:.4f} | Stage={m.model_stage}")
        passed += 1
    except Exception as e:
        print(f"  --> FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------
    # 3. POST /predict (VALID: Loyal Customer)
    # -------------------------------------------------------------
    print("\n[TEST 3] POST /predict with VALID Payload (Loyal Customer)...")
    valid_loyal = {
        "customerID": "LOYAL-7712",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
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
        "MonthlyCharges": 58.40,
        "TotalCharges": 2803.20
    }
    try:
        inp = CustomerInput(**valid_loyal)
        res = predict_churn(inp)
        assert res.churn_prediction == "No"
        assert res.risk_level == "Low Risk"
        print(f"  --> PASS [200 OK]: Prediction={res.churn_prediction} | Churn Prob={res.churn_probability:.2%} | Risk={res.risk_level}")
        passed += 1
    except Exception as e:
        print(f"  --> FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------
    # 4. POST /predict (VALID: At-Risk Customer)
    # -------------------------------------------------------------
    print("\n[TEST 4] POST /predict with VALID Payload (At-Risk Customer)...")
    valid_risk = {
        "customerID": "RISK-9914",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
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
        "MonthlyCharges": 85.50,
        "TotalCharges": 85.50
    }
    try:
        inp = CustomerInput(**valid_risk)
        res = predict_churn(inp)
        assert res.churn_prediction == "Yes"
        assert res.risk_level == "High Risk"
        print(f"  --> PASS [200 OK]: Prediction={res.churn_prediction} | Churn Prob={res.churn_probability:.2%} | Risk={res.risk_level}")
        print(f"      Risk Drivers: {res.risk_factors}")
        passed += 1
    except Exception as e:
        print(f"  --> FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------
    # 5. POST /predict (INVALID: Negative Tenure -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 5] POST /predict with INVALID Payload (Negative Tenure: -5)...")
    bad_tenure = valid_loyal.copy()
    bad_tenure["tenure"] = -5
    try:
        CustomerInput(**bad_tenure)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except ValidationError as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught error: {e.errors()[0]['msg']} (Field: {e.errors()[0]['loc']})")
        passed += 1

    # -------------------------------------------------------------
    # 6. POST /predict (INVALID: Excessive Tenure > 120 -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 6] POST /predict with INVALID Payload (Excessive Tenure: 250)...")
    bad_excessive_tenure = valid_loyal.copy()
    bad_excessive_tenure["tenure"] = 250
    try:
        CustomerInput(**bad_excessive_tenure)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except ValidationError as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught error: {e.errors()[0]['msg']} (Field: {e.errors()[0]['loc']})")
        passed += 1

    # -------------------------------------------------------------
    # 7. POST /predict (INVALID: Negative Monthly Charges -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 7] POST /predict with INVALID Payload (Negative MonthlyCharges: -45.0)...")
    bad_monthly = valid_loyal.copy()
    bad_monthly["MonthlyCharges"] = -45.0
    try:
        CustomerInput(**bad_monthly)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except ValidationError as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught error: {e.errors()[0]['msg']} (Field: {e.errors()[0]['loc']})")
        passed += 1

    # -------------------------------------------------------------
    # 8. POST /predict (INVALID: Unsupported Contract Category -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 8] POST /predict with INVALID Payload (Contract: 'Lifetime')...")
    bad_cat = valid_loyal.copy()
    bad_cat["Contract"] = "Lifetime Free Trial"
    try:
        CustomerInput(**bad_cat)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except ValidationError as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught error: {e.errors()[0]['msg']} (Field: {e.errors()[0]['loc']})")
        passed += 1

    # -------------------------------------------------------------
    # 9. POST /predict (INVALID: Negative TotalCharges -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 9] POST /predict with INVALID Payload (Negative TotalCharges: -200.0)...")
    bad_total = valid_loyal.copy()
    bad_total["TotalCharges"] = -200.0
    try:
        CustomerInput(**bad_total)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except (ValidationError, ValueError) as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught error: TotalCharges cannot be negative.")
        passed += 1

    # -------------------------------------------------------------
    # 10. POST /predict (INVALID: Missing Required Fields -> 422)
    # -------------------------------------------------------------
    print("\n[TEST 10] POST /predict with INVALID Payload (Missing Required Fields)...")
    missing_fields = {"customerID": "INCOMPLETE-01"}
    try:
        CustomerInput(**missing_fields)
        print("  --> FAIL: Expected 422 ValidationError, but validation passed.")
        failed += 1
    except ValidationError as e:
        print(f"  --> PASS [422 Unprocessable Entity]: Caught missing required fields (Total errors caught: {len(e.errors())})")
        passed += 1

    print("\n" + "=" * 75)
    print(f"SUMMARY: ALL {passed} TESTS PASSED | {failed} FAILED")
    print("=" * 75)
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_api_pipeline()
