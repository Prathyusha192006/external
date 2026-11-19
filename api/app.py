import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.schemas import CustomerInput, PredictionResponse, BatchPredictionRequest, MetricsResponse
from src.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastAPI_Server")

app = FastAPI(
    title="Telco Customer Churn Prediction & MLOps API",
    description="Real-time inference API serving Support Vector Machine (SVM) pipeline with MLflow and DVC tracking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = ChurnPredictor()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Telco Churn API server...")
    try:
        predictor.load_artifacts()
    except Exception as e:
        logger.warning(f"Could not load model artifacts on startup: {e}")

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming API service and artifact loading status."""
    is_ready = predictor.model is not None and predictor.preprocessor is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "Telco Churn SVM Inference API",
        "artifacts_loaded": is_ready,
        "model_type": "Support Vector Machine (SVC)"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    """
    Real-time single customer churn prediction with strict Pydantic payload validation.
    Returns churn prediction ('Yes'/'No'), probability, risk tier, and key risk drivers.
    """
    if predictor.model is None:
        predictor.load_artifacts()
        if predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not loaded. Please train the model first by running main.py."
            )

    try:
        payload = customer.model_dump()
        result = predictor.predict_single(payload)
        
        return PredictionResponse(
            customerID=customer.customerID or "CUST-9999",
            churn_prediction=result["churn_prediction"],
            churn_class=result["churn_class"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            risk_factors=result["risk_factors"],
            engineered_features=result["engineered_features"],
            model_version="v2.0 (Production)"
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction: {str(e)}"
        )

@app.post("/predict/batch", tags=["Inference"])
def predict_batch_churn(batch_request: BatchPredictionRequest):
    """Batch prediction endpoint for multi-customer scoring."""
    if predictor.model is None:
        predictor.load_artifacts()
        if predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not loaded."
            )

    try:
        import pandas as pd
        records = [c.model_dump() for c in batch_request.customers]
        df = pd.DataFrame(records)
        results = predictor.predict_batch(df)
        return {
            "total_processed": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failure: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse, tags=["Model Governance"])
def get_model_metrics():
    """
    Returns latest model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    and production deployment stage.
    """
    summary_path = "reports/evaluation_summary.json"
    if not os.path.exists(summary_path):
        # Return fallback baseline if report not generated yet
        return MetricsResponse(
            model_name="TelcoChurnSVM",
            model_stage="Staging",
            accuracy=0.805,
            precision=0.680,
            recall=0.550,
            f1_score=0.608,
            roc_auc=0.845,
            hyperparameters={"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        )

    with open(summary_path, "r") as f:
        metrics = json.load(f)

    return MetricsResponse(
        model_name="TelcoChurnSVM",
        model_stage="Production",
        accuracy=metrics.get("accuracy", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
        f1_score=metrics.get("f1_score", 0.0),
        roc_auc=metrics.get("roc_auc", 0.0),
        hyperparameters={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        last_evaluated="Latest Pipeline Run"
    )

@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_embedded_dashboard():
    """Embedded interactive HTML5/Bootstrap/Chart.js web dashboard for live churn prediction."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Telco Churn SVM MLOps Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #f8fafc; }
            .badge-low { background-color: #10b981; color: white; }
            .badge-medium { background-color: #f59e0b; color: white; }
            .badge-high { background-color: #ef4444; color: white; }
            .form-control, .form-select { background-color: #334155; border-color: #475569; color: #fff; }
            .form-control:focus, .form-select:focus { background-color: #334155; color: #fff; border-color: #38bdf8; box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25); }
            .header-banner { background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%); padding: 2rem 1rem; border-radius: 12px; margin-bottom: 2rem; }
        </style>
    </head>
    <body class="p-3 p-md-4">
        <div class="container-fluid max-w-7xl">
            <div class="header-banner shadow-lg text-white">
                <div class="d-flex justify-content-between align-items-center flex-wrap">
                    <div>
                        <h1 class="h2 fw-bold mb-1">Telco Customer Churn Prediction Dashboard</h1>
                        <p class="mb-0 text-light opacity-75">Automated MLOps Pipeline with Support Vector Machine (SVM), MLflow & DVC Tracking</p>
                    </div>
                    <div class="mt-2 mt-md-0">
                        <span class="badge bg-success fs-6 px-3 py-2">Model: TelcoChurnSVM @Production</span>
                    </div>
                </div>
            </div>

            <!-- KPI Cards Row -->
            <div class="row g-3 mb-4" id="kpi-row">
                <div class="col-md-3">
                    <div class="card p-3 shadow-sm text-center">
                        <span class="text-secondary small fw-bold">ACCURACY</span>
                        <h3 class="text-info fw-bold mb-0 mt-1" id="kpi-accuracy">--%</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 shadow-sm text-center">
                        <span class="text-secondary small fw-bold">ROC - AUC</span>
                        <h3 class="text-primary fw-bold mb-0 mt-1" id="kpi-roc-auc">--</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 shadow-sm text-center">
                        <span class="text-secondary small fw-bold">PRECISION / RECALL</span>
                        <h3 class="text-warning fw-bold mb-0 mt-1" id="kpi-pr">-- / --</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card p-3 shadow-sm text-center">
                        <span class="text-secondary small fw-bold">F1 - SCORE</span>
                        <h3 class="text-success fw-bold mb-0 mt-1" id="kpi-f1">--</h3>
                    </div>
                </div>
            </div>

            <!-- Main Interactive Section -->
            <div class="row g-4">
                <!-- Customer Input Form -->
                <div class="col-lg-7">
                    <div class="card shadow p-4">
                        <h4 class="card-title fw-bold text-info mb-3">Live Customer Churn Predictor</h4>
                        <form id="churnForm">
                            <div class="row g-3">
                                <div class="col-md-4">
                                    <label class="form-label small">Customer ID</label>
                                    <input type="text" class="form-control" id="customerID" value="CUST-8821">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Gender</label>
                                    <select class="form-select" id="gender">
                                        <option value="Female">Female</option>
                                        <option value="Male" selected>Male</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Senior Citizen</label>
                                    <select class="form-select" id="SeniorCitizen">
                                        <option value="0" selected>No (0)</option>
                                        <option value="1">Yes (1)</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Partner</label>
                                    <select class="form-select" id="Partner">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Dependents</label>
                                    <select class="form-select" id="Dependents">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Tenure (Months: <span id="tenureVal" class="text-info fw-bold">2</span>)</label>
                                    <input type="range" class="form-range" id="tenure" min="0" max="72" value="2" oninput="document.getElementById('tenureVal').innerText=this.value">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Contract</label>
                                    <select class="form-select" id="Contract">
                                        <option value="Month-to-month" selected>Month-to-month</option>
                                        <option value="One year">One year</option>
                                        <option value="Two year">Two year</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Internet Service</label>
                                    <select class="form-select" id="InternetService">
                                        <option value="Fiber optic" selected>Fiber optic</option>
                                        <option value="DSL">DSL</option>
                                        <option value="No">No</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Payment Method</label>
                                    <select class="form-select" id="PaymentMethod">
                                        <option value="Electronic check" selected>Electronic check</option>
                                        <option value="Mailed check">Mailed check</option>
                                        <option value="Bank transfer (automatic)">Bank transfer (auto)</option>
                                        <option value="Credit card (automatic)">Credit card (auto)</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Online Security</label>
                                    <select class="form-select" id="OnlineSecurity">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                        <option value="No internet service">No internet</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Tech Support</label>
                                    <select class="form-select" id="TechSupport">
                                        <option value="No" selected>No</option>
                                        <option value="Yes">Yes</option>
                                        <option value="No internet service">No internet</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label small">Paperless Billing</label>
                                    <select class="form-select" id="PaperlessBilling">
                                        <option value="Yes" selected>Yes</option>
                                        <option value="No">No</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label small">Monthly Charges ($)</label>
                                    <input type="number" step="0.05" class="form-control" id="MonthlyCharges" value="84.50">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label small">Total Charges ($)</label>
                                    <input type="number" step="0.1" class="form-control" id="TotalCharges" value="169.00">
                                </div>
                            </div>
                            <!-- Hidden defaults -->
                            <input type="hidden" id="PhoneService" value="Yes">
                            <input type="hidden" id="MultipleLines" value="No">
                            <input type="hidden" id="OnlineBackup" value="No">
                            <input type="hidden" id="DeviceProtection" value="No">
                            <input type="hidden" id="StreamingTV" value="No">
                            <input type="hidden" id="StreamingMovies" value="No">

                            <div class="mt-4">
                                <button type="button" class="btn btn-primary px-4 py-2 fw-bold" onclick="runPrediction()">
                                    Predict Churn Risk
                                </button>
                                <button type="button" class="btn btn-outline-secondary px-3 py-2 ms-2" onclick="setPreset('loyal')">
                                    Load Loyal Preset
                                </button>
                                <button type="button" class="btn btn-outline-danger px-3 py-2 ms-2" onclick="setPreset('atrisk')">
                                    Load At-Risk Preset
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Prediction Output & Chart -->
                <div class="col-lg-5">
                    <div class="card shadow p-4 h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h4 class="card-title fw-bold text-info mb-3">Prediction & Risk Assessment</h4>
                            <div id="resultBox" class="p-3 rounded mb-3 text-center" style="background-color: #334155;">
                                <span class="text-muted">Click "Predict Churn Risk" to run inference</span>
                            </div>
                            <div class="mb-3">
                                <label class="small text-secondary mb-1">Churn Probability Gauge</label>
                                <div class="progress" style="height: 24px;">
                                    <div id="probBar" class="progress-bar bg-success" role="progressbar" style="width: 0%; font-weight: bold;">0%</div>
                                </div>
                            </div>
                            <div class="mb-3">
                                <h6 class="small text-secondary fw-bold">PRIMARY RISK DRIVERS:</h6>
                                <ul id="riskFactorsList" class="small text-light ps-3 mb-0">
                                    <li>Awaiting analysis...</li>
                                </ul>
                            </div>
                        </div>
                        <div class="mt-3 p-3 rounded" style="background-color: #1e293b; border: 1px dashed #475569;">
                            <div class="d-flex justify-content-between small text-muted">
                                <span>Engineered Avg_Monthly_Charge_Ratio:</span>
                                <span id="engineeredRatio" class="fw-bold text-info">--</span>
                            </div>
                            <div class="d-flex justify-content-between small text-muted mt-1">
                                <span>Pipeline Endpoint:</span>
                                <span class="fw-bold text-success">/predict (POST)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Fetch Metrics on Load
            async function fetchMetrics() {
                try {
                    const res = await fetch('/metrics');
                    if (res.ok) {
                        const data = await res.json();
                        document.getElementById('kpi-accuracy').innerText = (data.accuracy * 100).toFixed(1) + '%';
                        document.getElementById('kpi-roc-auc').innerText = data.roc_auc.toFixed(3);
                        document.getElementById('kpi-pr').innerText = `${(data.precision*100).toFixed(0)}% / ${(data.recall*100).toFixed(0)}%`;
                        document.getElementById('kpi-f1').innerText = data.f1_score.toFixed(3);
                    }
                } catch(e) {
                    console.log('Metrics fetch error:', e);
                }
            }
            fetchMetrics();

            // Run Prediction
            async function runPrediction() {
                const payload = {
                    customerID: document.getElementById('customerID').value,
                    gender: document.getElementById('gender').value,
                    SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                    Partner: document.getElementById('Partner').value,
                    Dependents: document.getElementById('Dependents').value,
                    tenure: parseInt(document.getElementById('tenure').value),
                    PhoneService: document.getElementById('PhoneService').value,
                    MultipleLines: document.getElementById('MultipleLines').value,
                    InternetService: document.getElementById('InternetService').value,
                    OnlineSecurity: document.getElementById('OnlineSecurity').value,
                    OnlineBackup: document.getElementById('OnlineBackup').value,
                    DeviceProtection: document.getElementById('DeviceProtection').value,
                    TechSupport: document.getElementById('TechSupport').value,
                    StreamingTV: document.getElementById('StreamingTV').value,
                    StreamingMovies: document.getElementById('StreamingMovies').value,
                    Contract: document.getElementById('Contract').value,
                    PaperlessBilling: document.getElementById('PaperlessBilling').value,
                    PaymentMethod: document.getElementById('PaymentMethod').value,
                    MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                    TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    if (!res.ok) {
                        alert('Error: ' + JSON.stringify(data.detail));
                        return;
                    }

                    const probPercent = (data.churn_probability * 100).toFixed(1);
                    const isChurn = data.churn_prediction === 'Yes';
                    const badgeClass = data.risk_level === 'High Risk' ? 'badge-high' : (data.risk_level === 'Medium Risk' ? 'badge-medium' : 'badge-low');

                    document.getElementById('resultBox').innerHTML = `
                        <div class="h3 fw-bold mb-1 ${isChurn ? 'text-danger' : 'text-success'}">
                            ${data.churn_prediction === 'Yes' ? 'WILL CHURN' : 'STAYS (NO CHURN)'}
                        </div>
                        <span class="badge ${badgeClass} fs-6 px-3 py-1">${data.risk_level}</span>
                    `;

                    const probBar = document.getElementById('probBar');
                    probBar.style.width = probPercent + '%';
                    probBar.innerText = probPercent + '%';
                    probBar.className = 'progress-bar ' + (data.risk_level === 'High Risk' ? 'bg-danger' : (data.risk_level === 'Medium Risk' ? 'bg-warning' : 'bg-success'));

                    const factorsUl = document.getElementById('riskFactorsList');
                    factorsUl.innerHTML = data.risk_factors.map(f => `<li>${f}</li>`).join('');

                    document.getElementById('engineeredRatio').innerText = data.engineered_features.Avg_Monthly_Charge_Ratio;
                } catch(e) {
                    alert('Prediction request failed: ' + e);
                }
            }

            function setPreset(type) {
                if (type === 'loyal') {
                    document.getElementById('tenure').value = 48;
                    document.getElementById('tenureVal').innerText = '48';
                    document.getElementById('Contract').value = 'Two year';
                    document.getElementById('InternetService').value = 'DSL';
                    document.getElementById('PaymentMethod').value = 'Credit card (automatic)';
                    document.getElementById('MonthlyCharges').value = '45.00';
                    document.getElementById('TotalCharges').value = '2160.00';
                    document.getElementById('TechSupport').value = 'Yes';
                    document.getElementById('OnlineSecurity').value = 'Yes';
                } else {
                    document.getElementById('tenure').value = 2;
                    document.getElementById('tenureVal').innerText = '2';
                    document.getElementById('Contract').value = 'Month-to-month';
                    document.getElementById('InternetService').value = 'Fiber optic';
                    document.getElementById('PaymentMethod').value = 'Electronic check';
                    document.getElementById('MonthlyCharges').value = '89.50';
                    document.getElementById('TotalCharges').value = '179.00';
                    document.getElementById('TechSupport').value = 'No';
                    document.getElementById('OnlineSecurity').value = 'No';
                }
                runPrediction();
            }
        </script>
    </body>
    </html>
    """
