# Telco Customer Churn: Automated End-to-End MLOps Pipeline (SVM)

Enterprise-grade, automated MLOps pipeline for predicting customer churn on the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) using a **Support Vector Machine (SVM)** classifier, hyperparameter tuning, **MLflow** experiment tracking & model registry, **Git & DVC** remote dataset versioning, **FastAPI** REST microservice with Pydantic validation, and an interactive **Streamlit** dashboard.

---

## 📌 Project Architecture

```
ml project/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Downloaded raw dataset (7,043 rows)
│   ├── versions/
│   │   ├── dataset_v1.csv                         # Version 1 cohort (6,000 rows)
│   │   └── dataset_v2.csv                         # Version 2 cohort (7,043 rows)
│   └── active_dataset.csv.dvc                     # DVC tracking pointer for active version
├── dvc_remote_storage/                            # Local DVC remote storage
├── src/
│   ├── __init__.py
│   ├── data_loader.py                             # Stage 1: Data Loading & Version Ingestion
│   ├── validation.py                              # Stage 2: Data Validation & Schema Quality Checks
│   ├── preprocessing.py                           # Stage 3: Cleaning, Imputation, Encoding, StandardScaler
│   ├── feature_engineering.py                     # Stage 4 & 5: Avg_Monthly_Charge_Ratio, Correlation & LASSO
│   ├── train.py                                   # Stage 6: SVM Training with GridSearchCV (C, kernel, gamma)
│   ├── evaluate.py                                # Stage 7: Accuracy, Precision, Recall, F1, ROC-AUC & Curves
│   └── predict.py                                 # Stage 8: Real-time & Batch Inference Engine
├── api/
│   ├── __init__.py
│   ├── schemas.py                                 # Pydantic schemas with strict payload validation
│   └── app.py                                     # FastAPI endpoints (/predict, /metrics, /health, /dashboard)
├── dashboard/
│   └── app.py                                     # Interactive Streamlit Web Application
├── models/
│   ├── preprocessor.joblib                        # Fitted cleaning, encoding & scaling transformer
│   ├── feature_selector.joblib                    # Fitted Correlation + LASSO feature selector
│   └── svm_model.joblib                           # Tuned production Support Vector Machine model
├── reports/
│   ├── confusion_matrix.png                       # Heatmap visualization
│   ├── roc_curve.png                              # Receiver Operating Characteristic plot (AUC = 0.843)
│   ├── precision_recall_curve.png                 # Precision-Recall curve
│   ├── evaluation_summary.json                    # Metrics JSON
│   └── classification_report.txt                  # Full classification breakdown
├── scripts/
│   ├── prepare_versions.py                        # Prepares v1 and v2 cohort splits
│   ├── setup_dvc_versions.py                      # Initializes DVC tracking and Git release tags
│   ├── demo_dvc_versions.py                       # Automated version switching verification
│   ├── run_api_tests.py                           # Automated test suite for valid and invalid payloads
│   ├── start_api.py                               # Launches FastAPI on port 8000
│   ├── start_dashboard.py                         # Launches Streamlit Dashboard on port 8501
│   └── start_mlflow.py                            # Launches MLflow UI on port 5000
├── tests/
│   └── test_api.py                                # Pytest-compatible API integration tests
├── main.py                                        # Master orchestrator automating the entire workflow
├── requirements.txt                               # Environment dependencies
└── README.md
```

---

## 🏆 Key Requirements Implemented & Verified

### a) Data Preprocessing & Cleaning
- **TotalCharges to Numeric:** Handled empty strings (`' '`) when `tenure == 0` using `pd.to_numeric(errors='coerce')` and imputed with `0.0`.
- **Missing Value Imputation:** Handled missing numerical values with median imputation and categoricals with mode imputation.
- **Duplicate Removal:** Automated deduplication on customer records via `drop_duplicates(subset=['customerID'])`.
- **Categorical Encoding:** Binary mapping for binary attributes (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) and `OneHotEncoder(handle_unknown='ignore')` for multi-class attributes.
- **Feature Scaling:** Applied `StandardScaler` to numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `Avg_Monthly_Charge_Ratio`).

### b) Feature Engineering, Selection & SVM Modeling
- **Domain Feature Engineering (`Avg_Monthly_Charge_Ratio`):**
  $$\text{Avg\_Monthly\_Charge\_Ratio} = \frac{\text{TotalCharges} / (\text{tenure} + 1)}{\text{MonthlyCharges} + 1e-5}$$
  *Captures customer price shocks and discount expirations directly correlating with churn.*
- **Two-Stage Feature Selection:**
  1. *Correlation-based filter:* Retains features with target correlation $|r| \ge 0.02$ and drops pairwise multicollinear features ($|r| > 0.85$).
  2. *LASSO (L1 Regularization):* Sparse model selection eliminating redundant parameters.
- **SVM Classifier & GridSearchCV Tuning:**
  - Evaluated hyperparameters: `C ∈ [0.1, 1.0, 10.0]`, `kernel ∈ ['linear', 'rbf']`, `gamma ∈ ['scale', 'auto']`.
  - Best parameters identified: `{'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}` with CV ROC-AUC: **0.8457**.
- **Metrics Reported:**
  - **Accuracy:** `80.27%`
  - **Precision:** `65.48%`
  - **Recall:** `54.28%`
  - **F1 - Score:** `0.5936`
  - **ROC - AUC:** `0.8430`

### c) DVC & Git Artifact & Dataset Version Tracking
- **Version 1 (`v1.0`):** Baseline cohort of 6,000 customers tracked under Git tag `v1.0` and DVC storage.
- **Version 2 (`v2.0`):** Full updated cohort of 7,043 customers tracked under Git tag `v2.0` and DVC storage.
- **Automated Version Switching Verified (`scripts/demo_dvc_versions.py`):**
  - `git checkout v1.0 && python -m dvc checkout` -> Restores exact 6,000 row dataset.
  - `git checkout v2.0 && python -m dvc checkout` -> Restores full 7,043 row dataset.

### d) MLflow Tracking & Model Registry
- **Experiment:** `Telco_Customer_Churn_SVM`
- **Logged Entities:** Hyperparameters, dataset versions, metrics, confusion matrix, ROC curve, PR curve, preprocessor artifact, and serialized scikit-learn SVM model.
- **Model Registry Promotion:** Registered model `TelcoChurnSVM`. Version 2 promoted to `Production` stage with aliases `@champion` and `@production`.

### e) Automated Pipeline Orchestration (`main.py`)
- Executes the complete workflow sequentially:
  $$\text{Data Loading} \longrightarrow \text{Validation} \longrightarrow \text{Preprocessing} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Training} \longrightarrow \text{Evaluation} \longrightarrow \text{Prediction}$$

### f) FastAPI Real-Time Serving & Pydantic Validation
- Strict Pydantic model (`CustomerInput`) enforcing:
  - `tenure`: integer between `0` and `120` months.
  - `MonthlyCharges`: non-negative float ($\ge 0.0$).
  - `TotalCharges`: non-negative float or empty string.
  - Categorical enums: only authorized values accepted (`Contract`, `InternetService`, etc.).
- **Tested Endpoints (`scripts/run_api_tests.py`):**
  - `POST /predict`: Valid loyal payload -> `200 OK` (Prediction: No, Risk: Low Risk, Prob: 1.94%).
  - `POST /predict`: Valid at-risk payload -> `200 OK` (Prediction: Yes, Risk: High Risk, Prob: 86.17%).
  - `POST /predict`: Invalid negative tenure (`-5`) -> `422 Unprocessable Entity`.
  - `POST /predict`: Invalid excessive tenure (`250`) -> `422 Unprocessable Entity`.
  - `POST /predict`: Invalid negative monthly charges (`-45.0`) -> `422 Unprocessable Entity`.
  - `POST /predict`: Invalid contract category (`'Lifetime'`) -> `422 Unprocessable Entity`.
  - `POST /predict`: Invalid negative total charges (`-200.0`) -> `422 Unprocessable Entity`.
  - `POST /predict`: Missing required fields -> `422 Unprocessable Entity`.
  - `GET /metrics`: Returns `200 OK` with production metrics.
  - `GET /health`: Returns `200 OK` with service health.

---

## 🚀 How to Run the Project

### 1. Execute the Pipeline
Run Version 1:
```powershell
python main.py --version v1
```
Run Version 2 and promote winning model to Production:
```powershell
python main.py --version v2 --promote
```

### 2. Verify DVC Version Switching
```powershell
python scripts/demo_dvc_versions.py
```

### 3. Run FastAPI Endpoint Validation Tests
```powershell
python scripts/run_api_tests.py
```

### 4. Start the Services
- **FastAPI REST API & Web Dashboard:**
  ```powershell
  python scripts/start_api.py
  ```
  - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  - Embedded Web Dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

- **Interactive Streamlit Dashboard:**
  ```powershell
  python scripts/start_dashboard.py
  ```
  - Access at: [http://localhost:8501](http://localhost:8501)

- **MLflow Tracking UI:**
  ```powershell
  python scripts/start_mlflow.py
  ```
  - Access at: [http://127.0.0.1:5000](http://127.0.0.1:5000)
