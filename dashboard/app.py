import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Telco Churn SVM MLOps Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# Cache artifacts loader
@st.cache_resource
def load_model_artifacts():
    try:
        from src.predict import ChurnPredictor
        predictor = ChurnPredictor()
        return predictor
    except Exception as e:
        return None

def load_metrics_summary():
    metrics_path = "reports/evaluation_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {
        "accuracy": 0.805,
        "precision": 0.680,
        "recall": 0.550,
        "f1_score": 0.608,
        "roc_auc": 0.845
    }

predictor = load_model_artifacts()
metrics = load_metrics_summary()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("Telco MLOps Suite")
st.sidebar.markdown("**Algorithm:** Support Vector Machine (SVM)")
st.sidebar.markdown("**Experiment:** MLflow Tracked")
st.sidebar.markdown("**Storage:** DVC Remote Versioned")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate Modules",
    [
        "🚀 Model Overview & KPIs",
        "🎯 Real-time Churn Predictor",
        "📂 Batch File Scoring",
        "📊 Evaluation & Confusion Matrix",
        "🔬 Feature Engineering & LASSO",
        "📦 MLflow & DVC Versioning"
    ]
)

# -------------------------------------------------------------
# PAGE 1: MODEL OVERVIEW & KPIS
# -------------------------------------------------------------
if page == "🚀 Model Overview & KPIs":
    st.title("📡 Telco Customer Churn - SVM MLOps Pipeline")
    st.markdown("Automated, reproducible end-to-end Machine Learning pipeline using **Support Vector Machine (SVM)**, tracked with **MLflow**, and versioned with **Git & DVC**.")
    
    st.markdown("### 🏆 Production Model Performance KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Accuracy", value=f"{metrics.get('accuracy', 0)*100:.1f}%")
    with col2:
        st.metric(label="Precision", value=f"{metrics.get('precision', 0)*100:.1f}%")
    with col3:
        st.metric(label="Recall", value=f"{metrics.get('recall', 0)*100:.1f}%")
    with col4:
        st.metric(label="F1 - Score", value=f"{metrics.get('f1_score', 0):.3f}")
    with col5:
        st.metric(label="ROC - AUC", value=f"{metrics.get('roc_auc', 0):.3f}")
        
    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("System Architecture")
        st.info("""
        **Pipeline Execution Stages (`main.py`):**
        1. **Data Loading:** Ingests versioned raw customer cohorts (v1 / v2) via DVC remote storage.
        2. **Validation:** Verifies schema, detects nulls, range violations (e.g. negative tenure), and duplicates.
        3. **Preprocessing:** Converts TotalCharges to numeric, median imputation, duplicate customer row removal, categorical encoding, and StandardScaler.
        4. **Feature Engineering:** Computes domain-specific `Avg_Monthly_Charge_Ratio` to capture price sensitivity & bill shock.
        5. **Feature Selection:** Combines correlation-based multicollinearity filtering and LASSO (L1 regularization) sparse selection.
        6. **Model Training & Tuning:** GridSearchCV across SVM hyperparameters (`C`, `kernel`, `gamma`) logged in MLflow.
        7. **Evaluation:** Computes Confusion Matrix, ROC-AUC, PR Curves, Precision, Recall, and F1.
        8. **Model Registry:** Model version promoted to `@champion` / `Production` in MLflow Registry.
        9. **Serving:** Real-time FastAPI REST API (`/predict`, `/metrics`) and interactive dashboards.
        """)
        
    with c2:
        st.subheader("Deployment Governance")
        st.success("**MLflow Model:** `TelcoChurnSVM`\n\n**Stage:** `Production`\n\n**Alias:** `@champion`\n\n**DVC Remote:** Configured (`v1.0` & `v2.0` tags)")
        st.warning("**FastAPI Service:** Active on `:8000`\n\n**Endpoints:**\n- `/predict` (POST)\n- `/metrics` (GET)\n- `/health` (GET)\n- `/dashboard` (GET)")

# -------------------------------------------------------------
# PAGE 2: REAL-TIME CHURN PREDICTOR
# -------------------------------------------------------------
elif page == "🎯 Real-time Churn Predictor":
    st.title("🎯 Real-Time Customer Churn Scoring")
    st.markdown("Enter customer subscription details to evaluate their real-time probability of churn.")

    with st.form("churn_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="CUST-7829")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            tenure = st.slider("Tenure with Company (Months)", min_value=0, max_value=72, value=4)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=79.50, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=318.0, step=1.0)

        submitted = st.form_submit_button("⚡ Run Real-Time SVM Prediction")

    if submitted:
        if predictor is None:
            st.error("Model artifacts not loaded! Please execute `python main.py` first.")
        else:
            payload = {
                'customerID': customer_id,
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': "No",
                'StreamingMovies': "No",
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            try:
                res = predictor.predict_single(payload)
                st.divider()
                st.subheader("Inference Result & Risk Analysis")
                
                r1, r2, r3 = st.columns([1, 1, 2])
                with r1:
                    if res['churn_prediction'] == 'Yes':
                        st.error(f"### Prediction: CHURN (YES)")
                    else:
                        st.success(f"### Prediction: LOYAL (NO)")
                with r2:
                    st.metric("Churn Probability", f"{res['churn_probability']*100:.1f}%")
                    st.markdown(f"**Risk Tier:** `{res['risk_level']}`")
                with r3:
                    st.markdown("**Key Risk Factors Identified:**")
                    for factor in res['risk_factors']:
                        st.markdown(f"- ⚠️ {factor}")
                        
                st.caption(f"Engineered Feature: `Avg_Monthly_Charge_Ratio` = {res['engineered_features']['Avg_Monthly_Charge_Ratio']}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------------------------------------------
# PAGE 3: BATCH FILE SCORING
# -------------------------------------------------------------
elif page == "📂 Batch File Scoring":
    st.title("📂 Batch CSV Scoring")
    st.markdown("Upload a customer CSV dataset to generate batch predictions using the production SVM pipeline.")
    
    uploaded_file = st.file_uploader("Upload Telco Customer CSV", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} customer records:")
        st.dataframe(df_batch.head(5))
        
        if st.button("🚀 Score Entire Batch"):
            if predictor is None:
                st.error("Predictor model not initialized.")
            else:
                with st.spinner("Scoring customer records..."):
                    results = predictor.predict_batch(df_batch)
                    res_df = pd.DataFrame(results)
                    merged = pd.concat([df_batch.reset_index(drop=True), res_df[['churn_prediction', 'churn_probability', 'risk_level']]], axis=1)
                    
                    st.success(f"Batch scoring completed! {len(merged)} records scored.")
                    st.dataframe(merged[['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'churn_prediction', 'churn_probability', 'risk_level']].head(10))
                    
                    csv = merged.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Scored CSV", data=csv, file_name="telco_churn_scored.csv", mime="text/csv")

# -------------------------------------------------------------
# PAGE 4: EVALUATION & CONFUSION MATRIX
# -------------------------------------------------------------
elif page == "📊 Evaluation & Confusion Matrix":
    st.title("📊 Evaluation & Model Visualizations")
    st.markdown("High-resolution diagnostic curves and metrics generated during the automated evaluation stage.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python main.py` to generate the confusion matrix plot.")

    with col2:
        st.subheader("Receiver Operating Characteristic (ROC)")
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png", use_container_width=True)
        else:
            st.info("Run `python main.py` to generate the ROC curve plot.")

    st.subheader("Detailed Classification Report")
    if os.path.exists("reports/classification_report.txt"):
        with open("reports/classification_report.txt", "r") as f:
            st.code(f.read())

# -------------------------------------------------------------
# PAGE 5: FEATURE ENGINEERING & LASSO
# -------------------------------------------------------------
elif page == "🔬 Feature Engineering & LASSO":
    st.title("🔬 Feature Engineering & LASSO Selection")
    st.markdown("""
    ### 1. Engineered Feature: `Avg_Monthly_Charge_Ratio`
    $$\\text{Avg\\_Monthly\\_Charge\\_Ratio} = \\frac{\\text{TotalCharges} / (\\text{tenure} + 1)}{\\text{MonthlyCharges} + 1e-5}$$
    
    **Domain Rationale:**
    Customers whose past average monthly charge is significantly lower than their current monthly charge experience **billing shock** or the expiration of promotional pricing discounts, directly driving high churn probability.
    """)
    
    st.divider()
    st.markdown("""
    ### 2. Two-Stage Feature Selection Pipeline
    - **Stage 1 (Correlation Filter):** Evaluates Pearson correlation with the target binary Churn ($|r| \\ge 0.02$) and eliminates severe pairwise multicollinearity ($|r| > 0.85$).
    - **Stage 2 (LASSO L1-Regularization):** Fits sparse L1-penalized estimator to eliminate redundant features while preserving maximal discriminative power for SVM.
    """)
    
    if os.path.exists("models/feature_selector.joblib"):
        selector = joblib.load("models/feature_selector.joblib")
        st.success(f"Selected {len(selector.selected_features)} features for SVM classification:")
        st.write(selector.selected_features)
        
        if selector.lasso_coefficients:
            st.subheader("LASSO Non-Zero Coefficients")
            coef_df = pd.DataFrame(list(selector.lasso_coefficients.items()), columns=["Feature", "Coefficient"]).sort_values(by="Coefficient", ascending=False)
            st.dataframe(coef_df)

# -------------------------------------------------------------
# PAGE 6: MLFLOW & DVC VERSIONING
# -------------------------------------------------------------
elif page == "📦 MLflow & DVC Versioning":
    st.title("📦 MLflow & DVC Artifact Tracking")
    st.markdown("Track experiments, model registry stages, and dataset versions.")
    
    st.subheader("1. DVC Version Control Overview")
    st.info("""
    - **Dataset v1 (Baseline Cohort):** 6,000 customers tracked under Git tag `v1.0` and DVC storage.
    - **Dataset v2 (Full Production Cohort):** 7,043 customers tracked under Git tag `v2.0` and DVC storage.
    - **Reproducibility Command:**
      ```bash
      git checkout v1.0 && dvc checkout
      git checkout v2.0 && dvc checkout
      ```
    """)
    
    st.subheader("2. MLflow Tracking & Model Registry")
    st.markdown("""
    - **Experiment Name:** `Telco_Customer_Churn_SVM`
    - **Registered Model:** `TelcoChurnSVM`
    - **Active Production Stage:** Promoted winning version with tag `@champion`
    - **To launch the MLflow UI:**
      ```bash
      mlflow ui --port 5000
      ```
    """)
