import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from src.feature_engineering import add_engineered_features, FeatureSelector
from src.preprocessing import TelcoPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InferencePipeline")

class ChurnPredictor:
    """
    End-to-end inference engine for Telco Customer Churn.
    Combines preprocessing, engineered features, feature selection, and SVM prediction.
    """
    def __init__(
        self,
        preprocessor_path: str = "models/preprocessor.joblib",
        feature_selector_path: str = "models/feature_selector.joblib",
        model_path: str = "models/svm_model.joblib"
    ):
        self.preprocessor_path = preprocessor_path
        self.feature_selector_path = feature_selector_path
        self.model_path = model_path
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized pipeline artifacts."""
        if not (os.path.exists(self.preprocessor_path) and
                os.path.exists(self.feature_selector_path) and
                os.path.exists(self.model_path)):
            logger.warning("One or more model artifacts missing. Inference pipeline will be uninitialized until trained.")
            return

        logger.info("Loading inference artifacts...")
        self.preprocessor = TelcoPreprocessor.load(self.preprocessor_path)
        self.feature_selector = FeatureSelector.load(self.feature_selector_path)
        self.model = joblib.load(self.model_path)
        logger.info("All artifacts successfully loaded into ChurnPredictor.")

    def _determine_risk_level(self, probability: float) -> str:
        if probability >= 0.65:
            return "High Risk"
        elif probability >= 0.35:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _extract_risk_factors(self, row: pd.Series) -> List[str]:
        factors = []
        if str(row.get('Contract')) == 'Month-to-month':
            factors.append("Month-to-month contract (High flexibility to cancel)")
        if str(row.get('InternetService')) == 'Fiber optic' and str(row.get('TechSupport')) == 'No':
            factors.append("Fiber optic service without Tech Support")
        if str(row.get('PaymentMethod')) == 'Electronic check':
            factors.append("Electronic check payment method")
        if float(row.get('tenure', 0)) <= 12:
            factors.append(f"Short tenure ({row.get('tenure')} months)")
        if float(row.get('MonthlyCharges', 0)) > 75.0:
            factors.append(f"High monthly charges (${row.get('MonthlyCharges'):.2f})")
        if str(row.get('OnlineSecurity')) == 'No':
            factors.append("No online security add-on")
        return factors if factors else ["Standard account behavior"]

    def predict_single(self, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs real-time prediction on a single customer dictionary.
        """
        if self.model is None:
            self.load_artifacts()
            if self.model is None:
                raise RuntimeError("Model artifacts not loaded. Please run the training pipeline first.")

        df = pd.DataFrame([customer_dict])
        
        # 1. Feature Engineering
        df_feat = add_engineered_features(df)
        
        # 2. Preprocessing & Scaling
        df_proc = self.preprocessor.transform(df_feat)
        
        # 3. Feature Selection
        df_selected = self.feature_selector.transform(df_proc)
        
        # 4. Predict
        prob = float(self.model.predict_proba(df_selected)[0, 1])
        pred_class = int(self.model.predict(df_selected)[0])
        churn_label = "Yes" if pred_class == 1 else "No"
        risk_level = self._determine_risk_level(prob)
        risk_factors = self._extract_risk_factors(df.iloc[0])

        return {
            "churn_prediction": churn_label,
            "churn_class": pred_class,
            "churn_probability": round(prob, 4),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "engineered_features": {
                "Avg_Monthly_Charge_Ratio": round(float(df_feat['Avg_Monthly_Charge_Ratio'].iloc[0]), 4)
            }
        }

    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs batch prediction on a dataframe.
        """
        if self.model is None:
            self.load_artifacts()
            if self.model is None:
                raise RuntimeError("Model artifacts not loaded.")

        df_feat = add_engineered_features(df)
        df_proc = self.preprocessor.transform(df_feat)
        df_selected = self.feature_selector.transform(df_proc)
        
        probs = self.model.predict_proba(df_selected)[:, 1]
        preds = self.model.predict(df_selected)
        
        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            cust_id = df.iloc[i].get('customerID', f"Cust_{i+1}")
            prob_f = float(prob)
            results.append({
                "customerID": cust_id,
                "churn_prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(prob_f, 4),
                "risk_level": self._determine_risk_level(prob_f),
                "risk_factors": self._extract_risk_factors(df.iloc[i])
            })
        return results

if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample = {
        'customerID': 'TEST-001',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 79.85,
        'TotalCharges': 159.7
    }
    try:
        res = predictor.predict_single(sample)
        print("Prediction Result:", res)
    except Exception as e:
        print("Artifacts not yet created:", e)
