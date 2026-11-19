import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureEngineering")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features:
    - Avg_Monthly_Charge_Ratio: (TotalCharges / (tenure + 1)) / (MonthlyCharges + 1e-5)
    
    This ratio identifies price fluctuations and bill shock:
    A value < 1 indicates promotional discount expiring or recent price hike.
    """
    data = df.copy()
    
    # Ensure TotalCharges, tenure, and MonthlyCharges are numeric
    tot_charges = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0.0)
    tenure = pd.to_numeric(data['tenure'], errors='coerce').fillna(0.0)
    monthly = pd.to_numeric(data['MonthlyCharges'], errors='coerce').fillna(0.0)
    
    # Calculate Avg Monthly Charge over tenure
    avg_monthly_past = tot_charges / (tenure + 1.0)
    
    # Calculate Ratio of historical average to current monthly charges
    data['Avg_Monthly_Charge_Ratio'] = avg_monthly_past / (monthly + 1e-5)
    
    logger.info("Engineered feature 'Avg_Monthly_Charge_Ratio' successfully computed.")
    return data

class FeatureSelector:
    """
    Executes two-stage feature selection:
    1. Correlation-based filtering (target correlation + multicollinearity removal)
    2. LASSO (L1 regularization) sparse feature selection
    """
    def __init__(self, target_corr_thresh: float = 0.02, max_collinearity: float = 0.85, lasso_C: float = 0.15):
        self.target_corr_thresh = target_corr_thresh
        self.max_collinearity = max_collinearity
        self.lasso_C = lasso_C
        self.selected_features: List[str] = []
        self.correlation_scores: Dict[str, float] = {}
        self.lasso_coefficients: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"Starting feature selection on {X.shape[1]} input features...")
        
        # --- Stage 1: Correlation-based Selection ---
        corr_series = X.apply(lambda col: y.corr(col) if col.std() > 0 else 0.0)
        self.correlation_scores = {col: float(val) for col, val in corr_series.items() if not np.isnan(val)}
        
        # Filter features with correlation magnitude >= threshold
        candidate_features = [
            col for col, corr_val in self.correlation_scores.items()
            if abs(corr_val) >= self.target_corr_thresh
        ]
        logger.info(f"Stage 1a (Target Correlation): Retained {len(candidate_features)} features with |corr| >= {self.target_corr_thresh}")

        # Remove pairwise collinear features
        corr_matrix = X[candidate_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.max_collinearity)]
        
        filtered_by_corr = [c for c in candidate_features if c not in to_drop]
        logger.info(f"Stage 1b (Multicollinearity): Dropped {len(to_drop)} redundant features. Retained: {len(filtered_by_corr)}")

        # --- Stage 2: LASSO (L1-Penalty) Feature Selection ---
        X_stage1 = X[filtered_by_corr]
        lasso_model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=self.lasso_C,
            random_state=42,
            max_iter=1000
        )
        selector = SelectFromModel(lasso_model, prefit=False)
        selector.fit(X_stage1, y)
        
        # Get selected feature mask
        mask = selector.get_support()
        final_selected = [f for f, selected in zip(filtered_by_corr, mask) if selected]
        
        # Store non-zero coefficients
        coefs = selector.estimator_.coef_[0]
        self.lasso_coefficients = {f: float(c) for f, c in zip(filtered_by_corr, coefs) if f in final_selected}
        
        # Fallback safeguard in case sparse penalty drops too many
        if len(final_selected) < 5:
            logger.warning(f"LASSO selected only {len(final_selected)} features; falling back to correlation-filtered features.")
            final_selected = filtered_by_corr[:15]
            
        self.selected_features = final_selected
        self.is_fitted = True
        logger.info(f"Stage 2 (LASSO Selection): Final selected {len(self.selected_features)} features: {self.selected_features}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform.")
        # Fill missing selected features with 0 if inference payload didn't generate them
        missing = [f for f in self.selected_features if f not in X.columns]
        if missing:
            X_filled = X.copy()
            for col in missing:
                X_filled[col] = 0.0
            return X_filled[self.selected_features]
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str = "models/feature_selector.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Feature selector saved to: {filepath}")

    @staticmethod
    def load(filepath: str = "models/feature_selector.joblib") -> "FeatureSelector":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature selector not found at: {filepath}")
        return joblib.load(filepath)
