import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Preprocessing")

BINARY_COLS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1}
}

MULTICLASS_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Charge_Ratio']

class TelcoPreprocessor:
    """
    Handles end-to-end cleaning, imputation, encoding, and scaling for the Telco dataset.
    Fully serializable to ensure identical feature transformation in production inference.
    """
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.total_charges_median = 0.0
        self.encoded_feature_names: List[str] = []
        self.is_fitted = False

    def clean_raw_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Step 1: Converts TotalCharges to numeric, imputes missing values,
        and removes duplicate customer rows.
        """
        data = df.copy()
        
        # Remove duplicates
        if 'customerID' in data.columns and is_training:
            initial_count = len(data)
            data = data.drop_duplicates(subset=['customerID'], keep='first').copy()
            removed = initial_count - len(data)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate customer rows.")

        # Convert TotalCharges to numeric
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            if is_training:
                self.total_charges_median = float(data['TotalCharges'].median())
            # For tenure == 0, TotalCharges is typically 0; otherwise use median
            data['TotalCharges'] = data['TotalCharges'].fillna(
                data['tenure'].apply(lambda t: 0.0 if t == 0 else self.total_charges_median)
            )

        # Impute any other missing numerical values
        for col in ['MonthlyCharges', 'tenure']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Impute missing categoricals
        for col in list(BINARY_COLS.keys()) + MULTICLASS_COLS:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'No')

        return data

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits imputers, OneHotEncoder, and StandardScaler on training data.
        """
        logger.info("Fitting preprocessor pipeline...")
        data = self.clean_raw_data(df, is_training=True)

        # 1. Binary encoding
        for col, mapping in BINARY_COLS.items():
            if col in data.columns:
                data[col] = data[col].map(mapping).fillna(0).astype(int)

        # 2. Multi-class categorical encoding
        self.encoder.fit(data[MULTICLASS_COLS])
        ohe_feature_names = list(self.encoder.get_feature_names_out(MULTICLASS_COLS))

        # 3. Numeric feature scaling
        self.scaler.fit(data[NUMERIC_COLS])

        # Track complete ordered feature names
        self.encoded_feature_names = ['SeniorCitizen'] + list(BINARY_COLS.keys()) + ohe_feature_names + NUMERIC_COLS
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Total features after encoding: {len(self.encoded_feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw data into encoded, scaled features matching the fitted schema.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform!")

        data = self.clean_raw_data(df, is_training=False)

        # Binary columns
        binary_dfs = []
        for col, mapping in BINARY_COLS.items():
            if col in data.columns:
                binary_dfs.append(data[col].map(mapping).fillna(0).astype(float))
            else:
                binary_dfs.append(pd.Series(0.0, index=data.index, name=col))
        df_binary = pd.concat(binary_dfs, axis=1)

        # SeniorCitizen
        sr_citizen = data['SeniorCitizen'].astype(float).to_frame(name='SeniorCitizen') if 'SeniorCitizen' in data.columns else pd.DataFrame({'SeniorCitizen': 0.0}, index=data.index)

        # Multi-class OHE
        ohe_features = self.encoder.transform(data[MULTICLASS_COLS])
        df_ohe = pd.DataFrame(
            ohe_features,
            columns=self.encoder.get_feature_names_out(MULTICLASS_COLS),
            index=data.index
        )

        # Numeric Scaling
        scaled_numeric = self.scaler.transform(data[NUMERIC_COLS])
        df_scaled_num = pd.DataFrame(
            scaled_numeric,
            columns=NUMERIC_COLS,
            index=data.index
        )

        # Combine all processed features
        processed_df = pd.concat([sr_citizen, df_binary, df_ohe, df_scaled_num], axis=1)
        
        # Ensure column order matches
        processed_df = processed_df[self.encoded_feature_names]
        return processed_df

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def save(self, filepath: str = "models/preprocessor.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor artifact saved to: {filepath}")

    @staticmethod
    def load(filepath: str = "models/preprocessor.joblib") -> "TelcoPreprocessor":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        return joblib.load(filepath)

def preprocess_pipeline(df: pd.DataFrame, is_training: bool = True, preprocessor: Optional[TelcoPreprocessor] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], TelcoPreprocessor]:
    """
    Helper function to execute preprocessing pipeline and extract target Churn.
    """
    data = df.copy()
    y = None
    if 'Churn' in data.columns:
        y = data['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
        
    if is_training or preprocessor is None:
        preprocessor = TelcoPreprocessor()
        X_processed = preprocessor.fit_transform(data)
    else:
        X_processed = preprocessor.transform(data)
        
    return X_processed, y, preprocessor
