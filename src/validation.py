import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataValidation")

EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

VALID_CATEGORIES = {
    'gender': {'Female', 'Male'},
    'SeniorCitizen': {0, 1},
    'Partner': {'Yes', 'No'},
    'Dependents': {'Yes', 'No'},
    'PhoneService': {'Yes', 'No'},
    'MultipleLines': {'No phone service', 'No', 'Yes'},
    'InternetService': {'DSL', 'Fiber optic', 'No'},
    'OnlineSecurity': {'No', 'Yes', 'No internet service'},
    'OnlineBackup': {'Yes', 'No', 'No internet service'},
    'DeviceProtection': {'No', 'Yes', 'No internet service'},
    'TechSupport': {'No', 'Yes', 'No internet service'},
    'StreamingTV': {'No', 'Yes', 'No internet service'},
    'StreamingMovies': {'No', 'Yes', 'No internet service'},
    'Contract': {'Month-to-month', 'One year', 'Two year'},
    'PaperlessBilling': {'Yes', 'No'},
    'PaymentMethod': {
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    }
}

def validate_data(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates the dataset schema, data types, nulls, duplicates, and allowable ranges.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        is_training (bool): If True, checks for presence of target 'Churn'.
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_report)
    """
    report = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": df.shape[1],
        "missing_columns": [],
        "duplicate_rows": 0,
        "empty_strings_in_total_charges": 0,
        "range_violations": [],
        "invalid_categories": {},
        "warnings": []
    }
    
    # 1. Check for expected columns
    expected = list(EXPECTED_COLUMNS)
    if is_training:
        expected.append('Churn')
        
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["is_valid"] = False
        logger.error(f"Validation failed: Missing columns: {missing_cols}")
        return False, report

    # 2. Duplicate rows check
    if 'customerID' in df.columns:
        duplicate_count = df.duplicated(subset=['customerID']).sum()
        report["duplicate_rows"] = int(duplicate_count)
        if duplicate_count > 0:
            report["warnings"].append(f"Found {duplicate_count} duplicate customer IDs.")
            logger.warning(f"Found {duplicate_count} duplicate customer IDs.")

    # 3. Check TotalCharges whitespace
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object or str(df['TotalCharges'].dtype).startswith('str'):
        empty_count = (df['TotalCharges'].astype(str).str.strip() == '').sum()
        report["empty_strings_in_total_charges"] = int(empty_count)
        if empty_count > 0:
            report["warnings"].append(f"Found {empty_count} rows with blank 'TotalCharges'. Will be imputed.")

    # 4. Check numerical bounds
    if 'tenure' in df.columns:
        invalid_tenure = (df['tenure'] < 0).sum()
        if invalid_tenure > 0:
            report["range_violations"].append(f"Negative tenure found in {invalid_tenure} rows.")
            report["is_valid"] = False
            
    if 'MonthlyCharges' in df.columns:
        invalid_monthly = (df['MonthlyCharges'] < 0).sum()
        if invalid_monthly > 0:
            report["range_violations"].append(f"Negative MonthlyCharges found in {invalid_monthly} rows.")
            report["is_valid"] = False

    # 5. Check categorical values
    for col, valid_vals in VALID_CATEGORIES.items():
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            invalid = unique_vals - valid_vals
            if invalid:
                report["invalid_categories"][col] = list(invalid)
                report["warnings"].append(f"Column '{col}' contains unrecognized values: {invalid}")

    logger.info(f"Data validation completed. Status: {'PASSED' if report['is_valid'] else 'FAILED'}")
    return report["is_valid"], report

if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    is_valid, report = validate_data(df)
    print("Valid:", is_valid)
    print("Report:", report)
