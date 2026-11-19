import os
import sys
import argparse
import logging
from typing import Dict, Any
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.validation import validate_data
from src.feature_engineering import add_engineered_features, FeatureSelector
from src.preprocessing import TelcoPreprocessor, preprocess_pipeline
from src.train import train_svm, EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from src.evaluate import evaluate_model
from src.predict import ChurnPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a")
    ]
)
logger = logging.getLogger("MLOpsPipeline")

def run_pipeline(
    version: str = "v1",
    file_path: str = None,
    tune_hyperparameters: bool = True,
    promote_to_production: bool = False
) -> Dict[str, Any]:
    """
    Executes the automated end-to-end MLOps pipeline:
    Data Loading -> Validation -> Preprocessing -> Feature Engineering ->
    Training -> Evaluation -> Prediction & MLflow Logging
    """
    logger.info("=" * 60)
    logger.info(f"STARTING TELCO CHURN SVM MLOPS PIPELINE [Dataset Version: {version}]")
    logger.info("=" * 60)

    # MLflow Setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"SVM_Run_{version}"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Active MLflow Run ID: {run_id}")
        mlflow.set_tag("pipeline_version", version)
        mlflow.set_tag("algorithm", "SVM")

        # -----------------------------------------------------------------
        # STAGE 1: DATA LOADING
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 1: DATA LOADING")
        raw_df = load_data(file_path=file_path, version=version)
        mlflow.log_param("raw_data_rows", len(raw_df))
        mlflow.log_param("raw_data_columns", raw_df.shape[1])

        # -----------------------------------------------------------------
        # STAGE 2: DATA VALIDATION
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 2: DATA VALIDATION")
        is_valid, val_report = validate_data(raw_df, is_training=True)
        if not is_valid:
            logger.error(f"Validation failed with critical violations: {val_report['range_violations']}")
            raise ValueError(f"Data validation failed: {val_report}")
        logger.info(f"Validation summary: {val_report['row_count']} rows valid, {val_report['duplicate_rows']} duplicates detected.")

        # -----------------------------------------------------------------
        # STAGE 3: PREPROCESSING & CLEANING (Requirement a)
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 3: PREPROCESSING & CLEANING")
        # Initialize Preprocessor (handles TotalCharges numeric conversion, missing value imputation, duplicates removal)
        preprocessor = TelcoPreprocessor()
        cleaned_df = preprocessor.clean_raw_data(raw_df, is_training=True)

        # -----------------------------------------------------------------
        # STAGE 4: FEATURE ENGINEERING (Requirement b)
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 4: FEATURE ENGINEERING")
        # Engineer Avg_Monthly_Charge_Ratio
        df_feat = add_engineered_features(cleaned_df)
        
        # Fit Preprocessor (One-Hot Encoding, StandardScaler)
        preprocessor.fit(df_feat)
        X_all = preprocessor.transform(df_feat)
        y_all = df_feat['Churn'].map({'No': 0, 'Yes': 1}).astype(int)

        # Save preprocessor artifact
        preprocessor.save("models/preprocessor.joblib")
        mlflow.log_artifact("models/preprocessor.joblib")

        # Train / Test Stratified Split (80% Train, 20% Test)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        logger.info(f"Train set: {len(X_train_raw)} samples | Test set: {len(X_test_raw)} samples")

        # -----------------------------------------------------------------
        # STAGE 5: CORRELATION & LASSO FEATURE SELECTION (Requirement b)
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 5: FEATURE SELECTION (Correlation & LASSO)")
        selector = FeatureSelector(target_corr_thresh=0.02, max_collinearity=0.85, lasso_C=0.15)
        selector.fit(X_train_raw, y_train)
        
        X_train = selector.transform(X_train_raw)
        X_test = selector.transform(X_test_raw)
        
        selector.save("models/feature_selector.joblib")
        mlflow.log_artifact("models/feature_selector.joblib")
        mlflow.log_param("selected_features_count", len(selector.selected_features))
        mlflow.log_param("selected_features_list", ",".join(selector.selected_features))

        # -----------------------------------------------------------------
        # STAGE 6: SVM TRAINING & HYPERPARAMETER TUNING (Requirement b, d)
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 6: MODEL TRAINING & TUNING (SVM)")
        model, train_info = train_svm(
            X_train=X_train,
            y_train=y_train,
            tune_hyperparameters=tune_hyperparameters,
            version=version
        )

        # -----------------------------------------------------------------
        # STAGE 7: EVALUATION & REPORTING (Requirement b)
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 7: EVALUATION & METRICS REPORTING")
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_dir="reports",
            log_to_mlflow=True
        )

        # -----------------------------------------------------------------
        # STAGE 8: PREDICTION PIPELINE VERIFICATION
        # -----------------------------------------------------------------
        logger.info("\n>>> STAGE 8: REAL-TIME INFERENCE VERIFICATION")
        predictor = ChurnPredictor()
        test_sample = raw_df.iloc[0].to_dict()
        prediction_result = predictor.predict_single(test_sample)
        logger.info(f"Sample prediction for {test_sample.get('customerID')}: {prediction_result['churn_prediction']} "
                    f"(Prob: {prediction_result['churn_probability']:.2%}, Risk: {prediction_result['risk_level']})")

        # -----------------------------------------------------------------
        # STAGE 9: MODEL REGISTRY & PRODUCTION PROMOTION (Requirement d)
        # -----------------------------------------------------------------
        if promote_to_production:
            logger.info("\n>>> STAGE 9: PROMOTING MODEL TO PRODUCTION IN MODEL REGISTRY")
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # Fetch latest model versions for TelcoChurnSVM
                latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
                if latest_versions:
                    latest_v = latest_versions[-1].version
                    # MLflow stage transition and alias setting
                    try:
                        client.transition_model_version_stage(
                            name=REGISTERED_MODEL_NAME,
                            version=latest_v,
                            stage="Production",
                            archive_existing_versions=True
                        )
                        logger.info(f"Model {REGISTERED_MODEL_NAME} version {latest_v} transitioned to Stage: Production!")
                    except Exception as stage_err:
                        logger.warning(f"Stage transition notice: {stage_err}")
                        
                    try:
                        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", latest_v)
                        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", latest_v)
                        logger.info(f"Model {REGISTERED_MODEL_NAME} version {latest_v} tagged with aliases @champion and @production!")
                    except Exception as alias_err:
                        logger.warning(f"Alias setting notice: {alias_err}")
            except Exception as e:
                logger.warning(f"Model registry promotion notice: {e}")

        logger.info("=" * 60)
        logger.info(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY FOR VERSION {version}!")
        logger.info("=" * 60)

        return {
            "run_id": run_id,
            "version": version,
            "metrics": metrics,
            "train_info": train_info,
            "sample_prediction": prediction_result
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco Customer Churn SVM MLOps Pipeline Orchestrator")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Dataset version (v1 or v2)")
    parser.add_argument("--file_path", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--no_tune", action="store_true", help="Skip hyperparameter grid search (faster run)")
    parser.add_argument("--promote", action="store_true", help="Promote trained model to Production in MLflow Registry")
    
    args = parser.parse_args()
    run_pipeline(
        version=args.version,
        file_path=args.file_path,
        tune_hyperparameters=not args.no_tune,
        promote_to_production=args.promote
    )
