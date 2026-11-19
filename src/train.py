import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Training")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Telco_Customer_Churn_SVM"
REGISTERED_MODEL_NAME = "TelcoChurnSVM"

def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = True,
    version: str = "v1"
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Trains Support Vector Machine classifier with hyperparameter tuning over C and kernel.
    Logs run metadata, parameters, and serialized model to MLflow.
    """
    os.makedirs("models", exist_ok=True)
    
    # Configure MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    if tune_hyperparameters:
        logger.info("Executing GridSearchCV over SVM hyperparameters (C, kernel, gamma)...")
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        base_svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = float(grid_search.best_score_)
        logger.info(f"GridSearchCV Best Params: {best_params} | Best CV ROC-AUC: {best_cv_score:.4f}")
    else:
        logger.info("Training standard RBF SVM classifier (C=1.0)...")
        best_model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
        best_model.fit(X_train, y_train)
        best_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
        best_cv_score = 0.0

    # Save local artifact
    model_path = "models/svm_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"SVM model saved locally to: {model_path}")

    # Prepare metadata
    training_info = {
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "n_train_samples": len(X_train),
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "version": version
    }

    # Log to MLflow if an active run exists
    if mlflow.active_run():
        mlflow.log_params(best_params)
        mlflow.log_param("dataset_version", version)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("cv_best_roc_auc", best_cv_score)
        
        # Log serialized model artifact and register to model registry
        logger.info(f"Logging serialized model to MLflow with registry name: {REGISTERED_MODEL_NAME}")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="svm_churn_model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

    return best_model, training_info
