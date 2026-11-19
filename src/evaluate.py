import os
import json
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless background plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluation")

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "reports",
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Computes all standard classification metrics:
    Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    Generates Confusion Matrix and ROC Curve plots and saves them to reports/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Scale decision function to [0, 1] using sigmoid
        y_prob = 1 / (1 + np.exp(-model.decision_function(X_test)))
    else:
        y_prob = y_pred

    # Compute key metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4))
    }

    logger.info("================ MODEL EVALUATION METRICS ================")
    logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"Precision : {metrics['precision']:.4f}")
    logger.info(f"Recall    : {metrics['recall']:.4f}")
    logger.info(f"F1 - Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC - AUC : {metrics['roc_auc']:.4f}")
    logger.info("==========================================================")

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'])
    plt.title('Confusion Matrix - Telco Churn SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'SVM ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curve Plot
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(pr_rec, pr_prec, color='#2ca02c', lw=2, label=f'Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # 4. Save metrics summary json
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Save detailed classification report
    report_text = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Log to active MLflow run if available
    if log_to_mlflow:
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(pr_path)
                mlflow.log_artifact(summary_path)
                mlflow.log_artifact(report_path)
                logger.info("Successfully logged metrics and plots to active MLflow run.")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")

    return metrics
