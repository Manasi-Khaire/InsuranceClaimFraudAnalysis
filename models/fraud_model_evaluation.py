"""
fraud_model_evaluation.py — Model Threshold Evaluation

Purpose:
    Evaluates the existing RandomForestClassifier at different prediction 
    thresholds to find an optimal balance for Fraud Recall and F1.

Used in: MVP Model Tuning
Requires training: No (Uses existing trained model)
Inputs: 
    - models/fraud_model.pkl
    - models/preprocessing_pipeline.pkl
    - data/raw/fraud_dataset/fraud.csv
Outputs:
    - Prints threshold comparison table (Precision, Recall, F1, ROC-AUC)
Dependencies: pandas, scikit-learn, joblib, utils.feature_engineering
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path so utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import load_and_preprocess_data


def evaluate_thresholds():
    csv_path = os.path.join("data", "raw", "fraud_dataset", "fraud.csv")
    model_path = os.path.join("models", "fraud_model.pkl")
    preprocessor_path = os.path.join("models", "preprocessing_pipeline.pkl")

    if not all(os.path.exists(p) for p in [csv_path, model_path, preprocessor_path]):
        print("ERROR: Required artifacts not found.")
        sys.exit(1)

    print("Loading data and artifacts...")
    X, y = load_and_preprocess_data(csv_path)
    clf = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Use same split as training for consistent evaluation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_processed = preprocessor.transform(X_test)

    print("Predicting probabilities...")
    y_proba = clf.predict_proba(X_test_processed)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Base ROC-AUC: {roc_auc:.4f}\n")

    thresholds = [0.5, 0.4, 0.3, 0.2]
    
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'ROC-AUC':<10}")
    print("-" * 65)

    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        
        print(f"{thresh:<10.1f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {roc_auc:<10.4f}")


if __name__ == "__main__":
    evaluate_thresholds()
