"""
models/train_fraud_model.py — Fraud Model Training Script

Purpose:
    Trains a RandomForestClassifier on the Vehicle Insurance Fraud Detection
    dataset and persists all artifacts needed for inference.

Used in: MVP (Phase 1) — run once to produce model artifacts
Requires training: Yes (this IS the training script)
Inputs: data/raw/fraud_dataset/fraud.csv
Outputs:
    models/fraud_model.pkl            — trained RandomForestClassifier
    models/preprocessing_pipeline.pkl — fitted ColumnTransformer
    models/feature_names.json         — feature schema for inference
    models/model_metrics.json         — accuracy, f1, roc_auc, classification report
Dependencies: pandas, scikit-learn, joblib, utils.feature_engineering
Validation: Run 'py models/train_fraud_model.py' from project root
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path so utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import (
    build_preprocessing_pipeline,
    load_and_preprocess_data,
    save_feature_names,
)


def train_model():
    csv_path = os.path.join("data", "raw", "fraud_dataset", "fraud.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at '{csv_path}'")
        print("Ensure fraud.csv is located at data/raw/fraud_dataset/fraud.csv")
        sys.exit(1)

    # Step 1: Load data
    print("[1/6] Loading dataset...")
    X, y = load_and_preprocess_data(csv_path)
    print(f"       Loaded {X.shape[0]} rows, {X.shape[1]} features")
    print(f"       Fraud rate: {y.mean():.4f} ({y.sum()} / {len(y)})")

    # Step 2: Train/test split
    print("[2/6] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Step 3: Build and fit preprocessing pipeline
    print("[3/6] Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"       Transformed feature dimensions: {X_train_processed.shape[1]}")

    # Step 4: Train model
    print("[4/6] Training RandomForestClassifier (class_weight=balanced)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_processed, y_train)

    # Step 5: Evaluate model
    print("[5/6] Evaluating model on test set...")
    y_pred = clf.predict(X_test_processed)
    y_proba = clf.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"       Accuracy:  {accuracy:.4f}")
    print(f"       F1 Score:  {f1:.4f}")
    print(f"       ROC AUC:   {roc_auc:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    # Step 6: Save all artifacts
    print("[6/6] Saving artifacts to models/...")
    os.makedirs("models", exist_ok=True)

    joblib.dump(clf, os.path.join("models", "fraud_model.pkl"))
    print("       Saved: models/fraud_model.pkl")

    joblib.dump(preprocessor, os.path.join("models", "preprocessing_pipeline.pkl"))
    print("       Saved: models/preprocessing_pipeline.pkl")

    feature_path = save_feature_names("models")
    print(f"       Saved: {feature_path}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "classification_report": report,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X_train_processed.shape[1],
        "model_params": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
        },
    }
    metrics_path = os.path.join("models", "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"       Saved: {metrics_path}")

    print()
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
