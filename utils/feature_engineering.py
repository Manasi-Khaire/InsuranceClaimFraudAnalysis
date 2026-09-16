"""
utils/feature_engineering.py — Fraud Model Feature Definitions & Preprocessing

Purpose:
    Single source of truth for feature names and the shared preprocessing
    pipeline. Both training (train_fraud_model.py) and inference
    (fraud_prediction_agent.py) import from this module to guarantee
    identical feature ordering and transformations.

Used in: MVP (Phase 1)
Requires training: No (defines the pipeline; fitting happens in train_fraud_model.py)
Inputs: fraud.csv (via load_and_preprocess_data)
Outputs: X (DataFrame of 30 features), y (target array)
Dependencies: pandas, scikit-learn
Validation: Run 'py -c "from utils.feature_engineering import ALL_FEATURES; print(len(ALL_FEATURES))"'
             Expected output: 30
"""

import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================================
# Feature lists derived from fraud.csv inspection (15420 x 33).
# Dropped: PolicyNumber (identifier), RepNumber (identifier),
#          FraudFound_P (target). Remaining: 30 features.
# =====================================================================

NUMERICAL_FEATURES = [
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Age",
    "Deductible",
    "DriverRating",
    "Year",
]

CATEGORICAL_FEATURES = [
    "Month",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "BasePolicy",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "FraudFound_P"

DROP_COLUMNS = ["PolicyNumber", "RepNumber"]


def load_and_preprocess_data(csv_path: str):
    """
    Loads fraud.csv, drops identifiers, and returns feature matrix X
    and target vector y.

    Returns:
        X: pd.DataFrame with 30 feature columns
        y: np.ndarray with binary target values
    """
    df = pd.read_csv(csv_path)

    y = df[TARGET_COLUMN].values

    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])

    # Verify column alignment
    missing = set(ALL_FEATURES) - set(X.columns)
    extra = set(X.columns) - set(ALL_FEATURES)
    if missing:
        raise ValueError(f"Missing expected features in dataset: {missing}")
    if extra:
        raise ValueError(f"Unexpected extra columns in dataset: {extra}")

    # Enforce column order
    X = X[ALL_FEATURES]

    return X, y


def build_preprocessing_pipeline():
    """
    Builds a scikit-learn ColumnTransformer that applies:
      - StandardScaler to numerical features
      - OneHotEncoder to categorical features

    This exact pipeline is used for both training and inference.

    Returns:
        ColumnTransformer (unfitted)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def save_feature_names(output_dir: str = "models"):
    """
    Persists the feature schema to JSON so that inference agents can
    verify they are supplying the correct columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    schema = {
        "all_features": ALL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "drop_columns": DROP_COLUMNS,
    }
    path = os.path.join(output_dir, "feature_names.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=4)
    return path
