"""
1. Purpose: Dynamically parses claim_data to predict fraud probability and extract risk features.
2. Related PRD Section: 4. Agentic Workflow (Fraud Prediction Agent)
3. Related Architecture Section: 3. Agent Definitions (Agent 3)
4. Dependencies: pandas, joblib, json, os, graph.state.ClaimState
5. Validation approach: End-to-end inference test confirming 0-100 scale and 4 new state fields.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any

from graph.state import ClaimState

FRAUD_THRESHOLD = 30.0

def fraud_prediction_agent(state: ClaimState) -> ClaimState:
    """
    Loads saved artifacts, builds inference data dynamically from state["claim_data"],
    applies the saved preprocessing pipeline, and predicts fraud probability.
    
    Generates: fraud_probability, fraud_label, model_confidence, and top_risk_features.
    """
    
    claim_data = state.get("claim_data", {})
    if not claim_data:
        return {"status": "ERROR", "error_message": "Missing claim_data for prediction."}

    # Paths to artifacts
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "fraud_model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessing_pipeline.pkl")
    feature_names_path = os.path.join(base_dir, "models", "feature_names.json")
    
    # Validate artifacts
    if not os.path.exists(model_path):
        return {"status": "ERROR", "error_message": "fraud_model.pkl not found"}
    if not os.path.exists(preprocessor_path):
        return {"status": "ERROR", "error_message": "preprocessing_pipeline.pkl not found"}
    if not os.path.exists(feature_names_path):
        return {"status": "ERROR", "error_message": "feature_names.json not found"}
        
    try:
        # Load artifacts
        clf = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        with open(feature_names_path, "r") as f:
            schema = json.load(f)
            
        all_features = schema["all_features"]
        
        # Build DataFrame directly from claim_data using feature_names
        input_dict = {}
        for feature in all_features:
            if feature not in claim_data:
                return {"status": "ERROR", "error_message": f"Feature missing from claim_data: {feature}"}
            input_dict[feature] = [claim_data[feature]]
            
        df = pd.DataFrame(input_dict)[all_features]
        
        # Apply EXACT saved preprocessing pipeline (no fit, no rebuilding)
        X_processed = preprocessor.transform(df)
        
        # Prediction
        proba = clf.predict_proba(X_processed)[0][1]
        fraud_probability = float(proba * 100)
        
        # Label and confidence
        fraud_label = 1 if fraud_probability >= FRAUD_THRESHOLD else 0
        model_confidence = fraud_probability if fraud_label == 1 else (100.0 - fraud_probability)
        
        # Top risk features (Extract global feature importance mapping)
        # We extract top 3 original features that influenced the model globally
        try:
            feature_names_out = preprocessor.get_feature_names_out()
            importances = clf.feature_importances_
            top_indices = importances.argsort()[-3:][::-1]
            top_risk_features = [feature_names_out[i] for i in top_indices]
        except Exception:
            top_risk_features = ["Feature importances unavailable"]
            
        return {
            "fraud_probability": fraud_probability,
            # Returning extra keys that aren't strictly typed in ClaimState
            # works because ClaimState has total=False (or LangGraph accepts standard dict merging).
            "fraud_label": fraud_label,
            "model_confidence": model_confidence,
            "top_risk_features": top_risk_features,
            "status": "Fraud Prediction Complete"
        }
        
    except Exception as e:
        return {"status": "ERROR", "error_message": f"Prediction failed: {str(e)}"}
