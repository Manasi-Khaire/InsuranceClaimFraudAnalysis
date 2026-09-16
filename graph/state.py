"""
Purpose:
Defines the shared state schema used by the LangGraph workflow.

Related PRD:
- Section 4: Agentic Workflow
- FR-03: Fraud Prediction
- FR-04: Risk Assessment

Related Architecture:
- Agent 1: Claim Intake Agent
- Agent 3: Fraud Prediction Agent
- Agent 4: Risk Assessment Agent

Notes:
- No fraud dataset columns are hardcoded.
- claim_data stores dynamic features discovered from fraud.csv.
"""

from typing import Any, Dict, List, Optional, TypedDict


class ClaimState(TypedDict, total=False):
    """
    Shared state object that flows between LangGraph nodes.

    Phase 1:
        Claim Intake
        → Fraud Prediction
        → Risk Assessment

    Future:
        Damage Assessment
        → Investigation Agent
    """

    # ==========================================================
    # INPUT DATA
    # ==========================================================

    # Dynamic feature values collected from UI or dataset
    # Feature names come from feature_names.json generated
    # after fraud dataset inspection.
    claim_data: Dict[str, Any]

    # Optional image path (used in future damage agent phase)
    vehicle_image_path: Optional[str]

    # ==========================================================
    # FRAUD PREDICTION OUTPUT
    # ==========================================================

    fraud_probability: Optional[float]
    fraud_prediction: Optional[str]

    # Top contributors returned from RF feature importance
    top_risk_features: Optional[List[Dict[str, Any]]]

    # ==========================================================
    # RISK ASSESSMENT OUTPUT
    # ==========================================================

    risk_score: Optional[float]
    risk_level: Optional[str]

    # ==========================================================
    # INVESTIGATION OUTPUT
    # ==========================================================

    investigation_recommendation: Optional[str]

    # ==========================================================
    # EXECUTION METADATA
    # ==========================================================

    status: Optional[str]

    current_step: Optional[str]

    error_message: Optional[str]

    pipeline_log: List[Dict[str, Any]]