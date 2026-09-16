"""
1. Purpose: Applies the exact architecture formula to consolidate risk.
2. Related PRD Section: 4. Agentic Workflow (Risk Assessment Agent)
3. Related Architecture Section: 3. Agent Definitions (Agent 4)
4. Dependencies: graph.state.ClaimState
5. Validation approach: Unit test confirming (0.60 * prob) + (0.40 * damage * 100).
"""

from graph.state import ClaimState

FRAUD_THRESHOLD = 30.0

def risk_assessment_agent(state: ClaimState) -> ClaimState:
    """
    Applies the exact weighting formula:
    risk_score = (0.60 * fraud_probability) + (0.40 * damage_severity_score * 100)
    
    Since damage_assessment is Phase 2, damage_severity_score defaults to 0.0 for MVP.
    """
    
    fraud_prob = state.get("fraud_probability")
    if fraud_prob is None:
        return {"status": "ERROR", "error_message": "Missing fraud_probability in state"}
        
    damage_score = state.get("damage_severity_score")
    if damage_score is None:
        damage_score = 0.0
        
    # Exact PRD/Architecture Formula
    risk_score = (0.60 * fraud_prob) + (0.40 * damage_score * 100)
    
    # Approved Risk Level mapping
    if risk_score <= 30:
        risk_level = "LOW"
        recommendation = "Auto-approve claim. No further review required."
    elif risk_score <= 60:
        risk_level = "MEDIUM"
        recommendation = (
            "Assign to claims reviewer and request supporting documentation."
        )
    elif risk_score <= 80:
        risk_level = "HIGH"
        recommendation = (
            "Escalate to senior adjuster. Field inspection is recommended."
        )
    else:
        risk_level = "CRITICAL"
        recommendation = (
            "Escalate immediately to Special Investigation Unit (SIU)."
        )

    return {
        "risk_score": round(float(risk_score), 2),
        "risk_level": risk_level,
        "investigation_recommendation": recommendation,
        "status": "Risk Assessment Complete",
    }
