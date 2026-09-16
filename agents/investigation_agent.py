"""
Purpose:
    Produces the final investigation recommendation and detailed report based on the
    fraud probability, risk score, and damage-assessment proxy summary.

Related PRD:
    - FR-05: Generate investigation report summarizing findings.
    - Section 4: Investigation Recommendation Agent.

Related Architecture:
    - Agent 5: Investigation Recommendation Agent

Inputs:
    - fraud_probability
    - risk_score
    - risk_level
    - damage_assessment_summary

Outputs:
    - investigation_recommendation
    - investigation_report
    - status

Dependencies:
    - graph.state.ClaimState
    - utils.report_generator.generate_investigation_report

Integration:
    This agent is compatible with the existing state model and can be invoked after the
    fraud and risk stages without changing the existing workflow or state definition.
"""

from __future__ import annotations

from graph.state import ClaimState
from utils.report_generator import generate_investigation_report


def investigation_agent(state: ClaimState) -> ClaimState:
    """
    Construct the final investigation summary from the risk outputs already present
    in the shared state.
    """
    required_fields = [
        "fraud_probability",
        "risk_score",
        "risk_level",
    ]

    damage_assessment_summary = state.get(
        "damage_assessment_summary",
        "No image assessment available.",
    )

    missing_fields = [field for field in required_fields if state.get(field) is None]
    if missing_fields:
        return {
            "status": "ERROR",
            "error_message": f"Missing required investigation inputs: {', '.join(missing_fields)}",
        }

    report_data = generate_investigation_report(
        fraud_probability=state["fraud_probability"],
        risk_score=state["risk_score"],
        risk_level=state["risk_level"],
        damage_assessment_summary=damage_assessment_summary,
        top_risk_features=state.get("top_risk_features", []),
        claim_data=state.get("claim_data", {}),
        consistency_summary=state.get("consistency_summary", ""),
    )

    updated_state = dict(state)
    updated_state.update(report_data)
    updated_state["status"] = "Investigation Complete"

    return updated_state
