"""
Purpose:
Defines the LangGraph node wrappers for Phase 1/Phase 2 agents.
Acts as the interface between the LangGraph StateGraph and the core agent logic.

Dependencies:
- agents.claim_intake_agent
- agents.fraud_prediction_agent
- agents.risk_assessment_agent
- graph.state.ClaimState
"""

from graph.state import ClaimState
from agents.claim_intake_agent import claim_intake_agent
from agents.fraud_prediction_agent import fraud_prediction_agent
from agents.risk_assessment_agent import risk_assessment_agent

def claim_intake_node(state: ClaimState) -> ClaimState:
    """
    Executes the Claim Intake Agent.
    Updates the current_step in the state.
    """
    result = claim_intake_agent(state)
    result["current_step"] = "claim_intake"
    return result

def fraud_prediction_node(state: ClaimState) -> ClaimState:
    """
    Executes the Fraud Prediction Agent.
    Updates the current_step in the state.
    """
    result = fraud_prediction_agent(state)
    result["current_step"] = "fraud_prediction"
    return result

def risk_assessment_node(state: ClaimState) -> ClaimState:
    """
    Executes the Risk Assessment Agent.
    Updates the current_step in the state.
    """
    result = risk_assessment_agent(state)
    result["current_step"] = "risk_assessment"
    return result
