"""
1. Purpose: Validates that initial claim_data was provided in the state and marks intake as complete.
2. Related PRD Section: 4. Agentic Workflow (Claim Intake Agent)
3. Related Architecture Section: 3. Agent Definitions (Agent 1)
4. Dependencies: graph.state.ClaimState
5. Validation approach: Unit test ensuring missing claim_data sets error_message.
"""

from graph.state import ClaimState

def claim_intake_agent(state: ClaimState) -> ClaimState:
    """
    Validates the presence of the claim_data dictionary containing dataset features.
    Passes data down the LangGraph workflow.
    """
    if "claim_data" not in state or not state["claim_data"]:
        return {
            "status": "ERROR",
            "error_message": "Missing claim_data dictionary in input state."
        }
        
    return {"status": "Intake Complete"}
