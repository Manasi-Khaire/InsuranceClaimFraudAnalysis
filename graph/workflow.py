"""
Purpose:
Compiles the LangGraph StateGraph connecting the Phase 2 agents sequentially.

Related PRD Section: 4. Agentic Workflow (MVP Implementation Workflow)
Related Architecture Section: 4. MVP Sprint Plan

Dependencies:
- langgraph
- graph.state.ClaimState
- graph.nodes.*
"""

from langgraph.graph import StateGraph, END
from graph.state import ClaimState
from graph.nodes import claim_intake_node, fraud_prediction_node, risk_assessment_node

def build_workflow():
    """
    Builds the sequential LangGraph workflow.
    
    Flow:
    Claim Intake -> Fraud Prediction -> Risk Assessment -> END
    """
    
    # Initialize the graph with the structured state
    workflow = StateGraph(ClaimState)
    
    # Add nodes using the wrapper functions
    workflow.add_node("claim_intake", claim_intake_node)
    workflow.add_node("fraud_prediction", fraud_prediction_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    
    # Define strictly sequential edges
    workflow.set_entry_point("claim_intake")
    workflow.add_edge("claim_intake", "fraud_prediction")
    workflow.add_edge("fraud_prediction", "risk_assessment")
    workflow.add_edge("risk_assessment", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# Example test invocation block
if __name__ == "__main__":
    import json
    import sys
    import os
    
    # Add project root to path so we can import 'graph' and 'agents' modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Compile the graph
    app = build_workflow()
    
    # 2. Create a dummy state representing a new claim
    # These fields match the exact requirements found in feature_names.json
    dummy_claim_data = {
        "WeekOfMonth": 1,
        "WeekOfMonthClaimed": 2,
        "Age": 45,
        "Deductible": 400,
        "DriverRating": 3,
        "Year": 1994,
        "Month": "Jan",
        "DayOfWeek": "Monday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Jan",
        "Sex": "Male",
        "MaritalStatus": "Single",
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Liability",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "31 to 35",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "BasePolicy": "Liability"
    }

    initial_state = ClaimState(
        claim_data=dummy_claim_data,
        vehicle_image_path=None,
        pipeline_log=[]
    )
    
    print("--- Starting Agentic Workflow ---")
    
    # 3. Invoke the LangGraph workflow
    final_state = app.invoke(initial_state)
    
    print("\n--- Workflow Execution Complete ---\n")
    print(f"Status: {final_state.get('status')}")
    print(f"Error Message: {final_state.get('error_message')}")
    print(f"Fraud Probability: {final_state.get('fraud_probability')}")
    print(f"Risk Score: {final_state.get('risk_score')}")
    print(f"Risk Level: {final_state.get('risk_level')}")
    print(f"Recommendation: {final_state.get('investigation_recommendation')}")
    print(f"Top Risk Features: {final_state.get('top_risk_features')}")
