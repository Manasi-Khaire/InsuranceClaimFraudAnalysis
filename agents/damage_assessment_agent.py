"""
Purpose:
    Wraps the image-based damage-assessment proxy and updates the shared claim state.

Related PRD:
    - FR-02: MVP image validation using a pretrained YOLOv8n-based proxy.
    - Section 4: Claim Intake Agent -> Fraud Prediction Agent -> Risk Assessment Agent.

Related Architecture:
    - Agent 2: Damage Assessment Agent (MVP proxy only)

Important design note:
    The MVP explicitly uses a pretrained YOLOv8n model as a proxy for damage assessment.
    It does not claim true dent/scratch/crack detection. A true VehiDE-based damage
    classification model is a future enhancement.

Inputs:
    - state["vehicle_image_path"]

Outputs:
    - damage_detected
    - damage_confidence
    - damage_severity_score
    - damage_assessment_summary
    - status

Dependencies:
    - graph.state.ClaimState
    - utils.image_utils.assess_vehicle_damage

Integration:
    This agent is intentionally compatible with the existing ClaimState and can be
    invoked manually or in future workflows without changing graph/state.py or
    graph/workflow.py.
"""

from __future__ import annotations

from typing import Any, Dict

from graph.state import ClaimState
from utils.image_utils import assess_vehicle_damage


def damage_assessment_agent(state: ClaimState) -> ClaimState:
    """
    Evaluate a local vehicle image using a CPU-only hybrid MVP:
        - pretrained YOLOv8n for vehicle localization
        - OpenCV image-structure analysis for damage proxy scoring

    The function intentionally returns a proxy-level result rather than a claim of
    true damage classification. This keeps the MVP aligned with the PRD and
    architecture documents while supporting future VehiDE integration.
    """
    vehicle_image_path = state.get("vehicle_image_path")

    if not vehicle_image_path:
        return {
            "damage_detected": False,
            "damage_confidence": 0.0,
            "damage_severity_score": 0.0,
            "damage_assessment_summary": "No image provided.",
            "status": "Damage Assessment Complete",
        }

    try:
        assessment = assess_vehicle_damage(vehicle_image_path)
    except Exception as exc:
        return {
            "damage_detected": False,
            "damage_confidence": 0.0,
            "damage_severity_score": 0.0,
            "damage_assessment_summary": (
                "YOLO + OpenCV proxy failed while processing the image. "
                "This is a local demo placeholder and not a true dent/scratch/crack detector."
            ),
            "status": "Damage Assessment Complete",
        }

    updated_state = dict(state)
    updated_state.update(assessment)
    updated_state["damage_area_percentage"] = assessment.get("damage_area_percentage", 0.0)
    updated_state["contour_coverage_percentage"] = assessment.get("contour_coverage_percentage", 0.0)
    updated_state["edge_density"] = assessment.get("edge_density", 0.0)
    updated_state["contour_density"] = assessment.get("contour_density", 0.0)
    updated_state["damaged_area_ratio"] = assessment.get("damaged_area_ratio", 0.0)
    updated_state["damage_severity_label"] = assessment.get("damage_severity_label", "Minor")
    updated_state["status"] = "Damage Assessment Complete"

    return updated_state
