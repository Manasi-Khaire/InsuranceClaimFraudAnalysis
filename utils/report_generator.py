"""
Purpose:
    Creates the final investigation narrative and recommendation for a claim.

Related PRD:
    - FR-05: Detailed investigation report summarizing findings.
    - Section 4 (Target Architecture): Investigation Recommendation Agent.

Related Architecture:
    - Agent 5: Investigation Recommendation Agent

Important design note:
    The report generator prefers local Ollama with the model phi3:mini when available.
    If Ollama is unavailable, it falls back to a Jinja2-generated report so the MVP
    remains fully local and offline.

Inputs:
    - fraud_probability
    - risk_score
    - risk_level
    - damage_assessment_summary

Outputs:
    - investigation_recommendation
    - investigation_report

Dependencies:
    - Jinja2
    - urllib.request (for local Ollama calls)
    - json
    - typing

Integration:
    This helper is consumed by the investigation agent without modifying the existing
    workflow or claim-state schema. It adds the final investigation outputs to the
    same state object used throughout the project.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict

try:
    from jinja2 import Environment, BaseLoader
except ImportError:  # pragma: no cover
    Environment = None
    BaseLoader = None


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PREFERRED_MODEL = "phi3:mini"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_recommendation(
    fraud_probability: float,
    risk_level: str,
    consistency_review: str = "",
) -> str:
    """Create a deterministic recommendation from fraud probability, risk level, and consistency review."""
    probability = _safe_float(fraud_probability, 0.0)
    normalized_level = str(risk_level or "LOW").upper()
    consistency_text = (consistency_review or "").lower()

    if normalized_level == "LOW":
        return (
            "Recommendation Type: Auto Approval Candidate\n"
            "Reason: Claim information, evidence, and risk assessment indicate low investigation risk.\n"
            "Required Action: Proceed with standard claim processing."
        )
    if normalized_level == "MEDIUM":
        return (
            "Recommendation Type: Manual Review Recommended\n"
            "Reason: Additional review is advised due to moderate investigation indicators.\n"
            "Required Action: Review supporting documentation."
        )
    if normalized_level == "HIGH":
        return (
            "Recommendation Type: Senior Claims Review Recommended\n"
            "Reason: Several elevated-risk indicators justify enhanced claims review.\n"
            "Required Action: Complete senior claims review before final approval."
        )
    if probability < 20:
        return (
            "Recommendation Type: Manual Review Recommended\n"
            "Reason: Additional review is advised due to moderate investigation indicators.\n"
            "Required Action: Review supporting documentation."
        )
    return (
            "Recommendation Type: SIU Investigation Candidate\n"
            "Reason: Multiple significant fraud indicators and inconsistencies require specialized investigation.\n"
            "Required Action: Refer the claim for SIU investigation."
    )


def _build_report_template() -> str:
    """Return the fallback Jinja2 template for the investigation report."""
    return """
Executive Summary
=================
This claim has a fraud probability of {{ fraud_probability }}% and a composite risk score of {{ risk_score }}. The current risk level is {{ risk_level }}.

Fraud Analysis
==============
The fraud model indicates a probability of {{ fraud_probability }}% that this claim is suspicious. This output should be interpreted as a model-based signal and reviewed with supporting documentation.

Damage Assessment Summary
========================
{{ damage_assessment_summary }}

Risk Level
==========
{{ risk_level }}

Recommendation
=============
{{ recommendation }}
""".strip()


def _render_fallback_report(
    fraud_probability: float,
    risk_score: float,
    risk_level: str,
    damage_assessment_summary: str,
    recommendation: str,
) -> str:
    """Render the local Jinja2 fallback report."""
    if Environment is None or BaseLoader is None:
        return (
            "Executive Summary\n=================\n"
            f"Fraud probability: {fraud_probability}%\n"
            f"Risk score: {risk_score}\n"
            f"Risk level: {risk_level}\n\n"
            f"Damage Assessment Summary\n========================\n{damage_assessment_summary}\n\n"
            f"Recommendation\n=============\n{recommendation}\n"
        )

    template = Environment(loader=BaseLoader()).from_string(_build_report_template())
    return template.render(
        fraud_probability=round(float(fraud_probability), 2),
        risk_score=round(float(risk_score), 2),
        risk_level=(risk_level or "UNKNOWN").upper(),
        damage_assessment_summary=damage_assessment_summary,
        recommendation=recommendation,
    )


def _call_ollama_local(
    fraud_probability: float,
    risk_score: float,
    risk_level: str,
    damage_assessment_summary: str,
) -> str:
    """Ask a local Ollama instance to generate a narrative investigation report."""
    prompt = (
        "You are a claims investigation assistant. Generate a concise but informative insurance investigation report with these sections:\n"
        "- Executive Summary\n"
        "- Fraud Analysis\n"
        "- Damage Assessment Summary\n"
        "- Risk Level\n"
        "- Recommendation\n\n"
        f"Fraud probability: {fraud_probability}%\n"
        f"Risk score: {risk_score}\n"
        f"Risk level: {risk_level}\n"
        f"Damage assessment summary: {damage_assessment_summary}\n"
        "Keep the tone professional, factual, and local/offline. Remove technical model metrics and present only business-friendly claim findings."
    )

    payload = json.dumps({
        "model": PREFERRED_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }).encode("utf-8")

    request = urllib.request.Request(OLLAMA_ENDPOINT, data=payload, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=20) as response:
        details = json.loads(response.read().decode("utf-8"))

    if not isinstance(details, dict):
        raise ValueError("Unexpected Ollama response shape")

    generated_text = details.get("response")
    if not generated_text:
        raise ValueError("Ollama returned an empty response")

    return str(generated_text).strip()


def _build_consistency_review(
    fraud_probability: float,
    claim_data: Any,
    damage_assessment_summary: str,
) -> str:
    """Produce investor-friendly language describing claim/evidence consistency."""
    probability = float(fraud_probability or 0.0)
    claim_data_obj = claim_data or {}
    previous_claims = claim_data_obj.get("PastNumberOfClaims", "none")
    police_report = claim_data_obj.get("PoliceReportFiled", "No")
    witness_present = claim_data_obj.get("WitnessPresent", "No")
    damage_text = str(damage_assessment_summary or "").lower()

    factors = []
    if previous_claims not in {"none", "None"}:
        factors.append("previous claim activity")
    if police_report == "No":
        factors.append("lack of police report")
    if witness_present == "No":
        factors.append("no witness statement")
    if "minor" in damage_text or "limited" in damage_text:
        factors.append("limited visible damage")

    factor_text = ", ".join(factors) if factors else "the claim profile and submitted evidence"

    if probability >= 30 or (previous_claims not in {"none", "None"} and (police_report == "No" or witness_present == "No")):
        return (
            "The visual evidence appears inconsistent with the reported claim characteristics. "
            f"Additional review is recommended because {factor_text} do not fully align with the submitted evidence."
        )

    return (
        "The submitted vehicle image shows limited damage and is broadly consistent with the claim characteristics. "
        f"The main factors reviewed were {factor_text}."
    )


def _business_damage_summary(damage_assessment_summary: str) -> str:
    """Keep generated reports limited to investigator-facing damage language."""
    text = str(damage_assessment_summary or "").lower()
    if "critical" in text or "extensive" in text:
        return "Extensive visible damage requiring immediate investigation attention was identified."
    if "severe" in text or "significant" in text:
        return "Significant visible damage affecting multiple vehicle areas was observed."
    if "moderate" in text or "noticeable" in text:
        return "Noticeable visible damage was identified and may require review."
    if "no image" in text or "no damage" in text:
        return "No visible damage was identified in the submitted image."
    return "Limited visible damage identified in the submitted image."


def generate_investigation_report(
    fraud_probability: float,
    risk_score: float,
    risk_level: str,
    damage_assessment_summary: str,
    top_risk_features: Any = None,
    claim_data: Any = None,
    consistency_summary: str = "",
) -> Dict[str, Any]:
    """
    Generate a final investigation recommendation and a human-readable report.
    """
    risk_score_value = _safe_float(risk_score, 0.0)
    risk_level_value = str(risk_level or "LOW").upper()
    business_damage_summary = _business_damage_summary(damage_assessment_summary)
    consistency_review = consistency_summary or _build_consistency_review(
        fraud_probability=fraud_probability,
        claim_data=claim_data,
        damage_assessment_summary=business_damage_summary,
    )
    recommendation = _risk_recommendation(
        fraud_probability=fraud_probability,
        risk_level=risk_level_value,
        consistency_review=consistency_review,
    )

    claim_data_obj = claim_data or {}
    vehicle_category = claim_data_obj.get("VehicleCategory", "vehicle")
    previous_claims = claim_data_obj.get("PastNumberOfClaims", "none")
    policy_type = claim_data_obj.get("PolicyType", "standard policy")
    risk_feature_text = "the structured claim information and submitted evidence"

    key_findings = [
        f"Vehicle category: {vehicle_category}.",
        f"Previous claims pattern: {previous_claims}.",
        f"Policy type: {policy_type}.",
        f"Image assessment: {business_damage_summary}",
        f"Fraud probability: {float(fraud_probability):.2f}%.",
        f"Current risk level: {risk_level_value}.",
        f"Key review factors: {risk_feature_text if risk_feature_text else 'No specific feature list available.'}",
        f"Consistency review: {consistency_review}",
    ]

    report_text = "\n".join([
        "Claim Investigation Summary:",
        *[f"- {finding}" for finding in key_findings],
        "",
        "Recommendation:",
        f"- {recommendation}",
    ])

    try:
        investigation_report = _call_ollama_local(
            fraud_probability=fraud_probability,
            risk_score=risk_score_value,
            risk_level=risk_level_value,
            damage_assessment_summary=business_damage_summary,
        )
        if investigation_report and "Claim Investigation Summary:" not in investigation_report:
            investigation_report = report_text + "\n\n" + investigation_report
    except Exception:
        investigation_report = report_text

    return {
        "investigation_recommendation": recommendation,
        "investigation_report": investigation_report,
    }
