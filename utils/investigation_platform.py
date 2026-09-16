from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from agents.damage_assessment_agent import damage_assessment_agent
from agents.fraud_prediction_agent import fraud_prediction_agent
from agents.risk_assessment_agent import risk_assessment_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_dataset" / "fraud.csv"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "sample_claim_images"

CLAIM_IMAGE_MAP = {
    "Low": [
        SAMPLE_IMAGE_DIR / "low_risk_claim_1.jpg",
        SAMPLE_IMAGE_DIR / "low_risk_claim_2.jpg",
    ],
    "Medium": [
        SAMPLE_IMAGE_DIR / "medium_risk_claim_1.jpg",
        SAMPLE_IMAGE_DIR / "medium_risk_claim_2.jpg",
    ],
    "High": [
        SAMPLE_IMAGE_DIR / "high_risk_claim_1.jpg",
        SAMPLE_IMAGE_DIR / "high_risk_claim_2.jpg",
    ],
    "Fraudulent": [
        SAMPLE_IMAGE_DIR / "fraudulent_claim_1.jpg",
        SAMPLE_IMAGE_DIR / "fraudulent_claim_2.jpg",
    ],
}


def fraud_risk_level(fraud_probability: float) -> str:
    """Map model probability to the investigator-facing risk band."""
    probability = max(0.0, min(100.0, float(fraud_probability or 0.0)))
    if probability <= 20:
        return "LOW"
    if probability <= 50:
        return "MEDIUM"
    if probability <= 80:
        return "HIGH"
    return "CRITICAL"


def _coerce_claim_row(row: Dict[str, Any]) -> Dict[str, Any]:
    clean = {str(key): value for key, value in row.items()}
    clean["Fault"] = clean.get("Fault", "Policy Holder")
    clean["PolicyType"] = clean.get("PolicyType", "Sedan - Liability")
    clean["VehicleCategory"] = clean.get("VehicleCategory", "Sedan")
    clean["PastNumberOfClaims"] = clean.get("PastNumberOfClaims", "none")
    clean["PoliceReportFiled"] = clean.get("PoliceReportFiled", "No")
    clean["WitnessPresent"] = clean.get("WitnessPresent", "No")
    clean["AgeOfVehicle"] = clean.get("AgeOfVehicle", "3 years")
    clean["VehiclePrice"] = clean.get("VehiclePrice", "20000 to 29000")
    clean["Age"] = int(clean.get("Age", 0) or 0)
    clean["Deductible"] = int(clean.get("Deductible", 0) or 0)
    clean["DriverRating"] = int(clean.get("DriverRating", 0) or 0)
    return clean


def load_claim_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df


def _select_demo_rows() -> List[Dict[str, Any]]:
    df = load_claim_dataset()
    selected = []
    selected_bands = set()
    for raw_row in df.to_dict("records"):
        row = _coerce_claim_row(raw_row)
        prediction = fraud_prediction_agent({"claim_data": row})
        probability = prediction.get("fraud_probability")
        if probability is None:
            continue
        band = fraud_risk_level(probability)
        if band not in selected_bands:
            selected.append((row, float(probability)))
            selected_bands.add(band)
        if len(selected_bands) == 4:
            break

    selected.sort(key=lambda item: item[1])

    claims = []
    for index, (row, fraud_probability) in enumerate(selected[:4], start=1):
        risk_level = fraud_risk_level(fraud_probability)
        claim = {
            "claim_id": f"CLM-{index:03d}",
            "status": "Pending",
            "fraud_probability": fraud_probability,
            "fraud_risk": risk_level,
            "claim_data": row,
            "documented_claim_status": "Pending Investigation",
        }
        image_candidates = CLAIM_IMAGE_MAP.get(risk_level.title(), list(CLAIM_IMAGE_MAP.values())[0])
        claim["image_path"] = str(image_candidates[0]) if image_candidates else ""
        claims.append(claim)
    return claims


def build_claim_queue() -> List[Dict[str, Any]]:
    queue = _select_demo_rows()
    for claim in queue:
        claim["status"] = claim.get("status", "Pending")
    return queue


def ensure_claim_queue() -> List[Dict[str, Any]]:
    if "claim_queue" not in st.session_state:
        st.session_state.claim_queue = build_claim_queue()
    return st.session_state.claim_queue


def get_claim_by_id(claim_id: str) -> Dict[str, Any] | None:
    queue = ensure_claim_queue()
    for claim in queue:
        if claim.get("claim_id") == claim_id:
            return claim
    return queue[0] if queue else None


def set_selected_claim(claim_id: str) -> None:
    st.session_state.selected_claim_id = claim_id


def get_selected_claim() -> Dict[str, Any] | None:
    queue = ensure_claim_queue()
    selected_id = st.session_state.get("selected_claim_id")
    if selected_id:
        for claim in queue:
            if claim.get("claim_id") == selected_id:
                return claim
    return queue[0] if queue else None


def key_indicator_summary(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    indicators: List[Dict[str, Any]] = []
    previous_claims = claim_data.get("PastNumberOfClaims", "none")
    police_report = claim_data.get("PoliceReportFiled", "No")
    witness = claim_data.get("WitnessPresent", "No")
    fault = claim_data.get("Fault", "Policy Holder")
    vehicle_value = claim_data.get("VehiclePrice", "unknown")

    if previous_claims in {"none", "none"}:
        indicators.append({"label": "Previous Claims", "value": "No Previous Claims", "concern": "Low Concern"})
    elif str(previous_claims).lower() in {"1", "1 claim"}:
        indicators.append({"label": "Previous Claims", "value": "One Prior Claim", "concern": "Moderate Concern"})
    else:
        indicators.append({"label": "Previous Claims", "value": str(previous_claims), "concern": "High Concern"})

    if police_report == "No":
        indicators.append({"label": "Police Report Filed", "value": "Not Filed", "concern": "Moderate Concern"})
    else:
        indicators.append({"label": "Police Report Filed", "value": "Filed", "concern": "Low Concern"})

    if witness == "No":
        indicators.append({"label": "Witness Present", "value": "No Witnesses Reported", "concern": "Moderate Concern"})
    else:
        indicators.append({"label": "Witness Present", "value": "Witnesses Reported", "concern": "Low Concern"})

    if fault == "Third Party":
        indicators.append({"label": "Fault", "value": "Third Party Reported At Fault", "concern": "Contextual Factor"})
    else:
        indicators.append({"label": "Fault", "value": "Policyholder Reported At Fault", "concern": "Contextual Factor"})

    if any(token in str(vehicle_value).lower() for token in ["70000", "50000", "60000", "54000", "75000"]):
        indicators.append({"label": "Vehicle Value", "value": "High Value Vehicle", "concern": "Contextual Factor"})
    else:
        indicators.append({"label": "Vehicle Value", "value": "Standard Value Vehicle", "concern": "Contextual Factor"})

    return indicators[:5]


def _build_consistency_review(claim_data: Dict[str, Any], fraud_probability: float, damage_summary: str) -> str:
    previous_claims = claim_data.get("PastNumberOfClaims", "none")
    witness = claim_data.get("WitnessPresent", "No")
    police = claim_data.get("PoliceReportFiled", "No")
    damage_word = (damage_summary or "").lower()

    if fraud_probability >= 50 or (previous_claims not in {"none", "None"} and witness == "No" and police == "No"):
        return "Significant inconsistencies between claim information and visual evidence were identified."
    if "minor" in damage_word or "limited" in damage_word:
        return "The submitted claim information and supporting image appear broadly consistent."
    if "moderate" in damage_word or "noticeable" in damage_word:
        return "The submitted image and claim characteristics show moderate inconsistencies. Additional review is recommended."
    return "The submitted claim information and supporting image appear broadly consistent."


def _business_evidence_summary(severity: str, damage_detected: bool) -> str:
    if not damage_detected:
        return "No visible damage was identified in the submitted image."

    severity_text = str(severity or "Minor").lower()
    if severity_text == "moderate":
        return "Noticeable visible damage identified and may require review."
    if severity_text == "severe":
        return "Significant visible damage affecting multiple vehicle areas."
    if severity_text == "critical":
        return "Extensive visible damage requiring immediate investigation attention."
    return "Limited visible damage identified in the front bumper region."


def _recommendation_from_analysis(
    fraud_probability: float,
    key_indicators: List[Dict[str, Any]],
    consistency_text: str,
    risk_level: str | None = None,
) -> str:
    normalized_level = str(risk_level or "LOW").upper()

    if normalized_level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        normalized_level = "LOW"

    if normalized_level == "LOW":
        return "Recommendation Type: Auto Approval Candidate\nReason: Claim information, evidence, and risk assessment indicate low investigation risk.\nRequired Action: Proceed with standard claim processing."
    if normalized_level == "MEDIUM":
        return "Recommendation Type: Manual Review Recommended\nReason: Additional review is advised due to moderate investigation indicators.\nRequired Action: Review supporting documentation."
    if normalized_level == "HIGH":
        return "Recommendation Type: Senior Claims Review Recommended\nReason: Several elevated-risk indicators justify enhanced claims review.\nRequired Action: Complete senior claims review before final approval."
    return "Recommendation Type: SIU Investigation Candidate\nReason: Multiple significant fraud indicators and inconsistencies require specialized investigation.\nRequired Action: Refer the claim for SIU investigation."


def _build_pdf_report(analysis: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    claim = analysis["claim"]
    claim_data = claim["claim_data"]
    claim_id = claim.get("claim_id", "CLAIM")
    pdf_buffer = io.BytesIO()
    document = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph("Insurance Claim Investigation Report", styles['Title']))
    story.append(Paragraph(f"Claim ID: {claim_id}", styles['Heading2']))
    story.append(Paragraph("Prepared for stakeholder review", styles['BodyText']))
    story.append(Spacer(1, 18))

    summary_rows = [
        ["Claim Summary", ""],
        ["Status", str(claim.get("status", "Pending"))],
        ["Policy Type", str(claim_data.get("PolicyType", "N/A"))],
        ["Vehicle Category", str(claim_data.get("VehicleCategory", "N/A"))],
        ["Driver Age", str(claim_data.get("Age", "N/A"))],
        ["Previous Claims", str(claim_data.get("PastNumberOfClaims", "N/A"))],
        ["Fault", str(claim_data.get("Fault", "N/A"))],
        ["Police Report Filed", str(claim_data.get("PoliceReportFiled", "N/A"))],
        ["Witness Present", str(claim_data.get("WitnessPresent", "N/A"))],
        ["Age of Vehicle", str(claim_data.get("AgeOfVehicle", "N/A"))],
    ]
    summary_table = Table(summary_rows, colWidths=[200, 250])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#D9EAF7")),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
            ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("1. Fraud Assessment", styles['Heading2']))
    story.append(Paragraph(f"Fraud Probability: {analysis.get('fraud_probability', 0):.0f}%", styles['BodyText']))
    story.append(Paragraph(f"Risk Level: {analysis.get('risk_level', 'LOW')}", styles['BodyText']))
    story.append(Paragraph(analysis.get('fraud_assessment_text', 'Low likelihood of potentially fraudulent activity based on available structured claim information.'), styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Evidence Assessment", styles['Heading2']))
    story.append(Paragraph(f"Damage Detected: {'Yes' if analysis.get('damage_detected', False) else 'No'}", styles['BodyText']))
    story.append(Paragraph(f"Affected Area: {analysis.get('affected_area', 'N/A')}", styles['BodyText']))
    story.append(Paragraph(f"Estimated Affected Area: {analysis.get('estimated_damage_area', 0):.1f}%", styles['BodyText']))
    story.append(Paragraph(f"Severity: {analysis.get('damage_severity', 'Minor')}", styles['BodyText']))
    story.append(Paragraph(
        _business_evidence_summary(
            analysis.get("damage_severity", "Minor"),
            bool(analysis.get("damage_detected", False)),
        ),
        styles['BodyText'],
    ))

    image_path = claim.get("image_path")
    if image_path:
        try:
            story.append(Spacer(1, 12))
            story.append(Image(str(image_path), width=260, height=170))
        except Exception:
            story.append(Paragraph("Image unavailable for this claim.", styles['BodyText']))
    story.append(Spacer(1, 18))

    story.append(Paragraph("3. Key Fraud Indicators", styles['Heading2']))
    for item in analysis.get('key_indicators', []):
        story.append(Paragraph(f"- {item.get('label', '')}: {item.get('value', '')} -> {item.get('concern', '')}", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Consistency Review", styles['Heading2']))
    story.append(Paragraph(analysis.get('consistency_review', 'The submitted claim information and supporting image appear broadly consistent.'), styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. AI Recommendation", styles['Heading2']))
    recommendation = Paragraph(analysis.get('recommendation_text', 'No recommendation available.').replace("\n", "<br/>"), styles['BodyText'])
    story.append(recommendation)
    story.append(Spacer(1, 18))

    story.append(Paragraph("6. Investigator Decision", styles['Heading2']))
    story.append(Paragraph(str(claim.get("status", "Pending")), styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("7. Disclaimer", styles['Heading2']))
    story.append(Paragraph("This recommendation is generated by the AI Claims Investigation Assistant. Final claim decisions remain the responsibility of the assigned claims investigator.", styles['BodyText']))

    document.build(story)
    return pdf_buffer.getvalue()


def generate_claim_analysis(claim: Dict[str, Any]) -> Dict[str, Any]:
    claim_data = claim["claim_data"]
    claim_id = claim["claim_id"]
    image_path = claim.get("image_path") or ""

    state = {
        "claim_data": claim_data,
        "vehicle_image_path": image_path,
        "pipeline_log": [],
    }

    fraud_state = fraud_prediction_agent(state)
    fraud_state = fraud_state if isinstance(fraud_state, dict) else {}
    if image_path:
        damage_state = damage_assessment_agent({**state, **fraud_state})
    else:
        damage_state = {
            "damage_detected": False,
            "damage_summary": "No image provided.",
            "damage_severity_label": "Minor",
            "damage_area_percentage": 0.0,
            "damage_assessment_summary": "No image available for assessment.",
        }

    combined_state = {**state, **fraud_state, **damage_state}
    risk_state = risk_assessment_agent(combined_state)
    combined_state.update(risk_state)

    fraud_probability = float(combined_state.get("fraud_probability", 0.0) or 0.0)
    risk_level = fraud_risk_level(fraud_probability)
    damage_summary = str(combined_state.get("damage_assessment_summary") or "")
    consistency_review = _build_consistency_review(claim_data, fraud_probability, damage_summary)
    indicators = key_indicator_summary(claim_data)
    recommendation_text = _recommendation_from_analysis(
        fraud_probability,
        indicators,
        consistency_review,
        risk_level,
    )

    damage_detected = bool(combined_state.get("damage_detected", False))
    severity = str(combined_state.get("damage_severity_label", "Minor"))
    area_pct = float(combined_state.get("damage_area_percentage", 0.0) or 0.0)
    if area_pct >= 45:
        affected_area = "Multiple Areas"
    elif "rear" in damage_summary.lower():
        affected_area = "Rear"
    elif "side" in damage_summary.lower():
        affected_area = "Side"
    else:
        affected_area = "Front"

    assessment_text = {
        "LOW": "The structured claim information does not currently indicate elevated fraud risk. No significant high-risk indicators were identified.",
        "MEDIUM": "Some claim characteristics warrant additional review before final claim approval.",
        "HIGH": "Several claim characteristics are associated with elevated fraud risk and additional verification is recommended.",
        "CRITICAL": "Multiple claim characteristics indicate significant fraud risk requiring immediate investigation.",
    }
    model_assessment = assessment_text.get(risk_level, assessment_text["LOW"])

    report = {
        "claim": claim,
        "claim_id": claim_id,
        "status": claim.get("status", "Pending"),
        "fraud_probability": round(fraud_probability, 2),
        "risk_level": risk_level,
        "fraud_assessment_text": model_assessment,
        "damage_detected": damage_detected,
        "damage_severity": severity,
        "estimated_damage_area": round(area_pct, 1),
        "affected_area": affected_area,
        "evidence_summary": (
            "Limited visible damage identified in the submitted image."
            if severity.lower() == "minor" else
            "Noticeable visible damage was identified and may require review."
            if severity.lower() == "moderate" else
            "Significant visible damage affecting multiple vehicle areas was observed."
            if severity.lower() == "severe" else
            "Extensive visible damage requiring immediate investigation attention was identified."
        ),
        "key_indicators": indicators,
        "consistency_review": consistency_review,
        "recommendation_text": recommendation_text,
        "claim_summary": {
            "Claim ID": claim_id,
            "Policy Type": claim_data.get("PolicyType", "N/A"),
            "Vehicle Category": claim_data.get("VehicleCategory", "N/A"),
            "Driver Age": claim_data.get("Age", "N/A"),
            "Previous Claims": claim_data.get("PastNumberOfClaims", "N/A"),
            "Fault": claim_data.get("Fault", "N/A"),
            "Police Report Filed": claim_data.get("PoliceReportFiled", "N/A"),
            "Witness Present": claim_data.get("WitnessPresent", "N/A"),
            "Age Of Vehicle": claim_data.get("AgeOfVehicle", "N/A"),
        },
        "pdf_bytes": _build_pdf_report({
            "claim": claim,
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
            "damage_detected": damage_detected,
            "damage_severity": severity,
            "estimated_damage_area": area_pct,
            "affected_area": affected_area,
            "evidence_summary": _business_evidence_summary(severity, damage_detected),
            "key_indicators": indicators,
            "consistency_review": consistency_review,
            "recommendation_text": recommendation_text,
        }),
    }
    return report


def update_claim_status(claim_id: str, decision: str) -> None:
    queue = ensure_claim_queue()
    for claim in queue:
        if claim.get("claim_id") == claim_id:
            if decision == "Approve Claim":
                claim["status"] = "Completed"
            elif decision == "Send For Manual Review":
                claim["status"] = "Under Review"
            elif decision == "Escalate To SIU":
                claim["status"] = "Under Review"
            elif decision == "Mark Completed":
                claim["status"] = "Completed"
            break


def get_history_entries() -> List[Dict[str, Any]]:
    entries = st.session_state.get("history_entries", [])
    return entries


def add_history_entry(claim_id: str, decision: str, recommendation: str, fraud_probability: float) -> None:
    entries = st.session_state.get("history_entries", [])
    entries.insert(0, {
        "claim_id": claim_id,
        "decision": decision,
        "recommendation": recommendation,
        "fraud_probability": round(float(fraud_probability), 2),
    })
    st.session_state.history_entries = entries[:10]
