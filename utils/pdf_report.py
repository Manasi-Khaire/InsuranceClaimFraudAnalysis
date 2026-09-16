from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _business_evidence_summary(severity: Any, damage_detected: bool) -> str:
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


def generate_claim_pdf(
    claim_data: Dict[str, Any],
    final_state: Dict[str, Any],
    image_path: Optional[str] = None,
) -> bytes:
    """Create a stakeholder-ready PDF for a claim investigation."""
    output = []
    styles = getSampleStyleSheet()

    pdf_path = Path(__file__).resolve().parent.parent / "reports" / "claim_report.pdf"
    pdf_path.parent.mkdir(exist_ok=True)

    claim_id = final_state.get("claim_id") or claim_data.get("claim_id") or "CLAIM"
    title = f"Insurance Claim Investigation Report - {claim_id}"
    output.append(Paragraph(title, styles["Title"]))
    output.append(Spacer(1, 12))
    output.append(Paragraph("Prepared for stakeholder review", styles["BodyText"]))
    output.append(Spacer(1, 18))

    claim_summary = [
        ["Claim ID", str(claim_id)],
        ["Status", str(final_state.get("status", "Pending"))],
        ["Policy Type", str(claim_data.get("PolicyType", "N/A"))],
        ["Vehicle Category", str(claim_data.get("VehicleCategory", "N/A"))],
        ["Driver Age", str(claim_data.get("Age", "N/A"))],
        ["Previous Claims", str(claim_data.get("PastNumberOfClaims", "N/A"))],
        ["Fault", str(claim_data.get("Fault", "N/A"))],
        ["Police Report Filed", str(claim_data.get("PoliceReportFiled", "N/A"))],
        ["Witness Present", str(claim_data.get("WitnessPresent", "N/A"))],
        ["Age of Vehicle", str(claim_data.get("AgeOfVehicle", "N/A"))],
    ]
    output.append(Paragraph("Claim Summary", styles["Heading2"]))
    table = Table(claim_summary, colWidths=[180, 260])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ])
    )
    output.append(table)
    output.append(Spacer(1, 18))

    if image_path:
        output.append(Paragraph("Vehicle Image", styles["Heading2"]))
        try:
            image = Image(str(image_path), width=240, height=160)
            output.append(image)
        except Exception:
            output.append(Paragraph("Image attachment unavailable.", styles["BodyText"]))
        output.append(Spacer(1, 18))

    output.append(Paragraph("Fraud Assessment", styles["Heading2"]))
    fraud_table = Table(
        [
            ["Fraud Probability", f"{float(final_state.get('fraud_probability', 0.0) or 0.0):.2f}%"],
            ["Risk Level", str(final_state.get("risk_level", "LOW")).upper()],
            ["Assessment", str(final_state.get("fraud_assessment_text", "Assessment pending."))],
        ],
        colWidths=[200, 240],
    )
    fraud_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, colors.grey)]))
    output.append(fraud_table)
    output.append(Spacer(1, 18))

    output.append(Paragraph("Evidence Assessment", styles["Heading2"]))
    evidence_summary = (
        f"Damage Detected: {'Yes' if final_state.get('damage_detected', False) else 'No'}\n"
        f"Affected Area: {final_state.get('affected_area', 'N/A')}\n"
        f"Estimated Affected Area: {float(final_state.get('estimated_damage_area', 0.0) or 0.0):.1f}%\n"
        f"Severity: {final_state.get('damage_severity', 'Minor')}\n"
        f"Assessment: {_business_evidence_summary(final_state.get('damage_severity', 'Minor'), bool(final_state.get('damage_detected', False)))}"
    )
    output.append(Paragraph(evidence_summary.replace("\n", "<br/>"), styles["BodyText"]))
    output.append(Spacer(1, 18))

    output.append(Paragraph("Consistency Review", styles["Heading2"]))
    output.append(Paragraph(str(final_state.get("consistency_review", "Claim information and submitted evidence appear broadly consistent.")), styles["BodyText"]))
    output.append(Spacer(1, 18))

    output.append(Paragraph("Recommendation", styles["Heading2"]))
    output.append(Paragraph(str(final_state.get("recommendation_text", "No recommendation available.")).replace("\n", "<br/>"), styles["BodyText"]))
    output.append(Spacer(1, 18))

    output.append(Paragraph("Investigator Decision", styles["Heading2"]))
    investigator_decision = final_state.get("investigator_decision") or "Pending decision"
    output.append(Paragraph(str(investigator_decision), styles["BodyText"]))
    output.append(Spacer(1, 18))

    output.append(Paragraph("Disclaimer", styles["Heading2"]))
    output.append(
        Paragraph(
            "This recommendation is generated by the AI Claims Investigation Assistant. Final claim decisions remain the responsibility of the assigned claims investigator.",
            styles["BodyText"],
        )
    )

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    document.build(output)
    return pdf_path.read_bytes()
