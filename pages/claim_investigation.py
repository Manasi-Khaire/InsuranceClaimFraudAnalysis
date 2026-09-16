from __future__ import annotations

import streamlit as st

from utils.investigation_platform import (
    add_history_entry,
    generate_claim_analysis,
    get_selected_claim,
    update_claim_status,
)

st.set_page_config(page_title="Claim Investigation", layout="wide")

claim = get_selected_claim()
if claim is None:
    st.warning("No claim selected. Please return to the queue and open a claim.")
    st.stop()

st.title(f"Claim Investigation - {claim['claim_id']}")
st.caption(f"Status: {claim.get('status', 'Pending Investigation')}")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] > div { padding: 0.4rem 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

analysis = st.session_state.analysis_cache.get(claim["claim_id"])

if "selected_action" not in st.session_state:
    st.session_state.selected_action = ""

risk_badges = {
    "LOW": "background: #d4edda; color: #155724;",
    "MEDIUM": "background: #fff3cd; color: #856404;",
    "HIGH": "background: #ffe5d0; color: #9c4f06;",
    "CRITICAL": "background: #f8d7da; color: #842029;",
}

if st.button("Generate Investigation Analysis", use_container_width=True):
    progress = st.progress(0)
    status = st.status("Loading Claim Data", state="running")
    progress.progress(20)
    status.update(label="Loading Claim Data")

    progress.progress(40)
    status.update(label="Running Fraud Analysis")
    analysis = generate_claim_analysis(claim)
    st.session_state.analysis_cache[claim["claim_id"]] = analysis
    progress.progress(70)
    status.update(label="Running Damage Assessment")

    progress.progress(85)
    status.update(label="Performing Consistency Review")
    progress.progress(100)
    status.update(label="Generating Investigation Report", state="complete")

    analysis = st.session_state.analysis_cache[claim["claim_id"]]
    st.success("Investigation analysis generated.")
else:
    analysis = st.session_state.analysis_cache.get(claim["claim_id"])

if analysis:
    risk_level = str(analysis.get("risk_level", "LOW")).upper()
    status_style = risk_badges.get(risk_level, risk_badges["LOW"])
    summary = analysis["claim_summary"]

    st.subheader("Claim Summary")
    summary_cols = st.columns(4)
    for idx, key in enumerate(["Claim ID", "Policy Type", "Vehicle Category", "Driver Age"]):
        with summary_cols[idx]:
            st.write(f"**{key}:** {summary[key]}")
    st.write("---")
    second = st.columns(4)
    for idx, key in enumerate(["Previous Claims", "Fault", "Police Report Filed", "Witness Present"]):
        with second[idx]:
            st.write(f"**{key}:** {summary[key]}")
    st.write("---")

    col_left, col_right = st.columns([1.4, 0.9])
    with col_left:
        st.subheader("Claim Details")
        feature_df = []
        for key, value in claim["claim_data"].items():
            feature_df.append({"Feature": key, "Value": value})
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Key Facts")
        summary_fields = [
            ("Driver Age", claim["claim_data"].get("Age", "N/A")),
            ("Deductible", claim["claim_data"].get("Deductible", "N/A")),
            ("Driver Rating", claim["claim_data"].get("DriverRating", "N/A")),
            ("Policy Type", claim["claim_data"].get("PolicyType", "N/A")),
            ("Vehicle Category", claim["claim_data"].get("VehicleCategory", "N/A")),
            ("Past Number of Claims", claim["claim_data"].get("PastNumberOfClaims", "N/A")),
            ("Police Report Filed", claim["claim_data"].get("PoliceReportFiled", "N/A")),
            ("Witness Present", claim["claim_data"].get("WitnessPresent", "N/A")),
            ("Age Of Vehicle", claim["claim_data"].get("AgeOfVehicle", "N/A")),
            ("Fault", claim["claim_data"].get("Fault", "N/A")),
        ]
        for label, value in summary_fields:
            st.metric(label, str(value))

    st.subheader("Vehicle Image")
    image_path = claim.get("image_path")
    if image_path and image_path.strip():
        st.image(image_path, use_container_width=True)
    else:
        st.info("No image attached to this claim.")

    with st.container(border=True):
        st.subheader("Fraud Assessment")
        fraud_cols = st.columns(2)
        with fraud_cols[0]:
            st.markdown(f"**Fraud Probability:** {analysis['fraud_probability']:.0f}%")
        with fraud_cols[1]:
            st.markdown(
                f"<span style=\"{status_style}; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;\">{risk_level}</span>",
                unsafe_allow_html=True,
            )
        st.write(analysis["fraud_assessment_text"])

    with st.container(border=True):
        st.subheader("Evidence Assessment")
        evidence_cols = st.columns(4)
        with evidence_cols[0]:
            st.write(f"**Damage Detected:** {'Yes' if analysis['damage_detected'] else 'No'}")
        with evidence_cols[1]:
            st.write(f"**Affected Area:** {analysis['affected_area']}")
        with evidence_cols[2]:
            st.write(f"**Estimated Affected Area:** {analysis['estimated_damage_area']}%")
        with evidence_cols[3]:
            st.write(f"**Severity:** {analysis['damage_severity']}")
        st.write(analysis["evidence_summary"])

    st.subheader("Key Fraud Indicators")
    for item in analysis["key_indicators"]:
        st.write(f"- {item['label']}: {item['value']} → {item['concern']}")

    with st.container(border=True):
        st.subheader("Consistency Review")
        st.write(analysis["consistency_review"])

    with st.container(border=True):
        st.subheader("AI Recommendation")
        st.info(analysis["recommendation_text"])
        st.caption("This recommendation is generated by the AI Claims Investigation Assistant. Final claim decisions remain the responsibility of the assigned claims investigator.")

    st.subheader("Investigator Decision")
    decision_cols = st.columns(4)
    decisions = ["Approve Claim", "Send For Manual Review", "Escalate To SIU", "Mark Completed"]
    for idx, decision in enumerate(decisions):
        with decision_cols[idx]:
            if st.button(decision, key=f"decision_{decision.replace(' ', '_').lower()}", use_container_width=True):
                st.session_state.selected_action = decision
                st.session_state.investigator_decision = decision
                update_claim_status(claim["claim_id"], decision)
                add_history_entry(
                    claim["claim_id"],
                    decision,
                    analysis["recommendation_text"],
                    analysis["fraud_probability"],
                )
                st.success(f"Decision recorded: {decision}")

    action_row = st.columns(3)
    with action_row[0]:
        if st.button("Start New Investigation", use_container_width=True):
            for key in ["selected_claim_id", "claim_queue", "history_entries", "analysis_cache", "selected_action", "investigator_decision"]:
                st.session_state.pop(key, None)
            st.switch_page("pages/claim_queue.py")
    with action_row[1]:
        if st.button("Reset Current Investigation", use_container_width=True):
            for key in ["selected_claim_id", "claim_queue", "history_entries", "analysis_cache", "selected_action", "investigator_decision"]:
                st.session_state.pop(key, None)
            st.rerun()
    with action_row[2]:
        if st.button("Return To Claim Queue", use_container_width=True):
            st.session_state.pop("selected_claim_id", None)
            st.switch_page("pages/claim_queue.py")

    st.subheader("Download PDF")
    try:
        pdf_bytes = analysis["pdf_bytes"]
        st.download_button(
            label="Download Investigation Report (PDF)",
            data=pdf_bytes,
            file_name=f"{claim['claim_id']}_investigation_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"PDF generation is unavailable: {exc}")

    if st.session_state.get("selected_action"):
        st.success(f"Current decision: {st.session_state.selected_action}")
