from __future__ import annotations

import streamlit as st

from utils.investigation_platform import ensure_claim_queue, set_selected_claim


st.set_page_config(page_title="Claim Queue", layout="wide")

st.title("Claim Queue")
st.caption("Review active claims as they enter the investigation queue.")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

risk_badges = {
    "LOW": "background: #d4edda; color: #155724;",
    "MEDIUM": "background: #fff3cd; color: #856404;",
    "HIGH": "background: #ffe5d0; color: #9c4f06;",
    "CRITICAL": "background: #f8d7da; color: #842029;",
}

queue = ensure_claim_queue()
queue_table = [
    {
        "Claim ID": claim["claim_id"],
        "Status": claim["status"],
        "Fraud Risk": claim["fraud_risk"],
        "Policy Type": claim["claim_data"].get("PolicyType", "N/A"),
        "Vehicle Category": claim["claim_data"].get("VehicleCategory", "N/A"),
        "Driver Age": claim["claim_data"].get("Age", "N/A"),
    }
    for claim in queue
]

st.dataframe(queue_table, use_container_width=True, hide_index=True)

st.subheader("Open Claim")
for claim in queue:
    with st.container():
        cols = st.columns([2, 1.2, 1.2, 1.5, 1.5])
        with cols[0]:
            st.markdown(f"### {claim['claim_id']}")
            st.write(f"{claim['claim_data'].get('PolicyType', 'N/A')} • {claim['claim_data'].get('VehicleCategory', 'N/A')}")
        with cols[1]:
            st.write("Status")
            st.write(claim["status"])
        with cols[2]:
            st.write("Fraud Risk")
            risk_label = str(claim.get("fraud_risk", "LOW")).upper()
            risk_style = risk_badges.get(risk_label, risk_badges["LOW"])
            st.markdown(
                f"<span style=\"{risk_style}; padding: 5px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;\">{risk_label}</span>",
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.write("Driver Age")
            st.write(claim["claim_data"].get("Age", "N/A"))
        with cols[4]:
            if st.button("Open claim", key=f"open_{claim['claim_id']}", use_container_width=True):
                set_selected_claim(claim["claim_id"])
                st.switch_page("pages/claim_investigation.py")
