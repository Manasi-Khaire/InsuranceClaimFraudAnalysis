from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Reports / History", layout="wide")

st.title("Recent Investigations")
st.caption("Claim decisions and investigation outcomes from the active investigation queue.")

badge_colors = {
    "LOW": "background: #d4edda; color: #155724;",
    "MEDIUM": "background: #fff3cd; color: #856404;",
    "HIGH": "background: #ffe5d0; color: #9c4f06;",
    "CRITICAL": "background: #f8d7da; color: #842029;",
}


def outcome_risk_level(decision: str) -> str:
    decision_levels = {
        "Escalate To SIU": "CRITICAL",
        "Send For Manual Review": "MEDIUM",
        "Approve Claim": "LOW",
    }
    return decision_levels.get(str(decision), "LOW")

history = st.session_state.get("history_entries", [])
if not history:
    st.info("No claim decisions have been recorded yet.")
else:
    for entry in history[:6]:
        decision = str(entry.get("decision", "Pending"))
        risk_level = outcome_risk_level(decision)
        badge = badge_colors.get(risk_level, badge_colors["LOW"])
        recommendation = str(entry.get("recommendation", "No recommendation")).replace("\n", " | ")
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid #dfe3e8; border-radius: 12px; padding: 16px; margin-bottom: 12px; background: #ffffff; box-shadow: 0 1px 2px rgba(15,23,42,0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0;">Claim {entry.get('claim_id', 'N/A')}</h4>
                        <span style="{badge}; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;">{risk_level}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 8px;">
                        <div><strong>Status:</strong> {decision}</div>
                        <div><strong>Recommendation:</strong> {recommendation}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.subheader("Investigation Queue")
queue = st.session_state.get("claim_queue", [])
if not queue:
    st.info("No current claim queue available.")
else:
    for claim in queue:
        status = str(claim.get("status", "Pending"))
        decision = st.session_state.get(f"decision_{claim['claim_id']}", "Not recorded")
        recorded_entry = next(
            (item for item in history if item.get("claim_id") == claim.get("claim_id")),
            None,
        )
        decision = recorded_entry.get("decision", decision) if recorded_entry else decision
        risk_label = (
            outcome_risk_level(decision)
            if decision in {"Escalate To SIU", "Send For Manual Review", "Approve Claim"}
            else str(claim.get("fraud_risk", "Low")).upper()
        )
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid #dfe3e8; border-radius: 10px; padding: 12px; margin-bottom: 10px; background: #f8fafc; box-shadow: 0 1px 2px rgba(15,23,42,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{claim.get('claim_id', 'N/A')}</strong>
                        <span style="{badge_colors.get(risk_label, badge_colors['LOW'])}; padding: 5px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;">{risk_label}</span>
                    </div>
                    <div style="margin-top: 8px;">Status: <strong>{status}</strong></div>
                    <div>Decision: <strong>{decision}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
