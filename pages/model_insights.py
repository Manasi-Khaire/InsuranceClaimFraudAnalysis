from __future__ import annotations

import streamlit as st

from utils.investigation_platform import load_claim_dataset

st.set_page_config(page_title="Model Insights", layout="wide")

st.title("Model Insights")
st.caption("Business-focused explanation of the claim risk patterns used by the investigation workflow.")

df = load_claim_dataset()

risk_summary = df.groupby("FraudFound_P").size().to_dict()
low_risk = risk_summary.get(0, 0)
high_risk = risk_summary.get(1, 0)

col1, col2 = st.columns(2)
with col1:
    st.metric("Low Risk Claims", int(low_risk))
with col2:
    st.metric("Fraudulent Claims", int(high_risk))

st.subheader("Key Investigation Patterns")
pattern_rows = [
    {"Pattern": "Previous claims", "Business Interpretation": "Repeated prior claims often increase scrutiny because they can signal repeat loss behavior."},
    {"Pattern": "No police report", "Business Interpretation": "A missing police report may reduce supporting evidence and increase the need for corroboration."},
    {"Pattern": "No witness", "Business Interpretation": "Limited independent corroboration can increase the need for a more detailed review."},
    {"Pattern": "Third-party fault", "Business Interpretation": "Context matters. Fault assignment can influence how the claim is reviewed, but it is not proof of fraud."},
    {"Pattern": "High vehicle value", "Business Interpretation": "Higher-value claims may warrant closer review because the loss exposure is more significant."},
]
st.dataframe(pattern_rows, use_container_width=True, hide_index=True)

st.subheader("Business Risk Signals")
insights = [
    "Low Concern: clean claim profile with minimal prior loss activity and supporting evidence.",
    "Moderate Concern: repeated claim activity or limited supporting documentation requires closer review.",
    "High Concern: multiple indicators combine to increase investigation priority and escalation consideration.",
    "Contextual Factor: claim features may influence review, but should be interpreted alongside evidence and policy context.",
]
for insight in insights:
    st.write(f"- {insight}")

st.subheader("Random Forest Model Status")
st.write("The fraud model remains unchanged and continues to use the previously trained random forest pipeline. The investigation platform interprets these outputs in business language for investigators.")
