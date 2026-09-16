from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def reset_investigation_state() -> None:
    for key in [
        "selected_claim_id",
        "claim_queue",
        "history_entries",
        "analysis_cache",
        "selected_action",
        "investigator_decision",
    ]:
        st.session_state.pop(key, None)


PAGES = [
    st.Page("pages/claim_queue.py", title="Claim Queue", icon="📋"),
    st.Page("pages/claim_investigation.py", title="Claim Investigation", icon="🔎"),
    st.Page("pages/reports_history.py", title="Reports / History", icon="📄"),
    st.Page("pages/model_insights.py", title="Model Insights", icon="📊"),
]

st.set_page_config(page_title="Insurance Claims Investigation Platform", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.header("Investigation Controls")
    if st.button("Start New Investigation", use_container_width=True):
        reset_investigation_state()
        st.switch_page("pages/claim_queue.py")
    if st.button("Reset Current Investigation", use_container_width=True):
        reset_investigation_state()
        st.rerun()
    if st.button("Return To Claim Queue", use_container_width=True):
        st.session_state.pop("selected_claim_id", None)
        st.switch_page("pages/claim_queue.py")

pg = st.navigation(PAGES)
pg.run()

