"""
Broadband Intelligence Platform — Streamlit UI
================================================
Multi-page app with three project dashboards.
Run: streamlit run ui/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(
    page_title="Broadband Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 📡 Broadband Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Platform Overview",
            "🔴 P1 — HFC Anomaly Detection",
            "🟣 P2 — Wi-Fi Experience",
            "🟢 P3 — Churn Prediction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("ML Platform · v1.0.0")
    st.caption("Data Science & AI Engineering")

# Route to page
if page == "🏠 Platform Overview":
    from ui.pages.overview import render
    render()
elif page == "🔴 P1 — HFC Anomaly Detection":
    from ui.pages.p1_hfc import render
    render()
elif page == "🟣 P2 — Wi-Fi Experience":
    from ui.pages.p2_wifi import render
    render()
elif page == "🟢 P3 — Churn Prediction":
    from ui.pages.p3_churn import render
    render()
