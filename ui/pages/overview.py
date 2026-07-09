"""Platform Overview — landing page."""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ui.components.helpers import load_p1, load_p2, load_p3


def render():
    st.title("📡 Broadband Intelligence Platform")
    st.markdown(
        "A unified ML system that traces the causal chain: "
        "**network degradation → Wi-Fi experience decline → customer churn**."
    )

    # ── Platform architecture ─────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔴 P1 — HFC Anomaly Detection")
        st.markdown(
            "**Model:** Isolation Forest + LOF Ensemble  \n"
            "**Data:** 4.32M modem readings @ 15-min  \n"
            "**Task:** Flag anomalous HFC signal gradations  \n"
            "**Features:** 10 (SNR, power, utilisation)"
        )

    with col2:
        st.markdown("### 🟣 P2 — Wi-Fi Experience")
        st.markdown(
            "**Model:** SARIMA(1,0,1)×(1,1,0,24) + ±3σ  \n"
            "**Data:** 3.24M device readings @ hourly  \n"
            "**Task:** 1-hour-ahead forecast + breach flag  \n"
            "**Features:** 10 (experience, RSSI, seasonality)"
        )

    with col3:
        st.markdown("### 🟢 P3 — Churn Prediction")
        st.markdown(
            "**Model:** Dense Neural Network (MLP)  \n"
            "**Data:** 500 customers (13 features)  \n"
            "**Task:** Predict churn + SHAP explanations  \n"
            "**Features:** 12 CRM + 4 from P1/P2 network signals"
        )

    # ── Causal chain diagram ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### Causal feature flow across projects")

    flow_md = """
    ```
    P1 HFC Anomaly Detection
        anomaly_count_30d ────────────────────────────┐
        anomaly_severity_score ──────────────────────┐│
        hfc_anomaly_flag ──────────────────────────► P2 Wi-Fi  
                                                        │         │
                                          wifi_breach_count_30d ─┘
                                          days_since_last_anomaly ─►  P3 Churn
    ```
    """
    st.code(
        "P1 → anomaly_count_30d, hfc_anomaly_flag → P2 Wi-Fi → wifi_breach_count_30d → P3 Churn",
        language="text",
    )

    # ── Live model metrics ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Live model performance")

    try:
        p1 = load_p1()
        p2 = load_p2()
        p3 = load_p3()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("P1 ROC-AUC",   f"{p1['metrics']['roc_auc']:.3f}")
        col2.metric("P1 Recall",     f"{p1['metrics']['recall']:.3f}")
        col3.metric("P2 MAE",        f"{p2['metrics']['forecast_metrics']['mae']:.2f} pts")
        col4.metric("P3 ROC-AUC",   f"{p3['metrics']['roc_auc']:.3f}")
        col5.metric("P3 Recall",     f"{p3['metrics']['recall']:.3f}")

    except Exception as e:
        st.warning(f"Models not yet loaded: {e}")

    # ── Dataset overview ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Dataset summary")

    summary = pd.DataFrame([
        {"Project": "P1 HFC Anomaly",    "Rows": "4,320,000", "Interval": "15 min",
         "Anomaly/Breach Rate": "~11.8%", "Model": "IF + LOF Ensemble"},
        {"Project": "P2 Wi-Fi Experience","Rows": "3,240,000", "Interval": "1 hour",
         "Anomaly/Breach Rate": "~3.9%",  "Model": "SARIMA + ±3σ rule"},
        {"Project": "P3 Churn",           "Rows": "500",       "Interval": "One per customer",
         "Anomaly/Breach Rate": "26% churn", "Model": "MLP Neural Network + SHAP"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Tech stack ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Tech stack")

    cols = st.columns(4)
    cols[0].markdown("**ML**\n- scikit-learn\n- statsmodels\n- SHAP\n- pandas / numpy")
    cols[1].markdown("**Data**\n- Parquet / PyArrow\n- Synthetic generation\n- S3 (artifact store)")
    cols[2].markdown("**Serving**\n- FastAPI\n- Pydantic\n- Docker")
    cols[3].markdown("**UI**\n- Streamlit\n- Plotly\n- Streamlit Cloud")
