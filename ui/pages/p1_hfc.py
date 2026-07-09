"""P1 — HFC Anomaly Detection UI Page."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pyarrow.parquet as pq

from ui.components.helpers import load_p1, p1_predict, metric_row
from config.settings import PROCESSED_DIR, P1


def render():
    st.title("🔴 P1 — HFC Modem Anomaly Detection")
    st.markdown(
        "Ensemble of **Isolation Forest + LOF** detects anomalous gradations "
        "in HFC coaxial metrics. Three injected fault patterns: SNR drift, "
        "power spike, and channel congestion."
    )

    models = load_p1()
    m      = models["metrics"]

    # ── Model metrics ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROC-AUC",    f"{m['roc_auc']:.4f}")
    col2.metric("Precision",   f"{m['precision']:.4f}")
    col3.metric("Recall",      f"{m['recall']:.4f}")
    col4.metric("F1",          f"{m['f1']:.4f}")
    col5.metric("Threshold",   f"{m['threshold']:.4f}")

    # ── Feature importance chart ──────────────────────────────
    st.markdown("---")
    st.markdown("#### Permutation feature importance (AUC drop)")
    st.caption("Larger AUC drop when a feature is shuffled = more important to the ensemble")

    importance = m.get("feature_importance", {})
    if importance:
        feat_df = pd.DataFrame(
            [{"Feature": k, "AUC Drop": v} for k, v in importance.items()]
        ).sort_values("AUC Drop", ascending=True)

        fig = px.bar(
            feat_df, x="AUC Drop", y="Feature", orientation="h",
            color="AUC Drop",
            color_continuous_scale=["#e8f5e9","#ef5350"],
            title="Permutation Feature Importance",
        )
        fig.update_layout(height=350, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Live prediction ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Live anomaly scorer")
    st.caption("Adjust HFC signal values — model scores in real time")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        ds_power  = st.slider("Downstream power (dBmV)", -15.0, 15.0, 0.5, 0.1)
        us_power  = st.slider("Upstream power (dBmV)",    30.0, 55.0, 38.0, 0.1)
        ds_snr    = st.slider("Downstream SNR (dB)",      20.0, 50.0, 38.0, 0.1)
        us_mer    = st.slider("Upstream MER (dB)",        20.0, 48.0, 36.0, 0.1)
        ds_util   = st.slider("DS channel utilisation (%)", 0.0, 100.0, 55.0, 1.0)

    with col_l:
        us_util   = st.slider("US channel utilisation (%)", 0.0, 100.0, 40.0, 1.0)
        peak_flag = st.selectbox("Peak hour (7–10pm)?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    features = {
        "ds_power_level_dbmv":        ds_power,
        "us_power_level_dbmv":        us_power,
        "ds_snr_db":                  ds_snr,
        "us_mer_db":                  us_mer,
        "ds_channel_utilization_pct": ds_util,
        "us_channel_utilization_pct": us_util,
        "ds_power_rolling_std_1h":    abs(ds_power) * 0.1,
        "snr_drop_rate_per_hour":     (38 - ds_snr) / 4 if ds_snr < 38 else 0,
        "us_ds_power_delta":          us_power - (ds_power + 38),
        "channel_util_peak_hour_flag":peak_flag,
    }

    result = p1_predict(models, features)

    with col_r:
        score    = result["anomaly_score"]
        severity = result["severity"]
        color    = {"High": "🔴", "Medium": "🟡", "Normal": "🟢"}[severity]

        st.markdown(f"### {color} {severity} — Score: `{score:.4f}`")
        st.markdown(f"Threshold: `{result['threshold']:.4f}` | "
                    f"Anomaly: **{'Yes' if result['is_anomaly'] else 'No'}**")

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = score,
            title = {"text": "Ensemble Anomaly Score"},
            gauge = {
                "axis": {"range": [0, 1]},
                "bar":  {"color": "#ef5350" if result["is_anomaly"] else "#66bb6a"},
                "steps": [
                    {"range": [0, result["threshold"]], "color": "#e8f5e9"},
                    {"range": [result["threshold"], 1], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "#c62828", "width": 3},
                    "thickness": 0.8,
                    "value": result["threshold"],
                },
            },
            number={"valueformat": ".4f"},
        ))
        fig_g.update_layout(height=280,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Sample data distribution ──────────────────────────────
    st.markdown("---")
    st.markdown("#### SNR distribution — normal vs anomalous readings")
    st.caption("Loaded from processed test set")

    try:
        df_test = pd.read_parquet(PROCESSED_DIR / "p1_test.parquet",
                                  columns=["ds_snr_db", "anomaly_flag"])
        df_sample = df_test.sample(min(5000, len(df_test)), random_state=42)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_sample[df_sample["anomaly_flag"]==0]["ds_snr_db"],
            name="Normal", opacity=0.7, nbinsx=60,
            marker_color="#66bb6a",
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_sample[df_sample["anomaly_flag"]==1]["ds_snr_db"],
            name="Anomaly", opacity=0.7, nbinsx=60,
            marker_color="#ef5350",
        ))
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="Downstream SNR (dB)",
            yaxis_title="Count",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.info(f"Sample data not available: {e}")

    # ── Model explainer ───────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How the ensemble works"):
        st.markdown("""
**Isolation Forest** isolates observations by randomly selecting features and split values.
Anomalies have *shorter average path lengths* in the forest — they are easier to isolate.

**Local Outlier Factor** compares the density of each point to its k nearest neighbours.
A point in a low-density region relative to neighbours gets a high LOF score.

**Ensemble** = 60% IF + 40% LOF (both min-max normalised to [0,1]).
We weight IF higher because HFC faults are mostly global (plant-wide) — IF captures
these better. LOF catches local gradients (a single modem drifting against its neighbours).

**Threshold** is set at the 88th percentile of ensemble scores on the training set.
""")
