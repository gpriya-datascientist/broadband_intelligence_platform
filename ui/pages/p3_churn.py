"""
P3 — Customer Churn Prediction UI Page
========================================
Sections:
  1. Model performance metrics
  2. Single prediction form + churn probability ring + SHAP bar
  3. Batch predictions table with filters + CSV export
  4. Learning curve chart
  5. Model explainer expander
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ui.components.helpers import load_p3, p3_predict, risk_color, risk_label


def render():
    st.title("🟢 P3 — Customer Churn Prediction")
    st.markdown(
        "**MLP Neural Network** predicts churn probability using CRM data enriched "
        "with network quality signals from P1 and P2. "
        "**SHAP** explains each prediction with feature-level attribution."
    )

    models = load_p3()
    m      = models["metrics"]

    # Safe extraction — handle both flat and nested metric structures
    roc_auc   = float(m.get("roc_auc",   m.get("evaluation", {}).get("roc_auc",   0)))
    precision = float(m.get("precision", m.get("evaluation", {}).get("precision", 0)))
    recall    = float(m.get("recall",    m.get("evaluation", {}).get("recall",    0)))
    f1        = float(m.get("f1",        m.get("evaluation", {}).get("f1",        0)))
    threshold = m.get("threshold", 0.35)
    if isinstance(threshold, dict): threshold = threshold.get("value", 0.35)
    threshold = float(threshold)
    feat_list = m.get("features", m.get("features_used", []))

    # ── 1. Model performance ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model performance (held-out test set)")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROC-AUC",   f"{roc_auc:.4f}",
                help="Overall discriminative power (1.0 = perfect)")
    col2.metric("Precision", f"{precision:.4f}",
                help="Of customers we flag, how many actually churn")
    col3.metric("Recall",    f"{recall:.4f}",
                help="Of actual churners, how many we catch")
    col4.metric("F1",        f"{f1:.4f}")
    col5.metric("Threshold", f"{threshold:.4f}",
                help="Tuned for max F1 — lower than 0.5 to prioritise recall")

    # confusion matrix may be flat or nested under evaluation
    _ev = m.get("evaluation", {})
    cm  = m.get("confusion_matrix",
           {"tn": _ev.get("tn",0), "fp": _ev.get("fp",0),
            "fn": _ev.get("fn",0), "tp": _ev.get("tp",0)})
    if cm:
        st.caption(
            f"Confusion matrix — TN: {cm.get('tn',0)}  FP: {cm.get('fp',0)}  "
            f"FN: {cm.get('fn',0)}  TP: {cm.get('tp',0)}"
        )

    # ── 2. Single prediction ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Single customer prediction")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("**CRM features**")
        contract   = st.selectbox("Contract type",
                                  ["Month-to-month", "One year", "Two year"])
        tenure     = st.slider("Tenure (months)", 1, 72, 12)
        charges    = st.slider("Monthly charges ($)", 20.0, 120.0, 65.0, 1.0)
        payment    = st.selectbox("Payment method",
                                  ["Electronic check", "Mailed check",
                                   "Bank transfer", "Credit card"])
        internet   = st.selectbox("Internet service",
                                  ["Fiber optic", "DSL", "No"])
        tech_sup   = st.toggle("Tech support", value=False)
        paperless  = st.toggle("Paperless billing", value=True)
        svc_calls  = st.slider("Service calls (30d)", 0, 10, 1)

        st.markdown("**Network signals (from P1 & P2)**")
        anom_cnt   = st.slider("Modem anomaly count (30d)", 0, 300, 0, 5)
        wifi_breach= st.slider("Wi-Fi breach count (30d)", 0, 100, 0, 2)
        days_since = st.slider("Days since last network event", 0, 90, 90)

    features = {
        "contract_type":              contract,
        "tenure_months":              tenure,
        "monthly_charges":            charges,
        "payment_method":             payment,
        "tech_support_flag":          int(tech_sup),
        "internet_service_type":      internet,
        "paperless_billing_flag":     int(paperless),
        "service_call_frequency_30d": svc_calls,
        "charge_per_tenure_ratio":    round(charges / max(tenure, 1), 4),
        "anomaly_count_30d":          anom_cnt,
        "wifi_breach_count_30d":      wifi_breach,
        "days_since_last_anomaly":    days_since,
    }

    result = p3_predict(models, features)
    prob      = result["churn_probability"]
    threshold = result["threshold"]
    risk      = result["risk_level"]
    icon      = risk_color(prob, threshold)

    with col_result:
        st.markdown(f"### {icon} {risk} Risk")

        # Probability ring
        fig_ring = go.Figure(go.Pie(
            values=[prob, 1 - prob],
            labels=["Churn", "Stay"],
            hole=0.72,
            marker_colors=["#ef5350" if prob >= threshold else "#ffb74d"
                           if prob >= threshold - 0.1 else "#66bb6a", "#e0e0e0"],
            textinfo="none",
        ))
        fig_ring.add_annotation(
            text=f"<b>{prob:.1%}</b>",
            font=dict(size=28),
            showarrow=False,
        )
        fig_ring.update_layout(
            height=240,
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ring, use_container_width=True)

        st.markdown(f"**Churn probability:** `{prob:.4f}` | **Threshold:** `{threshold:.4f}`")

        if prob >= threshold:
            st.error("⚠️ High churn risk — retention action recommended")
            if contract == "Month-to-month":
                st.info("💡 Offer: 2-year contract discount (−15%)")
            elif not tech_sup:
                st.info("💡 Offer: Complimentary 3-month tech support trial")
            else:
                st.info("💡 Action: Proactive account manager call")
        else:
            st.success("✅ Low churn risk — no immediate action needed")

    # ── SHAP bar for this prediction ──────────────────────────
    st.markdown("---")
    st.markdown("#### Feature attribution (SHAP) — why this prediction?")
    st.caption("Global importance from test set · bars show mean |SHAP| value per feature")

    shap_importance = m.get("shap_importance", {})
    if shap_importance:
        shap_df = pd.DataFrame([
            {"Feature": k, "Mean |SHAP|": round(v, 4)}
            for k, v in list(shap_importance.items())[:10]
        ]).sort_values("Mean |SHAP|")

        fig_shap = px.bar(
            shap_df,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            color="Mean |SHAP|",
            color_continuous_scale=["#e3f2fd", "#1565c0"],
            title="Global SHAP Feature Importance",
        )
        fig_shap.update_layout(
            height=360,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # ── 3. Batch predictions table ────────────────────────────
    st.markdown("---")
    st.markdown("#### Batch predictions — test set customers")

    pred_df = models["predictions"].copy()
    pred_df["churn_probability"] = pred_df["churn_prob"].round(4)
    pred_df["risk"] = pred_df["churn_prob"].apply(
        lambda p: risk_label(p, threshold)
    )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect(
            "Risk level", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with col_f2:
        contract_opts = pred_df["contract_type"].unique().tolist() if "contract_type" in pred_df.columns else []
        contract_filter = st.multiselect("Contract", contract_opts, default=contract_opts)
    with col_f3:
        min_prob = st.slider("Min churn probability", 0.0, 1.0, 0.0, 0.05)

    mask = (
        pred_df["risk"].isin(risk_filter) &
        pred_df["churn_probability"].ge(min_prob)
    )
    if contract_opts and contract_filter:
        mask = mask & pred_df["contract_type"].isin(contract_filter)

    filtered = pred_df[mask].copy()

    # Metric cards
    total      = len(filtered)
    high_risk  = int((filtered["risk"] == "High").sum())
    med_risk   = int((filtered["risk"] == "Medium").sum())
    avg_prob   = filtered["churn_probability"].mean() if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", total)
    c2.metric("🔴 High risk",    high_risk)
    c3.metric("🟡 Medium risk",  med_risk)
    c4.metric("Avg churn prob",  f"{avg_prob:.1%}")

    # Display table
    display_cols = [
        c for c in ["customer_id", "churn_probability", "risk",
                    "contract_type", "tenure_months", "monthly_charges",
                    "anomaly_count_30d", "wifi_breach_count_30d", "churn"]
        if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols].sort_values("churn_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    # CSV export
    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered predictions (CSV)",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

    # ── 4. Learning curve ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Training learning curve")
    st.caption("Loss per epoch — gap between train/val indicates overfitting")

    hist_df = models["history"]

    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(
        x=hist_df["epoch"],
        y=hist_df["loss"],
        name="Train loss",
        line=dict(color="#29b6f6", width=2),
    ))
    if "val_loss" in hist_df.columns:
        fig_lc.add_trace(go.Scatter(
            x=hist_df["epoch"],
            y=hist_df["val_loss"],
            name="Val loss",
            line=dict(color="#ef5350", width=2, dash="dash"),
        ))
    if "val_score" in hist_df.columns:
        fig_lc.add_trace(go.Scatter(
            x=hist_df["epoch"],
            y=hist_df["val_score"],
            name="Val score",
            line=dict(color="#66bb6a", width=1.5, dash="dot"),
        ))

    fig_lc.update_layout(
        height=280,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_lc, use_container_width=True)

    # ── 5. Model explainer ───────────────────────────────────
    st.markdown("---")
    with st.expander("📖 Model architecture and key decisions"):
        st.markdown(f"""
**Architecture:** Input({len(feat_list)}) → Dense(128) → BatchNorm → Dropout(0.30)
→ Dense(64) → BatchNorm → Dropout(0.25) → Dense(32) → Dropout(0.20) → Sigmoid

**Class weights:** {m.get('class_weight', {})} — each churn sample contributes ~2.86× more
to the gradient. We use class weights instead of SMOTE because:
- SMOTE on 340 training rows risks overfitting to synthetic minority examples
- Class weights are transparent, interpretable, and natively supported

**Threshold tuning:** Default 0.50 misses many churners on imbalanced data.
We sweep the precision-recall curve and pick the threshold maximising F1.
Tuned threshold `{threshold:.4f}` prioritises recall — missing a churner costs
more than a wasted retention call.

**Multicollinearity fix:** `anomaly_severity_score` was dropped because it has
r=1.000 with `anomaly_count_30d` (it's a normalised version of the same value).
Keeping both would confuse the gradient and inflate importance scores.

**SHAP:** We use `shap.Explainer` (KernelSHAP path) with 100 background samples.
SHAP values are additive: sum of all SHAP values ≈ prediction − baseline.
""")
