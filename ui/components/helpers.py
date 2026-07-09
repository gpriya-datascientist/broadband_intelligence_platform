"""Shared UI helpers — model loading and inference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from config.settings import MODEL_DIR, PROCESSED_DIR, P1, P3

# Scaler was fit on all 13 features (CAT + NUM including anomaly_severity_score).
# Model was trained on 12 (anomaly_severity_score dropped due to r=1.0).
# So: scale 13 → drop anomaly_severity_score → feed 12 to model.
SCALER_FEATURES = P3["categorical_features"] + P3["numeric_features"]   # 13
MODEL_FEATURES  = [f for f in SCALER_FEATURES if f != "anomaly_severity_score"]  # 12


# ─────────────────────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading P1 models...")
def load_p1():
    m = json.load(open(MODEL_DIR / "p1_metrics.json"))
    if "metrics" in m and isinstance(m["metrics"], dict):
        m.update(m["metrics"])
    return {
        "scaler":    joblib.load(MODEL_DIR / "p1_scaler.joblib"),
        "if_model":  joblib.load(MODEL_DIR / "p1_isolation_forest.joblib"),
        "lof_model": joblib.load(MODEL_DIR / "p1_lof.joblib"),
        "metrics":   m,
    }


@st.cache_resource(show_spinner="Loading P2 models...")
def load_p2():
    m = json.load(open(MODEL_DIR / "p2_metrics.json"))
    st_block = m.get("stationarity_test", {})
    m["_adf_p"]      = st_block.get("adf_p", st_block.get("p_value", 0.0))
    m["_stationary"] = st_block.get("stationary", True)

    forecast = None
    fp = MODEL_DIR / "p2_forecast_sample.parquet"
    if fp.exists():
        forecast = pd.read_parquet(fp)

    return {
        "stats":    joblib.load(MODEL_DIR / "p2_baseline_stats.joblib"),
        "metrics":  m,
        "forecast": forecast,
    }


@st.cache_resource(show_spinner="Loading P3 models...")
def load_p3():
    m = json.load(open(MODEL_DIR / "p3_metrics.json"))

    # Flatten evaluation sub-dict to top level
    if "evaluation" in m and isinstance(m["evaluation"], dict):
        for k, v in m["evaluation"].items():
            if k not in m:
                m[k] = v

    # Flatten threshold sub-dict
    if "threshold" in m and isinstance(m["threshold"], dict):
        m["threshold"] = m["threshold"].get("value", 0.35)

    return {
        "model":       joblib.load(MODEL_DIR / "p3_dnn_model.joblib"),
        "scaler":      joblib.load(PROCESSED_DIR / "p3_scaler.joblib"),
        "encoders":    joblib.load(PROCESSED_DIR / "p3_encoders.joblib"),
        "metrics":     m,
        "predictions": pd.read_parquet(MODEL_DIR / "p3_predictions.parquet"),
        "shap":        pd.read_parquet(MODEL_DIR / "p3_shap_values.parquet"),
        "history":     pd.read_parquet(MODEL_DIR / "p3_training_history.parquet"),
    }


# ─────────────────────────────────────────────────────────────
# RISK HELPERS
# ─────────────────────────────────────────────────────────────

def risk_color(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.20: return "🔴"
    if prob >= threshold:        return "🟡"
    return "🟢"


def risk_label(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.20: return "High"
    if prob >= threshold:        return "Medium"
    return "Low"


def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


# ─────────────────────────────────────────────────────────────
# P1 INFERENCE
# ─────────────────────────────────────────────────────────────

def p1_predict(models: dict, features: dict) -> dict:
    X  = np.array([[features.get(f, 0) for f in P1["features"]]])
    Xs = models["scaler"].transform(X)

    if_raw  = -models["if_model"].score_samples(Xs)
    lof_raw = -models["lof_model"].score_samples(Xs)

    def _n(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
    score = float(0.6 * _n(if_raw)[0] + 0.4 * _n(lof_raw)[0])
    thr   = float(models["metrics"].get("threshold", 0.5))

    return {
        "anomaly_score": round(score, 4),
        "is_anomaly":    score > thr,
        "threshold":     thr,
        "severity":      "High"   if score > thr + 0.15 else
                         "Medium" if score > thr         else "Normal",
    }


# ─────────────────────────────────────────────────────────────
# P3 INFERENCE
# ─────────────────────────────────────────────────────────────

def p3_predict(models: dict, features: dict) -> dict:
    """
    Inference pipeline:
      1. Encode categoricals using saved LabelEncoders
      2. Build 13-feature row (what scaler expects)
      3. StandardScaler.transform() → 13 scaled features
      4. Drop anomaly_severity_score → 12 features
      5. model.predict_proba() → churn probability
    """
    enc       = models["encoders"]
    scaler    = models["scaler"]
    model     = models["model"]
    m         = models["metrics"]

    threshold = float(m.get("threshold", 0.35))

    # Step 1: encode categoricals
    row = features.copy()
    for col in P3["categorical_features"]:
        if col in enc and col in row:
            try:
                row[col] = int(enc[col].transform([str(row[col])])[0])
            except Exception:
                row[col] = 0

    # anomaly_severity_score not in UI form — derive from anomaly_count
    # (scaler needs it; model drops it afterward)
    anom_cnt = row.get("anomaly_count_30d", 0)
    row["anomaly_severity_score"] = round(min(anom_cnt, 600) / 600, 4)

    # Step 2: build 13-feature vector in scaler order
    X_13 = np.array([[row.get(f, 0) for f in SCALER_FEATURES]], dtype=float)

    # Step 3: scale all 13
    X_13_scaled = scaler.transform(X_13)

    # Step 4: drop anomaly_severity_score column (index 10)
    drop_idx = SCALER_FEATURES.index("anomaly_severity_score")
    X_12 = np.delete(X_13_scaled, drop_idx, axis=1)

    # Step 5: predict
    prob = float(model.predict_proba(X_12)[0, 1])

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  prob > threshold,
        "risk_level":        risk_label(prob, threshold),
        "threshold":         threshold,
    }
