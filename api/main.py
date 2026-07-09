"""
Broadband Intelligence Platform — FastAPI Backend
===================================================
Endpoints:
  GET  /health                    — liveness check
  GET  /metrics                   — all three model metrics
  POST /api/p1/predict            — HFC anomaly score
  POST /api/p2/predict            — Wi-Fi breach prediction
  POST /api/p3/predict            — Customer churn probability
  POST /api/p3/batch              — Batch churn predictions
  GET  /api/p3/predictions        — All test predictions for UI
  GET  /api/p3/shap/global        — Global SHAP importance
  GET  /api/p2/forecast-sample    — SARIMA forecast sample for UI

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import MODEL_DIR, P1, P3

app = FastAPI(
    title="Broadband Intelligence API",
    description="Unified ML platform: HFC Anomaly + Wi-Fi Forecasting + Churn Prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING ON STARTUP
# ─────────────────────────────────────────────────────────────

class ModelStore:
    p1_scaler    = None
    p1_if        = None
    p1_lof       = None
    p1_threshold = None
    p1_metrics   = None

    p2_stats     = None
    p2_metrics   = None

    p3_model     = None
    p3_scaler    = None
    p3_encoders  = None
    p3_threshold = None
    p3_metrics   = None
    p3_features  = None


@app.on_event("startup")
def load_models():
    """Load all model artifacts into memory at startup."""
    store = ModelStore

    # P1
    try:
        store.p1_scaler    = joblib.load(MODEL_DIR / "p1_scaler.joblib")
        store.p1_if        = joblib.load(MODEL_DIR / "p1_isolation_forest.joblib")
        store.p1_lof       = joblib.load(MODEL_DIR / "p1_lof.joblib")
        with open(MODEL_DIR / "p1_metrics.json") as f:
            store.p1_metrics = json.load(f)
        store.p1_threshold = store.p1_metrics.get("threshold", 0.5)
        print("[startup] P1 models loaded")
    except Exception as e:
        print(f"[startup] P1 load failed: {e}")

    # P2
    try:
        store.p2_stats = joblib.load(MODEL_DIR / "p2_baseline_stats.joblib")
        with open(MODEL_DIR / "p2_metrics.json") as f:
            store.p2_metrics = json.load(f)
        print("[startup] P2 stats loaded")
    except Exception as e:
        print(f"[startup] P2 load failed: {e}")

    # P3
    try:
        store.p3_model    = joblib.load(MODEL_DIR / "p3_dnn_model.joblib")
        store.p3_scaler   = joblib.load(MODEL_DIR / "p3_scaler.joblib")
        store.p3_encoders = joblib.load(MODEL_DIR / "p3_encoders.joblib")
        with open(MODEL_DIR / "p3_metrics.json") as f:
            store.p3_metrics = json.load(f)
        store.p3_threshold = store.p3_metrics.get("threshold", 0.35)
        store.p3_features  = store.p3_metrics.get("features", [])
        print("[startup] P3 model loaded")
    except Exception as e:
        print(f"[startup] P3 load failed: {e}")


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────

class HFCReading(BaseModel):
    ds_power_level_dbmv:        float = Field(0.5,  description="Downstream power level (dBmV)")
    us_power_level_dbmv:        float = Field(38.0, description="Upstream power level (dBmV)")
    ds_snr_db:                  float = Field(38.0, description="Downstream SNR (dB)")
    us_mer_db:                  float = Field(36.0, description="Upstream MER (dB)")
    ds_channel_utilization_pct: float = Field(55.0, description="Downstream channel utilisation (%)")
    us_channel_utilization_pct: float = Field(40.0, description="Upstream channel utilisation (%)")
    ds_power_rolling_std_1h:    float = Field(0.1,  description="1-hour rolling std of DS power")
    snr_drop_rate_per_hour:     float = Field(0.0,  description="SNR drop rate per hour")
    us_ds_power_delta:          float = Field(0.0,  description="US-DS power delta")
    channel_util_peak_hour_flag:int   = Field(0,    description="1 if peak hour (7-10pm)")


class WiFiReading(BaseModel):
    wifi_experience_score:      float = Field(75.0, description="Wi-Fi experience score (0-100)")
    rolling_mean_7d:            float = Field(75.0, description="7-day rolling mean of experience")
    rolling_std_7d:             float = Field(5.0,  description="7-day rolling std of experience")
    sigma_threshold:            float = Field(3.0,  description="σ multiplier for breach detection")


class CustomerFeatures(BaseModel):
    contract_type:              str   = Field("Month-to-month", description="Contract type")
    tenure_months:              int   = Field(12,  description="Customer tenure in months")
    monthly_charges:            float = Field(65.0,description="Monthly charges ($)")
    payment_method:             str   = Field("Electronic check", description="Payment method")
    tech_support_flag:          int   = Field(0,   description="1 if has tech support")
    internet_service_type:      str   = Field("Fiber optic", description="Internet service type")
    paperless_billing_flag:     int   = Field(1,   description="1 if paperless billing")
    service_call_frequency_30d: int   = Field(2,   description="Service calls in last 30 days")
    charge_per_tenure_ratio:    float = Field(5.0, description="Monthly charge / tenure")
    anomaly_count_30d:          int   = Field(0,   description="HFC anomaly count last 30 days")
    wifi_breach_count_30d:      int   = Field(0,   description="Wi-Fi breach count last 30 days")
    days_since_last_anomaly:    int   = Field(90,  description="Days since last network event")


# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────

def _p1_ensemble_score(X_scaled: np.ndarray) -> float:
    s = ModelStore
    if_raw  = -s.p1_if.score_samples(X_scaled)
    lof_raw = -s.p1_lof.score_samples(X_scaled)
    def _norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
    ens = 0.6 * _norm(if_raw) + 0.4 * _norm(lof_raw)
    return float(ens[0])


def _p3_preprocess(customer: CustomerFeatures) -> np.ndarray:
    s   = ModelStore
    enc = s.p3_encoders
    row = {
        "contract_type":             customer.contract_type,
        "payment_method":            customer.payment_method,
        "internet_service_type":     customer.internet_service_type,
        "tenure_months":             customer.tenure_months,
        "monthly_charges":           customer.monthly_charges,
        "charge_per_tenure_ratio":   customer.charge_per_tenure_ratio,
        "tech_support_flag":         customer.tech_support_flag,
        "paperless_billing_flag":    customer.paperless_billing_flag,
        "service_call_frequency_30d":customer.service_call_frequency_30d,
        "anomaly_count_30d":         customer.anomaly_count_30d,
        "wifi_breach_count_30d":     customer.wifi_breach_count_30d,
        "days_since_last_anomaly":   customer.days_since_last_anomaly,
    }
    for col in P3["categorical_features"]:
        if col in enc and col in row:
            try:
                row[col] = int(enc[col].transform([str(row[col])])[0])
            except Exception:
                row[col] = 0

    ordered = [row[f] for f in s.p3_features if f in row]
    X = np.array([ordered], dtype=float)
    return s.p3_scaler.transform(X)


def _risk_label(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.20: return "High"
    if prob >= threshold:        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "models": {
            "p1_hfc_anomaly": ModelStore.p1_if is not None,
            "p2_wifi":        ModelStore.p2_stats is not None,
            "p3_churn":       ModelStore.p3_model is not None,
        }
    }


@app.get("/metrics")
def get_all_metrics():
    """Return evaluation metrics for all three models."""
    return {
        "p1_hfc_anomaly":  ModelStore.p1_metrics,
        "p2_wifi":         ModelStore.p2_metrics,
        "p3_churn":        ModelStore.p3_metrics,
    }


@app.post("/api/p1/predict")
def predict_hfc_anomaly(reading: HFCReading):
    """Score a single HFC modem reading for anomaly probability."""
    s = ModelStore
    if s.p1_if is None:
        raise HTTPException(503, "P1 model not loaded")

    features = P1["features"]
    X = np.array([[getattr(reading, f, 0) for f in features]])
    X_scaled = s.p1_scaler.transform(X)
    score    = _p1_ensemble_score(X_scaled)
    is_anomaly = score > s.p1_threshold

    return {
        "anomaly_score": round(score, 4),
        "is_anomaly":    bool(is_anomaly),
        "threshold":     round(s.p1_threshold, 4),
        "severity":      "High" if score > s.p1_threshold + 0.15 else
                         "Medium" if is_anomaly else "Normal",
        "model":         "IF+LOF Ensemble",
    }


@app.post("/api/p2/predict")
def predict_wifi_breach(reading: WiFiReading):
    """Check if current Wi-Fi score breaches the ±3σ threshold."""
    sigma = reading.sigma_threshold
    lower = reading.rolling_mean_7d - sigma * reading.rolling_std_7d
    is_breach = reading.wifi_experience_score < lower
    deviation = (reading.rolling_mean_7d - reading.wifi_experience_score) / (
        reading.rolling_std_7d + 1e-6
    )

    return {
        "is_breach":          bool(is_breach),
        "current_score":      reading.wifi_experience_score,
        "lower_bound":        round(lower, 2),
        "deviation_sigmas":   round(float(deviation), 2),
        "sigma_threshold":    sigma,
        "model":              f"±{sigma}σ rule-based",
    }


@app.post("/api/p3/predict")
def predict_churn(customer: CustomerFeatures):
    """Predict churn probability for a single customer with top risk factors."""
    s = ModelStore
    if s.p3_model is None:
        raise HTTPException(503, "P3 model not loaded")

    X_scaled = _p3_preprocess(customer)
    prob     = float(s.p3_model.predict_proba(X_scaled)[0, 1])
    risk     = _risk_label(prob, s.p3_threshold)

    # Simple rule-based top reasons (SHAP in UI layer)
    reasons = []
    if customer.contract_type == "Month-to-month":
        reasons.append("Month-to-month contract (highest churn risk)")
    if customer.tenure_months < 12:
        reasons.append(f"Short tenure ({customer.tenure_months} months)")
    if customer.monthly_charges > 80:
        reasons.append(f"High monthly charges (${customer.monthly_charges})")
    if customer.tech_support_flag == 0:
        reasons.append("No tech support subscription")
    if customer.anomaly_count_30d > 100:
        reasons.append(f"High network anomaly count ({customer.anomaly_count_30d} in 30d)")
    if customer.wifi_breach_count_30d > 20:
        reasons.append(f"Frequent Wi-Fi degradations ({customer.wifi_breach_count_30d} breaches)")

    return {
        "customer_id":      None,
        "churn_probability":round(prob, 4),
        "churn_prediction": bool(prob > s.p3_threshold),
        "risk_level":       risk,
        "threshold":        round(s.p3_threshold, 4),
        "top_reasons":      reasons[:3],
        "model":            "MLP Neural Network",
        "retention_action": (
            "Offer 2-year contract discount" if customer.contract_type == "Month-to-month"
            else "Schedule proactive tech support call" if customer.tech_support_flag == 0
            else "Review pricing plan options"
        ),
    }


@app.post("/api/p3/batch")
def batch_churn(customers: list[CustomerFeatures]):
    """Score a batch of customers — returns sorted by churn probability."""
    results = []
    for i, c in enumerate(customers):
        try:
            r = predict_churn(c)
            r["customer_index"] = i
            results.append(r)
        except Exception as e:
            results.append({"customer_index": i, "error": str(e)})

    results.sort(key=lambda x: x.get("churn_probability", 0), reverse=True)
    return {"predictions": results, "total": len(results)}


@app.get("/api/p3/predictions")
def get_all_predictions(limit: int = 200):
    """Return saved test predictions for the batch UI view."""
    try:
        df = pd.read_parquet(MODEL_DIR / "p3_predictions.parquet")
        df["churn_probability"] = df["churn_prob"].round(4)
        df["risk_level"] = df["churn_prob"].apply(
            lambda p: _risk_label(p, ModelStore.p3_threshold or 0.35)
        )
        cols = [
            "customer_id", "churn_probability", "churn_pred",
            "risk_level", "contract_type", "tenure_months",
            "monthly_charges", "anomaly_count_30d",
            "wifi_breach_count_30d", "churn",
        ]
        available = [c for c in cols if c in df.columns]
        return df[available].head(limit).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, f"Failed to load predictions: {e}")


@app.get("/api/p3/shap/global")
def get_global_shap():
    """Return global SHAP feature importance."""
    s = ModelStore
    if s.p3_metrics is None:
        raise HTTPException(503, "P3 metrics not loaded")
    importance = s.p3_metrics.get("shap_importance", {})
    return {
        "feature_importance": [
            {"feature": k, "mean_abs_shap": round(v, 4)}
            for k, v in importance.items()
        ]
    }


@app.get("/api/p2/forecast-sample")
def get_forecast_sample(limit: int = 168):
    """Return SARIMA forecast sample for Wi-Fi time-series chart."""
    try:
        df = pd.read_parquet(MODEL_DIR / "p2_forecast_sample.parquet")
        df["timestamp"] = df["timestamp"].astype(str)
        return df.head(limit).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, f"Forecast sample unavailable: {e}")
