"""
Project 2 — Wi-Fi Experience Anomaly Detection
Model Training — SARIMA + ±3σ Rule-Based Flagging
===================================================
Steps:
  1. Load processed train/test data (single representative device)
  2. Stationarity test (ADF)
  3. ACF/PACF analysis for order selection
  4. Fit SARIMA model
  5. Residual diagnostics
  6. 1-hour-ahead forecast evaluation (MAE, RMSE, MAPE)
  7. ±3σ breach detection evaluation
  8. Save model artifacts, forecast sample, metrics

Run:
    python -m src.p2_wifi_anomaly.train_model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report,
)

from config.settings import PROCESSED_DIR, MODEL_DIR, P2
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p2_train.log")

SARIMA_ORDER    = tuple(P2["sarima_order"])
SEASONAL_ORDER  = tuple(P2["sarima_seasonal_order"])
SIGMA_THRESHOLD = P2["sigma_threshold"]


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_device_series(device_id: str = None) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load hourly Wi-Fi experience for one device.
    We train a representative model on one device;
    in production each device cluster gets its own model.
    """
    log.info("Step 1 — Loading processed Wi-Fi data")

    df_train = pd.read_parquet(PROCESSED_DIR / "p2_train.parquet")
    df_test  = pd.read_parquet(PROCESSED_DIR / "p2_test.parquet")

    # Use first laptop device as representative model
    if device_id is None:
        device_id = df_train[df_train["device_type"] == "laptop"]["device_id"].iloc[0]

    log.info("  Representative device: %s", device_id)

    train_dev = df_train[df_train["device_id"] == device_id].sort_values("timestamp")
    test_dev  = df_test[df_test["device_id"]   == device_id].sort_values("timestamp")

    train_series = train_dev.set_index("timestamp")["wifi_experience_score"]
    test_series  = test_dev.set_index("timestamp")["wifi_experience_score"]

    log.info("  Train series: %d hours | Test series: %d hours",
             len(train_series), len(test_series))
    log.info("  Train mean: %.2f | std: %.2f", train_series.mean(), train_series.std())

    return train_series, test_series, train_dev, test_dev


# ─────────────────────────────────────────────────────────────
# STEP 2 — STATIONARITY TEST
# ─────────────────────────────────────────────────────────────

def test_stationarity(series: pd.Series, report: dict) -> bool:
    """
    Augmented Dickey-Fuller test for stationarity.
    SARIMA requires the series (after differencing) to be stationary.

    H0: series has a unit root (non-stationary)
    H1: series is stationary

    If p > 0.05 we fail to reject H0 — series may need differencing.
    Our seasonal_order includes d=1 (seasonal differencing) which
    handles the weekly seasonality, so mild non-stationarity is OK.
    """
    log.info("Step 2 — ADF stationarity test")

    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < 0.05

    log.info("  ADF statistic : %.4f", adf_stat)
    log.info("  p-value       : %.4f", p_value)
    log.info("  Stationary    : %s (threshold p < 0.05)", is_stationary)

    if not is_stationary:
        log.info("  → Seasonal differencing in SARIMA order will address this")

    report["stationarity_test"] = {
        "adf_statistic":  round(adf_stat, 4),
        "p_value":        round(p_value, 4),
        "is_stationary":  is_stationary,
        "interpretation": "stationary" if is_stationary else "non-stationary (handled by SARIMA differencing)",
    }
    return is_stationary


# ─────────────────────────────────────────────────────────────
# STEP 3 — ORDER SELECTION NOTE
# ─────────────────────────────────────────────────────────────

def log_order_selection() -> None:
    """
    Document the reasoning for SARIMA order selection.
    In a full pipeline this would run auto_arima or a grid search.
    For this platform we use domain knowledge + AIC/BIC validation.
    """
    log.info("Step 3 — SARIMA order selection")
    log.info("  Order    (p,d,q)     = %s", SARIMA_ORDER)
    log.info("  Seasonal (P,D,Q,m)  = %s", SEASONAL_ORDER)
    log.info("  Rationale:")
    log.info("    p=1: one AR lag captures short-term autocorrelation")
    log.info("    d=0: series stationary after seasonal differencing")
    log.info("    q=1: one MA term for residual smoothing")
    log.info("    P=1: seasonal AR term (weekly pattern)")
    log.info("    D=1: seasonal differencing removes weekly trend")
    log.info("    Q=0: seasonal MA not needed given P=1 fits well")
    log.info("    m=24: 24-hour seasonality (daily cycle)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — FIT SARIMA
# ─────────────────────────────────────────────────────────────

def fit_sarima(train_series: pd.Series, report: dict):
    """
    Fit SARIMA with seasonal differencing for the daily cycle.
    enforce_stationarity=False allows fitting on mildly
    non-stationary series without crashing.
    """
    log.info("Step 4 — Fitting SARIMA%s × %s", SARIMA_ORDER, SEASONAL_ORDER)

    model = SARIMAX(
        train_series,
        order                 = SARIMA_ORDER,
        seasonal_order        = SEASONAL_ORDER,
        enforce_stationarity  = False,
        enforce_invertibility = False,
    )
    result = model.fit(disp=False, maxiter=200)

    log.info("  AIC: %.2f | BIC: %.2f | Log-likelihood: %.2f",
             result.aic, result.bic, result.llf)

    report["sarima_fit"] = {
        "order":          list(SARIMA_ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "aic":            round(result.aic, 2),
        "bic":            round(result.bic, 2),
        "llf":            round(result.llf, 2),
        "n_train":        len(train_series),
    }
    return result


# ─────────────────────────────────────────────────────────────
# STEP 5 — RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def residual_diagnostics(result, report: dict) -> None:
    """
    Check if residuals are white noise — if not, the model
    has not captured all structure in the series.

    Ljung-Box test:
    H0: residuals are uncorrelated (white noise)
    If p < 0.05 at any lag, residuals have autocorrelation —
    model may need more AR/MA terms.
    """
    log.info("Step 5 — Residual diagnostics (Ljung-Box test)")

    residuals = result.resid.dropna()
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)

    lb_pvals = lb_result["lb_pvalue"].values.tolist()
    white_noise = all(p > 0.05 for p in lb_pvals)

    log.info("  Ljung-Box p-values (lags 10, 20): %s",
             [round(p, 4) for p in lb_pvals])
    log.info("  Residuals are white noise: %s", white_noise)

    if not white_noise:
        log.warning("  Residuals show autocorrelation — model may underfit")

    report["residual_diagnostics"] = {
        "ljung_box_pvalues": [round(p, 4) for p in lb_pvals],
        "white_noise":       white_noise,
        "residual_mean":     round(float(residuals.mean()), 4),
        "residual_std":      round(float(residuals.std()),  4),
    }


# ─────────────────────────────────────────────────────────────
# STEP 6 — FORECAST EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_forecast(
    result,
    test_series: pd.Series,
    report:      dict,
) -> np.ndarray:
    """
    Forecast n_test steps ahead and evaluate accuracy.

    MAE  : mean absolute error in experience score points
    RMSE : root mean squared error (penalises large errors)
    MAPE : mean absolute percentage error
    """
    log.info("Step 6 — 1-hour-ahead forecast evaluation")

    n_test   = len(test_series)
    forecast = result.forecast(steps=n_test)

    mae  = mean_absolute_error(test_series, forecast)
    rmse = float(np.sqrt(mean_squared_error(test_series, forecast)))
    mape = float(
        np.mean(np.abs((test_series.values - forecast.values) /
                       (np.abs(test_series.values) + 1e-6))) * 100
    )

    log.info("  MAE  : %.4f  (avg error in score points)", mae)
    log.info("  RMSE : %.4f", rmse)
    log.info("  MAPE : %.2f%%", mape)

    report["forecast_metrics"] = {
        "mae":  round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "n_test": n_test,
    }
    return forecast.values


# ─────────────────────────────────────────────────────────────
# STEP 7 — BREACH DETECTION EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_breach_detection(
    forecast:     np.ndarray,
    test_dev:     pd.DataFrame,
    report:       dict,
) -> dict:
    """
    Apply ±3σ rule to forecast values to flag anomalous hours.

    A breach is flagged when the forecast falls below:
        mean_7d - 3 × std_7d

    This is the key business output — an operations team watches
    this flag to dispatch field engineers before customers complain.

    3σ corresponds to roughly 0.3% false alarm rate on a
    normal distribution. We use lower-bound only (experience
    degradation is the concern, not unusually good Wi-Fi).
    """
    log.info("Step 7 — ±3σ breach detection evaluation")

    test_sorted  = test_dev.sort_values("timestamp").reset_index(drop=True)
    roll_mean    = test_sorted["experience_rolling_mean_7d"].values
    roll_std     = test_sorted["experience_rolling_std_7d"].values
    y_true       = test_sorted["forecast_breach_flag"].values

    n = min(len(forecast), len(roll_mean), len(y_true))
    lower_bound  = roll_mean[:n] - SIGMA_THRESHOLD * roll_std[:n]
    y_pred       = (forecast[:n] < lower_bound).astype(int)
    y_true       = y_true[:n]

    if y_true.sum() > 0:
        rep = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Breach"],
            output_dict=True,
        )
        log.info("  Breach precision: %.4f", rep["Breach"]["precision"])
        log.info("  Breach recall   : %.4f", rep["Breach"]["recall"])
        log.info("  Breach F1       : %.4f", rep["Breach"]["f1-score"])
        log.info("  Business read: catching %.0f%% of Wi-Fi degradations",
                 rep["Breach"]["recall"] * 100)
    else:
        rep = {"Breach": {"precision": 0, "recall": 0, "f1-score": 0}}
        log.warning("  No breach events in test set — breach metrics unavailable")

    breach_metrics = {
        "sigma_threshold":  SIGMA_THRESHOLD,
        "breach_precision": round(rep["Breach"]["precision"], 4),
        "breach_recall":    round(rep["Breach"]["recall"], 4),
        "breach_f1":        round(rep["Breach"]["f1-score"], 4),
        "n_true_breaches":  int(y_true.sum()),
        "n_predicted":      int(y_pred.sum()),
    }
    report["breach_detection"] = breach_metrics

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "lower_bound": lower_bound,
        "forecast": forecast[:n],
        "timestamps": test_sorted["timestamp"].values[:n],
        "actual": test_sorted["wifi_experience_score"].values[:n],
    }


# ─────────────────────────────────────────────────────────────
# STEP 8 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────

def save_artifacts(result, baseline_stats: dict, report: dict, eval_data: dict) -> None:
    log.info("Step 8 — Saving model artifacts")

    result.save(str(MODEL_DIR / "p2_sarima_model.pkl"))
    joblib.dump(baseline_stats, MODEL_DIR / "p2_baseline_stats.joblib")

    # Save forecast sample for UI visualisation
    forecast_df = pd.DataFrame({
        "timestamp":  eval_data["timestamps"],
        "actual":     eval_data["actual"],
        "forecast":   eval_data["forecast"],
        "lower_bound":eval_data["lower_bound"],
        "breach_pred":eval_data["y_pred"],
        "breach_actual":eval_data["y_true"],
    })
    forecast_df.to_parquet(MODEL_DIR / "p2_forecast_sample.parquet", index=False)

    with open(MODEL_DIR / "p2_metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("  Saved: p2_sarima_model.pkl")
    log.info("  Saved: p2_baseline_stats.joblib")
    log.info("  Saved: p2_forecast_sample.parquet")
    log.info("  Saved: p2_metrics.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def train():
    log.info("=" * 60)
    log.info("P2 — Wi-Fi SARIMA Training")
    log.info("=" * 60)

    report = {}

    train_series, test_series, train_dev, test_dev = load_device_series()
    test_stationarity(train_series, report)
    log_order_selection()

    result   = fit_sarima(train_series, report)
    residual_diagnostics(result, report)
    forecast = evaluate_forecast(result, test_series, report)
    eval_data = evaluate_breach_detection(forecast, test_dev, report)

    baseline_stats = {
        "mean":        float(train_series.mean()),
        "std":         float(train_series.std()),
        "sigma":       SIGMA_THRESHOLD,
        "sarima_order": list(SARIMA_ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
    }

    save_artifacts(result, baseline_stats, report, eval_data)

    mae = report["forecast_metrics"]["mae"]
    log.info("=" * 60)
    log.info("P2 training complete — MAE: %.4f", mae)
    log.info("=" * 60)
    return report


if __name__ == "__main__":
    train()
