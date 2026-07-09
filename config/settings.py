"""
Central configuration for the Broadband Intelligence Platform.
All paths, constants, and model hyperparameters live here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR     = ROOT_DIR / "data" / "models"
REPORT_DIR    = ROOT_DIR / "data" / "reports"

for _d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Synthetic data parameters ─────────────────────────────────
DATA = {
    "n_modems":      500,
    "n_customers":   500,
    "n_days":         90,
    "hfc_interval_min": 15,      # 15-minute HFC readings
    "wifi_interval_h":   1,      # hourly Wi-Fi readings
    "start_date":    "2024-01-01",
    "random_seed":    42,
}

# ── Project 1 — HFC Anomaly Detection ─────────────────────────
P1 = {
    "features": [
        "ds_power_level_dbmv",
        "us_power_level_dbmv",
        "ds_snr_db",
        "us_mer_db",
        "ds_channel_utilization_pct",
        "us_channel_utilization_pct",
        "ds_power_rolling_std_1h",
        "snr_drop_rate_per_hour",
        "us_ds_power_delta",
        "channel_util_peak_hour_flag",
    ],
    # Normal operating ranges (used in cleaning + validation)
    "valid_ranges": {
        "ds_power_level_dbmv":       (-15,  15),
        "us_power_level_dbmv":       ( 30,  55),
        "ds_snr_db":                 ( 20,  50),
        "us_mer_db":                 ( 20,  48),
        "ds_channel_utilization_pct":(  0, 100),
        "us_channel_utilization_pct":(  0, 100),
    },
    "contamination":  0.12,      # expected anomaly fraction
    "if_n_estimators": 200,
    "lof_n_neighbors":  20,
    "if_weight":        0.60,    # ensemble blend
    "lof_weight":       0.40,
    "anomaly_threshold_percentile": 88,
}

# ── Project 2 — Wi-Fi Anomaly Detection ───────────────────────
P2 = {
    "features": [
        "wifi_experience_score",
        "rssi_dbm",
        "channel_interference_pct",
        "active_device_count",
        "hour_of_day",
        "day_of_week",
        "experience_rolling_mean_7d",
        "experience_rolling_std_7d",
        "hfc_anomaly_flag",
        "device_type_encoded",
    ],
    "valid_ranges": {
        "wifi_experience_score":   (0, 100),
        "rssi_dbm":                (-90, -20),
        "channel_interference_pct":(0,  100),
        "active_device_count":     (0,   50),
    },
    "sarima_order":          (1, 0, 1),
    "sarima_seasonal_order": (1, 1, 0, 24),
    "sigma_threshold":        3.0,
    "min_history_hours":      168,   # 7 days before flagging
}

# ── Project 3 — Churn Prediction ──────────────────────────────
P3 = {
    "categorical_features": [
        "contract_type",
        "payment_method",
        "internet_service_type",
    ],
    "numeric_features": [
        "tenure_months",
        "monthly_charges",
        "charge_per_tenure_ratio",
        "tech_support_flag",
        "paperless_billing_flag",
        "service_call_frequency_30d",
        "anomaly_count_30d",
        "anomaly_severity_score",
        "wifi_breach_count_30d",
        "days_since_last_anomaly",
    ],
    "target": "churn",
    "valid_ranges": {
        "tenure_months":           (0,  500),
        "monthly_charges":         (0, 500),
        "service_call_frequency_30d": (0, 50),
        "anomaly_severity_score":  (0,   1),
    },
    # DNN architecture
    "dnn": {
        "layers":       [128, 64, 32],
        "dropout":      [0.30, 0.25, 0.20],
        "learning_rate": 0.001,
        "batch_size":    32,
        "max_epochs":   100,
        "patience":      15,
    },
    "test_size":  0.20,
    "val_size":   0.15,
    "churn_rate_expected": 0.26,
}

# ── API ────────────────────────────────────────────────────────
API = {
    "host": "0.0.0.0",
    "port":  8000,
    "title": "Broadband Intelligence API",
    "version": "1.0.0",
}
