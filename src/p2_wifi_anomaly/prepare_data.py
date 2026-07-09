"""
Project 2 — Wi-Fi Experience Anomaly Detection
Data Preparation & Cleaning Pipeline
======================================
Steps:
  1. Load raw parquet (partitioned by date)
  2. Schema validation
  3. Range validation — flag impossible Wi-Fi readings
  4. Missing value audit and imputation
  5. Duplicate detection and removal
  6. Device type encoding
  7. Outlier capping (IQR, on normal readings only)
  8. Seasonality feature engineering
  9. Time-aware train/test split (per device)
  10. Save processed datasets + cleaning report

Run:
    python -m src.p2_wifi_anomaly.prepare_data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json

from config.settings import RAW_DIR, PROCESSED_DIR, P2
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p2_prepare.log")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────

def load_raw(customer_ids: list = None) -> pd.DataFrame:
    """
    Load Wi-Fi parquet dataset.
    Optionally filter to a subset of customers for dev/testing.
    """
    log.info("Step 1 — Loading raw Wi-Fi parquet data")

    cols_needed = [
        "timestamp", "customer_id", "device_id", "device_type",
        "hour_of_day", "day_of_week",
        "wifi_experience_score", "rssi_dbm",
        "channel_interference_pct", "active_device_count",
        "experience_rolling_mean_7d", "experience_rolling_std_7d",
        "hfc_anomaly_flag", "forecast_breach_flag",
    ]

    filters = None
    if customer_ids:
        filters = [("customer_id", "in", customer_ids)]

    table = pq.read_table(
        str(RAW_DIR / "wifi_metrics"),
        columns=cols_needed,
        filters=filters,
    )
    df = table.to_pandas()
    log.info("  Loaded %d rows | %d unique devices",
             len(df), df["device_id"].nunique())
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Enforce correct dtypes across all columns."""
    log.info("Step 2 — Schema validation")

    expected = {
        "timestamp":                  "datetime64[ns]",
        "customer_id":                "object",
        "device_id":                  "object",
        "device_type":                "object",
        "hour_of_day":                "int64",
        "day_of_week":                "int64",
        "wifi_experience_score":      "float64",
        "rssi_dbm":                   "float64",
        "channel_interference_pct":   "float64",
        "active_device_count":        "int64",
        "experience_rolling_mean_7d": "float64",
        "experience_rolling_std_7d":  "float64",
        "hfc_anomaly_flag":           "int64",
        "forecast_breach_flag":       "int64",
    }

    issues = []
    for col, exp_type in expected.items():
        if col not in df.columns:
            issues.append(f"MISSING: {col}")
            continue
        actual = str(df[col].dtype)
        if "datetime" in exp_type and "datetime" not in actual:
            df[col] = pd.to_datetime(df[col])
            issues.append(f"CAST {col} → datetime64")
        elif "float" in exp_type and "float" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            issues.append(f"CAST {col} → float64")
        elif "int" in exp_type and "int" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            issues.append(f"CAST {col} → int64")

    report["schema_issues"] = issues
    log.info("  Schema corrections applied: %d", len(issues))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — RANGE VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Null readings outside physically possible Wi-Fi signal ranges.

    RSSI below -90 dBm = device effectively disconnected (sensor error).
    RSSI above -20 dBm = physically impossible (too close to AP).
    Experience score must be 0-100.
    """
    log.info("Step 3 — Range validation")

    range_nulled = {}
    for col, (lo, hi) in P2["valid_ranges"].items():
        if col not in df.columns:
            continue
        mask  = (df[col] < lo) | (df[col] > hi)
        n_bad = int(mask.sum())
        if n_bad > 0:
            df.loc[mask, col] = np.nan
            range_nulled[col] = n_bad
            log.info("  Nulled %d impossible values in %s [%s, %s]", n_bad, col, lo, hi)

    report["range_validation"] = range_nulled
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — MISSING VALUE AUDIT & IMPUTATION
# ─────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Imputation strategy per column:

    - wifi_experience_score : forward-fill within device (time continuity),
                              then device-type median fallback
    - rssi_dbm              : forward-fill within device
    - channel_interference  : fill with device-type median (varies by location)
    - active_device_count   : fill with 1 (minimum plausible household)
    - rolling_mean/std      : recomputed in step 8 — fill 0 for now
    - flag columns          : fill 0 (absence of event)
    """
    log.info("Step 4 — Missing value audit and imputation")

    null_counts = df.isnull().sum()
    report["null_audit"] = null_counts.to_dict()

    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        log.info("  No missing values found")
        return df

    log.info("  Columns with nulls:")
    for col, cnt in null_cols.items():
        log.info("    %-40s %d (%.2f%%)", col, cnt, cnt / len(df) * 100)

    df = df.sort_values(["device_id", "timestamp"])

    # Forward-fill within device
    ffill_cols = ["wifi_experience_score", "rssi_dbm"]
    df[ffill_cols] = (
        df.groupby("device_id")[ffill_cols]
        .transform(lambda x: x.ffill())
    )

    # Device-type median fallback
    for col in ffill_cols + ["channel_interference_pct"]:
        if col in df.columns and df[col].isnull().any():
            medians = df.groupby("device_type")[col].transform("median")
            df[col] = df[col].fillna(medians)

    # Minimum household default
    df["active_device_count"] = df["active_device_count"].fillna(1).astype(int)

    # Rolling stats — will be recomputed; safe to zero-fill for now
    df["experience_rolling_mean_7d"] = df["experience_rolling_mean_7d"].fillna(0)
    df["experience_rolling_std_7d"]  = df["experience_rolling_std_7d"].fillna(1)

    # Flags
    for col in ["hfc_anomaly_flag", "forecast_breach_flag"]:
        df[col] = df[col].fillna(0).astype(int)

    remaining = int(df.isnull().sum().sum())
    report["nulls_after_imputation"] = remaining
    log.info("  Nulls remaining after imputation: %d", remaining)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    A device can only have one reading per hour.
    Duplicate device+timestamp rows indicate pipeline retry issues.
    """
    log.info("Step 5 — Duplicate detection and removal")

    n_before  = len(df)
    df        = df.drop_duplicates()
    n_exact   = n_before - len(df)

    df        = df.sort_values(["device_id", "timestamp"])
    df        = df.drop_duplicates(subset=["device_id", "timestamp"], keep="last")
    n_ts_dup  = n_before - n_exact - len(df)

    report["duplicates_removed"] = {
        "exact_rows": n_exact,
        "device_timestamp_duplicates": n_ts_dup,
        "rows_after": len(df),
    }
    log.info("  Exact duplicates: %d | device+timestamp duplicates: %d", n_exact, n_ts_dup)
    log.info("  Rows after deduplication: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — DEVICE TYPE ENCODING
# ─────────────────────────────────────────────────────────────

def encode_device_type(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Encode device_type as integer for model input.
    laptop = 0, iot = 1

    Kept as a separate step (not folded into schema) so the
    mapping is explicit, auditable, and reproducible.
    """
    log.info("Step 6 — Device type encoding")

    mapping = {"laptop": 0, "iot": 1}
    df["device_type_encoded"] = df["device_type"].map(mapping).fillna(-1).astype(int)

    unmapped = int((df["device_type_encoded"] == -1).sum())
    report["device_type_encoding"] = {"mapping": mapping, "unmapped_rows": unmapped}

    if unmapped > 0:
        log.warning("  %d rows with unknown device_type — encoded as -1", unmapped)
    else:
        log.info("  All device types encoded successfully: %s", mapping)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 7 — OUTLIER CAPPING
# ─────────────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    IQR-based capping on NORMAL readings (breach_flag == 0).
    Using 3× IQR bounds — Wi-Fi signals have legitimate heavy tails
    during peak hours. Same logic as P1 — compute on clean baseline.
    """
    log.info("Step 7 — Outlier capping (IQR 3x, computed on non-breach readings)")

    cap_cols = [
        "wifi_experience_score",
        "rssi_dbm",
        "channel_interference_pct",
        "active_device_count",
    ]

    normal      = df[df["forecast_breach_flag"] == 0]
    cap_summary = {}

    for col in cap_cols:
        if col not in df.columns:
            continue
        q1  = normal[col].quantile(0.25)
        q3  = normal[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 3 * iqr
        hi  = q3 + 3 * iqr

        n_capped    = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col]     = df[col].clip(lo, hi)
        cap_summary[col] = {"lower": round(lo,4), "upper": round(hi,4), "n_capped": n_capped}
        log.debug("  %s clipped [%.2f, %.2f] — %d capped", col, lo, hi, n_capped)

    report["outlier_capping"] = cap_summary
    total = sum(v["n_capped"] for v in cap_summary.values())
    log.info("  Total values capped: %d", total)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 8 — SEASONALITY FEATURES & ROLLING RECOMPUTE
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Recompute rolling mean/std on CLEANED scores.
    Also add seasonality indicators used by SARIMA exogenous inputs.

    Rolling window: 7 days = 168 hourly readings.
    min_periods=24 so we start flagging after 1 full day of history.
    """
    log.info("Step 8 — Seasonality features and rolling stats recompute")

    df = df.sort_values(["device_id", "timestamp"])

    # Recompute 7-day rolling baseline on clean scores
    df["experience_rolling_mean_7d"] = (
        df.groupby("device_id")["wifi_experience_score"]
        .transform(lambda x: x.rolling(168, min_periods=24).mean())
        .fillna(df["wifi_experience_score"])
    )
    df["experience_rolling_std_7d"] = (
        df.groupby("device_id")["wifi_experience_score"]
        .transform(lambda x: x.rolling(168, min_periods=24).std())
        .fillna(1.0)
    )

    # Seasonality indicators
    df["is_peak_hour"]    = ((df["hour_of_day"] >= 19) & (df["hour_of_day"] <= 22)).astype(int)
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]        = np.sin(2 * np.pi * df["hour_of_day"] / 24).round(4)
    df["hour_cos"]        = np.cos(2 * np.pi * df["hour_of_day"] / 24).round(4)
    df["dow_sin"]         = np.sin(2 * np.pi * df["day_of_week"] / 7).round(4)
    df["dow_cos"]         = np.cos(2 * np.pi * df["day_of_week"] / 7).round(4)

    # Recompute lower bound for ±3σ breach (on clean rolling stats)
    df["breach_lower_bound"] = (
        df["experience_rolling_mean_7d"] - 3.0 * df["experience_rolling_std_7d"]
    ).round(3)

    engineered = [
        "experience_rolling_mean_7d", "experience_rolling_std_7d",
        "is_peak_hour", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "breach_lower_bound",
    ]
    report["engineered_features"] = engineered
    log.info("  Features recomputed/added: %s", engineered)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 9 — TIME-AWARE TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on global time boundary: first 80% of dates = train.
    Applied consistently across all devices and customers.

    We do NOT split per-device to avoid identical timestamps
    appearing in both sets from different devices.
    """
    log.info("Step 9 — Time-aware train/test split (80/20 by timestamp)")

    df       = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.80)
    split_ts  = df["timestamp"].iloc[split_idx]

    df_train = df[df["timestamp"] <  split_ts]
    df_test  = df[df["timestamp"] >= split_ts]

    report["split"] = {
        "split_timestamp":     str(split_ts),
        "train_rows":          len(df_train),
        "test_rows":           len(df_test),
        "train_breach_rate":   round(df_train["forecast_breach_flag"].mean(), 4),
        "test_breach_rate":    round(df_test["forecast_breach_flag"].mean(),  4),
        "train_devices":       df_train["device_id"].nunique(),
        "test_devices":        df_test["device_id"].nunique(),
    }

    log.info("  Split at: %s", split_ts)
    log.info("  Train: %d rows | breach rate: %.2f%%",
             len(df_train), df_train["forecast_breach_flag"].mean() * 100)
    log.info("  Test : %d rows | breach rate: %.2f%%",
             len(df_test),  df_test["forecast_breach_flag"].mean()  * 100)
    return df_train, df_test


# ─────────────────────────────────────────────────────────────
# STEP 10 — SAVE
# ─────────────────────────────────────────────────────────────

def save_processed(df_train, df_test, report):
    log.info("Step 10 — Saving processed data and cleaning report")

    df_train.to_parquet(PROCESSED_DIR / "p2_train.parquet", index=False)
    df_test.to_parquet( PROCESSED_DIR / "p2_test.parquet",  index=False)

    report["output_files"] = {
        "train": str(PROCESSED_DIR / "p2_train.parquet"),
        "test":  str(PROCESSED_DIR / "p2_test.parquet"),
    }

    with open("data/reports/p2_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("  Saved: p2_train.parquet (%d rows)", len(df_train))
    log.info("  Saved: p2_test.parquet  (%d rows)", len(df_test))
    log.info("  Saved: data/reports/p2_cleaning_report.json")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(customer_ids: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("P2 — Data Preparation & Cleaning Pipeline")
    log.info("=" * 60)

    report = {}

    df              = load_raw(customer_ids)
    df              = validate_schema(df, report)
    df              = validate_ranges(df, report)
    df              = handle_missing(df, report)
    df              = remove_duplicates(df, report)
    df              = encode_device_type(df, report)
    df              = cap_outliers(df, report)
    df              = engineer_features(df, report)
    df_train, df_test = time_split(df, report)
    save_processed(df_train, df_test, report)

    log.info("=" * 60)
    log.info("P2 preparation complete")
    log.info("=" * 60)
    return df_train, df_test


if __name__ == "__main__":
    # Use first 20 customers for dev run; remove filter for full run
    sample_customers = [f"CUS-{i:04d}" for i in range(20)]
    run_pipeline(customer_ids=sample_customers)
