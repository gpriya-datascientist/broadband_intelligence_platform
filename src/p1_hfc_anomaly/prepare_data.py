"""
Project 1 — HFC Anomaly Detection
Data Preparation & Cleaning Pipeline
=====================================
Steps:
  1. Load raw parquet (partitioned by date)
  2. Schema validation — correct dtypes
  3. Range validation — flag out-of-spec readings
  4. Missing value audit and imputation
  5. Duplicate detection and removal
  6. Outlier capping (IQR-based, separate from anomaly label)
  7. Feature engineering
  8. Train/test split (time-aware)
  9. Save processed datasets + cleaning report

Run:
    python -m src.p1_hfc_anomaly.prepare_data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json

from config.settings import RAW_DIR, PROCESSED_DIR, P1
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p1_prepare.log")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────

def load_raw(sample_rows: int = None) -> pd.DataFrame:
    """
    Load HFC parquet dataset.
    Uses stratified sampling in dev; full load in production.
    """
    log.info("Step 1 — Loading raw HFC parquet data")

    cols_needed = (
        ["timestamp", "modem_id", "customer_id", "anomaly_flag"]
        + P1["features"]
    )

    table = pq.read_table(
        str(RAW_DIR / "hfc_metrics"),
        columns=cols_needed,
    )
    df = table.to_pandas()

    if sample_rows:
        # Stratified sample preserving anomaly ratio
        normal  = df[df["anomaly_flag"] == 0].sample(
            n=min(sample_rows, len(df[df["anomaly_flag"] == 0])),
            random_state=42,
        )
        anomaly = df[df["anomaly_flag"] == 1].sample(
            n=min(sample_rows // 5, len(df[df["anomaly_flag"] == 1])),
            random_state=42,
        )
        df = pd.concat([normal, anomaly]).sample(frac=1, random_state=42)
        log.info("  Sampled %d rows (stratified)", len(df))
    else:
        log.info("  Loaded %d rows (full dataset)", len(df))

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Enforce correct data types.
    Wrong dtypes can cause silent errors in sklearn/statsmodels.
    """
    log.info("Step 2 — Schema validation")

    expected_dtypes = {
        "timestamp":                     "datetime64[ns]",
        "modem_id":                      "object",
        "customer_id":                   "object",
        "ds_power_level_dbmv":           "float64",
        "us_power_level_dbmv":           "float64",
        "ds_snr_db":                     "float64",
        "us_mer_db":                     "float64",
        "ds_channel_utilization_pct":    "float64",
        "us_channel_utilization_pct":    "float64",
        "ds_power_rolling_std_1h":       "float64",
        "snr_drop_rate_per_hour":        "float64",
        "us_ds_power_delta":             "float64",
        "channel_util_peak_hour_flag":   "int64",
        "anomaly_flag":                  "int64",
    }

    issues = []
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            issues.append(f"MISSING column: {col}")
            continue
        actual = str(df[col].dtype)
        if "datetime" in expected and "datetime" not in actual:
            df[col] = pd.to_datetime(df[col])
            issues.append(f"CAST {col}: {actual} → datetime64")
        elif "float" in expected and "float" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            issues.append(f"CAST {col}: {actual} → float64")
        elif "int" in expected and "int" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            issues.append(f"CAST {col}: {actual} → int64")

    report["schema_issues"] = issues
    log.info("  Schema issues found and corrected: %d", len(issues))
    for iss in issues:
        log.debug("    %s", iss)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — RANGE VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Flag readings outside physically plausible ranges.
    These are NOT anomalies — they are sensor errors or data corruption.
    Key distinction: a modem with ds_snr = -999 is a bad sensor reading,
    not a real network anomaly. We null them before imputation.
    """
    log.info("Step 3 — Range validation (nulling impossible sensor values)")

    range_nulled = {}
    for col, (lo, hi) in P1["valid_ranges"].items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = int(mask.sum())
        if n_bad > 0:
            df.loc[mask, col] = np.nan
            range_nulled[col] = n_bad
            log.info("  Nulled %d out-of-range values in %s [%s, %s]", n_bad, col, lo, hi)

    report["range_validation"] = range_nulled
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — MISSING VALUE AUDIT & IMPUTATION
# ─────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Audit nulls then impute per column strategy.

    Strategy rationale:
    - Rolling std / drop rate: fill 0 (no history = no variance)
    - Numeric signals: forward-fill within modem then median fallback
      (forward-fill preserves time continuity better than mean/median)
    - Flag columns: fill 0 (absence of flag = normal)
    """
    log.info("Step 4 — Missing value audit and imputation")

    null_audit = df.isnull().sum()
    null_cols  = null_audit[null_audit > 0]
    report["null_audit"] = null_audit.to_dict()

    if len(null_cols) == 0:
        log.info("  No missing values found")
        return df

    log.info("  Columns with nulls:")
    for col, cnt in null_cols.items():
        pct = cnt / len(df) * 100
        log.info("    %-40s %d (%.2f%%)", col, cnt, pct)

    # Zero-fill engineered features (rolling computations)
    zero_fill_cols = ["ds_power_rolling_std_1h", "snr_drop_rate_per_hour"]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Forward-fill within modem (preserves time-series continuity)
    signal_cols = [
        "ds_power_level_dbmv", "us_power_level_dbmv",
        "ds_snr_db", "us_mer_db",
        "ds_channel_utilization_pct", "us_channel_utilization_pct",
        "us_ds_power_delta",
    ]
    df = df.sort_values(["modem_id", "timestamp"])
    df[signal_cols] = (
        df.groupby("modem_id")[signal_cols]
        .transform(lambda x: x.ffill())
    )

    # Median fallback for any remaining nulls
    for col in signal_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.info("  Median fallback applied to %s (median=%.3f)", col, median)

    # Flag columns → 0
    for col in ["channel_util_peak_hour_flag", "anomaly_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    remaining_nulls = df.isnull().sum().sum()
    report["nulls_after_imputation"] = int(remaining_nulls)
    log.info("  Nulls remaining after imputation: %d", remaining_nulls)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Remove exact duplicates and modem+timestamp duplicates.
    A modem can only have one reading per 15-minute window.
    """
    log.info("Step 5 — Duplicate detection and removal")

    n_before = len(df)

    # Exact row duplicates
    df = df.drop_duplicates()
    n_exact = n_before - len(df)

    # Modem + timestamp duplicates (keep last — most recent write wins)
    df = df.sort_values(["modem_id", "timestamp"])
    df = df.drop_duplicates(subset=["modem_id", "timestamp"], keep="last")
    n_ts_dup = n_before - n_exact - len(df)

    report["duplicates_removed"] = {
        "exact_rows": n_exact,
        "modem_timestamp_duplicates": n_ts_dup,
        "rows_after": len(df),
    }
    log.info("  Exact duplicates removed:           %d", n_exact)
    log.info("  Modem+timestamp duplicates removed: %d", n_ts_dup)
    log.info("  Rows after deduplication:           %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — OUTLIER CAPPING (IQR-based)
# ─────────────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Cap extreme values using IQR method on NORMAL readings only.

    IMPORTANT: We compute IQR on normal readings (anomaly_flag==0)
    so that genuine anomalies are not included in the percentile
    calculation. This avoids the IQR being inflated by the very
    signals we want to detect.

    Capping, not removal: we preserve the row but clip to
    [Q1 - 3*IQR, Q3 + 3*IQR]. Using 3× instead of 1.5× because
    telecom signals have heavy but legitimate tails.
    """
    log.info("Step 6 — Outlier capping (IQR, 3x, computed on normal readings)")

    cap_cols = [
        "ds_power_level_dbmv", "us_power_level_dbmv",
        "ds_snr_db", "us_mer_db",
        "ds_power_rolling_std_1h", "snr_drop_rate_per_hour",
        "us_ds_power_delta",
    ]

    normal_df   = df[df["anomaly_flag"] == 0]
    cap_summary = {}

    for col in cap_cols:
        if col not in df.columns:
            continue
        q1  = normal_df[col].quantile(0.25)
        q3  = normal_df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 3 * iqr
        hi  = q3 + 3 * iqr

        n_capped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col]  = df[col].clip(lo, hi)

        cap_summary[col] = {"lower": round(lo, 4), "upper": round(hi, 4), "n_capped": n_capped}
        log.debug("  %s clipped to [%.3f, %.3f] — %d values capped", col, lo, hi, n_capped)

    report["outlier_capping"] = cap_summary
    total_capped = sum(v["n_capped"] for v in cap_summary.values())
    log.info("  Total values capped across all columns: %d", total_capped)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 7 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Recompute rolling and derived features on the CLEANED signals.
    After capping/imputation the rolling values from raw data
    may have been computed on dirty values — recompute on clean.
    """
    log.info("Step 7 — Recomputing engineered features on cleaned signals")

    df = df.sort_values(["modem_id", "timestamp"])

    # Rolling std of downstream power (1-hour = 4 × 15-min windows)
    df["ds_power_rolling_std_1h"] = (
        df.groupby("modem_id")["ds_power_level_dbmv"]
        .transform(lambda x: x.rolling(4, min_periods=1).std())
        .fillna(0)
    )

    # Rate of SNR change per hour (mean of 4-window diff)
    df["snr_drop_rate_per_hour"] = (
        df.groupby("modem_id")["ds_snr_db"]
        .transform(lambda x: x.diff().rolling(4, min_periods=1).mean())
        .fillna(0)
    )

    # Upstream-downstream power delta (imbalance signal)
    df["us_ds_power_delta"] = (
        df["us_power_level_dbmv"] - (df["ds_power_level_dbmv"] + 38.0)
    ).round(3)

    engineered = [
        "ds_power_rolling_std_1h",
        "snr_drop_rate_per_hour",
        "us_ds_power_delta",
    ]
    report["engineered_features"] = engineered
    log.info("  Recomputed: %s", engineered)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 8 — TRAIN / TEST SPLIT (time-aware)
# ─────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on time boundary (not random) to prevent data leakage.
    Training = first 80% of timeline.
    Test     = last 20% of timeline.

    Why time-aware and not random?
    Random split leaks future readings into training — the model
    would see data from the same time window in both sets,
    inflating evaluation metrics unrealistically.
    """
    log.info("Step 8 — Time-aware train/test split (80/20)")

    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.80)
    split_ts  = df["timestamp"].iloc[split_idx]

    df_train = df[df["timestamp"] <  split_ts]
    df_test  = df[df["timestamp"] >= split_ts]

    report["split"] = {
        "split_timestamp": str(split_ts),
        "train_rows": len(df_train),
        "test_rows":  len(df_test),
        "train_anomaly_rate": round(df_train["anomaly_flag"].mean(), 4),
        "test_anomaly_rate":  round(df_test["anomaly_flag"].mean(),  4),
    }

    log.info("  Split at: %s", split_ts)
    log.info("  Train: %d rows | anomaly rate: %.1f%%",
             len(df_train), df_train["anomaly_flag"].mean() * 100)
    log.info("  Test : %d rows | anomaly rate: %.1f%%",
             len(df_test),  df_test["anomaly_flag"].mean()  * 100)
    return df_train, df_test


# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE
# ─────────────────────────────────────────────────────────────

def save_processed(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    report:   dict,
) -> None:
    log.info("Step 9 — Saving processed data and cleaning report")

    df_train.to_parquet(PROCESSED_DIR / "p1_train.parquet", index=False)
    df_test.to_parquet( PROCESSED_DIR / "p1_test.parquet",  index=False)

    report["output_files"] = {
        "train": str(PROCESSED_DIR / "p1_train.parquet"),
        "test":  str(PROCESSED_DIR / "p1_test.parquet"),
    }

    with open("data/reports/p1_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("  Saved: p1_train.parquet (%d rows)", len(df_train))
    log.info("  Saved: p1_test.parquet  (%d rows)", len(df_test))
    log.info("  Saved: data/reports/p1_cleaning_report.json")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(sample_rows: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("P1 — Data Preparation & Cleaning Pipeline")
    log.info("=" * 60)

    report = {}

    df              = load_raw(sample_rows)
    df              = validate_schema(df, report)
    df              = validate_ranges(df, report)
    df              = handle_missing(df, report)
    df              = remove_duplicates(df, report)
    df              = cap_outliers(df, report)
    df              = engineer_features(df, report)
    df_train, df_test = time_split(df, report)
    save_processed(df_train, df_test, report)

    log.info("=" * 60)
    log.info("P1 preparation complete")
    log.info("=" * 60)
    return df_train, df_test


if __name__ == "__main__":
    run_pipeline(sample_rows=200_000)
