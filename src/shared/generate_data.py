"""
Broadband Intelligence Platform — Synthetic Data Generator
==========================================================
Generates realistic time-series data for all three projects.

Causal chain:
    P1 HFC modem anomalies
        → degrade Wi-Fi experience (P2)
        → drive customer churn (P3)

Run:
    python -m src.shared.generate_data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import DATA, RAW_DIR, P1
from src.shared.logger import get_logger

log = get_logger(__name__)

np.random.seed(DATA["random_seed"])

N_MODEMS    = DATA["n_modems"]
N_CUSTOMERS = DATA["n_customers"]
N_DAYS      = DATA["n_days"]
START_DATE  = pd.Timestamp(DATA["start_date"])


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_timestamps(freq_minutes: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=START_DATE,
        periods=N_DAYS * (1440 // freq_minutes),
        freq=f"{freq_minutes}min",
    )


def _inject_snr_drift(arr: np.ndarray, start: int, length: int = 16) -> np.ndarray:
    """Gradual SNR drop over ~4 hours — simulates plant/amplifier issue."""
    end = min(start + length, len(arr))
    arr[start:end] += np.linspace(0, -12, end - start)
    return arr


def _inject_power_spike(arr: np.ndarray, start: int, length: int = 4) -> np.ndarray:
    """Sudden upstream power spike — simulates field fault."""
    end = min(start + length, len(arr))
    arr[start:end] += np.random.uniform(6, 10, end - start)
    return arr


def _inject_util_congestion(arr: np.ndarray, start: int, length: int = 8) -> np.ndarray:
    """Sustained channel utilisation surge — simulates peak congestion."""
    end = min(start + length, len(arr))
    arr[start:end] = np.clip(
        arr[start:end] + np.random.uniform(20, 35, end - start), 0, 100
    )
    return arr


# ─────────────────────────────────────────────────────────────
# PROJECT 1 — HFC MODEM METRICS
# ─────────────────────────────────────────────────────────────

def generate_hfc_data() -> tuple[pd.DataFrame, dict]:
    """
    Generate HFC modem metrics at 15-minute intervals.

    Returns
    -------
    df_hfc      : full dataframe
    anomaly_log : dict mapping modem_id → list of anomaly timestamps
    """
    log.info("Generating P1: HFC modem metrics (%d modems × %d days)", N_MODEMS, N_DAYS)

    timestamps = _make_timestamps(DATA["hfc_interval_min"])
    n_ts       = len(timestamps)
    hour_arr   = pd.Series(timestamps).dt.hour.values
    peak_mask  = ((hour_arr >= 19) & (hour_arr <= 22)).astype(int)

    modem_ids    = [f"MDM-{i:04d}" for i in range(N_MODEMS)]
    customer_ids = [f"CUS-{i:04d}" for i in range(N_CUSTOMERS)]
    modem_to_cust = dict(zip(modem_ids, customer_ids))

    anomaly_log: dict[str, list] = {}
    frames = []

    for modem_id in modem_ids:
        # ── Base raw signals ──────────────────────────────────
        ds_power = np.random.normal(0.0,  1.5, n_ts)
        us_power = np.random.normal(38.0, 2.0, n_ts)
        ds_snr   = np.random.normal(38.0, 1.5, n_ts)
        us_mer   = np.random.normal(36.0, 1.5, n_ts)
        ds_util  = np.clip(
            np.random.normal(55, 10, n_ts) + peak_mask * 15, 0, 100
        )
        us_util  = np.clip(
            np.random.normal(40,  8, n_ts) + peak_mask * 10, 0, 100
        )

        # ── Inject three fault pattern types (~4% of readings) ─
        n_faults   = max(1, int(n_ts * 0.04) // 3)
        fault_starts = np.random.choice(n_ts - 20, size=n_faults, replace=False)
        anom_idx: list[int] = []

        for k, fs in enumerate(fault_starts):
            pattern = k % 3
            if pattern == 0:                          # SNR drift (plant issue)
                ds_snr = _inject_snr_drift(ds_snr, fs)
                anom_idx.extend(range(fs, min(fs + 16, n_ts)))
            elif pattern == 1:                        # power spike (field fault)
                us_power = _inject_power_spike(us_power, fs)
                anom_idx.extend(range(fs, min(fs + 4, n_ts)))
            else:                                     # congestion (peak usage)
                ds_util = _inject_util_congestion(ds_util, fs)
                anom_idx.extend(range(fs, min(fs + 8, n_ts)))

        anom_idx_unique = sorted(set(anom_idx))
        anomaly_log[modem_id] = [timestamps[i] for i in anom_idx_unique]

        anomaly_flag          = np.zeros(n_ts, dtype=np.int8)
        anomaly_flag[anom_idx_unique] = 1

        # ── Engineered features ───────────────────────────────
        ds_power_std_1h  = (
            pd.Series(ds_power).rolling(4).std().fillna(0).values
        )
        snr_drop_rate    = (
            pd.Series(ds_snr).diff().rolling(4).mean().fillna(0).values
        )
        us_ds_delta      = us_power - (ds_power + 38.0)

        frames.append(pd.DataFrame({
            "timestamp":                  timestamps,
            "modem_id":                   modem_id,
            "customer_id":                modem_to_cust[modem_id],
            # Raw signals
            "ds_power_level_dbmv":        ds_power.round(3),
            "us_power_level_dbmv":        us_power.round(3),
            "ds_snr_db":                  ds_snr.round(3),
            "us_mer_db":                  us_mer.round(3),
            "ds_channel_utilization_pct": ds_util.round(2),
            "us_channel_utilization_pct": us_util.round(2),
            # Engineered
            "ds_power_rolling_std_1h":    ds_power_std_1h.round(4),
            "snr_drop_rate_per_hour":     snr_drop_rate.round(4),
            "us_ds_power_delta":          us_ds_delta.round(3),
            "channel_util_peak_hour_flag":peak_mask,
            # Label
            "anomaly_flag":               anomaly_flag,
        }))

    df_hfc = pd.concat(frames, ignore_index=True)

    # Partition parquet by date for efficient downstream reads
    df_hfc["date"] = df_hfc["timestamp"].dt.date.astype(str)
    pq.write_to_dataset(
        pa.Table.from_pandas(df_hfc),
        root_path=str(RAW_DIR / "hfc_metrics"),
        partition_cols=["date"],
        existing_data_behavior="overwrite_or_ignore",
    )

    anom_rate = df_hfc["anomaly_flag"].mean() * 100
    log.info(
        "P1 done — %d rows | anomaly rate: %.1f%% (%d flagged)",
        len(df_hfc), anom_rate, df_hfc["anomaly_flag"].sum(),
    )
    return df_hfc, anomaly_log


# ─────────────────────────────────────────────────────────────
# PROJECT 2 — WI-FI EXPERIENCE
# ─────────────────────────────────────────────────────────────

def generate_wifi_data(anomaly_log: dict) -> tuple[pd.DataFrame, dict]:
    """
    Generate hourly Wi-Fi experience scores per device per customer.
    60% of experience breaches co-occur with a P1 modem anomaly.

    Returns
    -------
    df_wifi    : full dataframe
    breach_log : dict mapping customer_id → list of breach timestamps
    """
    log.info("Generating P2: Wi-Fi experience metrics (%d customers × 3 devices)", N_CUSTOMERS)

    timestamps = _make_timestamps(DATA["wifi_interval_h"] * 60)
    n_ts       = len(timestamps)
    hours      = pd.Series(timestamps).dt.hour.values
    dow        = pd.Series(timestamps).dt.dayofweek.values

    breach_log: dict[str, list] = {}
    frames = []

    for cust_idx in range(N_CUSTOMERS):
        customer_id = f"CUS-{cust_idx:04d}"
        modem_id    = f"MDM-{cust_idx:04d}"

        # Map modem anomaly timestamps → hour buckets for fast lookup
        modem_anom_hours = {
            pd.Timestamp(ts).floor("h")
            for ts in anomaly_log.get(modem_id, [])
        }

        breach_log[customer_id] = []

        # 2 laptops + 1 IoT per customer
        devices = [
            (f"{customer_id}-LAP-01", "laptop"),
            (f"{customer_id}-LAP-02", "laptop"),
            (f"{customer_id}-IOT-01", "iot"),
        ]

        for device_id, device_type in devices:
            is_iot = device_type == "iot"

            # ── Base experience score ─────────────────────────
            score = np.random.normal(75, 8, n_ts)
            if not is_iot:
                # Laptops peak in evenings and weekends
                score += ((hours >= 19) & (hours <= 22)).astype(float) * 5
                score += (dow >= 5).astype(float) * 3
            else:
                # IoT devices busier on weekends
                score += (dow >= 5).astype(float) * 4

            score = np.clip(score, 20, 100)

            # ── Other raw signals ─────────────────────────────
            rssi          = np.random.normal(-55, 8, n_ts).round(1)
            interference  = np.clip(np.random.normal(15, 5, n_ts), 0, 60).round(1)
            active_count  = (np.random.poisson(3, n_ts) + 1).astype(int)

            # ── Rolling baseline (7-day window for ±3σ rule) ──
            series        = pd.Series(score)
            roll_mean     = series.rolling(24 * 7, min_periods=24).mean().fillna(series.mean())
            roll_std      = series.rolling(24 * 7, min_periods=24).std().fillna(series.std())

            # ── Apply modem anomaly degradation (60% linkage) ─
            hfc_flag = np.zeros(n_ts, dtype=np.int8)
            for i, ts in enumerate(timestamps):
                if ts in modem_anom_hours:
                    hfc_flag[i] = 1
                    if np.random.random() < 0.60:
                        score[i] -= np.random.uniform(15, 30)

            score = np.clip(score, 0, 100)

            # ── ±3σ breach detection ──────────────────────────
            lower       = roll_mean - 3.0 * roll_std
            breach_flag = (score < lower.values).astype(np.int8)

            for i, ts in enumerate(timestamps):
                if breach_flag[i] == 1:
                    breach_log[customer_id].append(ts)

            frames.append(pd.DataFrame({
                "timestamp":                timestamps,
                "customer_id":              customer_id,
                "device_id":               device_id,
                "device_type":             device_type,
                # Raw signals
                "hour_of_day":             hours,
                "day_of_week":             dow,
                "wifi_experience_score":   score.round(2),
                "rssi_dbm":                rssi,
                "channel_interference_pct":interference,
                "active_device_count":     active_count,
                # Engineered
                "experience_rolling_mean_7d": roll_mean.round(3),
                "experience_rolling_std_7d":  roll_std.round(3),
                # Cross-project signal from P1
                "hfc_anomaly_flag":        hfc_flag,
                # Label
                "forecast_breach_flag":    breach_flag,
            }))

    df_wifi = pd.concat(frames, ignore_index=True)

    df_wifi["date"] = df_wifi["timestamp"].dt.date.astype(str)
    pq.write_to_dataset(
        pa.Table.from_pandas(df_wifi),
        root_path=str(RAW_DIR / "wifi_metrics"),
        partition_cols=["date"],
        existing_data_behavior="overwrite_or_ignore",
    )

    breach_rate = df_wifi["forecast_breach_flag"].mean() * 100
    log.info(
        "P2 done — %d rows | breach rate: %.1f%% (%d flagged)",
        len(df_wifi), breach_rate, df_wifi["forecast_breach_flag"].sum(),
    )
    return df_wifi, breach_log


# ─────────────────────────────────────────────────────────────
# PROJECT 3 — CHURN CUSTOMER TABLE
# ─────────────────────────────────────────────────────────────

def generate_churn_data(anomaly_log: dict, breach_log: dict) -> pd.DataFrame:
    """
    Build one row per customer with CRM + aggregated network features.
    Churn label is causally derived — not randomly assigned.

    Causal drivers of churn:
        month-to-month contract  → +0.25
        short tenure (< 12 mo)   → +0.15
        high monthly charges     → +0.08
        electronic check payment → +0.06
        no tech support          → +0.06
        fiber optic (high expect)→ +0.04
        network anomaly severity → +0.12 (from P1)
        Wi-Fi breach frequency   → +0.10 (from P2)
        recent bad experience    → +0.04
    """
    log.info("Generating P3: Churn customer table (%d customers)", N_CUSTOMERS)

    # Load aggregated network signals from last 30 days of simulation
    cutoff = START_DATE + pd.Timedelta(days=60)

    # Read only the columns we need from parquet
    hfc_table = pq.read_table(
        str(RAW_DIR / "hfc_metrics"),
        columns=["customer_id", "timestamp", "anomaly_flag"],
    ).to_pandas()
    hfc_30d = hfc_table[hfc_table["timestamp"] >= cutoff]

    wifi_table = pq.read_table(
        str(RAW_DIR / "wifi_metrics"),
        columns=["customer_id", "timestamp", "forecast_breach_flag"],
    ).to_pandas()
    wifi_30d = wifi_table[wifi_table["timestamp"] >= cutoff]

    # Aggregate per customer
    anom_agg = (
        hfc_30d.groupby("customer_id")["anomaly_flag"]
        .sum()
        .reset_index()
        .rename(columns={"anomaly_flag": "anomaly_count_30d"})
    )
    breach_agg = (
        wifi_30d.groupby("customer_id")["forecast_breach_flag"]
        .sum()
        .reset_index()
        .rename(columns={"forecast_breach_flag": "wifi_breach_count_30d"})
    )

    # Last event times (recency feature)
    last_anom = (
        hfc_table[hfc_table["anomaly_flag"] == 1]
        .groupby("customer_id")["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "last_anomaly_ts"})
    )
    last_breach = (
        wifi_table[wifi_table["forecast_breach_flag"] == 1]
        .groupby("customer_id")["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "last_breach_ts"})
    )

    # CRM lookup tables
    contracts   = ["Month-to-month", "One year", "Two year"]
    payments    = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internets   = ["Fiber optic", "DSL", "No"]

    records = []
    sim_end = START_DATE + pd.Timedelta(days=N_DAYS)

    for cust_idx in range(N_CUSTOMERS):
        cid = f"CUS-{cust_idx:04d}"

        # ── CRM features ──────────────────────────────────────
        contract        = np.random.choice(contracts, p=[0.55, 0.25, 0.20])
        tenure          = max(1, int(np.random.exponential(24)))
        monthly_charges = round(float(np.random.normal(65, 20)), 2)
        payment         = np.random.choice(payments, p=[0.35, 0.20, 0.25, 0.20])
        tech_support    = int(np.random.choice([0, 1], p=[0.45, 0.55]))
        internet        = np.random.choice(internets, p=[0.50, 0.35, 0.15])
        paperless       = int(np.random.choice([0, 1], p=[0.40, 0.60]))
        service_calls   = int(np.random.poisson(1.5))

        # Engineered CRM feature
        charge_per_tenure = round(monthly_charges / max(tenure, 1), 4)

        # ── Network features from P1 ──────────────────────────
        anom_row      = anom_agg[anom_agg["customer_id"] == cid]
        raw_anom_cnt  = int(anom_row["anomaly_count_30d"].values[0]) if len(anom_row) else 0
        # Normalise to 0-1 (max realistic ~600 readings in 30 days)
        anomaly_severity = round(min(raw_anom_cnt, 600) / 600, 4)

        # ── Network features from P2 ──────────────────────────
        breach_row       = breach_agg[breach_agg["customer_id"] == cid]
        breach_count_30d = int(breach_row["wifi_breach_count_30d"].values[0]) if len(breach_row) else 0

        # ── Recency: days since last bad network event ─────────
        la_row = last_anom[last_anom["customer_id"] == cid]
        lb_row = last_breach[last_breach["customer_id"] == cid]
        event_times = []
        if len(la_row): event_times.append(pd.Timestamp(la_row["last_anomaly_ts"].values[0]))
        if len(lb_row): event_times.append(pd.Timestamp(lb_row["last_breach_ts"].values[0]))
        days_since_last = (
            int((sim_end - max(event_times)).days) if event_times else N_DAYS
        )

        # ── Causal churn score ────────────────────────────────
        score = 0.0
        if contract == "Month-to-month": score += 0.25
        elif contract == "One year":     score += 0.08
        if tenure < 12:                  score += 0.15
        elif tenure < 24:                score += 0.06
        if monthly_charges > 80:         score += 0.08
        if payment == "Electronic check":score += 0.06
        if not tech_support:             score += 0.06
        if internet == "Fiber optic":    score += 0.04
        score += anomaly_severity         * 0.12   # P1 signal
        score += min(breach_count_30d / 200, 0.10) # P2 signal
        if days_since_last < 7:          score += 0.04
        score += float(np.random.normal(0, 0.10))  # noise

        records.append({
            "customer_id":               cid,
            # CRM raw
            "contract_type":             contract,
            "tenure_months":             tenure,
            "monthly_charges":           monthly_charges,
            "payment_method":            payment,
            "tech_support_flag":         tech_support,
            "internet_service_type":     internet,
            "paperless_billing_flag":    paperless,
            "service_call_frequency_30d":service_calls,
            # CRM engineered
            "charge_per_tenure_ratio":   charge_per_tenure,
            # Network features (P1 → P3)
            "anomaly_count_30d":         raw_anom_cnt,
            "anomaly_severity_score":    anomaly_severity,
            # Network features (P2 → P3)
            "wifi_breach_count_30d":     breach_count_30d,
            "days_since_last_anomaly":   days_since_last,
            # Raw churn score (kept for threshold calibration)
            "_churn_score":              round(score, 6),
        })

    df = pd.DataFrame(records)

    # Threshold at 74th percentile → ~26% churn rate
    threshold = float(np.percentile(df["_churn_score"], 74))
    df["churn"] = (df["_churn_score"] > threshold).astype(int)
    df = df.drop(columns=["_churn_score"])

    df.to_parquet(RAW_DIR / "churn_customers.parquet", index=False)
    log.info(
        "P3 done — %d rows | churn rate: %.1f%% (%d churners)",
        len(df), df["churn"].mean() * 100, df["churn"].sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Broadband Intelligence Platform — Data Generation")
    log.info("=" * 60)

    df_hfc,  anomaly_log = generate_hfc_data()
    df_wifi, breach_log  = generate_wifi_data(anomaly_log)
    df_churn             = generate_churn_data(anomaly_log, breach_log)

    log.info("-" * 60)
    log.info("Summary")
    log.info("  P1 HFC   : %10d rows  →  data/raw/hfc_metrics/",    len(df_hfc))
    log.info("  P2 Wi-Fi : %10d rows  →  data/raw/wifi_metrics/",   len(df_wifi))
    log.info("  P3 Churn : %10d rows  →  data/raw/churn_customers.parquet", len(df_churn))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
