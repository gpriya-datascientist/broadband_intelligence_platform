"""
Project 3 — Customer Churn Prediction
Data Preparation & Cleaning Pipeline
======================================
Steps:
  1. Load raw churn customer parquet
  2. Schema validation
  3. Business rule validation
  4. Missing value audit and imputation
  5. Duplicate detection
  6. Class imbalance analysis
  7. Categorical encoding (Label Encoding)
  8. Numeric feature scaling (StandardScaler)
  9. Feature correlation analysis
  10. Stratified train/val/test split
  11. Save processed datasets + cleaning report

Run:
    python -m src.p3_churn.prepare_data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config.settings import RAW_DIR, PROCESSED_DIR, P3
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p3_prepare.log")

CAT_FEATURES = P3["categorical_features"]
NUM_FEATURES = P3["numeric_features"]
TARGET       = P3["target"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    log.info("Step 1 — Loading raw churn customer data")
    df = pd.read_parquet(RAW_DIR / "churn_customers.parquet")
    log.info("  Loaded %d customers | %d features | churn rate: %.1f%%",
             len(df), len(df.columns), df[TARGET].mean() * 100)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Enforce correct dtypes. Prevents silent type coercion in sklearn."""
    log.info("Step 2 — Schema validation")

    expected = {
        "customer_id":                "object",
        "contract_type":              "object",
        "payment_method":             "object",
        "internet_service_type":      "object",
        "tenure_months":              "int64",
        "monthly_charges":            "float64",
        "charge_per_tenure_ratio":    "float64",
        "tech_support_flag":          "int64",
        "paperless_billing_flag":     "int64",
        "service_call_frequency_30d": "int64",
        "anomaly_count_30d":          "int64",
        "anomaly_severity_score":     "float64",
        "wifi_breach_count_30d":      "int64",
        "days_since_last_anomaly":    "int64",
        "churn":                      "int64",
    }

    issues = []
    for col, exp in expected.items():
        if col not in df.columns:
            issues.append(f"MISSING: {col}")
            continue
        actual = str(df[col].dtype)
        if "float" in exp and "float" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            issues.append(f"CAST {col} → float64")
        elif "int" in exp and "int" not in actual:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            issues.append(f"CAST {col} → int64")

    report["schema_issues"] = issues
    log.info("  Schema corrections: %d", len(issues))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — BUSINESS RULE VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_business_rules(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Validate domain-specific business rules.
    These are constraints an interviewer expects you to enforce:

    - tenure must be ≥ 1 month (customer must have existed)
    - monthly_charges must be > 0 (no free plans in dataset)
    - anomaly_severity must be 0–1 (normalised score)
    - contract_type must be a known value
    - churn must be 0 or 1 (binary)

    Rows violating hard rules are flagged and removed.
    """
    log.info("Step 3 — Business rule validation")

    violations = {}
    n_before = len(df)

    # Range violations (from config)
    for col, (lo, hi) in P3["valid_ranges"].items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = int(mask.sum())
        if n_bad > 0:
            violations[col] = n_bad
            log.warning("  %d rows with %s outside [%s, %s] — removing", n_bad, col, lo, hi)
            df = df[~mask]

    # Categorical domain check
    valid_contracts = {"Month-to-month", "One year", "Two year"}
    bad_contracts   = ~df["contract_type"].isin(valid_contracts)
    n_bad_c         = int(bad_contracts.sum())
    if n_bad_c > 0:
        violations["contract_type_unknown"] = n_bad_c
        df = df[~bad_contracts]
        log.warning("  %d rows with unknown contract_type — removing", n_bad_c)

    # Binary target check
    bad_target = ~df[TARGET].isin([0, 1])
    n_bad_t    = int(bad_target.sum())
    if n_bad_t > 0:
        violations["churn_not_binary"] = n_bad_t
        df = df[~bad_target]
        log.warning("  %d rows with non-binary churn label — removing", n_bad_t)

    report["business_rule_violations"] = violations
    report["rows_removed_by_rules"]    = n_before - len(df)
    log.info("  Rows removed by business rules: %d | Remaining: %d",
             n_before - len(df), len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — MISSING VALUE AUDIT & IMPUTATION
# ─────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Imputation strategy:
    - Numeric features  : median imputation (robust to outliers)
    - Categorical       : mode imputation (most frequent category)
    - Network features  : fill 0 (no anomaly/breach = 0 events)

    We use median over mean for numeric because monthly_charges
    and tenure are right-skewed — mean would overestimate.
    """
    log.info("Step 4 — Missing value audit and imputation")

    null_counts = df.isnull().sum()
    report["null_audit"] = null_counts.to_dict()

    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        log.info("  No missing values detected")
        return df

    log.info("  Columns with nulls:")
    for col, cnt in null_cols.items():
        log.info("    %-40s %d (%.2f%%)", col, cnt, cnt / len(df) * 100)

    # Network features: no event = 0
    network_cols = [
        "anomaly_count_30d", "anomaly_severity_score",
        "wifi_breach_count_30d", "days_since_last_anomaly",
    ]
    for col in network_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Numeric: median imputation
    for col in NUM_FEATURES:
        if col in df.columns and df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.info("  Median imputed %s = %.4f", col, med)

    # Categorical: mode imputation
    for col in CAT_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            log.info("  Mode imputed %s = %s", col, mode)

    remaining = int(df.isnull().sum().sum())
    report["nulls_after_imputation"] = remaining
    log.info("  Nulls after imputation: %d", remaining)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Each customer_id must appear exactly once.
    Duplicates can arise from data pipeline retries.
    """
    log.info("Step 5 — Duplicate detection")

    n_before = len(df)
    df       = df.drop_duplicates()
    n_exact  = n_before - len(df)

    df       = df.drop_duplicates(subset=["customer_id"], keep="last")
    n_cust   = n_before - n_exact - len(df)

    report["duplicates_removed"] = {
        "exact_rows": n_exact,
        "customer_id_duplicates": n_cust,
        "rows_after": len(df),
    }
    log.info("  Exact: %d | customer_id: %d | Remaining: %d",
             n_exact, n_cust, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — CLASS IMBALANCE ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_class_balance(df: pd.DataFrame, report: dict) -> dict:
    """
    Compute class distribution and recommend class weights.

    We use class weights in the DNN loss function rather than
    oversampling (SMOTE) because:
    1. SMOTE on small datasets (500 rows) risks overfitting to
       synthetic minority examples.
    2. Class weights are natively supported in Keras and are
       transparent — easy to explain to interviewers.
    3. They don't change the data distribution, only the gradient
       contribution of each class.
    """
    log.info("Step 6 — Class imbalance analysis")

    n_total = len(df)
    n_pos   = int(df[TARGET].sum())
    n_neg   = n_total - n_pos
    ratio   = n_neg / n_pos

    class_weights = {0: 1.0, 1: round(ratio, 4)}

    log.info("  Class 0 (Stay)  : %d (%.1f%%)", n_neg, n_neg/n_total*100)
    log.info("  Class 1 (Churn) : %d (%.1f%%)", n_pos, n_pos/n_total*100)
    log.info("  Imbalance ratio : %.2f:1", ratio)
    log.info("  Class weights   : %s", class_weights)
    log.info("  Strategy: class-weighted loss (not SMOTE) — avoids small-dataset overfitting")

    report["class_balance"] = {
        "n_stay":       n_neg,
        "n_churn":      n_pos,
        "churn_rate":   round(n_pos / n_total, 4),
        "class_weights":class_weights,
        "strategy":     "class-weighted binary cross-entropy loss",
    }
    return class_weights


# ─────────────────────────────────────────────────────────────
# STEP 7 — CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical features.
    Returns encoders dict so inference pipeline can apply same mapping.

    Why Label Encoding over One-Hot here?
    The DNN handles ordinal relationships through its weight layers,
    so label encoding is sufficient. One-hot would add 6 extra
    dimensions for only 3 categoricals, increasing sparsity on
    a 500-row dataset unnecessarily.
    """
    log.info("Step 7 — Categorical label encoding")

    encoders    = {}
    encoding_map = {}

    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
        log.info("  %s → %s", col, encoding_map[col])

    report["categorical_encoding"] = encoding_map
    return df, encoders


# ─────────────────────────────────────────────────────────────
# STEP 8 — FEATURE SCALING
# ─────────────────────────────────────────────────────────────

def scale_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    report:   dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    StandardScaler fit ONLY on training data.
    Applied to validation and test sets separately.

    Critical rule: fitting the scaler on the full dataset
    before splitting = data leakage. The test set statistics
    would influence the scaling and inflate metrics.
    """
    log.info("Step 8 — Feature scaling (StandardScaler fit on train only)")

    scaler = StandardScaler()
    feature_cols = CAT_FEATURES + NUM_FEATURES

    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    report["scaling"] = {
        "method": "StandardScaler",
        "fit_on": "train only (no leakage)",
        "feature_means":  dict(zip(feature_cols, scaler.mean_.round(4).tolist())),
        "feature_stds":   dict(zip(feature_cols, scaler.scale_.round(4).tolist())),
    }

    log.info("  Scaled %d features | fit on %d training rows", len(feature_cols), len(df_train))
    return df_train, df_val, df_test, scaler


# ─────────────────────────────────────────────────────────────
# STEP 9 — FEATURE CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_correlations(df: pd.DataFrame, report: dict) -> None:
    """
    Compute Pearson correlation of all features with the churn target.
    Flags any pair of input features with |r| > 0.85 as potentially
    redundant (multicollinearity warning).

    This is a diagnostic step — we don't drop features here,
    we document the findings for the README and interview discussion.
    """
    log.info("Step 9 — Feature correlation analysis")

    feature_cols = CAT_FEATURES + NUM_FEATURES
    corr_with_target = (
        df[feature_cols + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )

    log.info("  Feature correlations with churn target:")
    for feat, r in corr_with_target.items():
        log.info("    %-40s r = %+.3f", feat, r)

    # Check for highly correlated feature pairs
    corr_matrix    = df[feature_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = [
        (col, row, round(upper_triangle.loc[row, col], 3))
        for col in upper_triangle.columns
        for row in upper_triangle.index
        if pd.notna(upper_triangle.loc[row, col]) and upper_triangle.loc[row, col] > 0.85
    ]

    if high_corr_pairs:
        log.warning("  High inter-feature correlations (|r| > 0.85):")
        for f1, f2, r in high_corr_pairs:
            log.warning("    %s ↔ %s : r=%.3f", f1, f2, r)
    else:
        log.info("  No high inter-feature correlations detected (threshold: 0.85)")

    report["correlations_with_target"] = corr_with_target.round(4).to_dict()
    report["high_corr_pairs"] = high_corr_pairs


# ─────────────────────────────────────────────────────────────
# STEP 10 — STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split to preserve churn rate across all three sets.

    Why stratified?
    With only 130 churners in 500 rows, a random split could put
    all churners in train, leaving the model unable to evaluate.
    Stratification guarantees class balance in each partition.
    """
    log.info("Step 10 — Stratified train/val/test split")

    X    = df[ALL_FEATURES]
    y    = df[TARGET]
    ids  = df["customer_id"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=P3["test_size"],
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_train, y_train, id_train,
        test_size=P3["val_size"],
        random_state=42,
        stratify=y_train,
    )

    def _make(X, y, ids):
        d = X.copy()
        d[TARGET]       = y.values
        d["customer_id"]= ids.values
        return d

    df_train = _make(X_train, y_train, id_train)
    df_val   = _make(X_val,   y_val,   id_val)
    df_test  = _make(X_test,  y_test,  id_test)

    report["split"] = {
        "train_rows": len(df_train), "train_churn_rate": round(df_train[TARGET].mean(), 4),
        "val_rows":   len(df_val),   "val_churn_rate":   round(df_val[TARGET].mean(),   4),
        "test_rows":  len(df_test),  "test_churn_rate":  round(df_test[TARGET].mean(),  4),
        "strategy": "stratified (preserves churn class ratio)",
    }

    log.info("  Train: %d rows | churn: %.1f%%", len(df_train), df_train[TARGET].mean()*100)
    log.info("  Val  : %d rows | churn: %.1f%%", len(df_val),   df_val[TARGET].mean()*100)
    log.info("  Test : %d rows | churn: %.1f%%", len(df_test),  df_test[TARGET].mean()*100)
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────
# STEP 11 — SAVE
# ─────────────────────────────────────────────────────────────

def save_processed(df_train, df_val, df_test, scaler, encoders, report):
    import joblib
    log.info("Step 11 — Saving processed data and artifacts")

    df_train.to_parquet(PROCESSED_DIR / "p3_train.parquet", index=False)
    df_val.to_parquet(  PROCESSED_DIR / "p3_val.parquet",   index=False)
    df_test.to_parquet( PROCESSED_DIR / "p3_test.parquet",  index=False)

    joblib.dump(scaler,   PROCESSED_DIR / "p3_scaler.joblib")
    joblib.dump(encoders, PROCESSED_DIR / "p3_encoders.joblib")

    report["output_files"] = {
        "train":    str(PROCESSED_DIR / "p3_train.parquet"),
        "val":      str(PROCESSED_DIR / "p3_val.parquet"),
        "test":     str(PROCESSED_DIR / "p3_test.parquet"),
        "scaler":   str(PROCESSED_DIR / "p3_scaler.joblib"),
        "encoders": str(PROCESSED_DIR / "p3_encoders.joblib"),
    }

    with open("data/reports/p3_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("  Saved train/val/test parquets")
    log.info("  Saved scaler and encoders")
    log.info("  Saved: data/reports/p3_cleaning_report.json")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 60)
    log.info("P3 — Data Preparation & Cleaning Pipeline")
    log.info("=" * 60)

    report = {}

    df            = load_raw()
    df            = validate_schema(df, report)
    df            = validate_business_rules(df, report)
    df            = handle_missing(df, report)
    df            = remove_duplicates(df, report)
    class_weights = analyse_class_balance(df, report)
    df, encoders  = encode_categoricals(df, report)

    # Correlation analysis before scaling (interpretable values)
    analyse_correlations(df, report)

    # Split before scaling to prevent leakage
    df_train, df_val, df_test = stratified_split(df, report)
    df_train, df_val, df_test, scaler = scale_features(df_train, df_val, df_test, report)

    save_processed(df_train, df_val, df_test, scaler, encoders, report)

    log.info("=" * 60)
    log.info("P3 preparation complete")
    log.info("=" * 60)
    return df_train, df_val, df_test, scaler, encoders, class_weights


if __name__ == "__main__":
    run_pipeline()
