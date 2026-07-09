"""
Project 1 — HFC Anomaly Detection
Model Training — Isolation Forest + LOF Ensemble
==================================================
Steps:
  1. Load processed train/test data
  2. Extract feature matrix
  3. Train Isolation Forest
  4. Train Local Outlier Factor (novelty mode)
  5. Ensemble score computation (weighted average)
  6. Threshold optimisation
  7. Evaluation (ROC-AUC, Precision, Recall, F1)
  8. Permutation feature importance
  9. Save model artifacts and metrics

Run:
    python -m src.p1_hfc_anomaly.train_model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve,
)

from config.settings import PROCESSED_DIR, MODEL_DIR, P1
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p1_train.log")

FEATURES   = P1["features"]
IF_WEIGHT  = P1["if_weight"]
LOF_WEIGHT = P1["lof_weight"]


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD PROCESSED DATA
# ─────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log.info("Step 1 — Loading processed P1 train/test data")

    df_train = pd.read_parquet(PROCESSED_DIR / "p1_train.parquet")
    df_test  = pd.read_parquet(PROCESSED_DIR / "p1_test.parquet")

    X_train = df_train[FEATURES].fillna(0).values
    y_train = df_train["anomaly_flag"].values
    X_test  = df_test[FEATURES].fillna(0).values
    y_test  = df_test["anomaly_flag"].values

    log.info("  Train: %d rows | anomalies: %d (%.1f%%)",
             len(X_train), y_train.sum(), y_train.mean()*100)
    log.info("  Test : %d rows | anomalies: %d (%.1f%%)",
             len(X_test), y_test.sum(), y_test.mean()*100)
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────
# STEP 2 — FEATURE SCALING
# ─────────────────────────────────────────────────────────────

def scale(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    StandardScaler fit on training data only.
    Both IF and LOF are distance-based — unscaled features
    with different magnitudes (e.g. SNR in dB vs utilisation in %)
    would bias distances toward higher-magnitude features.
    """
    log.info("Step 2 — Feature scaling (fit on train only)")
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    log.info("  Scaler fit on %d training samples", len(X_train))
    return X_train_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────────
# STEP 3 — TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Isolation Forest works by randomly isolating observations.
    Anomalies require fewer splits to isolate — they have shorter
    average path lengths in the forest.

    Key hyperparameters:
    - n_estimators=200  : more trees = more stable scores
    - contamination=0.12: expected anomaly fraction (from EDA)
    - max_samples='auto': subsampling for speed and diversity
    """
    log.info("Step 3 — Training Isolation Forest (n_estimators=200)")

    model = IsolationForest(
        n_estimators  = P1["if_n_estimators"],
        contamination = P1["contamination"],
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_train)
    log.info("  Isolation Forest trained on %d samples", len(X_train))
    return model


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN LOCAL OUTLIER FACTOR
# ─────────────────────────────────────────────────────────────

def train_lof(X_train: np.ndarray) -> LocalOutlierFactor:
    """
    LOF measures local density deviation of each point relative
    to its k-nearest neighbours. A point in a low-density
    region compared to its neighbours is anomalous.

    novelty=True allows inference on new data — required for
    production use. Without novelty=True, LOF can only score
    training data.

    Key hyperparameters:
    - n_neighbors=20   : balance between local and global structure
    - contamination=0.12: consistent with IF setting
    """
    log.info("Step 4 — Training Local Outlier Factor (n_neighbors=20, novelty=True)")

    model = LocalOutlierFactor(
        n_neighbors   = P1["lof_n_neighbors"],
        contamination = P1["contamination"],
        novelty       = True,
        n_jobs        = -1,
    )
    model.fit(X_train)
    log.info("  LOF trained on %d samples", len(X_train))
    return model


# ─────────────────────────────────────────────────────────────
# STEP 5 — ENSEMBLE SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_ensemble_score(
    if_model:  IsolationForest,
    lof_model: LocalOutlierFactor,
    X:         np.ndarray,
) -> np.ndarray:
    """
    Weighted average of normalised anomaly scores.

    Both models output different score scales — IF outputs
    negative values (more negative = more anomalous), LOF
    outputs positive values. We:
    1. Negate to make higher = more anomalous for both
    2. Min-max normalise each to [0, 1]
    3. Weighted average: IF 60%, LOF 40%

    Rationale for weights:
    IF detects global anomalies (rare events) better.
    LOF detects local density anomalies better.
    HFC faults are mostly global (plant-wide issues),
    so we weight IF slightly higher.
    """
    if_raw  = -if_model.score_samples(X)
    lof_raw = -lof_model.score_samples(X)

    # Min-max normalise
    def _normalise(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    if_norm  = _normalise(if_raw)
    lof_norm = _normalise(lof_raw)

    ensemble = IF_WEIGHT * if_norm + LOF_WEIGHT * lof_norm
    return ensemble


# ─────────────────────────────────────────────────────────────
# STEP 6 — THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────────────────────

def optimise_threshold(
    y_true:         np.ndarray,
    ensemble_score: np.ndarray,
) -> float:
    """
    Find threshold maximising F1 on the anomaly class.

    Using F1 rather than accuracy because class imbalance
    means a naive classifier (predict all normal) would get
    ~88% accuracy but catch 0 anomalies.

    We sweep the precision-recall curve and pick the threshold
    where F1 is highest — this balances catching real anomalies
    (recall) against false alert fatigue (precision).
    """
    log.info("Step 6 — Threshold optimisation (max F1 on anomaly class)")

    precisions, recalls, thresholds = precision_recall_curve(y_true, ensemble_score)
    f1_scores = (
        2 * precisions * recalls / (precisions + recalls + 1e-9)
    )
    best_idx  = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1   = float(f1_scores[best_idx])

    log.info("  Optimal threshold: %.4f | best F1: %.4f", threshold, best_f1)
    return threshold


# ─────────────────────────────────────────────────────────────
# STEP 7 — EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(
    y_true:         np.ndarray,
    ensemble_score: np.ndarray,
    threshold:      float,
) -> dict:
    log.info("Step 7 — Evaluation")

    y_pred = (ensemble_score > threshold).astype(int)
    auc    = roc_auc_score(y_true, ensemble_score)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomaly"],
        output_dict=True,
    )

    tn, fp, fn, tp = cm.ravel()

    log.info("  ROC-AUC  : %.4f", auc)
    log.info("  Precision: %.4f  (of flagged alerts, how many are real)", report["Anomaly"]["precision"])
    log.info("  Recall   : %.4f  (of real anomalies, how many we caught)", report["Anomaly"]["recall"])
    log.info("  F1       : %.4f", report["Anomaly"]["f1-score"])
    log.info("  Confusion matrix:")
    log.info("    TN=%d  FP=%d  (false alerts)", tn, fp)
    log.info("    FN=%d  TP=%d  (missed anomalies)", fn, tp)
    log.info("  Business read: catching %.0f%% of real faults with %.0f%% alert precision",
             report["Anomaly"]["recall"]*100, report["Anomaly"]["precision"]*100)

    return {
        "roc_auc":   round(auc, 4),
        "precision": round(report["Anomaly"]["precision"], 4),
        "recall":    round(report["Anomaly"]["recall"], 4),
        "f1":        round(report["Anomaly"]["f1-score"], 4),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


# ─────────────────────────────────────────────────────────────
# STEP 8 — PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def permutation_importance(
    if_model:       IsolationForest,
    lof_model:      LocalOutlierFactor,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
) -> dict:
    """
    Measure importance by shuffling each feature and observing
    ROC-AUC drop. Larger drop = more important feature.

    Used instead of SHAP because IF and LOF are tree/distance
    models without differentiable outputs — DeepSHAP does not apply.
    Permutation importance is model-agnostic and interpretable.
    """
    log.info("Step 8 — Permutation feature importance")

    baseline_score = compute_ensemble_score(if_model, lof_model, X_test)
    baseline_auc   = roc_auc_score(y_test, baseline_score)
    importance     = {}

    for i, feat in enumerate(FEATURES):
        X_perm = X_test.copy()
        np.random.seed(42)
        np.random.shuffle(X_perm[:, i])
        perm_score = compute_ensemble_score(if_model, lof_model, X_perm)
        perm_auc   = roc_auc_score(y_test, perm_score)
        importance[feat] = round(baseline_auc - perm_auc, 6)

    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    log.info("  Top 5 features by importance drop:")
    for feat, imp in list(importance.items())[:5]:
        log.info("    %-40s Δ AUC = %+.4f", feat, imp)

    return importance


# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────

def save_artifacts(
    scaler:     StandardScaler,
    if_model:   IsolationForest,
    lof_model:  LocalOutlierFactor,
    threshold:  float,
    metrics:    dict,
    importance: dict,
) -> None:
    log.info("Step 9 — Saving model artifacts")

    joblib.dump(scaler,    MODEL_DIR / "p1_scaler.joblib")
    joblib.dump(if_model,  MODEL_DIR / "p1_isolation_forest.joblib")
    joblib.dump(lof_model, MODEL_DIR / "p1_lof.joblib")

    output = {
        "model":            "IF + LOF Ensemble",
        "threshold":        round(threshold, 6),
        "if_weight":        IF_WEIGHT,
        "lof_weight":       LOF_WEIGHT,
        "contamination":    P1["contamination"],
        "if_n_estimators":  P1["if_n_estimators"],
        "lof_n_neighbors":  P1["lof_n_neighbors"],
        "features":         FEATURES,
        "metrics":          metrics,
        "feature_importance": importance,
    }

    with open(MODEL_DIR / "p1_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("  Saved: p1_scaler.joblib")
    log.info("  Saved: p1_isolation_forest.joblib")
    log.info("  Saved: p1_lof.joblib")
    log.info("  Saved: p1_metrics.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def train():
    log.info("=" * 60)
    log.info("P1 — HFC Anomaly Detection Model Training")
    log.info("=" * 60)

    X_train, y_train, X_test, y_test = load_data()
    X_train_s, X_test_s, scaler      = scale(X_train, X_test)
    if_model                          = train_isolation_forest(X_train_s)
    lof_model                         = train_lof(X_train_s)

    log.info("Step 5 — Computing ensemble anomaly scores")
    ensemble_score = compute_ensemble_score(if_model, lof_model, X_test_s)

    threshold = optimise_threshold(y_test, ensemble_score)
    metrics   = evaluate(y_test, ensemble_score, threshold)
    importance = permutation_importance(if_model, lof_model, X_test_s, y_test)
    save_artifacts(scaler, if_model, lof_model, threshold, metrics, importance)

    log.info("=" * 60)
    log.info("P1 training complete — ROC-AUC: %.4f", metrics["roc_auc"])
    log.info("=" * 60)
    return metrics


if __name__ == "__main__":
    train()
