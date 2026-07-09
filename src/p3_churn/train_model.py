"""
Project 3 — Customer Churn Prediction
Model Training — Dense Neural Network + SHAP
=============================================
Steps:
  1. Load processed train/val/test data
  2. Handle multicollinearity finding from preparation
  3. Build DNN architecture
  4. Train with minority oversampling + early stopping
  5. Learning curve analysis
  6. Threshold tuning (max F1 on churn class)
  7. Final evaluation (ROC-AUC, PR-AUC, F1, confusion matrix)
  8. SHAP explainability
  9. Save all artifacts

Run:
    python -m src.p3_churn.train_model
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
import shap

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve,
)

from config.settings import PROCESSED_DIR, MODEL_DIR, P3
from src.shared.logger import get_logger

log = get_logger(__name__, log_file="data/reports/p3_train.log")

CAT_FEATURES = P3["categorical_features"]
NUM_FEATURES = P3["numeric_features"]
TARGET       = P3["target"]
DNN_CFG      = P3["dnn"]

# Drop one of the perfectly correlated pair found in preparation
# anomaly_severity_score is normalised anomaly_count_30d (r=1.0)
FEATURES_TO_DROP = ["anomaly_severity_score"]
ALL_FEATURES = [
    f for f in CAT_FEATURES + NUM_FEATURES
    if f not in FEATURES_TO_DROP
]


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD PROCESSED DATA
# ─────────────────────────────────────────────────────────────

def load_data():
    log.info("Step 1 — Loading processed P3 train/val/test data")

    df_train = pd.read_parquet(PROCESSED_DIR / "p3_train.parquet")
    df_val   = pd.read_parquet(PROCESSED_DIR / "p3_val.parquet")
    df_test  = pd.read_parquet(PROCESSED_DIR / "p3_test.parquet")

    log.info("  Train: %d | Val: %d | Test: %d",
             len(df_train), len(df_val), len(df_test))

    def _xy(df):
        X = df[ALL_FEATURES].values.astype(np.float32)
        y = df[TARGET].values.astype(np.float32)
        return X, y

    X_train, y_train = _xy(df_train)
    X_val,   y_val   = _xy(df_val)
    X_test,  y_test  = _xy(df_test)

    log.info("  Features used: %d (dropped: %s)", len(ALL_FEATURES), FEATURES_TO_DROP)
    log.info("  Train churn: %.1f%% | Val: %.1f%% | Test: %.1f%%",
             y_train.mean()*100, y_val.mean()*100, y_test.mean()*100)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_test


# ─────────────────────────────────────────────────────────────
# STEP 2 — MULTICOLLINEARITY HANDLING
# ─────────────────────────────────────────────────────────────

def log_multicollinearity_handling():
    log.info("Step 2 — Multicollinearity handling")
    log.info("  Finding: anomaly_severity_score and anomaly_count_30d have r=1.000")
    log.info("  Decision: drop anomaly_severity_score")
    log.info("  Rationale: both encode same info; raw count is more interpretable for SHAP")


# ─────────────────────────────────────────────────────────────
# STEP 3 — BUILD MLP
# ─────────────────────────────────────────────────────────────

def build_dnn(input_dim: int) -> MLPClassifier:
    """
    Dense Neural Network: Input -> 128 -> 64 -> 32 -> Sigmoid

    sklearn MLPClassifier — same architecture as a Keras Dense network.
    early_stopping=True monitors validation loss and restores best weights.
    """
    log.info("Step 3 — Building MLP Neural Network")

    model = MLPClassifier(
        hidden_layer_sizes  = tuple(DNN_CFG["layers"]),
        activation          = "relu",
        solver              = "adam",
        learning_rate_init  = DNN_CFG["learning_rate"],
        max_iter            = DNN_CFG["max_epochs"],
        batch_size          = DNN_CFG["batch_size"],
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = DNN_CFG["patience"],
        random_state        = 42,
        verbose             = False,
    )

    log.info("  Architecture: Input(%d) -> %s -> Sigmoid",
             input_dim, " -> ".join(str(l) for l in DNN_CFG["layers"]))
    return model


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN
# ─────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, report):
    """
    sklearn MLPClassifier does NOT support sample_weight in fit().
    Workaround: oversample the minority (churn) class rows by the
    class weight ratio before fitting.

    e.g. weight = 2.86 → add 2 extra copies of every churn row.
    The optimizer then sees each churn sample ~3x more often,
    giving it equivalent gradient emphasis as class_weight would.
    """
    log.info("Step 4 — Training MLP Neural Network")

    n_neg        = int((y_train == 0).sum())
    n_pos        = int((y_train == 1).sum())
    class_weight = {0: 1.0, 1: round(n_neg / n_pos, 4)}
    log.info("  Class weights: %s", class_weight)
    log.info("  Strategy: minority class oversampling (weight ratio = %.2f)", class_weight[1])

    # Oversample minority rows
    minority_idx = np.where(y_train == 1)[0]
    n_copies     = max(1, round(class_weight[1])) - 1
    X_aug = np.vstack([X_train] + [X_train[minority_idx]] * n_copies)
    y_aug = np.concatenate([y_train] + [y_train[minority_idx]] * n_copies)

    # Shuffle augmented set
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    log.info("  Augmented train size: %d rows (%d original + %d minority copies)",
             len(X_aug), len(X_train), len(X_aug) - len(X_train))

    model.fit(X_aug, y_aug)

    epochs_run     = model.n_iter_
    best_val_score = max(model.validation_scores_) if model.validation_scores_ else 0.0
    log.info("  Trained %d epochs | best val score: %.4f", epochs_run, best_val_score)

    report["training"] = {
        "epochs_run":      epochs_run,
        "class_weight":    class_weight,
        "best_val_score":  round(float(best_val_score), 4),
        "oversampling_copies": n_copies,
    }

    history = {
        "loss":      model.loss_curve_,
        "val_score": model.validation_scores_,
    }
    return history, class_weight


# ─────────────────────────────────────────────────────────────
# STEP 5 — LEARNING CURVE
# ─────────────────────────────────────────────────────────────

def save_learning_curve(history, report):
    log.info("Step 5 — Learning curve analysis")

    n = len(history["loss"])
    hist_df = pd.DataFrame({
        "epoch":     range(1, n + 1),
        "loss":      history["loss"],
        "val_score": history["val_score"],
    })
    hist_df.to_parquet(MODEL_DIR / "p3_training_history.parquet", index=False)

    final_loss = history["loss"][-1]
    best_val   = max(history["val_score"]) if history["val_score"] else 0.0
    log.info("  Epochs: %d | Final loss: %.4f | Best val score: %.4f",
             n, final_loss, best_val)

    report["learning_curve"] = {
        "epochs":         n,
        "final_loss":     round(final_loss, 4),
        "best_val_score": round(float(best_val), 4),
    }


# ─────────────────────────────────────────────────────────────
# STEP 6 — THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────

def tune_threshold(y_true, y_prob, report):
    """
    Sweep precision-recall curve and find threshold maximising F1.
    Default 0.50 misses many churners on imbalanced data.
    Lower threshold = higher recall = catch more churners.
    """
    log.info("Step 6 — Threshold tuning (max F1 on churn class)")

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx  = int(np.argmax(f1))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.35

    log.info("  Optimal threshold: %.4f (default would be 0.50)", threshold)
    log.info("  At this threshold: precision=%.4f recall=%.4f F1=%.4f",
             prec[best_idx], rec[best_idx], f1[best_idx])

    report["threshold"] = {
        "value":     round(threshold, 4),
        "precision": round(float(prec[best_idx]), 4),
        "recall":    round(float(rec[best_idx]), 4),
        "f1":        round(float(f1[best_idx]), 4),
    }
    return threshold


# ─────────────────────────────────────────────────────────────
# STEP 7 — EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold, report):
    log.info("Step 7 — Final evaluation on held-out test set")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred,
                                 target_names=["Stay", "Churn"],
                                 output_dict=True)
    tn, fp, fn, tp = cm.ravel()

    log.info("  ROC-AUC    : %.4f", auc)
    log.info("  Avg Prec   : %.4f", ap)
    log.info("  Precision  : %.4f", rep["Churn"]["precision"])
    log.info("  Recall     : %.4f", rep["Churn"]["recall"])
    log.info("  F1         : %.4f", rep["Churn"]["f1-score"])
    log.info("  Confusion  : TN=%d FP=%d | FN=%d TP=%d", tn, fp, fn, tp)

    report["evaluation"] = {
        "roc_auc":   round(auc, 4),
        "avg_prec":  round(ap, 4),
        "precision": round(rep["Churn"]["precision"], 4),
        "recall":    round(rep["Churn"]["recall"], 4),
        "f1":        round(rep["Churn"]["f1-score"], 4),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    return y_prob


# ─────────────────────────────────────────────────────────────
# STEP 8 — SHAP
# ─────────────────────────────────────────────────────────────

def compute_shap(model, X_train, X_test, report):
    """
    KernelSHAP via shap.Explainer for sklearn MLPClassifier.
    100 background samples as reference distribution.
    """
    log.info("Step 8 — SHAP explainability")

    background = X_train[:100]
    explainer  = shap.Explainer(model.predict_proba, background, max_evals=500)
    shap_vals  = explainer(X_test[:50])

    sv = shap_vals.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs   = np.abs(sv).mean(axis=0)
    importance = dict(zip(ALL_FEATURES, mean_abs.round(6).tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    log.info("  Top SHAP features:")
    for feat, val in list(importance.items())[:5]:
        log.info("    %-40s %.4f", feat, val)

    pd.DataFrame(sv, columns=ALL_FEATURES).to_parquet(
        MODEL_DIR / "p3_shap_values.parquet", index=False
    )
    report["shap_importance"] = importance
    return sv


# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE
# ─────────────────────────────────────────────────────────────

def save_artifacts(model, y_prob, y_test, df_test, threshold, report):
    log.info("Step 9 — Saving all artifacts")

    joblib.dump(model, MODEL_DIR / "p3_dnn_model.joblib")

    pred_df = df_test.copy().reset_index(drop=True)
    pred_df["churn_prob"] = y_prob
    pred_df["churn_pred"] = (y_prob > threshold).astype(int)
    pred_df.to_parquet(MODEL_DIR / "p3_predictions.parquet", index=False)

    report["artifacts"] = {
        "model":       "p3_dnn_model.joblib",
        "predictions": "p3_predictions.parquet",
        "shap_values": "p3_shap_values.parquet",
        "history":     "p3_training_history.parquet",
        "metrics":     "p3_metrics.json",
    }
    report["features_used"]    = ALL_FEATURES
    report["features_dropped"] = FEATURES_TO_DROP

    # Copy threshold into top-level for easy loading
    report["roc_auc"]   = report["evaluation"]["roc_auc"]
    report["precision"] = report["evaluation"]["precision"]
    report["recall"]    = report["evaluation"]["recall"]
    report["f1"]        = report["evaluation"]["f1"]
    report["threshold"] = round(threshold, 4)
    report["class_weight"] = report["training"]["class_weight"]
    report["features"]  = ALL_FEATURES

    with open(MODEL_DIR / "p3_metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("  Saved: p3_dnn_model.joblib")
    log.info("  Saved: p3_predictions.parquet")
    log.info("  Saved: p3_shap_values.parquet")
    log.info("  Saved: p3_training_history.parquet")
    log.info("  Saved: p3_metrics.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def train():
    log.info("=" * 60)
    log.info("P3 — Churn Prediction MLP Training")
    log.info("=" * 60)

    report = {}

    X_train, y_train, X_val, y_val, X_test, y_test, df_test = load_data()
    log_multicollinearity_handling()

    model          = build_dnn(input_dim=len(ALL_FEATURES))
    history, _     = train_model(model, X_train, y_train, X_val, y_val, report)
    save_learning_curve(history, report)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold  = tune_threshold(y_val, y_prob_val, report)
    y_prob     = evaluate(model, X_test, y_test, threshold, report)
    compute_shap(model, X_train, X_test, report)
    save_artifacts(model, y_prob, y_test, df_test, threshold, report)

    log.info("=" * 60)
    log.info("P3 training complete — ROC-AUC: %.4f", report["evaluation"]["roc_auc"])
    log.info("=" * 60)
    return report


if __name__ == "__main__":
    train()
