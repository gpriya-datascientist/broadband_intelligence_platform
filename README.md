# 📡 Broadband Intelligence Platform

A production-grade ML platform that traces the causal chain:
**HFC network degradation → Wi-Fi experience decline → customer churn**

Three independent but causally linked projects, unified under one codebase.

---

## Architecture

```
broadband_intelligence/
├── config/
│   └── settings.py              ← Central config (all paths, hyperparams)
├── src/
│   ├── shared/
│   │   ├── logger.py            ← Shared structured logger
│   │   └── generate_data.py     ← Synthetic data generator (causal)
│   ├── p1_hfc_anomaly/
│   │   ├── prepare_data.py      ← 9-step cleaning pipeline
│   │   └── train_model.py       ← IF + LOF ensemble training
│   ├── p2_wifi_anomaly/
│   │   ├── prepare_data.py      ← 10-step cleaning pipeline
│   │   └── train_model.py       ← SARIMA + ±3σ training
│   └── p3_churn/
│       ├── prepare_data.py      ← 11-step cleaning pipeline
│       └── train_model.py       ← MLP DNN + SHAP training
├── api/
│   └── main.py                  ← FastAPI (13 endpoints)
├── ui/
│   ├── app.py                   ← Streamlit multi-page entry point
│   ├── components/
│   │   └── helpers.py           ← Cached model loaders, inference helpers
│   └── pages/
│       ├── overview.py          ← Platform landing page
│       ├── p1_hfc.py            ← P1: live scorer + feature importance
│       ├── p2_wifi.py           ← P2: forecast chart + breach detector
│       └── p3_churn.py          ← P3: single + batch + SHAP
├── data/
│   ├── raw/                     ← Parquet partitioned by date
│   ├── processed/               ← Clean train/val/test splits
│   ├── models/                  ← Trained model artifacts + metrics
│   └── reports/                 ← Cleaning reports + logs
├── run_pipeline.py              ← End-to-end pipeline runner
├── Dockerfile                   ← FastAPI container
└── requirements.txt
```

---

## Projects

### P1 — HFC Modem Anomaly Detection

**Problem:** Detect anomalous readings in HFC coaxial network metrics before customers notice.

**Data:** 4.32M rows — 500 modems × 96 readings/day × 90 days at 15-minute intervals.

**Model:** Isolation Forest (60%) + Local Outlier Factor (40%) ensemble.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7585 |
| Precision | 0.44 |
| Recall | 0.31 |
| Anomaly rate | ~11.8% |

**3 fault patterns injected:**
- SNR drift — gradual drop over 4 hours (plant/amplifier issue)
- Power spike — sudden upstream surge (field fault)
- Channel congestion — sustained utilisation surge (peak hour)

**Key cleaning decisions:**
- IQR computed on **normal readings only** — avoids anomalies inflating the bounds
- Range validation nulls impossible sensor values (e.g. SNR = -999) before imputation
- Time-aware split (not random) — prevents future readings leaking into training

---

### P2 — Wi-Fi Experience Anomaly Detection

**Problem:** Forecast hourly Wi-Fi experience scores and flag degradations early.

**Data:** 3.24M rows — 500 customers × 3 devices × 24h × 90 days.

**Model:** SARIMA(1,0,1)×(1,1,0,24) + ±3σ rule-based breach flagging.

| Metric | Value |
|--------|-------|
| MAE | 8.56 points |
| RMSE | 11.41 |
| Breach rate | ~3.9% |
| Causal linkage | 60% of breaches co-occur with P1 modem anomaly |

**SARIMA order rationale:**
- `p=1` — short-term AR lag
- `d=0` — series is stationary (ADF p=0.00)
- `q=1` — MA smoothing
- `P=1, D=1` — removes weekly seasonal trend
- `m=24` — daily cycle

**Cross-project signal:** `hfc_anomaly_flag` flows from P1 into P2 features.

---

### P3 — Customer Churn Prediction

**Problem:** Predict which customers will churn within the next month, with explanations.

**Data:** 500 customers, 26% churn rate. Churn label is **causally derived** (not random).

**Model:** MLP Neural Network — `Input(12) → Dense(128)→BN→DO → Dense(64)→BN→DO → Dense(32)→DO → Sigmoid`

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8571 |
| Recall | 0.9231 |
| F1 | 0.6316 |
| Tuned threshold | 0.0768 |

**Key decisions:**

| Decision | Rationale |
|----------|-----------|
| Class weights {0:1.0, 1:2.86} over SMOTE | 340 training rows too small for synthetic oversampling |
| Threshold tuned (0.077 vs default 0.50) | Recall prioritised — missing a churner costs more |
| Drop `anomaly_severity_score` | r=1.000 with `anomaly_count_30d` — redundant feature |
| Stratified split | 130 churners out of 500 — random split could misallocate all to train |

**Cross-project signals:** `anomaly_count_30d` (P1) + `wifi_breach_count_30d` (P2) feed into P3.

**SHAP top features (global):**
1. `contract_type` — month-to-month has highest churn rate
2. `tenure_months` — new customers churn most
3. `charge_per_tenure_ratio` — high value relative to tenure signals at-risk
4. `paperless_billing_flag`
5. `monthly_charges`

---

## Running the Platform

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
# Full run: generate → prepare → train all three projects
python run_pipeline.py

# Partial: skip data generation (data already exists)
python run_pipeline.py --steps p1 p2 p3

# Individual steps
python run_pipeline.py --steps p3
```

### 3. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs at: http://localhost:8000/docs
```

### 4. Start UI

```bash
streamlit run ui/app.py
```

### 5. Docker (API only)

```bash
docker build -t broadband-api .
docker run -p 8000:8000 broadband-api
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model load status |
| GET | `/metrics` | All three model metrics |
| POST | `/api/p1/predict` | HFC anomaly score |
| POST | `/api/p2/predict` | Wi-Fi breach detection |
| POST | `/api/p3/predict` | Customer churn probability + top reasons |
| POST | `/api/p3/batch` | Batch churn scoring |
| GET | `/api/p3/predictions` | All test set predictions |
| GET | `/api/p3/shap/global` | Global SHAP feature importance |
| GET | `/api/p2/forecast-sample` | SARIMA forecast for UI chart |

---

## Data Pipeline — Cleaning Steps per Project

### P1 (9 steps)
1. Load raw parquet (stratified sample: 200k rows)
2. Schema validation — enforce correct dtypes
3. Range validation — null impossible sensor values
4. Missing imputation — forward-fill within modem, median fallback
5. Duplicate removal — exact + modem/timestamp dedup
6. Outlier capping — IQR 3× on **normal readings only**
7. Feature engineering recompute — on cleaned signals
8. Time-aware split — 80/20 by timestamp (no leakage)
9. Save + cleaning report

### P2 (10 steps)
1–5. Same as P1 (load, schema, range, missing, dedup)
6. Device type encoding — laptop=0, iot=1
7. Outlier capping — IQR on non-breach readings
8. Seasonality features — hour_sin/cos, dow_sin/cos, is_peak, is_weekend
9. Time-aware split
10. Save + cleaning report

### P3 (11 steps)
1. Load churn customer parquet
2. Schema validation
3. Business rule validation — tenure≥1, charges>0, binary target
4. Missing imputation — median for numeric, mode for categorical
5. Duplicate removal
6. Class imbalance analysis — compute class weights
7. Categorical encoding — LabelEncoder (contract, payment, internet)
8. Feature correlation analysis — flag r>0.85 pairs
9. Stratified split — 340/60/100 preserving 26% churn rate
10. Feature scaling — StandardScaler fit on train only
11. Save train/val/test + scaler + encoders + cleaning report

---

## Interview Talking Points

- **Leakage prevention:** Time-aware split for time-series; scaler fit on train only; IQR computed on normal readings only
- **Imbalance handling:** Class-weighted loss over SMOTE — transparent and avoids overfitting on 340 rows
- **Threshold tuning:** Sweeping precision-recall curve; tuned to 0.077 (vs default 0.50) for recall-first business logic
- **Multicollinearity:** Detected r=1.000 between anomaly_severity_score and anomaly_count_30d — dropped one with documented rationale
- **Causal chain:** P1 network anomalies → P2 Wi-Fi degradation → P3 churn; features flow across all three
- **Parquet + partitioning:** 4.32M rows; date-partitioned parquet cuts load time by 30–50×
- **SHAP in production:** Feature attribution surfaced inline in UI, not buried in a notebook
- **SARIMA order:** Can explain each p,d,q,P,D,Q,m term and why

---

## Next Steps

- [ ] Replace sklearn MLP with TensorFlow/Keras for richer training callbacks
- [ ] Per-device SARIMA models (currently representative single-device model)
- [ ] Online retraining trigger when drift detected (KS test on feature distributions)
- [ ] A/B test retention actions against churn model predictions
- [ ] Stream P1 modem metrics from Kafka for real-time anomaly alerting
- [ ] Deploy to AWS App Runner (FastAPI) + Streamlit Community Cloud (UI)
