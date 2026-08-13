# ⚡ Energy Demand Forecasting — ML System

[![CI](https://github.com/narendrakalam2001/energy-demand-forecasting-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/narendrakalam2001/energy-demand-forecasting-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Champion](https://img.shields.io/badge/Champion-ElasticNet-brightgreen.svg)](#-champion-model-results)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange.svg)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-54%20passing-brightgreen.svg)](#-test-coverage)

> **Domain:** Energy / Utilities · Grid Operations
> **Problem:** Time Series Forecasting — 24-Hour Ahead Electricity Demand (MW)
> **Dataset:** Spain ESIOS/REE Grid Data (Kaggle) — hourly demand, weather, price, and generation-mix features
> **Industry Context:** Simulates a grid-operator dispatch pipeline (like NLDC/POSOCO) — 4-tier NORMAL / MONITOR / RESERVE / EMERGENCY dispatch decision engine

---

## 💡 Why This Project Matters

This system is used by utilities like **Tata Power, Adani, NTPC, Siemens India** to:

- Balance grid load in real time
- Avoid blackouts by activating spinning reserves early
- Optimize Power Purchase Agreements (PPAs)
- Reduce imbalance costs (India: ₹4–8/kWh penalty for grid deviation)

The system combines forecasting with grid governance:

- An **ElasticNet regressor** (selected from 14 tuned models, including a classical SARIMA
  baseline, via `TimeSeriesSplit` CV) forecasts hourly demand from lag, rolling-stat,
  cyclical-time, weather, and generation-mix features
- A **4-tier Dispatch Decision Engine** applies hard + soft grid rules **before** the forecast is
  acted on — demand crossing the peak threshold triggers `ACTIVATE_RESERVE` regardless of nuance
- Every model promotion goes through a **3-gate Champion vs Challenger** check (RMSE improvement
  ≥ 50 MW, R² ≥ 0.85, generalization gap ≤ 10%) — mirroring how grid operators govern forecast
  model changes
- A **Grid Imbalance Cost Evaluation** translates forecast error into real EUR terms — under- vs
  over-forecast cost, not just an abstract RMSE number

---

## 🏆 Champion Model Results

**Best Model — ElasticNet**

| Metric | Value |
|---|---|
| **Champion Model** | `ElasticNet` |
| **RMSE** | `712.04 MW` |
| **MAE** | `525.68 MW` |
| **MAPE** | `1.8675%` |
| **sMAPE** | `1.8602%` |
| **R²** | `0.9759` |
| **CVRMSE** | `2.4531%` |
| **CV Mean RMSE (TS split)** | `746.02 ± 35.56 MW` |
| **Train-Test Gap** | `0.0498` |
| **Demand Band Accuracy** | `96.17%` (5,053 / 5,254) |

> Industry benchmark: MAPE < 3% = excellent, < 5% = good, < 10% = acceptable — this system runs
> well inside the excellent range.
> *See `energy_models/model_card_ElasticNet_v1.json` for full metrics*

---

## 🔗 Live Links

| Service | URL |
|---|---|
| 🚀 **API Docs (Swagger UI)** | [energy-demand-forecast-api.onrender.com/docs](https://energy-demand-forecast-api.onrender.com/docs) |
| 📊 **Monitoring Dashboard** | [energy-demand-forecasting-mlops.streamlit.app](https://energy-demand-forecasting-mlops.streamlit.app) |
| 📓 **EDA Notebook** | [notebooks/energy_demand_eda.ipynb](notebooks/energy_demand_eda.ipynb) |

> ⚠️ Render free tier: first request may take 30–60 seconds (cold start).

---

## 🏗️ System Architecture

![System Architecture](docs/architecture/system_architecture.svg)

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║           ENERGY DEMAND FORECASTING — 5-LAYER PRODUCTION SYSTEM                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────── DATA LAYER ──────────────────────────────┐     ║
║  │  energy_data (hourly MW) + weather_data (temp, wind)  →  Spain ESIOS    │     ║
║  │  Merge + Validate  →  12 feature categories engineered                  │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌─────────────────────────── TRAINING PIPELINE ───────────────────────────┐     ║
║  │                                                                         │     ║
║  │  ┌──────────────────┐    ┌───────────────┐    ┌──────────────────────┐  │     ║
║  │  │  Scaled +        │    │  14 Models    │    │  Evaluation          │  │     ║
║  │  │  Unscaled Dual   │───▶│  TimeSeriesS- │───▶│  RMSE · MAE · MAPE  │  │     ║
║  │  │  Preprocessors   │    │  plit CV      │    │  sMAPE · R² · CVRMSE │  │     ║
║  │  └──────────────────┘    └───────────────┘    └──────────────────────┘  │     ║
║  │                                                                         │     ║
║  │  Ridge · Lasso · ElasticNet ⭐ · KNN · DecisionTree · RandomForest ·   │     ║
║  │  ExtraTrees · GradientBoosting · HistGBM · XGBoost · LightGBM ·         │     ║
║  │  CatBoost · MLP NeuralNet · SARIMA (classical statistical baseline)     │     ║
║  │                                                                         │     ║
║  │  Filter: gap ≤ 0.05 · cv_std ≤ 500 MW · R² ≥ 0.90                       │     ║
║  │  CHAMPION → ElasticNet  RMSE=712.04 MW  R²=0.9759  MAPE=1.87%           │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌──────────────────────── CHAMPION-CHALLENGER ────────────────────────────┐     ║
║  │                                                                         │     ║
║  │  Gate 1: RMSE improvement  ≥ 50 MW  →  ✅ PASS / ❌ FAIL               │     ║
║  │  Gate 2: R²                ≥ 0.85   →  ✅ PASS / ❌ FAIL               │     ║
║  │  Gate 3: Train-test gap    ≤ 10%    →  ✅ PASS / ❌ FAIL               │     ║
║  │                                                                         │     ║
║  │  ALL gates pass → PROMOTED (latest_model.json updated)                  │     ║
║  │  ANY gate fails → REJECTED (champion retained, result logged)           │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌──────────────────────────── SERVING LAYER ──────────────────────────────┐     ║
║  │                                                                         │     ║
║  │  Model Loader  →  Prediction Service  →  Dispatch Engine  →  FastAPI    │     ║
║  │                                                                         │     ║
║  │  POST /predict → single-hour forecast     GET /model_info → model card  │     ║
║  │  POST /predict/batch → multi-hour batch   GET /health → status          │     ║
║  │                                                                         │     ║
║  │  Dispatch Engine:  Grid Rules → ML Forecast → Demand Band               │     ║
║  │                    → NORMAL / MONITOR / RESERVE / REDUCE / EMERGENCY    │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌─────────────────── MONITORING LAYER — STREAMLIT DASHBOARD ──────────────┐     ║
║  │                                                                         │     ║
║  │  Section 1: Real-Time Alerts     → MAPE>10% · RMSE>3000MW · PSI drift   │     ║
║  │  Section 2: Champion-Challenger  → 3-gate ✅/❌ status · history       │     ║
║  │  Section 3: Forecast KPIs+Charts → RMSE·MAPE·forecast vs actual·bands   │     ║
║  │  Section 4: PSI Drift            → per-feature 🔴🟡🟢                 │     ║
║  │  Section 5: Recent API Predictions → demand band · dispatch action      │     ║
║  │                                                                         │     ║
║  │  Simulator: 3 scenarios (random · peak · offpeak) → hits /predict API   │     ║
║  └─────────────────────────────────────────────────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 📸 Dashboard Screenshots

### 🖥️ Full Dashboard UI

Real-time demand forecast + dispatch decisions + Champion vs Challenger status.

![Dashboard](docs/screenshots/dashboard_full_ui.png)

---

### 📊 Forecast Performance KPIs

RMSE, MAPE, normal-operation rate — live from the model card.

![KPIs](docs/screenshots/forecast_performance_kpis.png)

---

### 📉 Demand Band Breakdown (LOW / NORMAL / HIGH / PEAK)

Hour counts across demand bands.

![Demand Band](docs/screenshots/demand_band_breakdown.png)

---

### 📊 Forecast Error Distribution

Error histogram across the test period.

![Error Distribution](docs/screenshots/forecast_error_distribution.png)

---

### 📉 Feature Drift Report (PSI)

PSI drift monitoring with 🔴🟡🟢 status flags.

![Drift Report](docs/screenshots/feature_drift_report.png)
![Drift Scores](docs/screenshots/feature_drift_scores.png)

---

### 🔍 Recent API Predictions

Live prediction log with demand band and dispatch action taken.

![Predictions](docs/screenshots/recent_api_predictions.png)

---

## 📊 Training Reports

| All Models Summary | Top Feature Importances |
|---|---|
| ![Model Summary](docs/reports/training_model_summary.png) | ![Feature Importances](docs/reports/top_feature_importances.png) |

| Champion vs Challenger | Test Coverage |
|---|---|
| ![Challenger](docs/reports/challenger_evaluation.png) | ![Tests](docs/reports/test_coverage.png) |

| Forecast vs Actual — Scatter | Forecast vs Actual + Residuals |
|---|---|
| ![Scatter](docs/plots/forecast_scatter.png) | ![Residuals](docs/plots/forecast_vs_actual.png) |

---

## 🎬 System Demo

![System Demo](docs/gifs/system_demo.gif)

---

## 📁 Project Structure

```
energy-demand-forecasting-mlops/
│
├── src/                                 # Core ML system
│   ├── config.py                        # Constants, thresholds, demand bands
│   ├── data_loader.py                   # Validation, merge, feature engineering
│   ├── preprocessing.py                 # Clipper, dual ColumnTransformer, TS split
│   ├── model_tuning.py                  # Model grids, tune_models (TimeSeriesSplit)
│   ├── metrics.py                       # RMSE, MAE, MAPE, PSI, imbalance cost eval
│   ├── forecast_engine.py               # 4-tier dispatch decision engine
│   ├── evaluation.py                    # Eval, SHAP, MLflow, save
│   ├── leakage_check.py                 # TS leakage detection (future-leaky cols)
│   ├── model_card.py                    # Google Model Cards standard
│   ├── model_loader.py                  # Champion load + challenger comparison
│   └── training_pipeline.py             # Full orchestration
│
├── services/
│   └── prediction_service.py            # Feature prep + predict_demand
│
├── serving/
│   └── energy_demand_api.py             # FastAPI: /predict · /predict/batch · /health · /model_info
│
├── monitoring/
│   └── monitoring_dashboard.py          # Streamlit: 5 sections
│
├── simulation/
│   └── grid_simulator.py                # 3-scenario load simulator
│
├── scripts/
│   ├── train_model.py                   # Entry point for training pipeline
│   ├── run_api.py                       # Launch FastAPI server
│   ├── run_dashboard.py                 # Launch Streamlit dashboard
│   └── run_simulation.py                # Run grid simulation
│
├── tests/
│   └── test_pipeline_core.py            # 54 pytest unit tests (54/54 passing)
│
├── notebooks/
│   └── renergy_demand_eda.ipynb         # Professional EDA — 22 steps
│
├── data/
│   ├── sample_energy_dataset.csv        # A representative balanced sample dataset is provided in quick testing
│   └── sample_weather_features.csv      # Sample — download full from Kaggle
│
├── energy_models/                       # Model artifacts — all files pushed to GitHub
│   ├── energy_model_ElasticNet_v1.joblib
│   ├── latest_model.json                # Current champion registry
│   ├── model_card_ElasticNet_v1.json    # Structured model card
│   ├── challenger_log.json              # Champion vs Challenger history
│   ├── model_experiment_results.csv     # All 14 models comparison table
│   ├── monitor_scores.csv               # Test set scores for monitoring
│   └── feature_drift_report.csv         # PSI drift scores per feature
│
├── docs/
│   ├── architecture/
│   │   └── system_architecture.svg      # 5-layer system architecture
│   ├── gifs/
│   │   └── system_demo.gif              # End-to-end demo recording
│   ├── plots/
│   │   ├── elasticnet_scatter.png       # Forecast vs actual scatter
│   │   └── elasticnet_forecast_residuals.png  # Residuals time series
│   ├── reports/
│   │   ├── training_model_summary.png   # All models comparison table
│   │   ├── top_feature_importances.png  # SHAP / permutation importance
│   │   ├── challenger_evaluation.png    # Champion vs challenger gates
│   │   ├── simulation.png               # Grid simulator output
│   │   └── test_coverage.png            # pytest coverage report
│   └── screenshots/
│       ├── dashboard_full_ui.png
│       ├── forecast_performance_kpis.png
│       ├── demand_band_breakdown.png
│       ├── forecast_error_distribution.png
│       ├── feature_drift_report.png
│       ├── feature_drift_scores.png
│       └── recent_api_predictions.png
│
├── Dockerfile                           # API Docker image (multi-stage)
├── Dockerfile.dashboard                 # Streamlit Docker image
├── docker-compose.yml                   # API + Dashboard together
├── .dockerignore
├── .github/workflows/ci.yml             # GitHub Actions — pytest on every push
├── .gitignore
├── LICENSE                              # MIT License
├── README.md                            # This file
├── requirements.txt                     # Full dependencies
├── requirements_api.txt                 # API-only (Render deployment)
├── requirements_dashboard.txt           # Dashboard-only (Streamlit Cloud)
├── runtime.txt                          # Python version for Render
└── render.yaml                          # Render deployment config
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/narendrakalam2001/energy-demand-forecasting-mlops.git
cd energy-demand-forecasting-mlops
pip install -r requirements.txt
```

### 2. Download Dataset

Representative sample datasets are provided in `data/`. Download the full
[Spain Energy Grid Data](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
from Kaggle for full-scale training.

### 3. Train Model

```bash
python scripts/train_model.py
```

Expected output:
```
INFO  TS CV | mean_RMSE=746.02  std=35.56
INFO  Candidates found with gap<=0.05 cv_std<=500 r2>=0.90
INFO  Selected model: ElasticNet
INFO  Metrics | RMSE=712.04  MAE=525.68  MAPE=1.87%  R²=0.9759  CVRMSE=2.45%
INFO  CHAMPION vs CHALLENGER
      Champion  : ElasticNet   RMSE=712.04  R²=0.9759
      Challenger: ElasticNet   RMSE=712.04  R²=0.9759
❌ CHALLENGER REJECTED — champion 'ElasticNet' retained
INFO  TRAINING COMPLETE | Best model: ElasticNet
      RMSE: 712.04 MW  MAPE: 1.87%  R²: 0.9759
```

### 4. Start API

```bash
python scripts/run_api.py
# → http://localhost:8000/docs
```

### 5. Start Monitoring Dashboard

```bash
python scripts/run_dashboard.py
# → http://localhost:8501
```

### 6. Run Grid Simulator

```bash
python scripts/run_simulation.py
```

### 7. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# 54 collected · 54 passed
```

---

## 🐳 Docker

```bash
# API only
docker build -t energy-demand-api .
docker run -p 8000:8000 -v ./energy_models:/app/energy_models energy-demand-api

# API + Dashboard together
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | `http://localhost:8000` |
| Streamlit Dashboard | `http://localhost:8501` |

---

## 🔌 API Reference

| Method | Route | Description |
|---|---|---|
| GET | `/` | Health + links |
| GET | `/health` | Model loaded status |
| GET | `/model_info` | Model card JSON |
| POST | `/predict` | Single-hour forecast |
| POST | `/predict/batch` | Multi-hour batch forecast |

### POST /predict — Single-Hour Forecast

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 20,
    "day_of_week": 2,
    "month": 7,
    "is_weekend": 0,
    "demand_lag_1h": 38500.0,
    "demand_lag_24h": 37200.0,
    "demand_lag_168h": 36800.0,
    "demand_roll_mean_24h": 35000.0,
    "demand_roll_std_24h": 2500.0,
    "temp_celsius": 32.0
  }'
```

**Response:**
```json
{
  "forecast_mw": 41230.5,
  "demand_band": "HIGH",
  "dispatch_action": "ACTIVATE_RESERVE",
  "rule_triggered": "PEAK_DEMAND_THRESHOLD",
  "latency_seconds": 0.031
}
```

### GET /health

```json
{"status": "running", "model_loaded": true}
```

---

## 🧠 Technical Standards

| Component | Implementation |
|---|---|
| **Validation Strategy** | Chronological train/val/test split + `TimeSeriesSplit` CV — no leakage across time |
| **Dual Preprocessors** | `preprocessor_scaled` (linear models) + `preprocessor_unscaled` (tree models) |
| **Feature Engineering** | 12 categories — calendar, cyclical, lag (1h–168h), rolling stats, temperature, price, renewables |
| **Models Tuned** | Ridge · Lasso · **ElasticNet** · KNN · DecisionTree · RandomForest · ExtraTrees · GradientBoosting · HistGBM · XGBoost · LightGBM · CatBoost · MLP · SARIMA |
| **Selection Filter** | `train_test_gap ≤ 0.05` · `cv_std_rmse ≤ 500 MW` · `R² ≥ 0.90` |
| **Champion-Challenger** | 3-gate: RMSE improvement ≥ 50 MW · R² ≥ 0.85 · Train-test gap ≤ 10% |
| **Explainability** | SHAP `KernelExplainer` — top raw feature: `demand_lag_1h` (importance ≈ 7460) |
| **Experiment Tracking** | MLflow — metrics · params · model artifact per run |
| **Model Card** | Google Model Cards standard — metrics + grid imbalance cost |
| **Leakage Detection** | Time-series-aware — future-leaky column check |
| **Dispatch Engine** | Hard + soft grid rules → NORMAL / MONITOR / RESERVE / REDUCE / EMERGENCY |
| **CI/CD** | GitHub Actions — pytest on every push |
| **Deployment** | Render.com (FastAPI) + Streamlit Cloud (Dashboard) |

---

## 📊 All 14 Models — Comparison Table

Full evaluation run, sorted by test RMSE:

| Model | Train RMSE | Test RMSE | Gen. Gap | CV Mean RMSE | MAPE | R² | CVRMSE |
|---|---|---|---|---|---|---|---|
| RandomForest | 180.58 | 540.40 | 0.6658 | 530.99 ± 50.90 | 1.2854% | **0.9861** | 1.8618% |
| ExtraTrees | 0.00 | 559.71 | 1.0000 ⚠️ | 550.85 ± 55.31 | 1.3649% | 0.9851 | 1.9283% |
| LightGBM | 395.10 | 598.70 | 0.3401 | 528.87 ± 42.40 | 1.5142% | 0.9830 | 2.0627% |
| GradientBoosting | 379.99 | 620.02 | 0.3871 | 583.80 ± 71.48 | 1.5810% | 0.9818 | 2.1361% |
| HistGBM | 418.00 | 639.96 | 0.3468 | 566.13 ± 64.57 | 1.6374% | 0.9806 | 2.2048% |
| XGBoost | 377.42 | 648.64 | 0.4181 | 591.53 ± 50.70 | 1.6685% | 0.9800 | 2.2347% |
| CatBoost | 576.08 | 669.43 | 0.1395 | 627.43 ± 53.35 | 1.7133% | 0.9787 | 2.3063% |
| Lasso | 746.58 | 705.81 | 0.0578 | 741.10 ± 34.95 | 1.8410% | 0.9764 | 2.4317% |
| Ridge | 746.58 | 705.82 | 0.0577 | 741.10 ± 34.95 | 1.8410% | 0.9764 | 2.4317% |
| **ElasticNet** ⭐ | 747.50 | **712.04** | 0.0498 | 746.02 ± 35.56 | 1.8675% | 0.9759 | 2.4531% |
| DecisionTree | 337.61 | 717.00 | 0.5291 | 767.35 ± 64.73 | 1.6321% | 0.9756 | 2.4702% |
| NeuralNet (MLP) | 342.59 | 1013.44 | 0.6620 | 2232.01 ± 824.65 | 2.8591% | 0.9513 | 3.4915% |
| KNN | 0.00 | 1182.00 | 1.0000 ⚠️ | 1202.81 ± 134.49 | 3.1317% | 0.9337 | 4.0723% |
| SARIMA | 751.82 | 20,625.12 🔴 | 0.9635 ⚠️ | 95,690.02 ± 96,041.70 | 64.209% | **-19.1807** | 71.0581% |

> ⚠️ **RandomForest, ExtraTrees, LightGBM, and several tree models had lower raw test RMSE** than
> ElasticNet, but were excluded by the selection filter (`gap ≤ 0.05`, `cv_std ≤ 500 MW`) —
> RandomForest and ExtraTrees show large train-test gaps (0.66–1.00), indicating overfitting to
> the training period despite good test-set RMSE. **ElasticNet** was selected as the
> lowest-RMSE candidate that passed the stricter generalization filter.
>
> 🔴 **SARIMA — classical statistical baseline — was added specifically to close the "no pure
> time-series statistical model" gap in the model roster.** It performs catastrophically worse
> than every ML candidate (RMSE 29× higher than champion, negative R² of -19.18, meaning it's far
> worse than simply predicting the mean). This is expected: SARIMA models a single series with
> one seasonal period, while this grid-demand series has multiple overlapping seasonalities
> (daily, weekly, and annual) plus strong exogenous drivers (temperature, price, generation mix)
> that SARIMA cannot use as regressors the way the ML models do via the 12 engineered feature
> categories (lag, rolling, calendar, cyclical, temperature, price, renewables, etc.). It is
> kept in the comparison table as an honest baseline, not excluded, to demonstrate why the
> ML-with-feature-engineering approach was chosen over classical time-series statistics for this
> problem.
>
> Full comparison generated at training time in `energy_models/model_experiment_results.csv`
> and visualised in `docs/reports/training_model_summary.png`.

---

## 💰 Grid Imbalance Cost Evaluation

Cost model — EUR/MWh imbalance penalty (equivalent to Credit Risk ECL):

| Metric | Value |
|---|---|
| Total Forecast Error | `2,761,935 MW` |
| Mean Error | `525.68 MW` |
| Under-forecast Hours | `2,112` |
| Over-forecast Hours | `3,142` |
| Peak Hours Affected | `4` |
| Estimated Under-forecast Cost | `€91,658,130` |
| Estimated Over-forecast Cost | `€143,342,410` |
| **Total Estimated Cost** | **`€235,000,541`** |

> Every 1% MAPE improvement is estimated to reduce imbalance costs by roughly €2M/year at this scale.

**Demand band accuracy:** `96.17%` (5,053 / 5,254 hours correctly classified LOW/NORMAL/HIGH/PEAK)

---

## 🎯 Dispatch Decision Engine

4-tier grid dispatch decision system mimicking real NLDC/POSOCO operations:

| Decision | Trigger | Grid Action |
|---|---|---|
| `NORMAL_OPERATION` | Demand in baseline range | Maintain scheduled generation |
| `MONITOR_HIGH` | Demand approaching peak (≥ 85% of threshold) | Watch closely, pre-position reserves |
| `ACTIVATE_RESERVE` | Demand ≥ 42,000 MW (peak threshold) | Call spinning reserves, notify DSO |
| `REDUCE_GENERATION` | Demand ≤ 20,000 MW (low demand) | Ramp down thermal plants, store hydro |
| `EMERGENCY_PROCUREMENT` | Price spike AND near-peak demand | Buy spot power, emergency procurement |

Rules are checked **before** ML forecast — mirroring real grid control-centre logic.

**Real run — forecast engine decisions** (test set, `n=5,254` hours):

| Decision | Count |
|---|---|
| `NORMAL_OPERATION` | `4,918` |
| `MONITOR_HIGH` | `324` |
| `REDUCE_GENERATION` | `12` |

---

## 🏆 Champion vs Challenger System

Every new training run is compared against the production champion using **3 promotion gates**:

| Gate | Condition | Rationale |
|---|---|---|
| RMSE Improvement | Challenger must beat champion by ≥ 50 MW | Meaningful forecast improvement only |
| R² Gate | Challenger must have R² ≥ 0.85 | Minimum variance explained for grid decisions |
| Generalization Gap | Train-test RMSE gap ≤ 10% | No overfitting to training period |

**Real run example:**

```
CHAMPION vs CHALLENGER
  Champion  : ElasticNet   RMSE=712.04  R²=0.9759
  Challenger: ElasticNet   RMSE=712.04  R²=0.9759
❌ CHALLENGER REJECTED — champion 'ElasticNet' retained
   Reason: Gates failed: RMSE improvement +0.00 MW < 50.0 MW required
```

Results logged to `energy_models/challenger_log.json` and visible in the dashboard.

---

## 📈 Monitoring Dashboard — 5 Sections

| Section | What it shows |
|---|---|
| **1. Real-Time Alerts** | MAPE > 10% · RMSE > 3000 MW · PSI drift · emergency rate |
| **2. Champion vs Challenger** | 3-gate promotion history + gate results |
| **3. Forecast KPIs + Charts** | RMSE, MAPE, forecast vs actual, dispatch distribution, demand bands, error histogram |
| **4. PSI Drift Report** | Feature drift (train vs live) with 🔴🟡🟢 status |
| **5. Recent API Predictions** | Live prediction log table |

---

## 🔬 Feature Importance & Drift

**Top feature importances (from tree-based comparison model):**

| Feature | Importance |
|---|---|
| `demand_lag_1h` | 7460.49 |
| `demand_lag_2h` | 2078.40 |
| `price_lag_1h` | 1174.09 |
| `price_actual` | 1092.26 |
| `hour_cos` | 522.96 |
| `demand_roll_mean_12h` | 450.14 |
| `demand_roll_mean_6h` | 420.45 |
| `is_overnight` | 69.19 |
| `generation_hydro_water_reservoir` | 61.08 |
| `generation_fossil_gas` | 33.91 |
| `generation_solar` | 18.82 |
| `price_day_ahead` | 15.20 |
| `demand_lag_24h` | 13.48 |
| `demand_roll_mean_24h` | 11.85 |
| `demand_lag_3h` | 11.41 |

**Top PSI drift (train vs test), SHAP via `KernelExplainer`:**

| Feature | PSI |
|---|---|
| `day_of_year` | 4.2660 |
| `month` | 3.8947 |
| `month_sin` | 3.3690 |
| `week_of_year` | 3.0153 |
| `price_demand_ratio` | 2.3057 |
| `price_lag_1h` | 1.9678 |
| `price_actual` | 1.9676 |
| `price_day_ahead` | 1.3716 |
| `month_cos` | 1.1359 |
| `temp_min` | 0.1831 |

> ⚠️ Several calendar/seasonal features show elevated PSI (well above the typical 0.10–0.25
> stability range) — this is largely expected for a chronological train/test split where the
> test period covers different calendar months/seasons than training, but should still be
> reviewed to confirm it isn't masking a genuine distributional shift.

---

## 🧪 Test Coverage — 54/54 Passing

54 unit tests across 12 test classes:

| Class | Tests | What it covers |
|---|---|---|
| `TestClipper` | 6 | Shape, outlier removal, normal data, 1D, feature names, train/test bounds |
| `TestSafePowerTransformer` | 3 | Normal data, zero-variance passthrough, feature names |
| `TestBuildPreprocessors` | 5 | 3-tuple return, feature order, no-NaN scaled, no-NaN unscaled, all-NaN crash fix |
| `TestAddTimeFeatures` | 4 | Hour column, cyclical range, weekend flag, season range |
| `TestAddLagFeatures` | 3 | Lag columns created, lag-1h value, first row NaN |
| `TestDetectFeatureTypes` | 3 | Binary detection, target excluded, continuous detection |
| `TestDetectLeakage` | 4 | Identical feature, clean data, known leaky cols, empty df |
| `TestRegressionMetrics` | 8 | RMSE, MAE, MAPE, sMAPE, R², CVRMSE (perfect + nonzero cases) |
| `TestPSI` | 3 | Identical distributions, shifted distribution, disjoint distribution |
| `TestCostEvaluation` | 3 | Required keys, perfect=zero cost, peak costs more than off-peak |
| `TestDemandBandAccuracy` | 2 | Perfect accuracy, wrong band lowers accuracy |
| `TestForecastEngine` | 6 | LOW/NORMAL/HIGH/PEAK bands, NORMAL_OPERATION, ACTIVATE_RESERVE |
| `TestConfig` | 4 | PSI thresholds ordered, gate constants reasonable, band boundaries, positive thresholds |

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🔬 12 Feature Categories

| Category | Features |
|---|---|
| Calendar | hour, dow, month, quarter, day_of_year, week_of_year |
| Cyclical | hour_sin/cos, dow_sin/cos, month_sin/cos |
| Boolean | is_weekend, is_morning_peak, is_evening_peak, is_overnight |
| Season | season (0=Winter, 1=Spring, 2=Summer, 3=Autumn) |
| Lag demand | demand_lag_1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h |
| Rolling stats | roll_mean/std over 6h, 12h, 24h, 48h, 168h windows |
| Temperature | temp_celsius, cooling_degree, heating_degree, temp_x_hour |
| Price | price_lag_1h, price_change_1h, price_demand_ratio |
| Renewable | renewable_total, renewable_penetration |
| Demand delta | demand_delta_1h, demand_delta_24h |
| Peak flags | is_peak_demand, is_low_demand |

---

## 🔁 Grid Simulator

```bash
python scripts/run_simulation.py
```

Supports **3 scenarios**:

| Scenario | Description |
|---|---|
| `random` | Mixed realistic hours across all seasons |
| `peak` | Evening peak hour profiles (high load stress test) |
| `offpeak` | Overnight / weekend low-demand profiles |

---

## 🛡️ Ethical Considerations

- Forecast-driven dispatch actions (especially `EMERGENCY_PROCUREMENT` and `ACTIVATE_RESERVE`)
  should be confirmed by grid control-centre operators, not fully automated
- Elevated PSI on calendar/seasonal features should be periodically reviewed — chronological
  splits naturally show some drift, but sustained high PSI can also signal a genuine regime shift
  (new demand pattern, policy change, major consumer behavior shift)
- Cost assumptions (€/MWh imbalance penalty) are illustrative and should be replaced with the
  operating grid's actual regulatory penalty structure before use in production
- Trained on the Spain ESIOS/REE grid — re-validate thresholds (peak demand MW, price spike
  definitions) before use on a different national or regional grid
- Not designed for: fully automated procurement/reserve-activation decisions without human sign-off

---

## 👨‍💻 About

**Narendra Kalam** — MSc Computer Science (Gold Medalist — NASSCOM, Full Stack Data Science + AI)

> Building 20+ industry-level, end-to-end ML systems targeting **30+ LPA** at top MNCs in India.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/narendra-kalam/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/narendrakalam)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?logo=github)](https://narendrakalam2001.github.io/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:kalamnarendra2001@gmail.com)

### Portfolio Projects

| # | Project | Domain | Champion Model | Key Metric |
|---|---|---|---|---|
| 1 | Credit Card Fraud Detection | BFSI / Fintech | ExtraTrees | F1 = 0.8962 · 284K transactions |
| 2 | Credit Risk Prediction | BFSI / Lending | LightGBM | F1 = 0.9741 · ROC-AUC = 0.9991 |
| 3 | Customer Churn Prediction | Telecom / BFSI | CatBoost | F1 = 0.634 · Recall = 0.7312 |
| 4 | House Price Prediction | Real Estate | CatBoost | RMSE = $20,128 · R² = 0.9053 |
| 5 | Store Sales Forecasting | Retail / Supply Chain | LightGBM (Ensemble) | RMSLE = 0.3739 · R² = 0.9761 |
| 6 | **Energy Demand Forecasting** | **Energy / Utilities** | **ElasticNet** | **RMSE = 712.04 MW · R² = 0.9759** |
| 7 | Stock Price & Risk Forecasting | Fintech / Capital Markets | Ridge | DirAcc = 53.44% · Sharpe = 0.80 |
| 8 | Resume Screener AI | HR Tech | LightGBM | F1 = 0.7608 · Top-3 = 0.9416 |
| 9 | ABSA Sentiment Analysis | E-Commerce / Banking | RidgeClassifier | Macro-F1 = 0.6212 · ROC-AUC = 0.823 |
| 10 | Fake News Detector | Media Tech / Gov Tech | XGBoost | F1 = 0.9993 · ROC = 1.0000 |
| 11 | BC5CDR Clinical NER | Biomedical NLP | BioBERT | F1 = 0.8847 · Chemical F1 = 0.9239 |
| 12 | News Topic Modeling | Media Analytics | LDA (Gensim) | Cv = 0.6225 · Diversity = 0.92 |
| 13 | Chest X-Ray Diagnosis | Healthcare AI | DenseNet121 | Mean AUC = 0.7864 · 14 classes |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)