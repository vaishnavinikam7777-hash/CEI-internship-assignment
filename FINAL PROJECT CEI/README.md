# AI-Powered Energy Analytics System

Uses smart meter data to **forecast consumption**, **identify usage patterns**,
and deliver **optimization insights** for smarter energy management — built
against the schema of the Kaggle
["Smart meters in London"](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)
dataset.

> **Note on data access:** this environment can't reach kaggle.com directly, so
> the project ships with a synthetic data generator that reproduces the real
> dataset's schema and realistic consumption behavior (daily/weekly
> seasonality, weather-driven seasonal drift, appliance spikes). The loader
> auto-detects and uses real Kaggle CSVs if you drop them into `data/raw/` —
> nothing else in the pipeline needs to change. See "Using the real dataset" below.

## Architecture

```
Smart meter data (real or synthetic, same schema)
        │
        ▼
 data_loader.py ──► normalizes to LCLid / tstp / energy_kwh_hh (+ daily weather)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                              ▼                             │
 feature_engineering.py         pattern_analysis.py                  │
 (lags, rolling stats,          (KMeans clustering on daily          │
  calendar/cyclical features)    load-shape profiles;                │
        │                         peak/off-peak/weekday-weekend)     │
        ▼                              │                             │
 forecasting_model.py                  │                             │
  ML:  Gradient Boosting Regressor     │                             │
  DL:  MLP neural network              │                             │
  (+ naive persistence baseline)       │                             │
        │                              ▼                             │
        │                    optimization_insights.py                │
        │                    (rule-based, explainable                │
        │                     recommendations grounded in            │
        │                     the pattern-analysis numbers)          │
        ▼                              ▼                             ▼
                        app.py (Streamlit dashboard: 3 tabs)
```

### Forecasting (ML + DL)
`forecasting_model.py` trains two models on the same lag/rolling/calendar
feature table:
- **ML**: `GradientBoostingRegressor` (tree-based, classical)
- **DL**: `MLPRegressor` (feed-forward neural network)

Both are benchmarked against a **naive persistence baseline** (predict = same
half-hour yesterday) so the models' value is easy to verify — on the
synthetic dataset both beat the naive baseline on MAE/RMSE/R².

Forecasts are generated iteratively (each predicted step feeds back in as the
next lag), giving a genuine multi-step-ahead forecast rather than a single
one-step prediction.

### Usage pattern identification
`pattern_analysis.py` builds each household's average 48-slot daily load
profile and clusters households with **KMeans** into behavioral segments
(e.g. "Evening-peak", "Daytime/work-from-home", "Night-owl"), auto-labeled by
where each cluster's centroid peaks. It also computes per-household
peak-to-average ratio and weekday-vs-weekend averages — standard grid-planning
metrics.

### Optimization insights
`optimization_insights.py` turns those pattern-analysis numbers into
concrete, explainable recommendations (e.g. shifting deferrable loads off
the 5-9pm grid peak, tariff suitability, appliance-audit suggestions) — every
recommendation is directly traceable to a computed number, not a black box.

## Project structure

```
energy_analytics/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── data/raw/                  # drop real Kaggle CSVs here (see below)
├── models/                    # cached trained model artifacts (.joblib)
└── src/
    ├── synthetic_data.py       # realistic synthetic smart-meter data generator
    ├── data_loader.py          # loads real data if present, else synthetic
    ├── feature_engineering.py  # lag/rolling/calendar feature construction
    ├── forecasting_model.py    # ML (GBR) + DL (MLP) forecasters
    ├── pattern_analysis.py     # clustering + peak/off-peak analysis
    └── optimization_insights.py# rule-based recommendation engine
```

## Setup & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run trains and caches the forecasting models (~15-20s for the
synthetic demo dataset); later runs reuse the cached `.joblib` files.

## Using the real Kaggle dataset

1. Download from Kaggle: https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london
2. Place the files as:
   ```
   data/raw/halfhourly_dataset/*.csv       (the per-block half-hourly CSVs)
   data/raw/weather_daily_darksky.csv      (daily weather)
   ```
3. Delete any cached files in `models/` (they were trained on synthetic data)
   and restart the app — `data_loader.py` auto-detects the real files and
   `load_data()` will report `source = "kaggle_real_data"` in the sidebar.

The real dataset is large (~5000+ households, multiple years); `load_real_data()`
caps how many household files and rows it reads by default (`max_files`,
`max_rows_per_file` in `src/data_loader.py`) — raise those once you've
confirmed everything runs.

## Extending it

- **True per-household models**: currently forecasters are trained globally
  across all households pooled together; swap in per-`LCLid` models if you
  need household-specific accuracy over generalization.
- **Deep sequence models**: `MLPRegressor` stands in for "DL" here since this
  environment has no GPU/torch access; swap in an LSTM/GRU (PyTorch/TensorFlow)
  using the same lag-feature table if you have that available.
- **Real tariff data**: `optimization_insights.py` uses a generic 5-9pm UK
  peak window; wire in an actual time-of-use tariff schedule for precise
  £-savings estimates instead of directional recommendations.
