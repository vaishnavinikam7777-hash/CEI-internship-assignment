"""
Forecasting Models
------------------
Two models predict next-interval household energy consumption from
the engineered feature table:

  1. ML model:  GradientBoostingRegressor (classical, tree-based)
  2. DL model:  MLPRegressor (compact feed-forward neural network)

Both are trained on pooled half-hourly data across households (global
model) so they generalize across a whole portfolio of smart meters,
which mirrors how a real utility-scale forecasting system would work.
A naive persistence baseline (predict = same time yesterday) is also
reported so the ML/DL gains are easy to sanity-check.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from feature_engineering import build_supervised_table, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_training_table(consumption: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for hh_id, group in consumption.groupby("LCLid"):
        t = build_supervised_table(group, weather)
        tables.append(t)
    full = pd.concat(tables, ignore_index=True)
    # guard against missing temperature col if weather was empty
    for col in FEATURE_COLS:
        if col not in full.columns:
            full[col] = 0.0
    return full


def train_forecasters(consumption: pd.DataFrame, weather: pd.DataFrame, force=False):
    gbr_path = os.path.join(MODELS_DIR, "gbr_forecaster.joblib")
    mlp_path = os.path.join(MODELS_DIR, "mlp_forecaster.joblib")
    scaler_path = os.path.join(MODELS_DIR, "forecast_scaler.joblib")
    metrics_path = os.path.join(MODELS_DIR, "forecast_metrics.joblib")

    if not force and all(os.path.exists(p) for p in [gbr_path, mlp_path, scaler_path]):
        return (joblib.load(gbr_path), joblib.load(mlp_path), joblib.load(scaler_path),
                joblib.load(metrics_path) if os.path.exists(metrics_path) else {})

    table = build_training_table(consumption, weather)
    X = table[FEATURE_COLS].values
    y = table["energy_kwh_hh"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,
                                      random_state=42)
    gbr.fit(X_train, y_train)  # tree models don't need scaling
    gbr_pred = gbr.predict(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
                         random_state=42, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    mlp_pred = mlp.predict(X_test_s)

    # naive persistence baseline: predict = same time yesterday (lag_48)
    lag48_idx = FEATURE_COLS.index("lag_48")
    naive_pred = X_test[:, lag48_idx]

    def eval_model(y_true, y_pred):
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    metrics = {
        "gbr": eval_model(y_test, gbr_pred),
        "mlp": eval_model(y_test, mlp_pred),
        "naive_baseline": eval_model(y_test, naive_pred),
        "feature_importance_gbr": dict(zip(FEATURE_COLS, np.round(gbr.feature_importances_, 3).tolist())),
    }

    joblib.dump(gbr, gbr_path)
    joblib.dump(mlp, mlp_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(metrics, metrics_path)

    return gbr, mlp, scaler, metrics


def forecast_next(household_df: pd.DataFrame, weather: pd.DataFrame,
                    gbr, mlp, scaler, horizon: int = 48) -> pd.DataFrame:
    """
    Iteratively forecasts `horizon` half-hour steps ahead for one
    household by feeding each prediction back in as the next lag_1.
    Returns a DataFrame with timestamps + both models' predictions.
    """
    history = household_df.sort_values("tstp").copy().reset_index(drop=True)
    preds_gbr, preds_mlp, timestamps = [], [], []

    working = history.copy()
    for step in range(horizon):
        table = build_supervised_table(working, weather)
        if table.empty:
            break
        last_row = table.iloc[[-1]]
        for col in FEATURE_COLS:
            if col not in last_row.columns:
                last_row[col] = 0.0
        X = last_row[FEATURE_COLS].values

        gbr_val = float(gbr.predict(X)[0])
        mlp_val = float(mlp.predict(scaler.transform(X))[0])

        next_ts = working["tstp"].iloc[-1] + pd.Timedelta(minutes=30)
        timestamps.append(next_ts)
        preds_gbr.append(max(gbr_val, 0))
        preds_mlp.append(max(mlp_val, 0))

        # feed the GBR prediction back in as ground truth for the next lag step
        new_row = working.iloc[[-1]].copy()
        new_row["tstp"] = next_ts
        new_row["energy_kwh_hh"] = gbr_val
        working = pd.concat([working, new_row[["LCLid", "tstp", "energy_kwh_hh", "household_type"]]],
                             ignore_index=True)

    return pd.DataFrame({
        "tstp": timestamps,
        "forecast_gbr": preds_gbr,
        "forecast_mlp": preds_mlp,
    })
