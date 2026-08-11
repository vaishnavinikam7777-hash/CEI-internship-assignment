"""
Feature Engineering
--------------------
Turns a raw half-hourly consumption series into a supervised-learning
table: lag features, rolling statistics, and calendar features, which
both the ML and DL forecasting models consume.
"""
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, ts_col: str = "tstp") -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[ts_col].dt.hour
    df["dow"] = df[ts_col].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df[ts_col].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "energy_kwh_hh",
                      lags=(1, 2, 48, 336)) -> pd.DataFrame:
    """
    lags are in number of half-hour steps:
      1  = 30 min ago
      2  = 1 hour ago
      48 = same time yesterday
      336 = same time last week
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    df["rolling_mean_6"] = df[target_col].shift(1).rolling(6).mean()   # last 3 hrs
    df["rolling_mean_48"] = df[target_col].shift(1).rolling(48).mean()  # last day
    df["rolling_std_48"] = df[target_col].shift(1).rolling(48).std()
    return df


def build_supervised_table(household_df: pd.DataFrame, weather: pd.DataFrame,
                             target_col: str = "energy_kwh_hh") -> pd.DataFrame:
    """
    household_df: single household's time-ordered rows (LCLid, tstp, energy_kwh_hh)
    Returns a feature table ready for model training/prediction, with
    NaNs from lagging dropped.
    """
    df = household_df.sort_values("tstp").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col=target_col)

    if weather is not None and not weather.empty:
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"]).dt.date
        df["date"] = df["tstp"].dt.date
        df = df.merge(w, on="date", how="left")
        df["temperature_c"] = df["temperature_c"].ffill().bfill()
        df = df.drop(columns=["date"])

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month",
    "lag_1", "lag_2", "lag_48", "lag_336",
    "rolling_mean_6", "rolling_mean_48", "rolling_std_48", "temperature_c",
]
