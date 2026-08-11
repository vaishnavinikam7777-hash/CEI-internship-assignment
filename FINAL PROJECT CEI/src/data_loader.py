"""
Data Loader
-----------
Provides a single entry point `load_data()` for the rest of the
pipeline. If real "Smart meters in London" CSVs from Kaggle are
present under data/raw/, they are loaded and normalized to the same
schema as the synthetic generator. Otherwise, synthetic data is
generated on the fly so the app always works out of the box.

Expected real-data files (from the Kaggle dataset), if present:
  data/raw/halfhourly_dataset/*.csv   (columns: LCLid, tstp, energy(kWh/hh))
  data/raw/weather_daily_darksky.csv  (has a 'time' + temperature column)

Place the Kaggle CSVs there and this loader will pick them up
automatically -- no other code needs to change.
"""
import os
import glob
import pandas as pd

import synthetic_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def _has_real_data() -> bool:
    hh_dir = os.path.join(RAW_DIR, "halfhourly_dataset")
    return os.path.isdir(hh_dir) and len(glob.glob(os.path.join(hh_dir, "*.csv"))) > 0


def load_real_data(max_files: int = 5, max_rows_per_file: int = 200_000):
    """Loads and normalizes real Kaggle CSVs if present under data/raw/."""
    hh_dir = os.path.join(RAW_DIR, "halfhourly_dataset")
    files = sorted(glob.glob(os.path.join(hh_dir, "*.csv")))[:max_files]

    frames = []
    for f in files:
        df = pd.read_csv(f, nrows=max_rows_per_file)
        df.columns = [c.strip() for c in df.columns]
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("lclid",):
                rename_map[c] = "LCLid"
            elif cl in ("tstp",):
                rename_map[c] = "tstp"
            elif "energy" in cl:
                rename_map[c] = "energy_kwh_hh"
        df = df.rename(columns=rename_map)
        df["tstp"] = pd.to_datetime(df["tstp"], errors="coerce")
        df["energy_kwh_hh"] = pd.to_numeric(
            df["energy_kwh_hh"].astype(str).str.replace("Null", ""), errors="coerce"
        )
        df = df.dropna(subset=["tstp", "energy_kwh_hh"])
        df["household_type"] = "unknown"
        frames.append(df[["LCLid", "tstp", "energy_kwh_hh", "household_type"]])

    consumption = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    weather_path = os.path.join(RAW_DIR, "weather_daily_darksky.csv")
    if os.path.exists(weather_path):
        w = pd.read_csv(weather_path)
        time_col = next((c for c in w.columns if "time" in c.lower()), None)
        temp_col = next((c for c in w.columns if "temperaturemax" in c.lower()
                          or c.lower() == "temperature"), w.columns[1])
        weather = pd.DataFrame({
            "date": pd.to_datetime(w[time_col], errors="coerce"),
            "temperature_c": pd.to_numeric(w[temp_col], errors="coerce"),
        }).dropna()
    else:
        start = consumption["tstp"].min().strftime("%Y-%m-%d") if not consumption.empty else "2024-01-01"
        days = max((consumption["tstp"].max() - consumption["tstp"].min()).days + 1, 1) \
            if not consumption.empty else 30
        _, weather = synthetic_data.generate_dataset(n_households=1, periods_days=days, start=start)

    return consumption, weather


def load_data(n_households: int = 12, periods_days: int = 120, use_synthetic: bool = None):
    """
    Main entry point. Auto-detects real Kaggle data under data/raw/;
    falls back to synthetic data matching the same schema otherwise.
    Pass use_synthetic=True to force synthetic even if real data exists
    (useful for fast demos).
    """
    if use_synthetic is False or (use_synthetic is None and _has_real_data()):
        consumption, weather = load_real_data()
        source = "kaggle_real_data"
    else:
        consumption, weather = synthetic_data.generate_dataset(
            n_households=n_households, periods_days=periods_days
        )
        source = "synthetic"
    return consumption, weather, source
