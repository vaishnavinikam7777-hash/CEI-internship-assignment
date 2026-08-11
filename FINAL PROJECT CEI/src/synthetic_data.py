"""
Synthetic Smart Meter Data Generator
-------------------------------------
Generates half-hourly household energy consumption data that mirrors
the schema of the Kaggle "Smart meters in London" dataset
(https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london):

    LCLid, tstp, energy(kWh/hh)

plus a companion weather table (temperature) since consumption is
weather-driven in the real dataset too.

This lets the rest of the pipeline (feature engineering, forecasting,
pattern analysis, optimization insights) be built and tested end-to-
end without needing network access to Kaggle. Swap in the real CSVs
via data_loader.py's `load_real_data()` and nothing downstream changes.
"""
import numpy as np
import pandas as pd


def generate_household_series(household_id: str, start: str, periods_days: int,
                                seed: int, household_type: str = "family") -> pd.DataFrame:
    """
    Generates one household's half-hourly kWh series with:
      - daily seasonality (morning + evening peaks)
      - weekly seasonality (weekend differs from weekday)
      - slow seasonal drift (winter > summer, via a temperature proxy)
      - random noise + occasional spikes (appliance use)
    household_type shapes the daily profile: 'family', 'night_owl', 'work_from_home'
    """
    rng = np.random.default_rng(seed)
    n = periods_days * 48
    ts = pd.date_range(start=start, periods=n, freq="30min")

    hour = ts.hour + ts.minute / 60
    dow = ts.dayofweek
    is_weekend = (dow >= 5).astype(float)
    day_of_year = ts.dayofyear.values

    # seasonal (winter higher heating load) - peak around day 15 (mid Jan)
    seasonal = 0.6 * np.cos(2 * np.pi * (day_of_year - 15) / 365) + 1.2

    if household_type == "family":
        morning_peak = np.exp(-((hour - 7.5) ** 2) / (2 * 1.0 ** 2))
        evening_peak = np.exp(-((hour - 19) ** 2) / (2 * 1.8 ** 2))
        base_profile = 0.15 + 0.5 * morning_peak + 0.9 * evening_peak
        weekend_boost = is_weekend * 0.15
    elif household_type == "night_owl":
        evening_peak = np.exp(-((hour - 23) ** 2) / (2 * 2.5 ** 2))
        late_peak = np.exp(-((((hour + 6) % 24) - 6) ** 2) / (2 * 2.0 ** 2))
        base_profile = 0.15 + 0.7 * evening_peak + 0.2 * late_peak
        weekend_boost = is_weekend * 0.05
    else:  # work_from_home
        daytime = np.exp(-((hour - 13) ** 2) / (2 * 5.0 ** 2))
        base_profile = 0.2 + 0.55 * daytime
        weekend_boost = is_weekend * -0.05  # slightly less on weekends (no separate "home" spike)

    noise = rng.normal(0, 0.06, size=n).clip(min=-0.15)
    spikes = (rng.random(n) < 0.015) * rng.uniform(0.5, 1.5, size=n)  # appliance bursts

    energy = (base_profile + weekend_boost) * seasonal + noise + spikes
    energy = np.clip(energy, 0.01, None)

    return pd.DataFrame({
        "LCLid": household_id,
        "tstp": ts,
        "energy_kwh_hh": np.round(energy, 4),
        "household_type": household_type,
    })


def generate_weather(start: str, periods_days: int, seed: int = 7) -> pd.DataFrame:
    """Daily mean temperature proxy (°C), London-like seasonal curve."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods_days, freq="D")
    day_of_year = dates.dayofyear.values
    seasonal_temp = 11 + 8 * -np.cos(2 * np.pi * (day_of_year - 15) / 365)
    noise = rng.normal(0, 2.0, size=periods_days)
    return pd.DataFrame({"date": dates, "temperature_c": np.round(seasonal_temp + noise, 1)})


def generate_dataset(n_households: int = 12, periods_days: int = 120,
                      start: str = "2024-01-01", seed: int = 42) -> tuple:
    """Generates the full multi-household dataset + weather table."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["family", "night_owl", "work_from_home"],
                        size=n_households, p=[0.5, 0.2, 0.3])
    frames = []
    for i in range(n_households):
        hh_id = f"MAC{100000 + i}"
        frames.append(generate_household_series(
            hh_id, start, periods_days, seed=seed + i, household_type=types[i]
        ))
    consumption = pd.concat(frames, ignore_index=True)
    weather = generate_weather(start, periods_days, seed=seed)
    return consumption, weather


if __name__ == "__main__":
    consumption, weather = generate_dataset()
    print(consumption.head())
    print(consumption.shape, weather.shape)
