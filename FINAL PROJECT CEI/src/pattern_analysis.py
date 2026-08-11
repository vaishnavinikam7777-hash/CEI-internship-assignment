"""
Usage Pattern Analysis
-----------------------
Identifies consumption patterns two ways:

  1. Household-level clustering: builds a 48-dimensional average daily
     load profile per household and clusters households (KMeans) into
     behavioral segments (e.g. "evening-peak", "daytime/WFH", "night-owl").
  2. Peak/off-peak detection: for a single household, flags the top
     consumption hours and weekday-vs-weekend differences.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_daily_load_profiles(consumption: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a household x 48-halfhour-slot matrix of average energy
    use, i.e. each household's "typical day" shape.
    """
    df = consumption.copy()
    df["slot"] = df["tstp"].dt.hour * 2 + (df["tstp"].dt.minute // 30)
    profile = df.groupby(["LCLid", "slot"])["energy_kwh_hh"].mean().unstack(fill_value=0)
    profile = profile.reindex(columns=range(48), fill_value=0)
    return profile


def cluster_households(consumption: pd.DataFrame, n_clusters: int = 3, seed: int = 42):
    profile = build_daily_load_profiles(consumption)
    scaler = StandardScaler()
    X = scaler.fit_transform(profile.values)

    k = min(n_clusters, len(profile))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    result = pd.DataFrame({"LCLid": profile.index, "cluster": labels})

    # label clusters descriptively based on peak slot of the cluster centroid
    centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
    cluster_names = {}
    for c in range(k):
        centroid = centroids_unscaled[c]
        peak_slot = int(np.argmax(centroid))
        peak_hour = peak_slot / 2
        if 6 <= peak_hour < 11:
            name = "Morning-peak"
        elif 11 <= peak_hour < 17:
            name = "Daytime / work-from-home"
        elif 17 <= peak_hour < 22:
            name = "Evening-peak"
        else:
            name = "Night-owl"
        cluster_names[c] = f"{name} (cluster {c})"

    result["cluster_label"] = result["cluster"].map(cluster_names)
    return result, profile, cluster_names


def peak_offpeak_analysis(household_df: pd.DataFrame) -> dict:
    """
    For a single household's series, returns peak hours, weekday vs
    weekend averages, and estimated peak-to-average ratio (a standard
    grid-planning metric — high ratios mean spikier, less efficient usage).
    """
    df = household_df.copy()
    df["hour"] = df["tstp"].dt.hour
    df["is_weekend"] = df["tstp"].dt.dayofweek >= 5

    hourly_avg = df.groupby("hour")["energy_kwh_hh"].mean()
    top_hours = hourly_avg.sort_values(ascending=False).head(3)

    weekday_avg = df.loc[~df["is_weekend"], "energy_kwh_hh"].mean()
    weekend_avg = df.loc[df["is_weekend"], "energy_kwh_hh"].mean()

    peak_to_avg = float(hourly_avg.max() / hourly_avg.mean()) if hourly_avg.mean() > 0 else 0.0

    return {
        "top_peak_hours": [(int(h), round(float(v), 3)) for h, v in top_hours.items()],
        "weekday_avg_kwh_hh": round(float(weekday_avg), 3),
        "weekend_avg_kwh_hh": round(float(weekend_avg), 3),
        "peak_to_average_ratio": round(peak_to_avg, 2),
        "hourly_profile": hourly_avg.round(3).to_dict(),
    }
