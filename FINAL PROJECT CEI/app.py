"""
AI-Powered Energy Analytics System
-------------------------------------
Forecasts smart-meter energy consumption, identifies household usage
patterns via clustering, and delivers optimization insights.

Data: schema-compatible with the Kaggle "Smart meters in London" dataset
(https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london).
Drop the real CSVs into data/raw/ (see src/data_loader.py) to use real
data instead of the built-in synthetic generator.

Run with: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import data_loader
import forecasting_model
import pattern_analysis
import optimization_insights

st.set_page_config(page_title="Energy Analytics", page_icon="⚡", layout="wide")


@st.cache_data
def get_data(n_households, periods_days):
    return data_loader.load_data(n_households=n_households, periods_days=periods_days)


@st.cache_resource
def get_forecasters(consumption, weather):
    return forecasting_model.train_forecasters(consumption, weather)


@st.cache_data
def get_clusters(consumption, n_clusters):
    return pattern_analysis.cluster_households(consumption, n_clusters=n_clusters)


st.sidebar.title("⚡ Energy Analytics")
st.sidebar.caption(
    "Forecasts consumption, clusters households by usage pattern, and "
    "generates optimization insights from smart meter data."
)

n_households = st.sidebar.slider("Households (synthetic demo)", 4, 20, 10)
periods_days = st.sidebar.slider("History (days)", 30, 180, 90)
n_clusters = st.sidebar.slider("Usage-pattern clusters", 2, 5, 3)

with st.spinner("Loading data..."):
    consumption, weather, source = get_data(n_households, periods_days)

st.sidebar.markdown(f"**Data source:** `{source}`")
if source == "synthetic":
    st.sidebar.info(
        "Using synthetic data matching the Kaggle smart-meter schema. "
        "Place the real CSVs under `data/raw/` to switch to real data."
    )

with st.spinner("Training forecasting models (ML + DL)..."):
    gbr, mlp, scaler, metrics = get_forecasters(consumption, weather)

st.title("AI-Powered Energy Analytics System")
st.write(
    "Forecasts household energy consumption, identifies usage patterns, "
    "and delivers optimization insights from half-hourly smart meter data."
)

tab1, tab2, tab3 = st.tabs(["🔮 Forecasting", "📊 Usage Patterns", "💡 Optimization Insights"])

household_ids = sorted(consumption["LCLid"].unique())

# ---------------------------------------------------------------------
# TAB 1: Forecasting
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Consumption forecasting: ML vs DL vs naive baseline")

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("**Model performance (held-out test set)**")
        perf_df = pd.DataFrame(metrics).T[["mae", "rmse", "r2"]].loc[
            ["naive_baseline", "gbr", "mlp"]
        ]
        perf_df.index = ["Naive (yesterday)", "ML: Gradient Boosting", "DL: Neural Net (MLP)"]
        st.dataframe(perf_df, use_container_width=True)
        st.caption("Lower MAE/RMSE and higher R² is better. Both models beat the naive baseline.")

        st.markdown("**Top predictive features (ML model)**")
        fi = pd.Series(metrics["feature_importance_gbr"]).sort_values(ascending=False).head(6)
        st.bar_chart(fi)

    with c1:
        hh_id = st.selectbox("Household", household_ids)
        horizon = st.slider("Forecast horizon (half-hour steps)", 12, 96, 48)
        hh_df = consumption[consumption["LCLid"] == hh_id]

        with st.spinner("Generating forecast..."):
            forecast = forecasting_model.forecast_next(hh_df, weather, gbr, mlp, scaler, horizon=horizon)

        recent = hh_df.sort_values("tstp").tail(96)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["tstp"], y=recent["energy_kwh_hh"],
                                  name="Actual (recent)", line=dict(color="#636EFA")))
        fig.add_trace(go.Scatter(x=forecast["tstp"], y=forecast["forecast_gbr"],
                                  name="Forecast (ML: Gradient Boosting)", line=dict(color="#00CC96", dash="dash")))
        fig.add_trace(go.Scatter(x=forecast["tstp"], y=forecast["forecast_mlp"],
                                  name="Forecast (DL: Neural Net)", line=dict(color="#EF553B", dash="dot")))
        fig.update_layout(height=450, xaxis_title="Time", yaxis_title="kWh / half-hour",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: Usage Patterns
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Household usage-pattern segmentation")

    clusters, profile, cluster_names = get_clusters(consumption, n_clusters)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Cluster assignments**")
        st.dataframe(clusters, use_container_width=True, hide_index=True)
        counts = clusters["cluster_label"].value_counts()
        fig_pie = px.pie(values=counts.values, names=counts.index, title="Segment sizes")
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("**Average daily load shape per segment**")
        profile_with_cluster = profile.copy()
        profile_with_cluster["cluster_label"] = clusters.set_index("LCLid")["cluster_label"]
        avg_by_cluster = profile_with_cluster.groupby("cluster_label").mean()

        fig = go.Figure()
        for label, row in avg_by_cluster.iterrows():
            hours = [s / 2 for s in row.index]
            fig.add_trace(go.Scatter(x=hours, y=row.values, mode="lines", name=label))
        fig.update_layout(height=450, xaxis_title="Hour of day", yaxis_title="Avg kWh / half-hour",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Per-household peak/off-peak breakdown**")
    hh_id_2 = st.selectbox("Household ", household_ids, key="pattern_hh")
    peak_info = pattern_analysis.peak_offpeak_analysis(consumption[consumption["LCLid"] == hh_id_2])

    m1, m2, m3 = st.columns(3)
    m1.metric("Peak-to-average ratio", f"{peak_info['peak_to_average_ratio']}x")
    m2.metric("Weekday avg", f"{peak_info['weekday_avg_kwh_hh']} kWh/hh")
    m3.metric("Weekend avg", f"{peak_info['weekend_avg_kwh_hh']} kWh/hh")

    hourly = pd.Series(peak_info["hourly_profile"]).sort_index()
    st.bar_chart(hourly)

# ---------------------------------------------------------------------
# TAB 3: Optimization Insights
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Optimization insights")
    hh_id_3 = st.selectbox("Household  ", household_ids, key="insight_hh")

    peak_info_3 = pattern_analysis.peak_offpeak_analysis(consumption[consumption["LCLid"] == hh_id_3])
    clusters_3, _, _ = get_clusters(consumption, n_clusters)
    cluster_label_3 = clusters_3.loc[clusters_3["LCLid"] == hh_id_3, "cluster_label"].iloc[0]

    result = optimization_insights.generate_insights(peak_info_3, cluster_label_3)

    st.markdown("**Insights**")
    for i in result["insights"]:
        st.markdown(f"- {i}")

    st.markdown("**Recommended actions**")
    for a in result["recommended_actions"]:
        st.markdown(f"- ✅ {a}")
