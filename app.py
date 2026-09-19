from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.flood_analysis import DATA_PATH, OUTPUT_DIR, clean_data, feature_importance, model_frame, run_analysis

st.set_page_config(page_title="India Flood Risk Intelligence", page_icon="🌧️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#64748b; --teal:#087f8c; --sky:#dff5f6; --amber:#f59e0b; --red:#dc2626; }
.block-container { padding-top: 1.5rem; max-width: 1450px; }
.hero { background: linear-gradient(120deg,#083344 0%,#0e7490 55%,#38bdf8 100%); padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.3rem; box-shadow: 0 10px 30px rgba(8,51,68,.18); }
.hero h1 { margin:0; font-size:2.5rem; letter-spacing:0; } .hero p { margin:.5rem 0 0; opacity:.9; }
.kpi { background:#ffffff; border:1px solid #e2e8f0; border-left:5px solid #0e7490; padding:1rem 1.1rem; border-radius:10px; min-height:105px; box-shadow:0 4px 14px rgba(15,23,42,.05); }
.kpi small { color:#64748b; text-transform:uppercase; font-size:.72rem; letter-spacing:.05em; } .kpi strong { display:block; color:#0f172a; font-size:1.7rem; margin-top:.3rem; }
.note { background:#fff7ed; border:1px solid #fed7aa; padding:1rem; border-radius:10px; color:#7c2d12; }
section[data-testid="stSidebar"] { background:#f0fdfa; }
</style>
""", unsafe_allow_html=True)


def kpi(label: str, value: str) -> None:
    st.markdown(f'<div class="kpi"><small>{label}</small><strong>{value}</strong></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning the inventory...")
def get_data() -> pd.DataFrame:
    return clean_data(DATA_PATH)[0]


@st.cache_resource(show_spinner="Training comparison models...")
def get_analysis() -> dict:
    return run_analysis(DATA_PATH)


@st.cache_resource(show_spinner="Loading trained models...")
def get_artifacts() -> dict:
    if not (OUTPUT_DIR / "models.joblib").exists():
        return get_analysis()["models"]
    return joblib.load(OUTPUT_DIR / "models.joblib")


def risk_label(probability: float) -> tuple[str, str]:
    if probability < .25: return "Low", "#15803d"
    if probability < .50: return "Moderate", "#d97706"
    if probability < .75: return "High", "#ea580c"
    return "Very High", "#b91c1c"


def chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, template="plotly_white", margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return fig


df = get_data()
analysis = get_analysis()
metrics = analysis["metrics"]
importance = analysis["importance"]
profile = analysis["profile"]

st.markdown('<div class="hero"><h1>India Flood Risk Intelligence</h1><p>Event inventory analysis, impact-risk estimation, and climate-action decision support</p></div>', unsafe_allow_html=True)
st.warning("Analytical limitation: this inventory has no measured rainfall, temperature, humidity, water-level, latitude, longitude, or severity values. Model outputs estimate recorded event-impact risk from historical inventory attributes; they are not guaranteed flood warnings or causal conclusions.")

with st.sidebar:
    st.header("Explore the dashboard")
    selected_states = st.multiselect("State filter", sorted(df["State"].dropna().astype(str).unique()), default=[])
    selected_seasons = st.multiselect("Season filter", sorted(df["season"].dropna().unique()), default=[])
    min_year = int(df["year"].min()) if df["year"].notna().any() else 1967
    max_year = int(df["year"].max()) if df["year"].notna().any() else 2023
    years = st.slider("Start year", min_year, max_year, (min_year, max_year))
    st.caption("Filters apply to charts and KPIs. The trained models use the full cleaned inventory.")

filtered = df.copy()
filtered = filtered[filtered["year"].between(years[0], years[1], inclusive="both")]
if selected_states:
    filtered = filtered[filtered["State"].astype(str).apply(lambda x: any(s in x for s in selected_states))]
if selected_seasons:
    filtered = filtered[filtered["season"].isin(selected_seasons)]

valid_duration = filtered["duration_days"].dropna()
impact_rate = filtered["impact_event"].mean() if len(filtered) else 0
avg_duration = valid_duration.mean() if len(valid_duration) else 0
max_duration = valid_duration.max() if len(valid_duration) else 0

st.subheader("Overview & KPI Dashboard")
cols = st.columns(6)
for col, label, value in zip(cols, ["Events", "Impact-risk rate", "Flood frequency / year", "Average duration", "Maximum duration", "High / very high observed"], [f"{len(filtered):,}", f"{impact_rate:.1%}", f"{len(filtered) / max(filtered['year'].nunique(), 1):,.1f}", f"{avg_duration:.1f} days", f"{max_duration:.0f} days", f"{filtered['risk_band_observed'].isin(['High','Very High']).mean():.1%}" if len(filtered) else "0.0%"]):
    with col: kpi(label, value)

st.subheader("Dataset Analysis")
left, right = st.columns([1.2, 1])
with left:
    profile_table = pd.DataFrame({"Metric": ["Rows", "Columns", "Exact duplicates", "Invalid start dates", "Date range", "Fully missing columns"], "Value": [profile["rows_raw"], profile["columns_raw"], profile["duplicates"], profile["invalid_start_dates"], f"{profile['date_min']} to {profile['date_max']}", ", ".join(profile["fully_missing_columns"])]})
    st.dataframe(profile_table, hide_index=True, use_container_width=True)
with right:
    missing = pd.Series(profile["missing_values"]).sort_values(ascending=False).head(12).reset_index()
    missing.columns = ["Column", "Missing values"]
    st.plotly_chart(chart_layout(px.bar(missing, x="Missing values", y="Column", orientation="h", color="Missing values", color_continuous_scale="Teal"), 360), use_container_width=True)

st.subheader("Climate & Flood Trends")
trend = filtered.dropna(subset=["year"]).groupby("year").agg(events=("UEI", "count"), impact_rate=("impact_event", "mean"), average_duration=("duration_days", "mean")).reset_index()
trend_long = trend.melt("year", value_vars=["events", "average_duration"], var_name="measure", value_name="value")
t1, t2 = st.columns(2)
with t1: st.plotly_chart(chart_layout(px.line(trend_long, x="year", y="value", color="measure", markers=True, title="Annual inventory volume and duration")), use_container_width=True)
with t2: st.plotly_chart(chart_layout(px.line(trend, x="year", y="impact_rate", markers=True, title="Recorded impact-risk rate by year").update_yaxes(tickformat=".0%")), use_container_width=True)

st.subheader("Exploratory Data Analysis")
e1, e2 = st.columns(2)
with e1:
    dist = filtered["risk_band_observed"].value_counts().reindex(["Low", "Moderate", "High", "Very High"], fill_value=0).reset_index()
    dist.columns = ["Risk band", "Events"]
    st.plotly_chart(chart_layout(px.bar(dist, x="Risk band", y="Events", color="Risk band", color_discrete_map={"Low":"#16a34a","Moderate":"#eab308","High":"#f97316","Very High":"#dc2626"}, title="Observed impact-derived risk bands")), use_container_width=True)
with e2:
    monthly = filtered.groupby("month", dropna=True).agg(events=("UEI", "count"), impact_rate=("impact_event", "mean")).reset_index()
    st.plotly_chart(chart_layout(px.bar(monthly, x="month", y="events", color="impact_rate", color_continuous_scale="YlOrRd", title="Events by start month", labels={"month":"Month", "events":"Events"})), use_container_width=True)

r1, r2 = st.columns(2)
with r1:
    st.plotly_chart(chart_layout(px.box(filtered.dropna(subset=["duration_days"]), x="risk_band_observed", y="duration_days", color="risk_band_observed", title="Duration distribution by observed risk band")), use_container_width=True)
with r2:
    rain = filtered.groupby("rain_related_flag").agg(events=("UEI", "count"), impact_rate=("impact_event", "mean")).reset_index()
    rain["rain_related_flag"] = rain["rain_related_flag"].map({0:"No rain wording", 1:"Rain-related wording"})
    st.plotly_chart(chart_layout(px.bar(rain, x="rain_related_flag", y="impact_rate", color="rain_related_flag", title="Rain-related wording and recorded impact rate").update_yaxes(tickformat=".0%")), use_container_width=True)

st.subheader("Correlation & Statistical Analysis")
cor_cols = ["duration_days", "district_count", "state_count", "year", "month", "extreme_weather_flag", "rain_related_flag", "impact_event"]
cor = filtered[cor_cols].corr(numeric_only=True)
st.plotly_chart(chart_layout(px.imshow(cor, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation among engineered numeric features"), 500), use_container_width=True)
st.caption("Correlation is association, not causation. In particular, text flags and inventory counts can reflect reporting practices rather than physical mechanisms.")

st.subheader("Flood Risk Prediction")
st.write("Predict the probability that a new inventory-style event will contain a recorded human or animal impact. This is an impact-risk estimate, not a rainfall or river forecast.")
with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1: input_state = st.selectbox("State", sorted(df["State"].dropna().astype(str).unique()))
    with p2: input_cause = st.selectbox("Main cause", sorted(df["cause_clean"].dropna().unique()))
    with p3: input_season = st.selectbox("Season", ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon", "Unknown"], index=2)
    p4, p5, p6 = st.columns(3)
    with p4: input_duration = st.number_input("Duration (days)", min_value=1.0, max_value=365.0, value=3.0)
    with p5: input_year = st.number_input("Start year", min_value=min_year, max_value=max_year + 10, value=max_year)
    with p6: input_districts = st.number_input("District count", min_value=0, max_value=800, value=1)
    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)

if submitted:
    month_lookup = {"Winter":1, "Pre-monsoon":4, "Monsoon":7, "Post-monsoon":10, "Unknown":np.nan}
    text = input_cause.lower()
    row = pd.DataFrame([{"Duration(Days)__num": input_duration, "year": input_year, "month": month_lookup[input_season], "district_count": input_districts, "state_count": 1, "extreme_weather_flag": int(bool(pd.Series([text]).str.contains(r"extreme|severe|cloud burst|flash|heavy rain|cyclone|storm", regex=True).iloc[0])), "rain_related_flag": int(bool(pd.Series([text]).str.contains(r"rain|monsoon|precipitation", regex=True).iloc[0])), "Main Cause": input_cause, "State": input_state, "season": input_season}])
    artifacts = get_artifacts()
    model = artifacts["models"][artifacts["best_model"]]
    probability = float(model.predict_proba(row)[:, 1][0])
    label, color = risk_label(probability)
    st.markdown(f'<div style="border-left:8px solid {color};padding:1rem;background:#f8fafc;border-radius:8px"><h3 style="color:{color};margin:0">{label} risk</h3><p style="font-size:1.2rem;margin:.4rem 0">Predicted recorded-impact probability: <b>{probability:.1%}</b></p><p style="margin:0">Model: {artifacts["best_model"]}. Major contributing signals are estimated from the fitted model, not proof of causation.</p></div>', unsafe_allow_html=True)
    st.write("Signals used: duration, year/month/season, district and state counts, normalized cause, and rain/extreme-weather wording flags. Actual casualty fields are intentionally excluded from prediction inputs.")

st.subheader("Model Comparison")
st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c != "Model"}), hide_index=True, use_container_width=True)
y_test = np.load(OUTPUT_DIR / "test_predictions.npz")
cm = np.array(analysis["details"]["confusion_matrix"])
m1, m2 = st.columns(2)
with m1:
    st.plotly_chart(chart_layout(px.imshow(cm, text_auto=True, x=["Predicted low impact", "Predicted impact"], y=["Actual low impact", "Actual impact"], color_continuous_scale="Blues", title="Best-model confusion matrix"), 400), use_container_width=True)
with m2:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test["y_test"], y_test["test_prob"])
    roc_fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    roc_fig.update_xaxes(title="False positive rate"); roc_fig.update_yaxes(title="True positive rate")
    st.plotly_chart(chart_layout(roc_fig, 400).update_layout(title="ROC curve"), use_container_width=True)

st.subheader("Feature Importance & Explainability")
st.plotly_chart(chart_layout(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal", title=f"Top signals from {analysis['details']['best_model']}"), 600), use_container_width=True)
actual_vs = pd.DataFrame({"Actual impact": y_test["y_test"], "Predicted probability": y_test["test_prob"]})
st.plotly_chart(chart_layout(px.scatter(actual_vs, x="Actual impact", y="Predicted probability", color="Predicted probability", color_continuous_scale="RdYlGn", title="Actual outcome vs predicted probability")), use_container_width=True)

st.subheader("Regional / Seasonal Analysis")
regional = filtered["State"].fillna("Unknown").astype(str).str.split(",").explode().str.strip().value_counts().head(20).reset_index()
regional.columns = ["State", "Event mentions"]
st.plotly_chart(chart_layout(px.bar(regional.sort_values("Event mentions"), x="Event mentions", y="State", orientation="h", color="Event mentions", color_continuous_scale="Viridis", title="Top state mentions in the inventory"), 550), use_container_width=True)

st.subheader("Key Insights")
for insight in analysis["insights"]: st.markdown(f"- {insight}")
st.markdown('<div class="note"><b>Decision-support recommendations:</b> use the dashboard to prioritize data-quality review, compare historical event concentration, investigate high predicted-impact cases, and combine this analysis with official hydrometeorological observations and local warning protocols before taking action.</div>', unsafe_allow_html=True)

st.subheader("SDG 13: Climate Action")
st.write("This project is a potential contribution toward SDG 13 through climate-risk analysis and preparedness support. The dataset can help users examine where and when recorded flood events and impacts appear in the inventory, identify recurring seasonal concentration, and test an explainable model for recorded-impact risk. It does not claim to achieve SDG 13, measure climate change, or prove that any climate variable caused a flood.")
st.write("Potential uses include climate-risk assessment, flood-risk monitoring, disaster preparedness, climate adaptation and resilience planning, identification of extreme-weather wording in event records, and data-informed environmental decision-making. Communities, planners, disaster-management teams, and authorities could use the results as one historical evidence layer alongside gauges, forecasts, vulnerability data, and local knowledge.")

st.subheader("Potential Impact & Final Conclusion")
st.write(f"Within this inventory, {profile['rows_raw']:,} event records span {profile['date_min']} to {profile['date_max']}. The project turns those records into quality diagnostics, seasonal and regional summaries, association analysis, and a model for the operational target of recorded human or animal impact. The current best model is {analysis['details']['best_model']} on the fixed evaluation split, but performance depends on this dataset, target definition, and sampling design.")
st.write("The main conclusion is therefore bounded: the dashboard can support historical flood-risk awareness and preparedness conversations, but it cannot replace official warnings. Missing climate measurements, incomplete impact reporting, inconsistent text and coding, possible reporting bias, and the event-inventory design limit causal interpretation and real-time generalization.")
