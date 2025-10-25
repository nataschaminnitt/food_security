# ============================ streamlit_app.py =============================
# Streamlit app: load artifacts, show history + forecasts with filters.
# Run with:
#   streamlit run streamlit_app.py
# ==========================================================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ----------------- Paths (match your training script) -----------------
ARTIFACT_DIR = Path("/Users/nataschajademinnitt/Documents/5_data/food_security/etl/artifacts")
MODEL_PATH   = ARTIFACT_DIR / "xgb_staples_model.joblib"
FEATS_PATH   = ARTIFACT_DIR / "xgb_staples_features.json"
LAST_HIST_PQ = ARTIFACT_DIR / "xgb_staples_last_history.parquet"
FUTURE_PQ    = ARTIFACT_DIR / "xgb_staples_future_forecast.parquet"
TEST_PRED_PQ = ARTIFACT_DIR / "xgb_staples_test_predictions.parquet"  # optional

st.set_page_config(page_title="Ethiopia Food Prices – Forecast", layout="wide")

st.title("📈 Ethiopia Food Prices – Staples Forecast")
st.caption("Monthly retail price forecasts by product and region (XGBoost model).")

@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feats = json.loads(Path(FEATS_PATH).read_text())
    last_hist = pd.read_parquet(LAST_HIST_PQ)
    future = pd.read_parquet(FUTURE_PQ)
    # Make sure month is datetime
    for df in (last_hist, future):
        df["month"] = pd.to_datetime(df["month"])
    # Prepare a history series (actuals)
    hist = last_hist[["admin_1","product","month","y"]].rename(columns={"y":"value"})
    hist["type"] = "Actual"
    # Future forecasts
    fut = future.rename(columns={"y_pred":"value"}).copy()
    fut["type"] = "Forecast"
    return model, feats, hist, fut

model, feats, hist, fut = load_artifacts()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    products = sorted(hist["product"].dropna().unique())
    regions  = sorted(hist["admin_1"].dropna().unique())
    sel_product = st.selectbox("Product", products, index=0)
    sel_region  = st.selectbox("Region (Admin 1)", regions, index=0)

# Filtered frames
h = hist[(hist["product"] == sel_product) & (hist["admin_1"] == sel_region)].copy()
f = fut[(fut["product"] == sel_product) & (fut["admin_1"] == sel_region)].copy()

# Combine for chart
chart_df = pd.concat([h, f], ignore_index=True)
chart_df = chart_df.sort_values("month")

# Display metrics (last actual vs last forecast)
import plotly.express as px

# chart_df has columns: ["month", "type", "value"]
# Make sure month is datetime
chart_df = chart_df.copy()
chart_df["month"] = pd.to_datetime(chart_df["month"], errors="coerce")

# Plotly line chart (Actual vs Forecast)
fig = px.line(
    chart_df.sort_values("month"),
    x="month",
    y="value",
    color="type",
    markers=True,
    labels={"month": "Month", "value": "Retail Price", "type": "Series"},
    title="Price: Actual vs Forecast"
)
fig.update_layout(height=420, legend_title_text="")

st.plotly_chart(fig, use_container_width=True)

# Optional: table
with st.expander("Show data"):
    st.dataframe(chart_df.sort_values("month").reset_index(drop=True))

st.caption("Note: Forecasts are recursive and use lag/rolling features; exogenous drivers are held at last observed values unless future values are provided.")
