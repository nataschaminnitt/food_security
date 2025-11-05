# ===================== streamlit_app.py =====================

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.cache_data.clear()


# ------------------------------------------------------------------------------
# Config / paths
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Ethiopia Food Prices Dashboard",
    page_icon="🌾"
)


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASE_DIR

PANEL_PATH = ROOT / "data" / "processed" / "ethiopia_foodprices_model_panel.parquet"

# Artifacts directory for forecasting
ART_DIR = BASE_DIR / "etl" / "artifacts"
P_TEST = ART_DIR / "xgb_staples_test_predictions.parquet"
P_FUTURE = ART_DIR / "xgb_staples_future_forecast.parquet"

# ArcGIS admin-1 GeoJSON (layer 3)
ADMIN1_GEOJSON_URL = (
    "https://services8.arcgis.com/oTalEaSXAuyNT7xf/ArcGIS/rest/services/"
    "Ethiopia_AdminBoundaries/FeatureServer/3/query"
    "?where=1%3D1&outFields=*&f=geojson"
)

FOCUS_PRODUCTS = [
    "Wheat Grain",
    "Wheat Flour",
    "Rice (Milled)",
    "Beans (Haricot)",
    "Maize Grain (White)",
    "Sorghum",
    "Mixed Teff",
    "Refined sugar",
    "Refined Vegetable Oil",
]

PRED_COL_CANDIDATES   = ["y_pred_hybrid", "y_pred_global", "y_pred", "pred", "forecast"]
ACTUAL_COL_CANDIDATES = ["y", "y_true", "actual", "obs"]

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ------------------------------------------------------------------------------
# Load precomputed exploration summaries
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading exploration summaries…")
def load_exploration_summaries():
    nat_trends = pd.read_parquet("data/processed/expl_nat_trends.parquet")
    regional_trends = pd.read_parquet("data/processed/expl_regional_trends.parquet")
    map_summary = pd.read_parquet("data/processed/expl_map_summary.parquet")
    pivot_cv = pd.read_parquet("data/processed/expl_stability_cv_wide.parquet")

    for df in (nat_trends, regional_trends):
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")

    return nat_trends, regional_trends, map_summary, pivot_cv

@st.cache_data(show_spinner="Loading panel…")
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["product"].isin(FOCUS_PRODUCTS)].copy()
    # Make sure month is datetime
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

@st.cache_data(show_spinner="Loading admin-1 boundaries…")
def load_admin1_geojson():
    r = requests.get(ADMIN1_GEOJSON_URL, timeout=60)
    r.raise_for_status()
    gj = r.json()
    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        region_name = props.get("REGIONNAME") or props.get("adm1_name") or ""
        props["admin_key"] = _norm(region_name)
    return gj

# ------------------------------------------------------------------------------
# Page 0 – Landing / Summary
# ------------------------------------------------------------------------------

def render_landing_page():
    st.markdown("""
    Welcome to the **Ethiopia Food Prices Dashboard**, an interactive platform 
    tracking and forecasting staple food prices across Ethiopian regions.

    This dashboard provides:
    - Historical and recent **market price trends** for key staples (wheat, maize, teff, beans, etc.).
    - **Regional variations** in prices across Ethiopia’s administrative regions.
    - **Forecasts** for staple prices based on historical data and machine learning models.

    ---
    **Note:** Data primarily comes from the World Food Programme’s VAM Data Repository (HDX), 
    complemented by FEWS NET price series. Forecasting uses an XGBoost model with hybrid residual correction.
    """)


# ------------------------------------------------------------------------------
# Page 1 – Methodology / Sources
# ------------------------------------------------------------------------------

def render_methodology_page():
    st.markdown("## Methodology and Data Sources")

    st.markdown("""
    ### Data Sources
    - **World Food Programme (WFP)** — Retail market price data via [HDX](https://data.humdata.org/dataset/2e4f1922-e446-4b57-a98a-d0e2d5e34afa).
    - **FEWS NET** — Market price data through [FEWS NET’s API](https://fdw.fews.net/api/marketpricefacts.csv).
    - **Official Exchange Rates** — Sourced from [Trading Economics](https://tradingeconomics.com/) for currency normalization.

    ### Data Processing
    - Both WFP and FEWS NET data are harmonized into a **FEWS-like schema** with columns:
      `period_date`, `admin_1`, `market`, `product`, `unit`, `price_type`, `value`.
    - Only **retail prices** and the **past five years** of data are retained.
    - Units are normalized to a common basis (e.g., per kg, per liter).

    ### Forecasting Model
    - A **global XGBoost model** is trained on all available product-region combinations.
    - A **per-product Ridge regression correction** (hybrid layer) refines residual biases.
    - Forecasts are produced only for region–product pairs with **≥12 months** of training data.

    ### Metrics
    - **MAE**, **RMSE**, and **sMAPE** are used to evaluate model accuracy on the test set.

    ### 💡 Data Availability
    - Only regions with sufficient data coverage (≥12 months) are forecasted.
    - The data availability visualization helps identify missing months by region/product.
    """)

    panel = load_panel()

    # Center content with margins on left/right
    margin_left, main, margin_right = st.columns([0.01, 0.98, 0.01])

    with main:
        # Region selection
        regions = sorted(panel["admin_1"].dropna().unique())
        default_region = "Addis Ababa" if "Addis Ababa" in regions else regions[0]
        selected_admin = st.selectbox(
            "Select region",
            options=regions,
            index=regions.index(default_region)
        )

        # Pick a price column
        if "value_imputed" in panel.columns:
            price_col = "value_imputed"
        elif "value_mean" in panel.columns:
            price_col = "value_mean"
        elif "value_median" in panel.columns:
            price_col = "value_median"
        else:
            price_col = "value"

        # Ensure month is datetime
        panel = panel.copy()
        panel["month"] = pd.to_datetime(panel["month"], errors="coerce")

        # Aggregate to monthly mean per region/product
        avail = (
            panel.groupby(["admin_1", "month", "product"], as_index=False)[price_col]
            .mean()
        )

        # Filter selected region
        region_df = avail[avail["admin_1"] == selected_admin].copy()

        # Full date range over panel
        full_months = pd.date_range(
            start=panel["month"].min(),
            end=panel["month"].max(),
            freq="MS"
        )

        products_region = region_df["product"].dropna().unique()
        all_combos = pd.MultiIndex.from_product(
            [full_months, products_region],
            names=["month", "product"]
        ).to_frame(index=False)

        region_full = all_combos.merge(region_df, on=["month", "product"], how="left")

        region_full["available"] = region_full[price_col].notna()
        region_full["color"] = region_full["available"].map({True: "green", False: "red"})
        region_full["avail_label"] = region_full["available"].map({True: "Yes", False: "No"})

        # Plotly scatter plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=region_full["month"],
                y=region_full["product"],
                mode="markers",
                marker=dict(
                    color=region_full["color"],
                    size=10,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Product: %{y}<br>"
                    "Data available: %{customdata}<extra></extra>"
                ),
                customdata=region_full["avail_label"],
            )
        )

        fig.update_layout(
            title=f"Data availability (past 5 years) – {selected_admin}",
            xaxis_title="Month",
            yaxis_title="Product",
            height=450,
            margin=dict(l=20, r=20, t=60, b=40),
            yaxis=dict(categoryorder="category ascending"),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    _Developed using Python, Streamlit, Plotly, and Scikit-learn._
    """)

# ------------------------------------------------------------------------------
# Page 2 – Data exploration
# ------------------------------------------------------------------------------

def render_exploration_page(panel: pd.DataFrame) -> None:
    nat_trends, regional_trends, map_summary, pivot_cv = load_exploration_summaries()

    margin_left, main, margin_right = st.columns([0.01, 0.98, 0.01])

    with main:
        
        # ───────────────────────── Product price trends ─────────────────────────
        st.markdown("### Product price trends")

        selected_products = st.multiselect(
            "Select products",
            options=FOCUS_PRODUCTS,
            default=FOCUS_PRODUCTS,
        )

        view_mode = st.radio(
            "View mode",
            ["National median across regions", "Regional breakdown"],
            index=0,
            horizontal=True,
        )

        if view_mode == "National median across regions":
            ts = nat_trends[nat_trends["product"].isin(selected_products)].copy()
            ts = ts.sort_values(["product", "month"])

            fig_trend = px.line(
                ts,
                x="month",
                y="price",
                color="product",
                markers=True,
                labels={
                    "month": "Month",
                    "price": "Price (ETB)",
                    "product": "Product",
                },
            )
            fig_trend.update_layout(
                height=450,
                title="National median prices across regions",
                legend_title_text="Product",
            )

        else:
            regions = (
                regional_trends["admin_1"].dropna().sort_values().unique().tolist()
            )
            default_region = "Addis Ababa" if "Addis Ababa" in regions else regions[0]

            region = st.selectbox(
                "Choose region",
                options=regions,
                index=regions.index(default_region),
            )

            ts = regional_trends[
                (regional_trends["admin_1"] == region)
                & (regional_trends["product"].isin(selected_products))
            ].copy()
            ts = ts.sort_values("month")

            fig_trend = px.line(
                ts,
                x="month",
                y="price",
                color="product",
                markers=True,
                labels={
                    "month": "Month",
                    "price": "Price (ETB)",
                    "product": "Product",
                },
            )
            fig_trend.update_layout(
                height=450,
                title=f"Prices in {region}",
                legend_title_text="Product",
            )

        st.plotly_chart(fig_trend, use_container_width=True)

        # ───────────────────────── Map: average prices by region ─────────────────────────
        st.markdown("### Average prices by region")

        map_product = st.selectbox(
            "Product for map",
            options=FOCUS_PRODUCTS,
            index=FOCUS_PRODUCTS.index("Maize Grain (White)")
            if "Maize Grain (White)" in FOCUS_PRODUCTS
            else 0,
        )

        admin1_geojson = load_admin1_geojson()

        map_df = (
            map_summary[map_summary["product"] == map_product]
            .copy()
            .rename(columns={"median_price": "median_price"})
        )
        map_df["admin_key"] = map_df["admin_1"].map(_norm)

        if map_df.empty:
            st.warning(f"No data found for product: {map_product}")
        else:
            fig_map = px.choropleth_mapbox(
                map_df,
                geojson=admin1_geojson,
                locations="admin_key",
                color="median_price",
                featureidkey="properties.admin_key",
                color_continuous_scale="Viridis",
                mapbox_style="open-street-map",   # OpenStreetMap tiles
                zoom=5.5,
                center={"lat": 9.145, "lon": 40.489},
                opacity=0.7,
                labels={"median_price": "Median price (ETB)"},
            )
            fig_map.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                title=f"Median {map_product} price by region",
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # ───────────────────────── Price stability heatmap ─────────────────────────
        st.markdown("### Price stability by region and product")

        if pivot_cv.empty:
            st.warning("Not enough data to compute stability (CV) yet.")
            return

        pivot_cv_local = pivot_cv.copy()
        cols_order = [p for p in FOCUS_PRODUCTS if p in pivot_cv_local.columns] + [
            c for c in pivot_cv_local.columns if c not in FOCUS_PRODUCTS
        ]
        pivot_cv_local = pivot_cv_local[cols_order]

        fig_cv = px.imshow(
            pivot_cv_local,
            color_continuous_scale="RdYlGn_r",  # red = volatile, green = stable
            aspect="auto",
            labels=dict(color="Coefficient of variation"),
        )
        fig_cv.update_layout(
            height=500,
            title="Coefficient of variation (price stability) by region and product",
            xaxis_title="Product",
            yaxis_title="Region",
        )

        st.plotly_chart(fig_cv, use_container_width=True)


# ------------------------------------------------------------------------------
# Page 2 – Forecasting (stub – plug your previous code back here)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_forecast_data():
    # ----- existence checks -----
    if not P_TEST.exists():
        raise FileNotFoundError(f"Missing test predictions parquet: {P_TEST}")
    if not P_FUTURE.exists():
        raise FileNotFoundError(f"Missing future forecast parquet: {P_FUTURE}")

    test_df   = pd.read_parquet(P_TEST)
    future_df = pd.read_parquet(P_FUTURE)

    # ensure datetime for month
    for d in (test_df, future_df):
        if "month" in d.columns:
            d["month"] = pd.to_datetime(d["month"], errors="coerce")

    # ----- keep only series that we actually forecast for -----
    eligible_pairs = (
        future_df[["admin_1", "product"]]
        .dropna()
        .drop_duplicates()
    )
    test_df = test_df.merge(
        eligible_pairs,
        on=["admin_1", "product"],
        how="inner",
        suffixes=("", "_elig"),
    )

    used_msgs = []

    # ----- unify prediction column -----
    pred_test_col = _pick_col(test_df, PRED_COL_CANDIDATES)
    if pred_test_col is None:
        raise KeyError(
            f"No prediction column found in test file. "
            f"Tried: {PRED_COL_CANDIDATES}"
        )
    if pred_test_col != "y_pred":
        test_df = test_df.rename(columns={pred_test_col: "y_pred"})

    pred_future_col = _pick_col(future_df, PRED_COL_CANDIDATES)
    if pred_future_col is None:
        raise KeyError(
            f"No prediction column found in future file. "
            f"Tried: {PRED_COL_CANDIDATES}"
        )
    if pred_future_col != "y_pred":
        future_df = future_df.rename(columns={pred_future_col: "y_pred"})

    # ----- unify actuals in test -----
    actual_col = _pick_col(test_df, ACTUAL_COL_CANDIDATES)
    if actual_col is None:
        test_df["y_actual"] = np.nan
        used_msgs.append(
            "Test actuals: none found (CI from residuals will be unavailable)."
        )
    else:
        if actual_col != "y_actual":
            test_df = test_df.rename(columns={actual_col: "y_actual"})

    # basic column sanity
    req = {"admin_1", "product", "month"}
    for name, frame in [("test", test_df), ("future", future_df)]:
        missing = req - set(frame.columns)
        if missing:
            raise KeyError(f"{name} file missing required columns: {sorted(missing)}")

    # ----- residual sigma per product (only where actuals exist) -----
    if test_df["y_actual"].notna().any():
        mask = test_df["y_actual"].notna() & test_df["y_pred"].notna()
        resids = test_df.loc[mask, "y_actual"] - test_df.loc[mask, "y_pred"]
        sigma_global = float(resids.std(ddof=1)) if len(resids) else 0.0

        s_by_prod = (
            test_df.loc[mask]
            .assign(resid=lambda d: d["y_actual"] - d["y_pred"])
            .groupby("product", observed=False)["resid"]
            .std(ddof=1)
        )
    else:
        sigma_global = 0.0
        s_by_prod = pd.Series(dtype=float)

    # attach sigma to future
    future_df = future_df.merge(s_by_prod.rename("sigma"), on="product", how="left")
    future_df["sigma"] = (
        future_df["sigma"]
        .fillna(sigma_global)
        .replace([np.inf, -np.inf], sigma_global)
        .astype(float)
    )

    # CI band (95% ~ 1.96σ). Clip lower at 0.
    k = 1.96
    future_df["y_lo"] = np.maximum(0.0, future_df["y_pred"] - k * future_df["sigma"])
    future_df["y_hi"] = future_df["y_pred"] + k * future_df["sigma"]

    # Build combined viz frame
    hist = (
        test_df.assign(y_lo=np.nan, y_hi=np.nan)[
            ["admin_1", "product", "month", "y_actual", "y_pred", "y_lo", "y_hi"]
        ]
    )
    fut = (
        future_df.assign(y_actual=np.nan)[
            ["admin_1", "product", "month", "y_actual", "y_pred", "y_lo", "y_hi"]
        ]
    )
    df_vis = (
        pd.concat([hist, fut], ignore_index=True)
        .sort_values(["admin_1", "product", "month"])
        .reset_index(drop=True)
    )

    return df_vis, used_msgs

def render_forecasting_page():
    try:
        df_vis, used_msgs = load_forecast_data()
    except Exception as e:
        st.error(f"Could not load forecast data: {e}")
        return

    # Side info about which columns were used
    for m in used_msgs:
        st.caption(f"ℹ️ {m}")

    st.header("Staples price forecast")
    st.caption(
        "Forecasts are only shown for region–product combinations with at least 12 "
        "months of history in the training data. CI bands are based on residual "
        "dispersion in the test set."
    )

    # Available admin/product choices (only those we actually forecast)
    admins  = sorted(df_vis["admin_1"].dropna().unique())
    products = sorted(df_vis["product"].dropna().unique())

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        admin = st.selectbox("Region (admin 1)", admins)
    with col_sel2:
        prod = st.selectbox("Product", products)

    d = (
        df_vis
        .query("admin_1 == @admin and product == @prod")
        .sort_values("month")
        .reset_index(drop=True)
    )

    if d.empty:
        st.warning("No forecast available for this region / product (likely <12 months of history).")
        return

    fig = go.Figure()

    # CI band (future part)
    if d["y_lo"].notna().any() and d["y_hi"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=d["month"],
                y=d["y_lo"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Lower CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["month"],
                y=d["y_hi"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                opacity=0.2,
                name="Uncertainty band",
                hovertemplate=(
                    "Month=%{x|%Y-%m}<br>CI upper: %{y:.2f}<extra></extra>"
                ),
            )
        )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=d["month"],
            y=d["y_pred"],
            mode="lines+markers",
            name="Forecast",
            hovertemplate="Month=%{x|%Y-%m}<br>Forecast=%{y:.2f}<extra></extra>",
        )
    )

    # Actuals (historical)
    if "y_actual" in d and d["y_actual"].notna().any():
        mask_hist = d["y_actual"].notna()
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask_hist, "month"],
                y=d.loc[mask_hist, "y_actual"],
                mode="markers",
                name="Actual",
                hovertemplate="Month=%{x|%Y-%m}<br>Actual=%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Month",
        yaxis_title="Price (ETB)",
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------------------

def main():
    st.title("Ethiopia Food Prices Dashboard")
    
    # Load main panel for data-driven pages
    panel = load_panel()

    # Tabs
    tab_home, tab_method, tab_exp, tab_fore = st.tabs([
        "Summary",
        "Methodology / Sources",
        "Data Exploration",
        "Data Forecasting"
    ])

    with tab_home:
        render_landing_page()

    with tab_method:
        render_methodology_page()

    with tab_exp:
        render_exploration_page(panel)

    with tab_fore:
        render_forecasting_page()

    # Footer (outside tab block)
    st.markdown(
        """
        <hr style="margin-top:2em;margin-bottom:0.5em;">
        <p style="text-align:center;color:gray;font-size:0.85em;">
        Built by <b>Natascha Minnitt</b> · Data sources: WFP VAM, FEWS NET · Updated monthly
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
