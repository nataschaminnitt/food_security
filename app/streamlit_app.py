# ===================== streamlit_app.py =====================

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
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
# Loaders
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
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

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
# Helpers
# ------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

# ------------------------------------------------------------------------------
# Page 0 – Data exploration (incl. STL admin × product)
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
            key="expl_products"
        )

        view_mode = st.radio(
            "View mode",
            ["National median across regions", "Regional breakdown"],
            index=0,
            horizontal=True,
            key="expl_viewmode"
        )

        if view_mode == "National median across regions":
            ts = nat_trends[nat_trends["product"].isin(selected_products)].copy()
            ts = ts.sort_values(["product", "month"])

            fig_trend = px.line(
                ts, x="month", y="price", color="product", markers=True,
                labels={"month": "Month", "price": "Price (ETB)", "product": "Product"},
            )
            fig_trend.update_layout(height=450, title="National median prices across regions",
                                    legend_title_text="Product")

        else:
            regions = regional_trends["admin_1"].dropna().sort_values().unique().tolist()
            default_region = "Addis Ababa" if "Addis Ababa" in regions else regions[0]

            region = st.selectbox(
                "Choose region",
                options=regions,
                index=regions.index(default_region),
                key="expl_trend_region"
            )

            ts = regional_trends[
                (regional_trends["admin_1"] == region)
                & (regional_trends["product"].isin(selected_products))
            ].copy().sort_values("month")

            fig_trend = px.line(
                ts, x="month", y="price", color="product", markers=True,
                labels={"month": "Month", "price": "Price (ETB)", "product": "Product"},
            )
            fig_trend.update_layout(height=450, title=f"Prices in {region}",
                                    legend_title_text="Product")

        st.plotly_chart(fig_trend, use_container_width=True)

        # ───────────────────────── Map: average prices by region ─────────────────────────
        st.markdown("### Average prices by region")

        map_product = st.selectbox(
            "Product for map",
            options=FOCUS_PRODUCTS,
            index=FOCUS_PRODUCTS.index("Maize Grain (White)") if "Maize Grain (White)" in FOCUS_PRODUCTS else 0,
            key="expl_map_product"
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
                mapbox_style="open-street-map",
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

        # ───────────────────────── Seasonal decomposition (admin × product) ─────────────────────────
        st.markdown("### Seasonal decomposition (admin × product)")

        colc1, colc2, colc3, colc4 = st.columns([1.2, 1.2, 1, 1])
        period = colc1.number_input("Seasonal period", min_value=2, max_value=60, value=12, step=1, key="expl_stl_period")
        recent_years = colc2.number_input("Recent window (years)", min_value=1, max_value=10, value=5, step=1, key="expl_stl_years")
        shade_imputed = colc3.checkbox("Shade imputed", value=True, key="expl_stl_shade")
        national_view = colc4.checkbox("National median", value=False, key="expl_stl_nat")

        # Restrict to recent window
        panel_recent = panel.copy()
        panel_recent["month"] = pd.to_datetime(panel_recent["month"], errors="coerce")
        start_date = panel_recent["month"].max() - pd.DateOffset(years=int(recent_years))
        d5 = panel_recent.loc[panel_recent["month"] >= start_date].copy()

        # Pick value column
        price_col = "value_imputed" if "value_imputed" in d5.columns else ("value_mean" if "value_mean" in d5.columns else "value_median")

        if national_view:
            admin_label = "National"
            prod_options = sorted(d5["product"].dropna().unique())
            prod = st.selectbox("Product", prod_options, index=0, key="expl_stl_product_nat")

            g = (
                d5[d5["product"] == prod]
                .groupby(["month"], as_index=False)
                .agg(
                    value=(price_col, "median"),
                    imputed_any=("impute_method", lambda x: (x != "observed").any() if "impute_method" in d5.columns else False)
                )
                .rename(columns={"month": "date"})
            )
            g["impute_method"] = np.where(g.get("imputed_any", False), "imputed", "observed")
            g = g[["date", "value", "impute_method"]]
        else:
            admins = sorted(d5["admin_1"].dropna().unique())
            default_admin = admins.index("Oromia") if "Oromia" in admins else 0
            admin_label = st.selectbox("Region (admin 1)", admins, index=default_admin, key="expl_stl_admin")

            prod_options = sorted(d5.loc[d5["admin_1"] == admin_label, "product"].dropna().unique())
            prod = st.selectbox("Product", prod_options, index=0, key="expl_stl_product")

            g = (
                d5[(d5["admin_1"] == admin_label) & (d5["product"] == prod)]
                .loc[:, ["month", price_col] + (["impute_method"] if "impute_method" in d5.columns else [])]
                .rename(columns={"month": "date", price_col: "value"})
                .sort_values("date")
                .copy()
            )
            if "impute_method" not in g.columns:
                g["impute_method"] = "observed"

        def stl_decompose_one(frame: pd.DataFrame, per: int):
            frame = frame.sort_values("date").copy()
            if frame["value"].notna().sum() < per * 2:
                return None
            s = frame["value"].astype(float).to_numpy()
            s_log = np.log1p(s)
            res = STL(s_log, period=per, robust=True).fit()
            return {
                "dates": frame["date"].to_numpy(),
                "log_data": s_log,
                "seasonal": res.seasonal,
                "trend": res.trend,
                "remainder": res.resid,
                "impute_mask": (frame["impute_method"].values != "observed") if "impute_method" in frame.columns else np.zeros(len(frame), bool),
                "ymin": float(np.nanmin(s_log)),
                "ymax": float(np.nanmax(s_log)),
            }

        out = stl_decompose_one(g, int(period))

        if out is None:
            st.warning("Not enough data points for STL (need at least ~2 seasonal periods).")
        else:
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                subplot_titles=("Log of Original Data", "Seasonal Component", "Trend Component", "Remainder Component")
            )

            # Panel 1
            fig.add_trace(
                go.Scatter(
                    x=out["dates"], y=out["log_data"], mode="lines", line=dict(color="purple"),
                    name="log1p(data)",
                    hovertemplate="<b>Log of Original Data</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>"
                ),
                row=1, col=1
            )

            # Shade imputed
            if shade_imputed and out["impute_mask"].any():
                idx = out["dates"]; mask = out["impute_mask"]
                starts, ends, in_run = [], [], False
                for i, m in enumerate(mask):
                    if m and not in_run:
                        starts.append(i); in_run = True
                    if in_run and (i == len(mask)-1 or not mask[i+1]):
                        ends.append(i); in_run = False
                shapes = []
                for s_i, e_i in zip(starts, ends):
                    shapes.append(dict(
                        type="rect", xref="x", yref="y1",
                        x0=idx[s_i], x1=idx[e_i],
                        y0=out["ymin"], y1=out["ymax"],
                        fillcolor="rgba(0,0,0,0.12)", line=dict(width=0)
                    ))
                fig.update_layout(shapes=shapes)

            # Other panels
            fig.add_trace(go.Scatter(x=out["dates"], y=out["seasonal"], mode="lines", line=dict(color="#90ED7D"),
                                     name="seasonal",
                                     hovertemplate="<b>Seasonal</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>"),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=out["dates"], y=out["trend"], mode="lines", line=dict(color="#7CB5EC"),
                                     name="trend",
                                     hovertemplate="<b>Trend</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>"),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=out["dates"], y=out["remainder"], mode="lines", line=dict(color="#F7A35C"),
                                     name="remainder",
                                     hovertemplate="<b>Remainder</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>"),
                          row=4, col=1)

            fig.update_layout(
                title=f"STL Decomposition — {'National' if national_view else admin_label} · {prod} (log1p), start={start_date.date()}",
                showlegend=False, autosize=True, height=820, width=None,
                margin=dict(l=60, r=40, t=70, b=40),
                hovermode="x unified",
                xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1),
            )
            fig.update_yaxes(title_text="log1p(data)", row=1, col=1)
            fig.update_yaxes(title_text="seasonal", row=2, col=1)
            fig.update_yaxes(title_text="trend", row=3, col=1)
            fig.update_yaxes(title_text="remainder", row=4, col=1)

            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# Page 1 – Forecasting
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

    for d in (test_df, future_df):
        if "month" in d.columns:
            d["month"] = pd.to_datetime(d["month"], errors="coerce")

    # ----- keep only series that we actually forecast for -----
    eligible_pairs = future_df[["admin_1", "product"]].dropna().drop_duplicates()
    test_df = test_df.merge(eligible_pairs, on=["admin_1", "product"], how="inner", suffixes=("", "_elig"))

    used_msgs = []

    # ----- unify prediction column -----
    pred_test_col = _pick_col(test_df, PRED_COL_CANDIDATES)
    if pred_test_col is None:
        raise KeyError(f"No prediction column found in test file. Tried: {PRED_COL_CANDIDATES}")
    if pred_test_col != "y_pred":
        test_df = test_df.rename(columns={pred_test_col: "y_pred"})

    pred_future_col = _pick_col(future_df, PRED_COL_CANDIDATES)
    if pred_future_col is None:
        raise KeyError(f"No prediction column found in future file. Tried: {PRED_COL_CANDIDATES}")
    if pred_future_col != "y_pred":
        future_df = future_df.rename(columns={pred_future_col: "y_pred"})

    # ----- unify actuals in test -----
    actual_col = _pick_col(test_df, ACTUAL_COL_CANDIDATES)
    if actual_col is None:
        test_df["y_actual"] = np.nan
        used_msgs.append("Test actuals: none found (CI from residuals will be unavailable).")
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

    # CI band (95% ~ 1.96σ)
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

    for m in used_msgs:
        st.caption(f"ℹ️ {m}")

    st.header("Staples price forecast")
    st.caption(
        "Forecasts are only shown for region–product combinations with at least 12 "
        "months of history in the training data. CI bands are based on residual "
        "dispersion in the test set."
    )

    admins  = sorted(df_vis["admin_1"].dropna().unique())
    products = sorted(df_vis["product"].dropna().unique())

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        admin = st.selectbox("Region (admin 1)", admins, key="forecast_admin")
    with col_sel2:
        prod = st.selectbox("Product", products, key="forecast_product")

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
                x=d["month"], y=d["y_lo"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip", name="Lower CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["month"], y=d["y_hi"], mode="lines",
                line=dict(width=0), fill="tonexty", opacity=0.2, name="Uncertainty band",
                hovertemplate=("Month=%{x|%Y-%m}<br>CI upper: %{y:.2f}<extra></extra>"),
            )
        )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=d["month"], y=d["y_pred"], mode="lines+markers",
            name="Forecast",
            hovertemplate="Month=%{x|%Y-%m}<br>Forecast=%{y:.2f}<extra></extra>",
        )
    )

    # Actuals (historical)
    if "y_actual" in d and d["y_actual"].notna().any():
        mask_hist = d["y_actual"].notna()
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask_hist, "month"], y=d.loc[mask_hist, "y_actual"],
                mode="markers", name="Actual",
                hovertemplate="Month=%{x|%Y-%m}<br>Actual=%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Month", yaxis_title="Price (ETB)",
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# Page 2 – Methodology / Sources
# ------------------------------------------------------------------------------
def render_methodology_page():
    st.markdown("## Methodology")

    st.markdown(
    """
### Data sources

This dashboard combines multiple data sources to support robust exploration and forecasting of food prices. 
The principal datasets are monthly staple food prices from the 
[World Food Programme (WFP)](https://data.humdata.org/dataset/wfp-food-prices-for-ethiopia) 
and the [Famine Early Warning Systems Network (FEWS NET)](https://fews.net/east-africa/ethiopia).

Supplementary data include the [FAO Food Price Index](https://www.fao.org/worldfoodsituation/foodpricesindex/en/), 
the [WFP Global Market Monitor](https://data.humdata.org/dataset/global-market-monitor), 
[CHIRPS monthly rainfall](https://data.humdata.org/dataset/eth-rainfall-subnational), 
[OCHA population estimates](https://data.humdata.org/dataset/cod-ps-eth), 
[ACLED conflict events](https://data.humdata.org/dataset/ethiopia-acled-conflict-data), 
and the [USD–ETB exchange rate (WFP VAM)](https://dataviz.vam.wfp.org/data-explorer/economic/exchange-rates).

Most sources are updated monthly. An automated pipeline refreshes the dashboard at the start of each month. 
Note that a **temporary pause** in early 2025 affected FEWS NET reporting; the FEWS NET Data Explorer currently includes the latest available data through January 2025.
"""
)

    st.write("")

    try:
        PANEL_PATH = ROOT / "img"
        logos = [
            PANEL_PATH / "fewsnet.jpeg",
            PANEL_PATH / "wfp.png",
            PANEL_PATH / "fao.png",
            PANEL_PATH / "ocha.png",
            PANEL_PATH / "chirps.png",
            PANEL_PATH / "acled.png",
        ]

        cols = st.columns(len(logos))
        for col, path in zip(cols, logos):
            with col:
                st.image(str(path), use_container_width=True)

    except Exception:
        st.caption("Logos unavailable.")


    st.markdown(
    """
### Data processing

We standardize units (per kg/liter), retain **retail prices**, and keep the **past five years**. 
For missing observations, we impute prices in three steps and record both the original value and the method used:

1) **Temporal interpolation (on log prices).**  
   For each region and product, if a few months are missing, we estimate the price by drawing a smooth line between nearby months. 
   We only fill short gaps (up to 2 months) and never make up values before the first or after the last real data point.

2) **Scaled national fallback.**  
   If a value is still missing, we estimate it as *(region’s typical ratio to the national median) × (national median for that month)*, 
   where the ratio is the median of region/national over periods with data.

3) **Cross-region same-month median.**  
   If still missing, we use the median across all regions for that product–unit in the same month.

For each record we keep: `value_orig` (observed), `value_imputed` (final series), and `impute_method` 
(one of: `observed`, `interp_time`, `scaled_national`, `cross_admin`). Imputed points add uncertainty; we surface this in forecast intervals and require ≥12 months of data for forecasting.
"""
)

    st.markdown(
        """
### Data availability

- We generate forecasts only for region–product pairs with **≥ 12 months** of observations.  
- The visualization below shows months with/without data for each product in the selected region.
"""
    )

    panel = load_panel()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")

    # --- Region picker (defines selected_admin) ---
    regions = sorted(panel["admin_1"].dropna().unique())
    if not regions:
        st.warning("No regions found in the data.")
        st.stop()

    default_region = "Addis Ababa" if "Addis Ababa" in regions else regions[0]
    selected_admin = st.selectbox(
        "Select region",
        options=regions,
        index=regions.index(default_region),
        key="method_region"
    )

    # --- Keep both original and imputed so we can classify status ---
    avail = (
        panel.groupby(["admin_1", "month", "product"], as_index=False)[["value_orig", "value_imputed"]]
            .mean()
    )

    # --- Filter selected region ---
    region_df = avail[avail["admin_1"] == selected_admin].copy()

    # --- Build full monthly grid for the selected region across its products ---
    full_months = pd.date_range(start=panel["month"].min(), end=panel["month"].max(), freq="MS")
    products_region = region_df["product"].dropna().unique()
    if len(products_region) == 0:
        st.info(f"No products found for {selected_admin}.")
        st.stop()

    all_combos = pd.MultiIndex.from_product(
        [full_months, products_region],
        names=["month", "product"]
    ).to_frame(index=False)

    region_full = all_combos.merge(region_df, on=["month", "product"], how="left")

    # --- Classify status: green observed, blue imputed, red missing ---
    observed = region_full["value_orig"].notna()
    imputed  = region_full["value_imputed"].notna() & ~observed
    missing  = region_full["value_imputed"].isna()

    region_full["status"] = np.select(
        [observed, imputed, missing],
        ["Observed", "Imputed", "Missing"],
        default="Missing"
    )

    color_map = {"Observed": "green", "Imputed": "blue", "Missing": "red"}

    # --- Plot (one trace per status so you get a legend) ---
    fig = go.Figure()
    for name, color in color_map.items():
        df_ = region_full[region_full["status"] == name]
        if df_.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_["month"],
                y=df_["product"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, line=dict(width=0)),
                hovertemplate="Month: %{x|%b %Y}<br>Product: %{y}<br>Status: " + name + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Data availability (past 5 years) – {selected_admin}",
        xaxis_title="Month",
        yaxis_title="Product",
        height=450,
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis=dict(categoryorder="category ascending"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0.25),
    )

    st.plotly_chart(fig, use_container_width=True)



    st.markdown(
        """
### Forecasting model

- A **global XGBoost** model is trained on all region–product pairs.  
- A **per-product Ridge regression** “bias-correction” layer refines residual errors.  
- Forecasts are produced only for pairs with **≥ 12 months** of training data.

### Metrics and confidence intervals

We evaluate performance on a hold-out test set using:

- **MAE (Mean Absolute Error):** average absolute difference between predictions and actual prices (lower is better).  
- **RMSE (Root Mean Squared Error):** emphasizes larger errors (lower is better).  
- **sMAPE (Symmetric Mean Absolute Percentage Error):** scale-free percentage error robust to zeros (lower is better).

We display an **80% confidence interval (CI)** around forecasts, meaning that under model assumptions, roughly 4 out of 5 realized values are expected to fall within the shaded band. Narrower bands indicate higher certainty; wider bands indicate greater uncertainty.
"""
    )

# ------------------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------------------
def main():
    st.title("🌾 Ethiopia Food Prices Dashboard")
    st.markdown("#### An interactive dashboard to track and forecast staple food prices across regions in Ethiopia.")

    
    # Load main panel for data-driven pages
    panel = load_panel()

    # Tabs
    tab_exp, tab_fore, tab_method = st.tabs([
        "Exploration",
        "Forecasting",
        "Methodology",
    ])

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
        Built by <b>Natascha Minnitt</b> · Updated monthly
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
