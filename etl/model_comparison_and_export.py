# ================= Ethiopia food prices – model comparison & export =================
#
# This script:
#   - compares StatsForecast baselines (Naive, SeasonalNaive, WindowAvg, ARIMA, SARIMA)
#     for horizons h = 3 and h = 6 months
#   - trains and evaluates a global XGBoost model (and a residual-corrected hybrid)
#     on the feature-engineered panel for horizon h = TEST_HORIZON_MONTHS (3)
#   - logs metrics to MLflow (if available)
#   - exports:
#       * etl/artifacts/model_comparison_results.csv
#       * etl/artifacts/best_model_summary.json
#
# Best model for the operational dashboard (h = 3) is selected as the one with
# the lowest sMAPE (ties broken by lower RMSE).
#
# Run
# micromamba activate foodsec_py311
# python etl/model_comparison_and_export.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # /.../food_security
ART_DIR = ROOT / "etl" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# ----- Optional MLflow logging (won't crash if missing) ----------------------
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # graceful fallback
    mlflow = None
    _HAS_MLFLOW = False

# ----- StatsForecast baselines ----------------------------------------------
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoARIMA, WindowAverage

# We only use utilsforecast for the models; metrics are done via sklearn / custom
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----- XGBoost feature-based modelling --------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from staples_model_core import (
    impute_features,
    build_features,
    encode_ids,
    pick_features,
    time_split,
    tune_xgb,
    fit_xgb_compat,
    smape as smape_vec,
    rmse as rmse_vec,
    TEST_HORIZON_MONTHS,
    N_TRIALS,
    SEED,
)

# =============================================================================
# Paths & basic setup
# =============================================================================

HERE = Path(__file__).resolve().parent              # .../food_security/etl
PROJECT_ROOT = HERE.parent                          # .../food_security
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH = DATA_DIR / "ethiopia_foodprices_model_panel_tierA.parquet"

ART_DIR = HERE / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = ART_DIR / "model_comparison_results.csv"
BEST_MODEL_JSON = ART_DIR / "best_model_summary.json"

H_SHORT = TEST_HORIZON_MONTHS  # expected 3
H_LONG = 6                     # optional long horizon for comparison


# =============================================================================
# Utilities
# =============================================================================

def compute_panel_metrics(y_true, y_pred):
    """
    Flat MAE, RMSE, sMAPE over all series & horizons.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "smape": np.nan}

    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-9) / 2.0
    smape_val = np.mean(np.abs(y_pred - y_true) / denom) * 100.0

    return {"mae": mae_val, "rmse": rmse_val, "smape": smape_val}


def log_run_to_mlflow(model_name, metrics, params=None, tags=None):
    """
    Helper to log a single model run in MLflow, if mlflow is installed.
    """
    if not _HAS_MLFLOW:
        return

    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def make_statsforecast_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the StatsForecast panel with [unique_id, ds, y], where:
      - unique_id = "admin_1 · product"
      - ds        = month-end timestamps
      - y         = value_imputed (mean if duplicates)
    """
    df_sf = (
        panel.rename(columns={"month": "ds", "value_imputed": "y"})
              .assign(
                  ds=lambda d: pd.to_datetime(d["ds"], errors="coerce")
                                 .dt.to_period("M").dt.to_timestamp("M"),
                  unique_id=lambda d: (
                      d["admin_1"].astype(str) + " · " + d["product"].astype(str)
                  ),
              )[["unique_id", "ds", "y"]]
              .groupby(["unique_id", "ds"], as_index=False)["y"].mean()
              .sort_values(["unique_id", "ds"])
    )
    return df_sf


def make_train_test_statsforecast(df_sf: pd.DataFrame, horizon: int):
    """
    For each unique_id, use the last `horizon` months as test, rest as train.
    Require more than `horizon` points.
    """
    counts = df_sf.groupby("unique_id")["ds"].count()
    ok_ids = counts[counts > horizon].index
    df_sub = df_sf[df_sf["unique_id"].isin(ok_ids)].reset_index(drop=True)

    test = df_sub.groupby("unique_id", group_keys=False).tail(horizon).reset_index(drop=True)
    train = (
        df_sub.groupby("unique_id", group_keys=False)
              .apply(lambda g: g.iloc[:-horizon])
              .reset_index(drop=True)
    )
    return train, test


def forecast_naive_drift(train_sf, test_sf, h, drift_window=6, col_name="NaiveDrift"):
    """
    Naive + drift model:
      - per unique_id, fit drift = mean of last `drift_window` differences
      - forecast h steps ahead: y_t + k * drift, k=1..h
    Returns DataFrame: [unique_id, ds, col_name].
    """
    preds = []

    for uid, g_train in train_sf.groupby("unique_id", observed=False):
        g_train = g_train.sort_values("ds")
        g_test = (
            test_sf.loc[test_sf["unique_id"] == uid]
                   .sort_values("ds")
        )
        if g_test.empty:
            continue

        y_train = g_train["y"].to_numpy()
        last_y = y_train[-1]

        diffs = np.diff(y_train)
        if len(diffs) == 0:
            drift = 0.0
        else:
            drift = diffs[-drift_window:].mean()

        horizon = len(g_test)
        y_forecast = [last_y + (i + 1) * drift for i in range(horizon)]

        tmp = g_test[["unique_id", "ds"]].copy()
        tmp[col_name] = y_forecast
        preds.append(tmp)

    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds", col_name])

    return pd.concat(preds, ignore_index=True)


# =============================================================================
# Group residual corrector (for XGB hybrid)
# =============================================================================

class GroupResidualCorrector:
    """
    Generic group-wise residual corrector.

    - group_cols: columns defining a series group (e.g. ["product"])
    - main_pred_col: base model prediction column
    - feature_cols: columns used as features to predict residuals
    - target_col: true target column
    """

    def __init__(
        self,
        group_cols,
        main_pred_col: str,
        feature_cols=None,
        target_col: str = "y",
        min_n: int = 12,
    ):
        self.group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        self.main_pred_col = main_pred_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.min_n = min_n

        self.models_ = {}

    def _base_estimator_factory(self):
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=2.0, fit_intercept=True, random_state=SEED)),
        ])

    def fit(self, df: pd.DataFrame) -> "GroupResidualCorrector":
        if self.feature_cols is None:
            self.feature_cols = [self.main_pred_col]

        required = set(self.group_cols + [self.target_col, self.main_pred_col] + self.feature_cols)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        self.models_.clear()

        for g_key, sub in df.groupby(self.group_cols, observed=False):
            if len(sub) < self.min_n:
                continue

            y_true = sub[self.target_col].to_numpy(dtype=float)
            y_base = sub[self.main_pred_col].to_numpy(dtype=float)
            residuals = y_true - y_base

            X = sub[self.feature_cols].to_numpy(dtype=float)
            model = self._base_estimator_factory()
            model.fit(X, residuals)
            self.models_[g_key] = model

        return self

    def predict(self, df: pd.DataFrame, new_col: str = "y_pred_corrected", inplace: bool = False):
        if not self.models_:
            raise RuntimeError("You must call .fit() before .predict().")

        if self.feature_cols is None:
            self.feature_cols = [self.main_pred_col]

        required = set(self.group_cols + [self.main_pred_col] + self.feature_cols)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        out = df if inplace else df.copy()
        out[new_col] = out[self.main_pred_col].astype(float)

        for g_key, sub_idx in out.groupby(self.group_cols, observed=False).groups.items():
            model = self.models_.get(g_key)
            if model is None:
                continue
            X = out.loc[sub_idx, self.feature_cols].to_numpy(dtype=float)
            corr = model.predict(X)
            base_vals = out.loc[sub_idx, self.main_pred_col].to_numpy(dtype=float)
            out.loc[sub_idx, new_col] = base_vals + corr

        return out


# =============================================================================
# Main script
# =============================================================================

def main():
    print(f"[{datetime.now().isoformat()}] Loading panel from {PARQUET_PATH}")
    panel = pd.read_parquet(PARQUET_PATH)
    assert {"admin_1", "product", "month", "value_imputed"}.issubset(panel.columns)

    # ------------------------------------------------------------------------------
    # 1) StatsForecast baselines (h = 3 and h = 6)
    # ------------------------------------------------------------------------------
    print("\n=== Building StatsForecast panel ===")
    df_sf = make_statsforecast_panel(panel)

    results_rows = []

    for horizon, label in [(H_SHORT, "h3"), (H_LONG, "h6")]:
        print(f"\n=== StatsForecast models for horizon = {horizon} ({label}) ===")
        train_sf, test_sf = make_train_test_statsforecast(df_sf, horizon=horizon)
        print("  train:", train_sf.shape, "test:", test_sf.shape)

        models_sf = [
            Naive(),  # alias: "Naive"
            SeasonalNaive(season_length=12, alias="SeasonalNaive_12"),
            WindowAverage(window_size=3, alias="WindowAvg_3"),
            AutoARIMA(seasonal=False, alias="ARIMA"),
            AutoARIMA(season_length=12, alias="SARIMA"),
        ]

        sf = StatsForecast(models=models_sf, freq="M", n_jobs=-1)
        sf.fit(df=train_sf)

        preds_sf = sf.predict(h=horizon)
        eval_sf = test_sf.merge(preds_sf, on=["unique_id", "ds"], how="left")

        model_cols = [c for c in eval_sf.columns if c not in ["unique_id", "ds", "y"]]

        for col in model_cols:
            model_name = f"sf_{col}_{label}"
            y_true = eval_sf["y"].values
            y_pred = eval_sf[col].values

            metrics = compute_panel_metrics(y_true, y_pred)
            print(f"  {model_name}: {metrics}")

            results_rows.append({
                "model": model_name,
                "horizon": horizon,
                **metrics,
            })

            log_run_to_mlflow(
                model_name=model_name,
                metrics=metrics,
                params={"family": "statsforecast", "horizon": horizon},
                tags={"kind": f"baseline_{label}"},
            )

        # --- Naive + drift on same horizon ---
        nd_preds = forecast_naive_drift(
            train_sf, test_sf, h=horizon, drift_window=6, col_name="NaiveDrift"
        )
        eval_nd = test_sf.merge(nd_preds, on=["unique_id", "ds"], how="left")
        metrics_nd = compute_panel_metrics(eval_nd["y"], eval_nd["NaiveDrift"])

        nd_name = f"sf_NaiveDrift_{label}"
        print(f"  {nd_name}: {metrics_nd}")

        results_rows.append({
            "model": nd_name,
            "horizon": horizon,
            **metrics_nd,
        })

        log_run_to_mlflow(
            model_name=nd_name,
            metrics=metrics_nd,
            params={"family": "naive_drift", "horizon": horizon, "drift_window": 6},
            tags={"kind": f"baseline_plus_{label}"},
        )

    # ------------------------------------------------------------------------------
    # 2) Global XGBoost + hybrid (feature-based) – only for h = TEST_HORIZON_MONTHS
    # ------------------------------------------------------------------------------
    print(f"\n=== Feature-based global XGB (h = {TEST_HORIZON_MONTHS}) ===")

    # Feature engineering pipeline
    staples = impute_features(panel)
    df_feat = build_features(staples)
    df_feat = encode_ids(df_feat)
    df_feat = df_feat.sort_values(["month", "admin_1", "product"]).reset_index(drop=True)

    feats = pick_features(df_feat)
    df_feat[feats] = df_feat[feats].apply(pd.to_numeric, errors="coerce").astype("float32")

    train_df, test_df = time_split(df_feat, horizon=TEST_HORIZON_MONTHS)
    X_train, X_test = train_df[feats], test_df[feats]
    y_train_log = np.log1p(train_df["y"].to_numpy())
    y_test_true = test_df["y"].to_numpy()

    # History length per series
    hist_month_counts = (
        train_df.groupby(["admin_1", "product"])["month"]
                .nunique()
                .rename("n_hist_months")
                .reset_index()
    )
    test_df = test_df.merge(hist_month_counts, on=["admin_1", "product"], how="left")
    test_df["has_12m_hist"] = test_df["n_hist_months"].fillna(0) >= 12

    print("  test rows with ≥12 months history:", int(test_df["has_12m_hist"].sum()))

    # Tune XGB
    print(f"  Tuning XGB with {N_TRIALS} Optuna trials…")
    best_params = tune_xgb(X_train, y_train_log, n_trials=N_TRIALS, seed=SEED)
    print("  Best params:", best_params)

    # Early-stopping split: last ~3 months of train as validation, if possible
    last_train_m = train_df["month"].max()
    val_cut = (last_train_m.to_period("M") - 2).to_timestamp()
    val_mask = train_df["month"] >= val_cut

    X_tr_es, y_tr_es = X_train[~val_mask], y_train_log[~val_mask]
    X_va_es, y_va_es = X_train[val_mask], y_train_log[val_mask]

    if len(X_va_es) == 0 or len(X_tr_es) == 0:
        X_tr_es, y_tr_es = X_train, y_train_log
        X_va_es = y_va_es = None

    final_xgb = fit_xgb_compat(
        best_params,
        X_tr_es,
        y_tr_es,
        X_va_es,
        y_va_es,
        early_rounds=200,
    )

    # Base predictions
    train_df["y_pred_xgb"] = np.expm1(final_xgb.predict(X_train))
    test_df["y_pred_xgb"] = np.expm1(final_xgb.predict(X_test))

    mask_ok = test_df["has_12m_hist"]
    eval_xgb = test_df[mask_ok].copy()
    valid_mask_base = eval_xgb["y_pred_xgb"].notna() & eval_xgb["y"].notna()

    metrics_xgb_base = compute_panel_metrics(
        eval_xgb.loc[valid_mask_base, "y"],
        eval_xgb.loc[valid_mask_base, "y_pred_xgb"],
    )
    print("  XGB global base metrics:", metrics_xgb_base)

    results_rows.append({
        "model": "xgb_global_base",
        "horizon": TEST_HORIZON_MONTHS,
        **metrics_xgb_base,
    })

    log_run_to_mlflow(
        model_name="xgb_global_base",
        metrics=metrics_xgb_base,
        params={
            "family": "global_ml",
            "horizon": TEST_HORIZON_MONTHS,
            "n_features": len(feats),
            **{f"xgb_{k}": v for k, v in best_params.items()},
        },
        tags={"kind": "candidate", "uses_residual_correction": False},
    )

    # Hybrid residual-corrected XGB (per product)
    corrector = GroupResidualCorrector(
        group_cols="product",
        main_pred_col="y_pred_xgb",
        feature_cols=["y_pred_xgb"],
        target_col="y",
        min_n=12,
    )
    corrector.fit(train_df)

    eval_xgb = corrector.predict(eval_xgb, new_col="y_pred_xgb_hybrid", inplace=True)
    valid_mask_hybrid = eval_xgb["y_pred_xgb_hybrid"].notna() & eval_xgb["y"].notna()

    metrics_xgb_hybrid = compute_panel_metrics(
        eval_xgb.loc[valid_mask_hybrid, "y"],
        eval_xgb.loc[valid_mask_hybrid, "y_pred_xgb_hybrid"],
    )
    print("  XGB hybrid metrics:", metrics_xgb_hybrid)

    results_rows.append({
        "model": "xgb_global_hybrid",
        "horizon": TEST_HORIZON_MONTHS,
        **metrics_xgb_hybrid,
    })

    log_run_to_mlflow(
        model_name="xgb_global_hybrid",
        metrics=metrics_xgb_hybrid,
        params={
            "family": "global_ml",
            "horizon": TEST_HORIZON_MONTHS,
            "n_features": len(feats),
            "residual_group": "product",
            "residual_features": "y_pred_xgb",
            **{f"xgb_{k}": v for k, v in best_params.items()},
        },
        tags={"kind": "production_candidate", "uses_residual_correction": True},
    )

    # ------------------------------------------------------------------------------
    # 3) Final results table & best model (for h = TEST_HORIZON_MONTHS)
    # ------------------------------------------------------------------------------
    results_df = (
        pd.DataFrame(results_rows)
        .drop_duplicates(subset=["model", "horizon"])
        .set_index(["model", "horizon"])
        .sort_values(["horizon", "smape", "rmse"])
    )

    print("\n=== Model comparison (all horizons) ===")
    print(results_df.round(3))

    # Export comparison table
    results_df.to_csv(RESULTS_CSV)
    print(f"\nSaved model comparison table to: {RESULTS_CSV}")

    # Best model for operational horizon (h = TEST_HORIZON_MONTHS)
    results_h_short = (
        results_df.reset_index()
        [results_df.reset_index()["horizon"] == TEST_HORIZON_MONTHS]
        .set_index("model")
        .sort_values(["smape", "rmse"])
    )

    best_model_name = results_h_short.index[0]
    best_model_metrics = results_h_short.iloc[0][["mae", "rmse", "smape"]].to_dict()

    print(f"\n=== Best model for horizon={TEST_HORIZON_MONTHS} ===")
    print("Model:", best_model_name)
    print("Metrics:", best_model_metrics)

    # Export best model summary as JSON
    import json

    best_summary = {
        "timestamp": datetime.now().isoformat(),
        "horizon_months": int(TEST_HORIZON_MONTHS),
        "best_model": best_model_name,
        "metrics": {k: float(v) for k, v in best_model_metrics.items()},
        "note": (
            "Best model by sMAPE (then RMSE) among StatsForecast baselines and "
            "global XGB / XGB-hybrid. Use this model as the operational default "
            "for the Streamlit forecasting dashboard."
        ),
    }

    with open(BEST_MODEL_JSON, "w") as f:
        json.dump(best_summary, f, indent=2)

    print(f"\nSaved best model summary to: {BEST_MODEL_JSON}")

    # ========================= 9. Export Naive h=3 test & future =========================

    H = TEST_HORIZON_MONTHS  # should be 3

    print(f"\nBuilding Naive(h={H}) test & future forecasts for dashboard…")

    # 1) Build clean monthly panel per (admin_1, product)
    df_naive = (
        panel.rename(columns={"month": "ds", "value_imputed": "y"})
            .assign(
                ds=lambda d: pd.to_datetime(d["ds"], errors="coerce")
                                .dt.to_period("M").dt.to_timestamp("M"),
            )[["admin_1", "product", "ds", "y"]]
            .groupby(["admin_1", "product", "ds"], as_index=False)["y"]
            .mean()
            .sort_values(["admin_1", "product", "ds"])
            .reset_index(drop=True)
    )

    def split_last_h(group: pd.DataFrame, h: int):
        """Return (train, test) where last h rows are test."""
        if len(group) <= h:
            # not enough history → no test points
            return group.iloc[:-h], group.iloc[0:0]
        return group.iloc[:-h], group.iloc[-h:]

    train_list = []
    test_list = []

    for (adm, prod), g in df_naive.groupby(["admin_1", "product"], observed=False):
        g_train, g_test = split_last_h(g, h=H)
        if not g_train.empty:
            g_train = g_train.copy()
            g_train["has_history"] = True
            train_list.append(g_train)
        if not g_test.empty:
            g_test = g_test.copy()
            g_test["has_history"] = True
            test_list.append(g_test)

    train_naive = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    test_naive  = pd.concat(test_list,  ignore_index=True) if test_list  else pd.DataFrame()

    print(f"Naive train rows: {len(train_naive)}, test rows: {len(test_naive)}")

    # 2) Naive forecasts on TEST (last value repeated)
    if not train_naive.empty and not test_naive.empty:
        naive_preds = []

        for (adm, prod), g_train in train_naive.groupby(["admin_1", "product"], observed=False):
            g_test = (
                test_naive.loc[
                    (test_naive["admin_1"] == adm) & (test_naive["product"] == prod)
                ].sort_values("ds")
            )
            if g_test.empty:
                continue

            last_y = g_train.sort_values("ds")["y"].iloc[-1]
            horizon = len(g_test)
            y_forecast = [last_y] * horizon

            tmp = g_test[["admin_1", "product", "ds"]].copy()
            tmp["y_pred"] = y_forecast
            naive_preds.append(tmp)

        if naive_preds:
            naive_test_pred = pd.concat(naive_preds, ignore_index=True)
        else:
            naive_test_pred = pd.DataFrame(columns=["admin_1", "product", "ds", "y_pred"])
    else:
        naive_test_pred = pd.DataFrame(columns=["admin_1", "product", "ds", "y_pred"])

    # Merge back to compute metrics if you like (optional)
    test_naive = test_naive.merge(
        naive_test_pred,
        on=["admin_1", "product", "ds"],
        how="left",
    )

    # 3) Build future forecasts (next H months, last value carried forward)
    future_rows = []
    last_obs = (
        df_naive.sort_values("ds")
                .groupby(["admin_1", "product"], observed=False)
                .tail(1)
                .reset_index(drop=True)
    )

    for _, row in last_obs.iterrows():
        adm  = row["admin_1"]
        prod = row["product"]
        last_ds = row["ds"]
        last_y  = row["y"]

        last_period = last_ds.to_period("M")
        future_dates = [
            (last_period + i).to_timestamp("M")
            for i in range(1, H + 1)
        ]

        for ds_f in future_dates:
            future_rows.append({
                "admin_1": adm,
                "product": prod,
                "ds": ds_f,
                "y_pred": last_y,
            })

    future_naive = pd.DataFrame(future_rows)

    print(f"Built {len(future_naive)} future forecast rows for Naive(h={H}).")

    # 4) Save in the exact format/filenames expected by Streamlit
    test_out = test_naive.rename(columns={"ds": "month"})[
        ["admin_1", "product", "month", "y", "y_pred"]
    ].copy()
    future_out = future_naive.rename(columns={"ds": "month"})[
        ["admin_1", "product", "month", "y_pred"]
    ].copy()

    test_pred_pq   = ART_DIR / "naive_h3_test_predictions.parquet"
    future_pred_pq = ART_DIR / "naive_h3_future_forecast.parquet"

    test_out.to_parquet(test_pred_pq, index=False)
    future_out.to_parquet(future_pred_pq, index=False)

    print("\nSaved Naive(h=3) dashboard artifacts:")
    print("  Test predictions  ->", test_pred_pq)
    print("  Future forecasts  ->", future_pred_pq)

    # 5) (Optional) also export comparison table for Streamlit methodology page
    comparison_path = ART_DIR / "model_comparison_results.parquet"
    results_df.to_parquet(comparison_path)
    print("  Model comparison  ->", comparison_path)


if __name__ == "__main__":
    main()
