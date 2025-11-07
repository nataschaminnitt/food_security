# ===================== train_and_export_with_hybrid.py =====================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
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
    smape,
    rmse,
    TEST_HORIZON_MONTHS,
    N_TRIALS,
    SEED,
)

from model_utils import recursive_forecast


PARQUET_PATH = "data/processed/ethiopia_foodprices_model_panel_tierA.parquet"
OUTPUT_DIR   = Path("/Users/nataschajademinnitt/Documents/5_data/food_security/etl/artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================== TRAIN & EXPORT ===============================
if __name__ == "__main__":
    # 1) Load
    panel = pd.read_parquet(PARQUET_PATH)
    assert {"admin_1", "product", "month", "value_imputed"}.issubset(panel.columns)
    
    # 3) Features (via staples_model_core)
    staples = impute_features(panel)
    df_feat = build_features(staples)
    df_feat = encode_ids(df_feat)
    df_feat = df_feat.sort_values(["month", "admin_1", "product"]).reset_index(drop=True)

    feats = pick_features(df_feat)
    df_feat[feats] = df_feat[feats].apply(pd.to_numeric, errors="coerce").astype("float32")

    # 4) Split (same horizon logic as modeling pipeline)
    train_df, test_df = time_split(df_feat, horizon=TEST_HORIZON_MONTHS)
    X_train, X_test = train_df[feats], test_df[feats]
    y_train_log = np.log1p(train_df["y"].to_numpy())
    y_test_true = test_df["y"].to_numpy()

    # 4b) Mark series with ≥12 months of history in TRAIN
    # We use distinct months with non-null y in the training window.
    hist_counts = (
        train_df[train_df["y"].notna()]
        .groupby(["admin_1", "product"], observed=False)["month"]
        .nunique()
        .reset_index(name="n_train_months")
    )

    eligible = hist_counts[hist_counts["n_train_months"] >= 12].copy()
    eligible["has_12m_history"] = True

    # Attach flag to train/test (for reference/diagnostics)
    train_df = train_df.merge(
        eligible[["admin_1", "product", "has_12m_history"]],
        on=["admin_1", "product"],
        how="left",
    )
    train_df["has_12m_history"] = train_df["has_12m_history"].fillna(False)

    test_df = test_df.merge(
        eligible[["admin_1", "product", "has_12m_history"]],
        on=["admin_1", "product"],
        how="left",
    )
    test_df["has_12m_history"] = test_df["has_12m_history"].fillna(False)

    # 5) Tune
    print(f"[{datetime.now()}] Tuning ({N_TRIALS} trials)…")
    best_params = tune_xgb(X_train, y_train_log, n_trials=N_TRIALS, seed=SEED)

    # 6) Early stopping on tail of TRAIN (same as pipeline)
    last_train_m = train_df["month"].max()
    val_cut = (last_train_m.to_period("M") - 2).to_timestamp()  # last ~3 months
    val_mask = train_df["month"] >= val_cut
    X_tr_es, y_tr_es = X_train[~val_mask], y_train_log[~val_mask]
    X_va_es, y_va_es = X_train[val_mask],  y_train_log[val_mask]
    if len(X_va_es) == 0 or len(X_tr_es) == 0:
        X_tr_es, y_tr_es = X_train, y_train_log
        X_va_es = y_va_es = None

    # 7) Fit final & predict test (global)
    final_model = fit_xgb_compat(
        best_params,
        X_tr_es,
        y_tr_es,
        X_va_es,
        y_va_es,
        early_rounds=200,
    )
    y_pred_test_global = np.expm1(final_model.predict(X_test))

    # ---- HYBRID: fit per-product Ridge residual corrections on TRAIN ----
    train_df = train_df.copy()
    train_df["y_pred_global"] = np.expm1(final_model.predict(X_train))
    train_df["resid"] = train_df["y"] - train_df["y_pred_global"]

    hybrid_models: dict[str, Pipeline] = {}
    for prod, g in train_df.groupby("product", observed=False):
        # Only fit a hybrid model where we have enough history for that product overall
        if len(g) < 12:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge",  Ridge(alpha=2.0, fit_intercept=True, random_state=SEED)),
        ])
        Xp = g[["y_pred_global"]]
        yp = g["resid"]
        pipe.fit(Xp, yp)
        hybrid_models[prod] = pipe

    # Apply hybrid on TEST
    test_df = test_df.copy()
    test_df["y_pred_global"] = y_pred_test_global

    corrections = []
    for _, row in test_df.iterrows():
        prod = row["product"]
        base = row["y_pred_global"]
        if prod in hybrid_models:
            corr = float(
                hybrid_models[prod].predict(
                    pd.DataFrame([[base]], columns=["y_pred_global"])
                )[0]
            )
            corrections.append(base + corr)
        else:
            corrections.append(base)
    test_df["y_pred_hybrid"] = corrections

    # ---- Metrics: only where we have ≥12 months of history ----
    valid_mask = (
        test_df["has_12m_history"]
        & test_df["y_pred_hybrid"].notna()
        & test_df["y"].notna()
    )

    print("\n=== STAPLES TEST METRICS (Hybrid, only series with ≥12 months history) ===")
    if valid_mask.any():
        mae_val = mean_absolute_error(
            test_df.loc[valid_mask, "y"],
            test_df.loc[valid_mask, "y_pred_hybrid"],
        )
        rmse_val = rmse(
            test_df.loc[valid_mask, "y"],
            test_df.loc[valid_mask, "y_pred_hybrid"],
        )
        smape_val = smape(
            test_df.loc[valid_mask, "y"],
            test_df.loc[valid_mask, "y_pred_hybrid"],
        )

        print(f"MAE : {mae_val:,.3f}")
        print(f"RMSE: {rmse_val:,.3f}")
        print(f"sMAPE: {smape_val:.2f}%")
    else:
        print("No eligible series (≥12 months history) in the test set; skipping metrics.")

    # Per-product breakdown (only for eligible series)
    gg = test_df.loc[
        valid_mask,
        ["product", "y", "y_pred_hybrid"],
    ]
    if not gg.empty:
        by_prod = (
            gg.assign(
                abs_err=lambda d: (d["y"] - d["y_pred_hybrid"]).abs(),
                sq_err=lambda d: (d["y"] - d["y_pred_hybrid"]) ** 2,
                denom=lambda d: (d["y"].abs() + d["y_pred_hybrid"].abs() + 1e-9) / 2.0,
                smape_i=lambda d: (d["y_pred_hybrid"] - d["y"]).abs() / d["denom"] * 100.0,
            )
            .groupby("product", observed=False)
            .agg(
                sMAPE=("smape_i", "mean"),
                MAE=("abs_err", "mean"),
                RMSE=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                N=("y", "size"),
            )
            .reset_index()
            .sort_values("sMAPE", ascending=False)
        )
        print("\nWorst staples by sMAPE (eligible series only):")
        print(by_prod.head(10).to_string(index=False))
    else:
        by_prod = pd.DataFrame()
        print("\nNo per-product breakdown (no eligible series).")

    # ---- Save artifacts ----
    model_path       = OUTPUT_DIR / "xgb_staples_model.joblib"
    feats_path       = OUTPUT_DIR / "xgb_staples_features.json"
    hybrid_path      = OUTPUT_DIR / "xgb_staples_hybrid_models.joblib"
    last_hist_pq     = OUTPUT_DIR / "xgb_staples_last_history.parquet"
    test_pred_pq     = OUTPUT_DIR / "xgb_staples_test_predictions.parquet"
    future_pred_pq   = OUTPUT_DIR / "xgb_staples_future_forecast.parquet"
    metadata_path    = OUTPUT_DIR / "xgb_staples_metadata.json"

    joblib.dump(final_model, model_path)
    pd.Series(feats, name="features").to_json(feats_path, orient="values")
    joblib.dump(hybrid_models, hybrid_path)

    test_out = test_df[[
        "admin_1",
        "product",
        "month",
        "y",
        "y_pred_global",
        "y_pred_hybrid",
        "has_12m_history",
    ]].copy()
    test_out.to_parquet(test_pred_pq, index=False)

    # Last 15 months per series for recursive forecasting
    last_blocks = (
        df_feat.sort_values(["admin_1", "product", "month"])
              .groupby(["admin_1", "product"], observed=False, as_index=False)
              .tail(15)
              .reset_index(drop=True)
    )

    # Keep only series with ≥12 months history in TRAIN
    eligible_pairs = eligible[["admin_1", "product"]].drop_duplicates()
    last_blocks = last_blocks.merge(eligible_pairs, on=["admin_1", "product"], how="inner")

    last_blocks.to_parquet(last_hist_pq, index=False)

    # Forward forecasts (hybrid-corrected) for eligible series only
    print(f"\n[{datetime.now()}] Building {TEST_HORIZON_MONTHS}-month recursive forecasts…")
    future_df = recursive_forecast(
        final_model,
        last_blocks,
        feats,
        hybrid_models=hybrid_models,
        horizon=TEST_HORIZON_MONTHS,
    )
    future_df.to_parquet(future_pred_pq, index=False)

    # Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "test_horizon_months": TEST_HORIZON_MONTHS,
        "forecast_horizon_months": TEST_HORIZON_MONTHS,
        "optuna_trials": N_TRIALS,
        "best_params": best_params,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_test_eligible": int(valid_mask.sum()),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print("  Model                 ->", model_path)
    print("  Hybrid models         ->", hybrid_path)
    print("  Feature list          ->", feats_path)
    print("  Last history          ->", last_hist_pq)
    print("  Test predictions      ->", test_pred_pq)
    print("  Future forecasts      ->", future_pred_pq)
    print("  Metadata              ->", metadata_path)

    # ---- Debug preview: sample rows for inspection ----
    try:
        preview_cols = ["admin_1", "product", "month", "y", "y_pred_global", "y_pred_hybrid", "has_12m_history"]
        preview = (
            test_df[preview_cols]
            .sort_values(["product", "admin_1", "month"])
            .head(20)
            .copy()
        )
        preview["y"] = preview["y"].round(2)
        preview["y_pred_global"] = preview["y_pred_global"].round(2)
        preview["y_pred_hybrid"] = preview["y_pred_hybrid"].round(2)

        print("\nSample test predictions (first 20 rows):")
        print(preview.to_string(index=False))
    except Exception as e:
        print(f"\n[WARN] Could not create debug preview table: {e}")

    if not by_prod.empty:
        metrics_preview = by_prod.head(15).copy()
        metrics_preview["sMAPE"] = metrics_preview["sMAPE"].round(2)
        metrics_preview["MAE"]   = metrics_preview["MAE"].round(2)
        metrics_preview["RMSE"]  = metrics_preview["RMSE"].round(2)

        print("\nPer-product metrics (top 15 worst by sMAPE, eligible series only):")
        print(metrics_preview.to_string(index=False))