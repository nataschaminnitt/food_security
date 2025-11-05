# ===================== model_utils.py =====================

import numpy as np
import pandas as pd

def future_months(start_month: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start_m = start_month.to_period("M")
    rng = pd.period_range(start=start_m + 1, periods=horizon, freq="M").to_timestamp()
    return pd.DatetimeIndex(rng)

def recursive_forecast(model, last_hist, feats, hybrid_models=None, horizon=6):
    """
    Recursively forecasts each (admin_1, product) series forward for `horizon` months.
    If `hybrid_models` dict is provided, applies per-product residual correction.
    """
    out_rows = []
    for (adm, prod), g in last_hist.groupby(["admin_1", "product"], observed=False, sort=False):
        g = g.sort_values("month").copy()
        last_m = g["month"].max()
        future_idx = future_months(last_m, horizon)
        work = g[["month", "y"]].copy()

        admin_code = int(g["admin_1_code"].iloc[-1]) if "admin_1_code" in g.columns else 0
        product_code = int(g["product_code"].iloc[-1]) if "product_code" in g.columns else 0

        exo_cols = [c for c in feats if c not in {
            "admin_1_code", "product_code",
            "y_lag1","y_lag3","y_lag6","y_lag12",
            "roll_mean_3","roll_std_3","roll_mean_12",
            "month_num","mo_sin","mo_cos"
        }]
        exo_defaults = {}
        if exo_cols:
            last_exo = g.iloc[-1:].reindex(columns=exo_cols).to_dict(orient="records")[0]
            exo_defaults = {k: (0.0 if pd.isna(v) else float(v)) for k, v in last_exo.items()}

        for m in future_idx:
            y_lag1  = work["y"].iloc[-1] if len(work) >= 1 else np.nan
            y_lag3  = work["y"].iloc[-3] if len(work) >= 3 else np.nan
            y_lag6  = work["y"].iloc[-6] if len(work) >= 6 else np.nan
            y_lag12 = work["y"].iloc[-12] if len(work) >= 12 else np.nan

            roll_mean_3  = work["y"].shift(1).rolling(3,  min_periods=1).mean().iloc[-1] if len(work) >= 1 else np.nan
            roll_std_3   = work["y"].shift(1).rolling(3,  min_periods=2).std().iloc[-1]  if len(work) >= 2 else np.nan
            roll_mean_12 = work["y"].shift(1).rolling(12, min_periods=3).mean().iloc[-1] if len(work) >= 3 else np.nan

            feat_row = {
                "admin_1_code": admin_code,
                "product_code": product_code,
                "y_lag1": y_lag1, "y_lag3": y_lag3, "y_lag6": y_lag6, "y_lag12": y_lag12,
                "roll_mean_3": roll_mean_3, "roll_std_3": roll_std_3, "roll_mean_12": roll_mean_12,
                "month_num": int(m.month),
                "mo_sin": float(np.sin(2*np.pi*m.month/12.0)),
                "mo_cos": float(np.cos(2*np.pi*m.month/12.0)),
            }
            feat_row.update(exo_defaults)

            X_f = pd.DataFrame([feat_row])[feats].apply(pd.to_numeric, errors="coerce")
            y_pred_global = float(np.expm1(model.predict(X_f)[0]))

            if hybrid_models and prod in hybrid_models:
                corr = float(hybrid_models[prod].predict(
                    pd.DataFrame([[y_pred_global]], columns=["y_pred_global"])
                )[0])
                y_hat = y_pred_global + corr
            else:
                y_hat = y_pred_global

            out_rows.append({"admin_1": adm, "product": prod, "month": m, "y_pred": y_hat})
            work = pd.concat([work, pd.DataFrame({"month": [m], "y": [y_hat]})], ignore_index=True)

    return pd.DataFrame(out_rows)
