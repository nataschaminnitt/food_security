# ===================== staples_model_core.py =====================

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from xgboost import XGBRegressor

TEST_HORIZON_MONTHS = 6
N_TRIALS = 40
SEED = 42

def month_start_series(s: pd.Series) -> pd.Series:
    """Coerce to timestamp at month start."""
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def smape(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ensure_cols(df: pd.DataFrame, cols: list[str], dtype: str | None = "float64") -> pd.DataFrame:
    """Add any missing columns with NaN; cast dtype if requested."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        if dtype is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    out[c] = out[c].astype(dtype)
                except Exception:
                    pass
    return out

def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Create <col>_was_missing flags. If a col is absent, create it as NaN first."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[f"{c}_was_missing"] = out[c].isna().astype("int8")
    return out

def national_same_month_fill(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Median over all rows at the same month."""
    out = df.copy()
    if col in out.columns:
        same_mo = out.groupby("month", observed=False)[col].transform("median")
        out[col] = out[col].fillna(same_mo)
    return out

def past_only_roll_fill(df: pd.DataFrame, key_cols: list[str], col: str, window: int = 3) -> pd.DataFrame:
    """Per group, fallback with past-only rolling mean."""
    out = df.sort_values(key_cols + ["month"]).copy()
    if col in out.columns:
        def _roll(g):
            s = g[col]
            s_fallback = s.shift(1).rolling(window=window, min_periods=1).mean()
            return s.fillna(s_fallback)
        out[col] = out.groupby(key_cols, group_keys=False)[col].apply(_roll)
    return out

# ---- data prep ----
KEY_CATS = {"admin_1", "product"}
NUMERIC_LIKELY = {
    "value_imputed","value_mean","value_median","value_orig","n_obs",
    "rfh_month","rfh_avg_month","rfq_month","rain_anom_pct",
    "fao_category_index","fao_food_price_index",
    "ptm_severity","population_2023",
    "y","y_lag1","y_lag3","y_lag6","y_lag12",
    "roll_mean_3","roll_std_3","roll_mean_12",
    "month_num","mo_sin","mo_cos",
}

def decategorize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Keep key IDs as category, coerce everything else plausibly numeric to numeric."""
    out = df.copy()
    for c in out.columns:
        if c in KEY_CATS:
            if not isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].astype("category")
            continue
        if (c in NUMERIC_LIKELY
            or isinstance(out[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(out[c])):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def impute_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Robust FE:
      - month to month-start + drop accidental duplicates
      - ensure optional columns exist (created if missing)
      - add missingness flags (pre-fill)
      - recompute rain_anom_pct where possible
      - same-month national median fill
      - past-only rolling per admin_1
      - ptm/pop flags & imputations
      - FAO: same-month fill then ffill
    """
    df = decategorize_numerics(df_in.copy())

    # Standardize time + remove accidental duplicates (keep latest)
    df["month"] = month_start_series(df["month"])
    df = (
        df.sort_values(["admin_1", "product", "month"])
          .drop_duplicates(subset=["admin_1", "product", "month"], keep="last")
          .reset_index(drop=True)
    )

    # --- Define optional columns and ensure they exist BEFORE first access
    rain_cols_raw = ["rfh_month", "rfh_avg_month", "rfq_month", "rain_anom_pct"]
    fao_cols_raw  = ["fao_category_index", "fao_food_price_index"]

    # Create missing columns as NaN (no KeyErrors later)
    for c in rain_cols_raw + fao_cols_raw:
        if c not in df.columns:
            df[c] = np.nan

    # Cast where possible (won't error if all NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in rain_cols_raw + fao_cols_raw:
            try:
                df[c] = df[c].astype("float64")
            except Exception:
                pass

    rain_cols = rain_cols_raw[:]  # (now guaranteed to exist)
    fao_cols  = fao_cols_raw[:]

    # --- Debug sanity checks (safe to leave in; prints once)
    if not hasattr(impute_features, "_printed_once"):
        print("[impute_features] columns at entry:", list(df.columns))
        missing_any = [c for c in rain_cols + fao_cols if c not in df.columns]
        if missing_any:
            print("[impute_features] (INFO) created missing columns:", missing_any)
        impute_features._printed_once = True

    # --- Missingness flags BEFORE filling
    for c in rain_cols:
        df[f"{c}_was_missing"] = df[c].isna().astype("int8")

    # --- Recompute rain anomaly where possible
    can = df["rfh_month"].notna() & df["rfh_avg_month"].gt(0)
    df.loc[can, "rain_anom_pct"] = 100.0 * (df.loc[can, "rfh_month"] / df.loc[can, "rfh_avg_month"] - 1.0)

    # --- Same-month national median fill
    for c in rain_cols + fao_cols:
        df[c] = df[c].fillna(df.groupby("month", observed=False)[c].transform("median"))

    # --- Past-only rolling fallback per admin_1 (gentle)
    df = df.sort_values(["admin_1", "month"])
    for c in rain_cols:
        df[c] = (
            df.groupby("admin_1", group_keys=False)[c]
              .apply(lambda s: s.fillna(s.shift(1).rolling(3, min_periods=1).mean()))
        )

    # --- PTM (GMM) & population
    if "ptm_severity" not in df.columns:
        df["ptm_severity"] = np.nan
    df["ptm_missing"] = df["ptm_severity"].isna().astype("int8")
    df["ptm_severity"] = df["ptm_severity"].fillna(0).astype("int8")

    if "population_2023" not in df.columns:
        df["population_2023"] = np.nan
    df["pop_missing"] = df["population_2023"].isna().astype("int8")
    adm_med = df.groupby("admin_1", observed=False)["population_2023"].transform("median")
    df["population_2023"] = df["population_2023"].fillna(adm_med)
    df["population_2023"] = df["population_2023"].fillna(df["population_2023"].median())

    # --- FAO: same-month fill then ffill by time
    df = df.sort_values(["month"])
    for c in fao_cols:
        df[c] = df[c].fillna(df.groupby("month", observed=False)[c].transform("median"))
        df[c] = df[c].ffill()

    return df

def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.sort_values(["admin_1","product","month"]).copy()
    df["y"] = df["value_imputed"].astype("float32")
    g = df.groupby(["admin_1","product"], observed=False, group_keys=False)["y"]
    df["y_lag1"]  = g.shift(1)
    df["y_lag3"]  = g.shift(3)
    df["y_lag6"]  = g.shift(6)
    df["y_lag12"] = g.shift(12)
    df["roll_mean_3"]  = g.apply(lambda s: s.shift(1).rolling(3,  min_periods=1).mean())
    df["roll_std_3"]   = g.apply(lambda s: s.shift(1).rolling(3,  min_periods=2).std())
    df["roll_mean_12"] = g.apply(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
    df["month_num"] = df["month"].dt.month
    df["mo_sin"] = np.sin(2*np.pi*df["month_num"]/12.0)
    df["mo_cos"] = np.cos(2*np.pi*df["month_num"]/12.0)
    return df[df["y_lag1"].notna()].copy()

def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["admin_1","product"]:
        out[c] = out[c].astype("category")
        out[f"{c}_code"] = out[c].cat.codes.astype("int32")
    return out

def pick_features(df: pd.DataFrame) -> list[str]:
    drop = {"y","month","admin_1","product",
            "value_imputed","value_median","value_mean","value_orig",
            "n_obs","impute_method","sources","period_date"}
    return [c for c in df.columns if c not in drop]

def time_split(df: pd.DataFrame, horizon=6):
    df = df.sort_values(["month","admin_1","product"]).reset_index(drop=True)
    last_m = df["month"].max().to_period("M").to_timestamp()
    cut_m  = (last_m.to_period("M") - horizon).to_timestamp()
    return df[df["month"] <= cut_m].copy(), df[df["month"] > cut_m].copy()

def fit_xgb_compat(params, X_tr, y_tr, X_val=None, y_val=None, early_rounds=100):
    """Compatible with both old/new XGBoost versions."""
    # Try newer API
    try:
        from xgboost.callback import EarlyStopping
        mdl = XGBRegressor(**params)
        if X_val is not None:
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=early_rounds, save_best=True)])
        else:
            mdl.fit(X_tr, y_tr)
        return mdl
    except (TypeError, ImportError):
        pass
    # Fallback: older API signatures
    try:
        mdl = XGBRegressor(**{k:v for k,v in params.items() if k!="eval_metric"})
        if X_val is not None:
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    eval_metric=params.get("eval_metric","rmse"),
                    early_stopping_rounds=early_rounds)
        else:
            mdl.fit(X_tr, y_tr)
        return mdl
    except TypeError:
        pass
    # Last resort
    mdl = XGBRegressor(**{k:v for k,v in params.items() if k!="eval_metric"})
    mdl.fit(X_tr, y_tr)
    return mdl

# ---- Optuna tuning ----
def tune_xgb(X_train, y_train, n_trials=40, seed=SEED):
    tscv = TimeSeriesSplit(n_splits=5)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist", "random_state": seed, "n_jobs": -1, "eval_metric": "rmse",
        }
        smapes = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_val = y_train[tr_idx],    y_train[va_idx]
            mdl = fit_xgb_compat(params, X_tr, y_tr, X_val, y_val, early_rounds=100)
            # y_* are in log space; evaluate on original scale
            smapes.append(smape(np.expm1(y_val), np.expm1(mdl.predict(X_val))))
            trial.report(float(np.mean(smapes)), step=len(smapes))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(smapes))
    study = optuna.create_study(direction="minimize", study_name="xgb-staples-smape")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial.params.copy()
    best.update({"tree_method": "hist", "random_state": seed, "n_jobs": -1, "eval_metric": "rmse"})
    print("Best sMAPE:", study.best_value)
    print("Best params:", study.best_trial.params)
    return best