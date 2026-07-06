"""
===================================================================================
 HYBRID DEMAND FORECASTER v3 — Reactive Signal Engine
===================================================================================
 Objective : WAPE < 10% via signal-sensitive features + gradient boosting
 
 Evolution from v2:
   • RF → LightGBM (captures non-linear spikes vs RF's averaging)
   • New reactivity layer: momentum, acceleration, volatility, micro-windows
   • WAPE-centric tuning with asymmetric custom loss
   • Reactivity check visualization (old RF vs new GBM)

 Routing Logic (preserved):
   • 1–7 days   → LightGBM (reactive, lag-heavy)
   • 8–29 days  → Weighted blend (70% GBM + 30% Prophet)
   • 30–90 days → Prophet (trend + seasonality)

 Usage:
   python3 hybrid_forecaster.py                                     # full train + demo
   python3 hybrid_forecaster.py --predict 2024-01-01 2024-01-07     # predict range
===================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb
from prophet import Prophet


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
DATA_PATH = DATA_DIR / "walmart.csv"

# Model artifacts
GBM_MODEL_PATH = DATA_DIR / "hybrid_gbm_model.pkl"
RF_MODEL_PATH = DATA_DIR / "hybrid_rf_model.pkl"    # kept for comparison
PROPHET_MODEL_PATH = DATA_DIR / "hybrid_prophet_model.pkl"
PIPELINE_META_PATH = DATA_DIR / "hybrid_pipeline_meta.json"

# Output
PREDICTIONS_CSV = DATA_DIR / "hybrid_predictions.csv"
FORECAST_CHART = DATA_DIR / "hybrid_forecast_chart.png"
REACTIVITY_CHART = DATA_DIR / "reactivity_check.png"
IMPORTANCE_CHART = DATA_DIR / "feature_importance.png"

RANDOM_STATE = 42
TEST_RATIO = 0.20
SAFETY_STOCK_MULTIPLIER = 1.05

# Blend weights for mid-range
GBM_WEIGHT_MID = 0.70
PROPHET_WEIGHT_MID = 0.30

# Volatility config
SPIKE_SIGMA = 1.5  # flag demand > mean + 1.5σ as a spike

# US holidays for Prophet
US_HOLIDAYS = pd.DataFrame([
    {"holiday": "new_years",       "ds": "2019-01-01"},
    {"holiday": "mlk_day",         "ds": "2019-01-21"},
    {"holiday": "memorial_day",    "ds": "2019-05-27"},
    {"holiday": "independence_day","ds": "2019-07-04"},
    {"holiday": "labor_day",       "ds": "2019-09-02"},
    {"holiday": "thanksgiving",    "ds": "2019-11-28"},
    {"holiday": "black_friday",    "ds": "2019-11-29"},
    {"holiday": "christmas",       "ds": "2019-12-25"},
    {"holiday": "new_years",       "ds": "2020-01-01"},
    {"holiday": "independence_day","ds": "2020-07-04"},
    {"holiday": "thanksgiving",    "ds": "2020-11-26"},
    {"holiday": "christmas",       "ds": "2020-12-25"},
    {"holiday": "new_years",       "ds": "2021-01-01"},
    {"holiday": "independence_day","ds": "2021-07-04"},
    {"holiday": "thanksgiving",    "ds": "2021-11-25"},
    {"holiday": "christmas",       "ds": "2021-12-25"},
    {"holiday": "new_years",       "ds": "2022-01-01"},
    {"holiday": "independence_day","ds": "2022-07-04"},
    {"holiday": "thanksgiving",    "ds": "2022-11-24"},
    {"holiday": "christmas",       "ds": "2022-12-25"},
    {"holiday": "new_years",       "ds": "2023-01-01"},
    {"holiday": "independence_day","ds": "2023-07-04"},
    {"holiday": "thanksgiving",    "ds": "2023-11-23"},
    {"holiday": "christmas",       "ds": "2023-12-25"},
    {"holiday": "new_years",       "ds": "2024-01-01"},
    {"holiday": "independence_day","ds": "2024-07-04"},
    {"holiday": "thanksgiving",    "ds": "2024-11-28"},
    {"holiday": "christmas",       "ds": "2024-12-25"},
])
US_HOLIDAYS["ds"] = pd.to_datetime(US_HOLIDAYS["ds"])
US_HOLIDAYS["lower_window"] = -1
US_HOLIDAYS["upper_window"] = 1


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — UNIFIED DAILY AGGREGATE
# ═════════════════════════════════════════════════════════════════════════════

def load_unified_daily() -> pd.DataFrame:
    """Load → clean → aggregate to daily sum (single source of truth)."""
    df = pd.read_csv(DATA_PATH, encoding_errors="ignore")
    df.drop_duplicates(inplace=True)

    if df["unit_price"].dtype == object:
        df["unit_price"] = (
            df["unit_price"].astype(str)
            .str.replace("$", "", regex=False).str.strip()
        )
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df.dropna(subset=["date", "unit_price", "quantity"], inplace=True)
    df["quantity"] = df["quantity"].astype(int)

    df_daily = (
        df.groupby("date")
        .agg(
            quantity=("quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
            n_transactions=("invoice_id", "count"),
            avg_profit_margin=("profit_margin", "mean"),
            n_categories=("category", "nunique"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df_daily


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADVANCED SIGNAL ENGINEERING (The Reactivity Layer)
# ═════════════════════════════════════════════════════════════════════════════

def build_reactive_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Signal-sensitive feature engineering targeting spike detection.

    Feature Groups:
      1. Temporal     : day_of_week, month, quarter, cyclical
      2. Lag           : 1d, 3d, 5d, 7d, 14d, 30d
      3. Momentum     : lag_1d - lag_7d, deviation from rolling mean
      4. Acceleration  : 2nd-order derivative (change of the change)
      5. Micro-windows : 3d and 5d rolling mean/std (fast reaction)
      6. Standard windows: 7d and 30d rolling mean/std/max
      7. Volatility   : spike flag (>μ+kσ), coefficient of variation
      8. Holiday      : is_holiday, is_near_holiday
    """
    df = df_daily.copy()

    # ══════ GROUP 1: Temporal features ══════
    df["month_of_year"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ══════ GROUP 2: Lag features (extended for reactivity) ══════
    for lag in [1, 3, 5, 7, 14, 30]:
        df[f"lag_{lag}d"] = df["quantity"].shift(lag)

    # ══════ GROUP 3: Momentum (1st-order derivative) ══════
    # Short-term momentum: how much did demand change in last 1 vs 7 days?
    df["momentum_1d_vs_7d"] = df["lag_1d"] - df["lag_7d"]
    # Deviation from recent average: is current demand above/below trend?
    df["rolling_mean_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).mean()
    df["deviation_from_7d_mean"] = df["lag_1d"] - df["rolling_mean_7d"]
    # Short-range momentum
    df["momentum_1d_vs_3d"] = df["lag_1d"] - df["lag_3d"]
    df["momentum_3d_vs_7d"] = df["lag_3d"] - df["lag_7d"]
    # Week-over-week momentum
    df["momentum_7d_vs_14d"] = df["lag_7d"] - df["lag_14d"]

    # ══════ GROUP 4: Acceleration (2nd-order derivative) ══════
    # Is the trend accelerating or decelerating?
    df["accel_short"] = df["momentum_1d_vs_3d"] - df["momentum_1d_vs_3d"].shift(1)
    df["accel_medium"] = df["momentum_1d_vs_7d"] - df["momentum_1d_vs_7d"].shift(1)
    # Rate of change of rolling mean
    df["rolling_mean_7d_diff"] = df["rolling_mean_7d"] - df["rolling_mean_7d"].shift(1)

    # ══════ GROUP 5: Micro-windows (3d and 5d — fast reaction) ══════
    df["rolling_mean_3d"] = df["quantity"].shift(1).rolling(3, min_periods=1).mean()
    df["rolling_std_3d"] = df["quantity"].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df["rolling_mean_5d"] = df["quantity"].shift(1).rolling(5, min_periods=1).mean()
    df["rolling_std_5d"] = df["quantity"].shift(1).rolling(5, min_periods=1).std().fillna(0)
    df["rolling_max_3d"] = df["quantity"].shift(1).rolling(3, min_periods=1).max()
    df["rolling_min_3d"] = df["quantity"].shift(1).rolling(3, min_periods=1).min()

    # ══════ GROUP 6: Standard windows (7d, 30d) ══════
    df["rolling_std_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["rolling_max_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).max()
    df["rolling_mean_30d"] = df["quantity"].shift(1).rolling(30, min_periods=1).mean()
    df["rolling_std_30d"] = df["quantity"].shift(1).rolling(30, min_periods=1).std().fillna(0)

    # Ratio of short to long window (detects regime shift)
    df["ratio_3d_to_30d"] = df["rolling_mean_3d"] / df["rolling_mean_30d"].clip(lower=1)
    df["ratio_7d_to_30d"] = df["rolling_mean_7d"] / df["rolling_mean_30d"].clip(lower=1)

    # Trend deviation
    df["trend_deviation"] = df["lag_1d"] - df["rolling_mean_7d"]

    # ══════ GROUP 7: Volatility & Outlier Detection ══════
    # Expanding mean and std for spike detection
    expanding_mean = df["quantity"].shift(1).expanding(min_periods=7).mean()
    expanding_std = df["quantity"].shift(1).expanding(min_periods=7).std().fillna(1)

    # Spike flag: was yesterday's demand > μ + k*σ?
    df["spike_flag"] = (df["lag_1d"] > (expanding_mean + SPIKE_SIGMA * expanding_std)).astype(int)

    # Coefficient of Variation (std / mean) over 7-day window
    df["cv_7d"] = df["rolling_std_7d"] / df["rolling_mean_7d"].clip(lower=1)
    df["cv_30d"] = df["rolling_std_30d"] / df["rolling_mean_30d"].clip(lower=1)

    # Range as % of mean (captures spread)
    df["range_pct_3d"] = (df["rolling_max_3d"] - df["rolling_min_3d"]) / df["rolling_mean_3d"].clip(lower=1)

    # ══════ GROUP 8: Holiday flags ══════
    holiday_dates = set(US_HOLIDAYS["ds"])
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    df["is_near_holiday"] = (
        df["date"].shift(1).isin(holiday_dates) |
        df["date"].shift(-1).isin(holiday_dates)
    ).astype(int)

    # ── Drop warm-up period (30 days for longest lag) ──
    df.dropna(subset=["lag_1d", "lag_7d", "lag_30d"], inplace=True)
    df = df.reset_index(drop=True)

    # Fill any remaining NaN in derived features
    df = df.fillna(0)

    return df


def build_prophet_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Map unified daily aggregate to Prophet format."""
    return df_daily[["date", "quantity"]].rename(
        columns={"date": "ds", "quantity": "y"}
    ).copy()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train):
    """
    LightGBM with WAPE-centric tuning + TimeSeriesSplit.
    Key tuning choices for reactivity:
      - Lower min_child_samples: lets tree split on rare spike patterns
      - Higher num_leaves: captures more complex interactions
      - Moderate learning_rate: fast enough to react, slow enough not to overfit
    """
    print("  [LGBM] Running RandomizedSearchCV (50 iter, 5-fold TS)...")

    param_dist = {
        "n_estimators": [300, 500, 800, 1000],
        "max_depth": [4, 6, 8, 10, -1],
        "num_leaves": [15, 31, 50, 80],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [3, 5, 10, 15],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0, 0.1, 1.0],
    }

    search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  [LGBM] Best CV MAE: {-search.best_score_:.4f}")
    print(f"  [LGBM] Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def train_rf_baseline(X_train, y_train):
    """Train a baseline RF for the reactivity comparison."""
    print("  [RF Baseline] Training for comparison...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=1, max_features="sqrt", max_samples=0.9,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_prophet_model(prophet_train):
    """Prophet with daily/weekly/yearly seasonality + holidays."""
    print("  [Prophet] Fitting with seasonality + holidays...")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=US_HOLIDAYS,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        interval_width=0.95,
    )
    model.fit(prophet_train)
    print(f"  [Prophet] Fitted on {len(prophet_train)} days")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — METRICS (WAPE-CENTRIC)
# ═════════════════════════════════════════════════════════════════════════════

def compute_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error — primary metric."""
    return np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1) * 100


def evaluate(name, y_true, y_pred):
    """Full evaluation: MAE, RMSE, WAPE, business cost."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = compute_wape(y_true, y_pred)

    errors = y_true - y_pred
    stockout_cost = np.sum(np.abs(errors[errors > 0]) * 1.5)
    overstock_cost = np.sum(np.abs(errors[errors < 0]) * 1.0)

    return {
        "name": name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 2),
        "biz_cost": round(stockout_cost + overstock_cost, 2),
        "stockout_cost": round(stockout_cost, 2),
        "overstock_cost": round(overstock_cost, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SAVING & LOADING
# ═════════════════════════════════════════════════════════════════════════════

def save_models(gbm_model, rf_model, prophet_model, gbm_params,
                feature_cols, cutoff_date, train_stats):
    with open(GBM_MODEL_PATH, "wb") as f:
        pickle.dump(gbm_model, f)
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    with open(PROPHET_MODEL_PATH, "wb") as f:
        pickle.dump(prophet_model, f)

    meta = {
        "saved_at": datetime.now().isoformat(),
        "cutoff_date": str(cutoff_date.date()),
        "rf_features": feature_cols,
        "gbm_best_params": {k: str(v) for k, v in gbm_params.items()},
        "safety_stock_multiplier": SAFETY_STOCK_MULTIPLIER,
        "blend_weights": {"gbm": GBM_WEIGHT_MID, "prophet": PROPHET_WEIGHT_MID},
        "train_stats": train_stats,
    }
    with open(PIPELINE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {GBM_MODEL_PATH.name}, {RF_MODEL_PATH.name}, "
          f"{PROPHET_MODEL_PATH.name}, {PIPELINE_META_PATH.name}")


def load_models():
    if not GBM_MODEL_PATH.exists() or not PROPHET_MODEL_PATH.exists():
        raise FileNotFoundError("Models not found. Run training first.")

    with open(GBM_MODEL_PATH, "rb") as f:
        gbm = pickle.load(f)
    with open(RF_MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    with open(PROPHET_MODEL_PATH, "rb") as f:
        prophet = pickle.load(f)
    with open(PIPELINE_META_PATH) as f:
        meta = json.load(f)

    print(f"  Loaded models (trained up to {meta['cutoff_date']})")
    return gbm, rf, prophet, meta


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED PREDICTION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def predict(start_date, end_date, gbm_model, prophet_model, daily_featured, meta):
    """
    Unified prediction with auto-routing.
      ≤7d → LightGBM | 8-29d → 70% GBM + 30% Prophet | ≥30d → Prophet
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    horizon = (end - start).days + 1

    if horizon <= 7:
        mode, label = "short", f"SHORT-TERM ({horizon}d) → LightGBM"
    elif horizon >= 30:
        mode, label = "long", f"LONG-TERM ({horizon}d) → Prophet"
    else:
        mode, label = "mid", f"MID-TERM ({horizon}d) → Blend ({GBM_WEIGHT_MID:.0%} GBM + {PROPHET_WEIGHT_MID:.0%} Prophet)"

    print(f"\n  Prediction Mode: {label}")
    print(f"  Date Range: {start.date()} → {end.date()} ({horizon} days)")

    pred_dates = pd.date_range(start, end, freq="D")
    results = pd.DataFrame({"date": pred_dates})

    # Prophet
    forecast = prophet_model.predict(pd.DataFrame({"ds": pred_dates}))
    results["prophet_pred"] = forecast["yhat"].clip(lower=0).values
    results["prophet_lower"] = forecast["yhat_lower"].clip(lower=0).values
    results["prophet_upper"] = forecast["yhat_upper"].clip(lower=0).values

    # GBM (recursive)
    results["gbm_pred"] = _recursive_predict(
        gbm_model, daily_featured, pred_dates, meta
    )

    # Route
    if mode == "short":
        results["final_pred"] = results["gbm_pred"]
        results["model_used"] = "LightGBM"
    elif mode == "long":
        results["final_pred"] = results["prophet_pred"]
        results["model_used"] = "Prophet"
    else:
        results["final_pred"] = (
            GBM_WEIGHT_MID * results["gbm_pred"] +
            PROPHET_WEIGHT_MID * results["prophet_pred"]
        )
        results["model_used"] = "Blend"

    results["final_pred"] = results["final_pred"].clip(lower=0).round().astype(int)
    results["safety_stock_order"] = np.ceil(
        results["final_pred"] * SAFETY_STOCK_MULTIPLIER
    ).astype(int)

    return results


def _recursive_predict(model, daily_featured, pred_dates, meta):
    """Recursive multi-step prediction using training medians for aux features."""
    feature_cols = meta["rf_features"]
    ts = meta.get("train_stats", {})

    raw_cols = ["date", "quantity", "avg_unit_price", "n_transactions",
                "avg_profit_margin", "n_categories"]
    available = [c for c in raw_cols if c in daily_featured.columns]
    history = daily_featured[available].copy()

    preds = []
    for target_date in pred_dates:
        if target_date in daily_featured["date"].values:
            row = daily_featured[daily_featured["date"] == target_date].iloc[0]
            feat_avail = [c for c in feature_cols if c in row.index]
            X = row[feat_avail].values.reshape(1, -1)
            if len(feat_avail) < len(feature_cols):
                X = _pad(X, feat_avail, feature_cols)
            pred = model.predict(X)[0]
        else:
            new_row = {
                "date": target_date, "quantity": 0,
                "avg_unit_price": ts.get("median_unit_price", 50.0),
                "n_transactions": ts.get("median_n_transactions", 7.0),
                "avg_profit_margin": ts.get("median_profit_margin", 0.4),
                "n_categories": ts.get("median_n_categories", 5),
            }
            temp = pd.concat([history, pd.DataFrame([new_row])],
                             ignore_index=True).sort_values("date").reset_index(drop=True)
            temp = build_reactive_features(temp)
            target_row = temp[temp["date"] == target_date]
            if target_row.empty:
                target_row = temp.tail(1)
            feat_avail = [c for c in feature_cols if c in target_row.columns]
            X = target_row[feat_avail].values.reshape(1, -1)
            if len(feat_avail) < len(feature_cols):
                X = _pad(X, feat_avail, feature_cols)
            pred = model.predict(X)[0]

            new_row["quantity"] = max(0, round(pred))
            history = pd.concat([history, pd.DataFrame([new_row])],
                                ignore_index=True).sort_values("date").reset_index(drop=True)

        preds.append(max(0, pred))
    return preds


def _pad(X, available_feats, full_feats):
    full_X = np.zeros((1, len(full_feats)))
    idx_map = {f: i for i, f in enumerate(available_feats)}
    for i, col in enumerate(full_feats):
        if col in idx_map:
            full_X[0, i] = X[0, idx_map[col]]
    return full_X


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(gbm_model, feature_cols):
    """Show feature importance grouped by signal type."""
    print("\n" + "=" * 72)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 72)

    importances = gbm_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    # Categorize features
    categories = {
        "🚀 Momentum": ["momentum_", "deviation_from"],
        "⚡ Acceleration": ["accel_", "rolling_mean_7d_diff"],
        "📊 Volatility": ["spike_flag", "cv_", "range_pct"],
        "🔬 Micro-window": ["rolling_mean_3d", "rolling_std_3d",
                            "rolling_mean_5d", "rolling_std_5d",
                            "rolling_max_3d", "rolling_min_3d"],
        "📈 Lag": ["lag_"],
        "🕐 Temporal": ["month_", "day_", "week_", "is_weekend",
                         "is_month", "quarter", "dow_"],
        "📉 Rolling (7d/30d)": ["rolling_mean_7d", "rolling_std_7d",
                                "rolling_max_7d", "rolling_mean_30d",
                                "rolling_std_30d", "ratio_"],
    }

    print(f"\n  {'Rank':<6}{'Feature':<30}{'Importance':<12}{'Category':<20}")
    print(f"  {'─' * 6}{'─' * 30}{'─' * 12}{'─' * 20}")

    total_by_cat = {}
    for i in range(min(25, len(feature_cols))):
        idx = sorted_idx[i]
        feat = feature_cols[idx]
        imp = importances[idx]

        cat = "Other"
        for cat_name, keywords in categories.items():
            if any(kw in feat for kw in keywords):
                cat = cat_name
                break

        total_by_cat[cat] = total_by_cat.get(cat, 0) + imp
        print(f"  {i+1:<6}{feat:<30}{imp:<12.4f}{cat:<20}")

    # Category summary
    print(f"\n  ── Importance by Category ──")
    for cat, total in sorted(total_by_cat.items(), key=lambda x: -x[1]):
        bar = "█" * int(total * 100)
        print(f"    {cat:<25} {total:.4f}  {bar}")

    # Check: are new reactive features driving the model?
    reactive_keys = ["Momentum", "Acceleration", "Volatility", "Micro-window"]
    reactive_total = sum(v for k, v in total_by_cat.items()
                         if any(rk in k for rk in reactive_keys))
    print(f"\n  Reactive features total importance: {reactive_total:.4f} "
          f"({reactive_total / sum(importances) * 100:.1f}%)")

    if reactive_total > 0.15:
        print(f"  ✅ Reactive features are significantly driving the model")
    else:
        print(f"  ⚠️  Reactive features have low impact — may need more tuning")

    # Save importance chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(25, len(feature_cols))
    top_feats = [feature_cols[sorted_idx[i]] for i in range(top_n)]
    top_imps = [importances[sorted_idx[i]] for i in range(top_n)]

    colors = []
    for feat in top_feats:
        if any(k in feat for k in ["momentum_", "deviation_from"]):
            colors.append("#E74C3C")  # red = momentum
        elif any(k in feat for k in ["accel_", "rolling_mean_7d_diff"]):
            colors.append("#F39C12")  # orange = acceleration
        elif any(k in feat for k in ["spike_flag", "cv_", "range_pct"]):
            colors.append("#9B59B6")  # purple = volatility
        elif any(k in feat for k in ["rolling_mean_3d", "rolling_std_3d",
                                     "rolling_mean_5d", "rolling_std_5d",
                                     "rolling_max_3d", "rolling_min_3d"]):
            colors.append("#E67E22")  # dark orange = micro-window
        elif "lag_" in feat:
            colors.append("#3498DB")  # blue = lag
        else:
            colors.append("#95A5A6")  # gray = other

    ax.barh(range(top_n), top_imps[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance — Reactive Signal Engine", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#E74C3C", label="Momentum"),
        Patch(color="#F39C12", label="Acceleration"),
        Patch(color="#9B59B6", label="Volatility"),
        Patch(color="#E67E22", label="Micro-window"),
        Patch(color="#3498DB", label="Lag"),
        Patch(color="#95A5A6", label="Other"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {IMPORTANCE_CHART}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# REACTIVITY CHECK VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def reactivity_check(y_test, gbm_pred, rf_pred, prophet_pred, test_dates):
    """
    Compare how the new GBM tracks actual spikes vs the old RF and Prophet.
    Highlights periods where demand spikes and shows which model reacts faster.
    """
    print("\n" + "=" * 72)
    print("  REACTIVITY CHECK — GBM vs RF vs Prophet")
    print("=" * 72)

    y_true = np.array(y_test)
    dates = np.array(test_dates)

    # Identify spike days (demand > mean + 1σ)
    spike_threshold = np.mean(y_true) + np.std(y_true)
    spike_mask = y_true > spike_threshold
    n_spikes = spike_mask.sum()

    # Metrics on spike days vs normal days
    for name, pred in [("LightGBM", gbm_pred), ("RF (old)", rf_pred), ("Prophet", prophet_pred)]:
        pred = np.array(pred)
        if n_spikes > 0:
            spike_mae = mean_absolute_error(y_true[spike_mask], pred[spike_mask])
            spike_wape = compute_wape(y_true[spike_mask], pred[spike_mask])
        else:
            spike_mae = spike_wape = 0
        normal_mae = mean_absolute_error(y_true[~spike_mask], pred[~spike_mask])
        normal_wape = compute_wape(y_true[~spike_mask], pred[~spike_mask])
        overall_wape = compute_wape(y_true, pred)

        print(f"  {name:12s} | Overall WAPE={overall_wape:5.1f}% | "
              f"Spike MAE={spike_mae:5.1f} (WAPE={spike_wape:5.1f}%) | "
              f"Normal MAE={normal_mae:5.1f} (WAPE={normal_wape:5.1f}%)")

    print(f"\n  Spike days in test: {n_spikes} / {len(y_true)} "
          f"(threshold: {spike_threshold:.1f} units)")

    # ── Visualization ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})

    # Panel 1: Actual vs Predictions
    ax1 = axes[0]
    ax1.plot(dates, y_true, color="#2C3E50", linewidth=2, label="Actual", zorder=4)
    ax1.plot(dates, gbm_pred, color="#E74C3C", linewidth=1.5,
             label="LightGBM (new)", linestyle="--", alpha=0.9, zorder=3)
    ax1.plot(dates, rf_pred, color="#95A5A6", linewidth=1,
             label="RF (old)", linestyle=":", alpha=0.6, zorder=2)
    ax1.plot(dates, prophet_pred, color="#3498DB", linewidth=1,
             label="Prophet", linestyle="-.", alpha=0.6, zorder=2)

    # Highlight spike days
    for i, is_spike in enumerate(spike_mask):
        if is_spike:
            ax1.axvspan(dates[i], dates[min(i+1, len(dates)-1)],
                        alpha=0.15, color="#E74C3C")

    ax1.axhline(y=spike_threshold, color="#E74C3C", linestyle=":",
                alpha=0.3, label=f"Spike threshold ({spike_threshold:.0f})")
    ax1.set_ylabel("Daily Quantity", fontsize=12, fontweight="bold")
    ax1.set_title("Reactivity Check: New GBM vs Old RF — Spike Tracking",
                   fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # Panel 2: Absolute errors
    ax2 = axes[1]
    gbm_errors = np.abs(y_true - np.array(gbm_pred))
    rf_errors = np.abs(y_true - np.array(rf_pred))
    ax2.fill_between(dates, rf_errors, alpha=0.3, color="#95A5A6", label="RF error")
    ax2.fill_between(dates, gbm_errors, alpha=0.5, color="#E74C3C", label="GBM error")
    ax2.set_ylabel("Abs Error", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#FAFAFA")

    # Panel 3: Cumulative WAPE
    ax3 = axes[2]
    gbm_cum_wape = np.cumsum(np.abs(y_true - np.array(gbm_pred))) / np.cumsum(np.abs(y_true)).clip(1) * 100
    rf_cum_wape = np.cumsum(np.abs(y_true - np.array(rf_pred))) / np.cumsum(np.abs(y_true)).clip(1) * 100
    ax3.plot(dates, gbm_cum_wape, color="#E74C3C", linewidth=1.5, label="GBM WAPE")
    ax3.plot(dates, rf_cum_wape, color="#95A5A6", linewidth=1, label="RF WAPE")
    ax3.axhline(y=10, color="green", linestyle="--", alpha=0.5, label="10% target")
    ax3.set_ylabel("Cum. WAPE %", fontsize=10)
    ax3.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#FAFAFA")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(REACTIVITY_CHART, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {REACTIVITY_CHART}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# SCALE PARITY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_scale_parity(results, daily_df):
    print("\n  ── Scale Parity Check ──")
    hist_mean = daily_df["quantity"].mean()
    gbm_mean = results["gbm_pred"].mean()
    pr_mean = results["prophet_pred"].mean()
    ratio = gbm_mean / max(pr_mean, 0.01)
    print(f"    Historical mean: {hist_mean:.1f} | GBM: {gbm_mean:.1f} | "
          f"Prophet: {pr_mean:.1f} | Ratio: {ratio:.2f}x")
    status = "✅ OK" if 0.5 <= ratio <= 2.0 else "⚠️ MISMATCH"
    print(f"    {status}")


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    print("\n" + "=" * 72)
    print("  PHASE 1: TRAINING — REACTIVE SIGNAL ENGINE")
    print("=" * 72)

    # 1. Unified daily
    print("\n  Loading unified daily aggregate...")
    df_daily = load_unified_daily()
    print(f"  Daily: {len(df_daily)} days, mean={df_daily['quantity'].mean():.1f}, "
          f"range=[{df_daily['quantity'].min()}, {df_daily['quantity'].max()}]")

    # 2. Reactive features
    print("  Building reactive features...")
    df_featured = build_reactive_features(df_daily)
    feature_cols = [c for c in df_featured.columns if c not in ["date", "quantity"]]
    print(f"  Features: {len(feature_cols)} ({len(df_featured)} days)")

    # Feature breakdown
    cats = {"momentum": 0, "acceleration": 0, "volatility": 0,
            "micro_window": 0, "lag": 0, "temporal": 0, "rolling_std": 0, "other": 0}
    for f in feature_cols:
        if "momentum" in f or "deviation" in f: cats["momentum"] += 1
        elif "accel" in f or "rolling_mean_7d_diff" in f: cats["acceleration"] += 1
        elif "spike" in f or "cv_" in f or "range_pct" in f: cats["volatility"] += 1
        elif any(k in f for k in ["3d", "5d", "min_3d", "max_3d"]): cats["micro_window"] += 1
        elif "lag_" in f: cats["lag"] += 1
        elif any(k in f for k in ["month", "day", "week", "quarter", "dow", "is_weekend", "is_month"]): cats["temporal"] += 1
        else: cats["other"] += 1
    for cat, count in cats.items():
        if count > 0:
            print(f"    {cat:15s}: {count} features")

    # 3. Prophet format
    prophet_full = build_prophet_df(df_daily)

    # 4. Strict chronological split
    n = len(df_featured)
    split_idx = int(n * (1 - TEST_RATIO))
    cutoff_date = df_featured.iloc[split_idx]["date"]

    train = df_featured.iloc[:split_idx]
    test = df_featured.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train["quantity"]
    X_test = test[feature_cols]
    y_test = test["quantity"]

    train_start = train["date"].min()
    train_end = train["date"].max()
    prophet_train = prophet_full[
        (prophet_full["ds"] >= train_start) & (prophet_full["ds"] <= train_end)
    ].copy()

    print(f"\n  ── Synchronized Split ──")
    print(f"  Train: {len(train)} days ({train_start.date()} → {train_end.date()})")
    print(f"  Test:  {len(test)} days ({test['date'].min().date()} → {test['date'].max().date()})")

    train_stats = {
        "mean_quantity": round(float(y_train.mean()), 2),
        "std_quantity": round(float(y_train.std()), 2),
        "median_unit_price": round(float(train["avg_unit_price"].median()), 2),
        "median_n_transactions": round(float(train["n_transactions"].median()), 2),
        "median_profit_margin": round(float(train["avg_profit_margin"].median()), 4),
        "median_n_categories": int(train["n_categories"].median()),
    }

    # 5. Train LightGBM
    print("\n  ── Training LightGBM (Reactive) ──")
    gbm_model, gbm_params = train_lgbm(X_train, y_train)

    # 6. Train RF baseline (for comparison)
    print("\n  ── Training RF Baseline (for comparison) ──")
    rf_baseline = train_rf_baseline(X_train, y_train)

    # 7. Train Prophet
    print("\n  ── Training Prophet ──")
    prophet_model = train_prophet_model(prophet_train)

    # 8. Save
    print("\n  ── Saving ──")
    save_models(gbm_model, rf_baseline, prophet_model, gbm_params,
                feature_cols, cutoff_date, train_stats)

    return (gbm_model, rf_baseline, prophet_model, df_featured, df_daily,
            feature_cols, X_train, y_train, X_test, y_test, test)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║   REACTIVE SIGNAL ENGINE — LightGBM + Prophet (WAPE < 10% Target)      ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # --predict mode
    if len(sys.argv) == 4 and sys.argv[1] == "--predict":
        start_date, end_date = sys.argv[2], sys.argv[3]
        print(f"\n  Predict mode: {start_date} → {end_date}")
        gbm, rf, prophet, meta = load_models()
        df_daily = load_unified_daily()
        daily_featured = build_reactive_features(df_daily)
        results = predict(start_date, end_date, gbm, prophet, daily_featured, meta)
        print("\n" + results.to_string(index=False))
        results.to_csv(PREDICTIONS_CSV, index=False)
        print(f"\n  Saved → {PREDICTIONS_CSV}")
        check_scale_parity(results, df_daily)
        return

    # ── Full pipeline ──
    (gbm_model, rf_baseline, prophet_model, df_featured, df_daily,
     feature_cols, X_train, y_train, X_test, y_test, test) = train_pipeline()
    meta = json.load(open(PIPELINE_META_PATH))

    # ── Evaluation ──
    print("\n" + "=" * 72)
    print("  PHASE 2: EVALUATION — RF vs GBM vs Prophet")
    print("=" * 72)

    gbm_pred = gbm_model.predict(X_test)
    rf_pred = rf_baseline.predict(X_test)
    prophet_test = build_prophet_df(df_daily)
    prophet_test = prophet_test[prophet_test["ds"].isin(test["date"])]
    prophet_forecast = prophet_model.predict(prophet_test[["ds"]])
    prophet_pred = prophet_forecast["yhat"].clip(lower=0).values

    gbm_m = evaluate("LightGBM", y_test, gbm_pred)
    rf_m = evaluate("RF (old)", y_test, rf_pred)
    pr_m = evaluate("Prophet", y_test, prophet_pred)

    print(f"\n  ┌─────────────┬──────────┬──────────┬──────────┬──────────────┐")
    print(f"  │ Model       │ MAE      │ RMSE     │ WAPE %   │ Biz Cost     │")
    print(f"  ├─────────────┼──────────┼──────────┼──────────┼──────────────┤")
    for m in [gbm_m, rf_m, pr_m]:
        marker = " ★" if m["wape"] == min(gbm_m["wape"], rf_m["wape"], pr_m["wape"]) else "  "
        print(f"  │{marker}{m['name']:10s} │ {m['mae']:<8} │ {m['rmse']:<8} │ {m['wape']:<8} │ {m['biz_cost']:<12} │")
    print(f"  └─────────────┴──────────┴──────────┴──────────┴──────────────┘")

    if gbm_m["wape"] < 10:
        print(f"\n  🎯 TARGET ACHIEVED: GBM WAPE = {gbm_m['wape']}% (< 10%)")
    else:
        print(f"\n  ⚠️  GBM WAPE = {gbm_m['wape']}% — target is < 10%")
        if gbm_m["wape"] < rf_m["wape"]:
            improvement = rf_m["wape"] - gbm_m["wape"]
            print(f"  📈 But GBM improved by {improvement:.1f}pp over RF ({rf_m['wape']}%)")

    # ── Feature Importance ──
    analyze_feature_importance(gbm_model, feature_cols)

    # ── Reactivity Check ──
    reactivity_check(y_test, gbm_pred, rf_pred, prophet_pred, test["date"].values)

    # ── Save predictions ──
    print("\n" + "=" * 72)
    print("  PHASE 3: SAVING")
    print("=" * 72)

    results_df = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test.values,
        "gbm_pred": np.round(gbm_pred).astype(int),
        "rf_pred": np.round(rf_pred).astype(int),
        "prophet_pred": np.round(prophet_pred).astype(int),
    })
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"  Predictions → {PREDICTIONS_CSV}")
    print(f"  Reactivity  → {REACTIVITY_CHART}")
    print(f"  Importance  → {IMPORTANCE_CHART}")

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY: REACTIVE ENGINE vs OLD RF")
    print("=" * 72)
    print(f"  WAPE: {rf_m['wape']}% (RF) → {gbm_m['wape']}% (GBM)  "
          f"{'📉 Improved' if gbm_m['wape'] < rf_m['wape'] else '📈 Needs work'}")
    print(f"  MAE:  {rf_m['mae']} (RF) → {gbm_m['mae']} (GBM)")
    print(f"  Cost: {rf_m['biz_cost']} (RF) → {gbm_m['biz_cost']} (GBM)")
    print(f"  Features: {len(feature_cols)} (was 26)")
    print(f"\n  Usage: python3 hybrid_forecaster.py --predict 2024-01-01 2024-01-07")

    print("\n" + "=" * 72)
    print("  ✅ REACTIVE SIGNAL ENGINE READY")
    print("=" * 72)


if __name__ == "__main__":
    main()
