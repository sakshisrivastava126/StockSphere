"""
===================================================================================
 HYBRID MODEL BENCHMARKING — Tuned Random Forest vs Prophet
===================================================================================
 Objective : Compare RF (hyperparameter-tuned) against Prophet for demand forecasting
 Dataset   : Walmart retail transactions (walmart.csv)
 Target    : Daily aggregated quantity (units sold)
 Output    : Metrics table, comparison chart, final deployment recommendation
===================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from prophet import Prophet

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
DATA_PATH = DATA_DIR / "walmart.csv"
CHART_PATH = DATA_DIR / "model_comparison_chart.png"
RESULTS_PATH = DATA_DIR / "benchmark_results.json"
RANDOM_STATE = 42
TEST_RATIO = 0.20

# US public holidays (major ones for retail)
US_HOLIDAYS = [
    {"holiday": "new_years",       "ds": "2019-01-01"},
    {"holiday": "mlk_day",         "ds": "2019-01-21"},
    {"holiday": "presidents_day",  "ds": "2019-02-18"},
    {"holiday": "memorial_day",    "ds": "2019-05-27"},
    {"holiday": "independence_day","ds": "2019-07-04"},
    {"holiday": "labor_day",       "ds": "2019-09-02"},
    {"holiday": "thanksgiving",    "ds": "2019-11-28"},
    {"holiday": "black_friday",    "ds": "2019-11-29"},
    {"holiday": "christmas_eve",   "ds": "2019-12-24"},
    {"holiday": "christmas",       "ds": "2019-12-25"},
    {"holiday": "new_years_eve",   "ds": "2019-12-31"},
    # Extend to other years in dataset
    {"holiday": "new_years",       "ds": "2020-01-01"},
    {"holiday": "independence_day","ds": "2020-07-04"},
    {"holiday": "thanksgiving",    "ds": "2020-11-26"},
    {"holiday": "black_friday",    "ds": "2020-11-27"},
    {"holiday": "christmas",       "ds": "2020-12-25"},
    {"holiday": "new_years",       "ds": "2021-01-01"},
    {"holiday": "independence_day","ds": "2021-07-04"},
    {"holiday": "thanksgiving",    "ds": "2021-11-25"},
    {"holiday": "black_friday",    "ds": "2021-11-26"},
    {"holiday": "christmas",       "ds": "2021-12-25"},
    {"holiday": "new_years",       "ds": "2022-01-01"},
    {"holiday": "independence_day","ds": "2022-07-04"},
    {"holiday": "thanksgiving",    "ds": "2022-11-24"},
    {"holiday": "black_friday",    "ds": "2022-11-25"},
    {"holiday": "christmas",       "ds": "2022-12-25"},
    {"holiday": "new_years",       "ds": "2023-01-01"},
    {"holiday": "independence_day","ds": "2023-07-04"},
    {"holiday": "thanksgiving",    "ds": "2023-11-23"},
    {"holiday": "black_friday",    "ds": "2023-11-24"},
    {"holiday": "christmas",       "ds": "2023-12-25"},
]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING & DAILY AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def load_and_aggregate():
    """Load raw data, clean, and aggregate to daily level for time-series modeling."""
    print("\n" + "=" * 72)
    print("  STEP 1: DATA LOADING & DAILY AGGREGATION")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH, encoding_errors="ignore")
    df.drop_duplicates(inplace=True)

    # Fix unit_price
    if df["unit_price"].dtype == object:
        df["unit_price"] = (
            df["unit_price"].astype(str)
            .str.replace("$", "", regex=False).str.strip()
        )
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    # Fix date — explicit format
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df.dropna(subset=["date", "unit_price", "quantity"], inplace=True)
    df["quantity"] = df["quantity"].astype(int)

    # Aggregate to daily level (Prophet and time-series models need this)
    daily = (
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

    print(f"  Raw transactions : {len(df)}")
    print(f"  Daily aggregated : {len(daily)} days")
    print(f"  Date range       : {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"  Avg daily demand : {daily['quantity'].mean():.1f} units")

    return daily


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADVANCED FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, lag, rolling, and holiday features."""
    print("\n" + "=" * 72)
    print("  STEP 2: ADVANCED FEATURE ENGINEERING")
    print("=" * 72)

    df = daily.copy()

    # ── Temporal features ──
    df["month_of_year"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek           # 0=Mon, 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # ── Cyclical encoding (captures wrap-around: Dec→Jan, Sun→Mon) ──
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Holiday flag ──
    holiday_dates = set(pd.to_datetime([h["ds"] for h in US_HOLIDAYS]))
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    # Days adjacent to holidays often see abnormal demand
    df["is_near_holiday"] = (
        df["date"].shift(1).isin(holiday_dates) |
        df["date"].shift(-1).isin(holiday_dates)
    ).astype(int)

    # ── Lag features (autocorrelation / historical momentum) ──
    for lag in [1, 7, 30]:
        df[f"lag_{lag}d"] = df["quantity"].shift(lag)

    # ── Rolling statistics ──
    df["rolling_mean_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).mean()
    df["rolling_std_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["rolling_mean_30d"] = df["quantity"].shift(1).rolling(30, min_periods=1).mean()
    df["rolling_max_7d"] = df["quantity"].shift(1).rolling(7, min_periods=1).max()

    # ── Trend: difference from rolling mean ──
    df["trend_deviation"] = df["quantity"].shift(1) - df["rolling_mean_7d"]

    # Drop warm-up rows (NaN from 30-day lag)
    n_before = len(df)
    df.dropna(subset=["lag_1d", "lag_7d", "lag_30d"], inplace=True)
    n_dropped = n_before - len(df)

    n_features = len([c for c in df.columns if c not in ["date", "quantity"]])
    print(f"  Temporal features  : month_of_year, day_of_week, quarter, cyclical encodings")
    print(f"  Holiday features   : is_holiday, is_near_holiday ({df['is_holiday'].sum()} holiday days)")
    print(f"  Lag features       : 1d, 7d, 30d")
    print(f"  Rolling features   : mean/std/max(7d), mean(30d), trend_deviation")
    print(f"  Warm-up dropped    : {n_dropped} rows → {len(df)} rows remain")
    print(f"  Total features     : {n_features}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHRONOLOGICAL TRAIN/TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def temporal_split(df):
    """Split chronologically; return RF features and Prophet dataframes."""
    print("\n" + "=" * 72)
    print("  STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT")
    print("=" * 72)

    n = len(df)
    split_idx = int(n * (1 - TEST_RATIO))
    cutoff_date = df.iloc[split_idx]["date"]

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # RF features
    feature_cols = [c for c in df.columns if c not in ["date", "quantity"]]
    X_train = train_df[feature_cols]
    y_train = train_df["quantity"]
    X_test = test_df[feature_cols]
    y_test = test_df["quantity"]

    # Prophet format (needs 'ds' and 'y' columns)
    prophet_train = train_df[["date", "quantity"]].rename(
        columns={"date": "ds", "quantity": "y"}
    )
    prophet_test = test_df[["date", "quantity"]].rename(
        columns={"date": "ds", "quantity": "y"}
    )

    print(f"  Cutoff date  : {cutoff_date.date()}")
    print(f"  Train        : {len(train_df)} days ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test         : {len(test_df)} days ({test_df['date'].min().date()} → {test_df['date'].max().date()})")
    print(f"  RF features  : {len(feature_cols)}")

    return (X_train, X_test, y_train, y_test, feature_cols,
            prophet_train, prophet_test, test_df)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — RANDOM FOREST HYPERPARAMETER TUNING
# ═════════════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning with RandomizedSearchCV + TimeSeriesSplit."""
    print("\n" + "=" * 72)
    print("  STEP 4: RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 72)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
        "max_samples": [0.7, 0.8, 0.9, None],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    print("  Param search space:")
    for k, v in param_dist.items():
        print(f"    {k:20s}: {v}")
    print(f"  CV strategy: TimeSeriesSplit (5 folds)")
    print(f"  Running RandomizedSearchCV (60 iterations)...")

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=60,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best = search.best_params_
    print(f"\n  ★ Best hyperparameters found:")
    for k, v in best.items():
        print(f"    {k:20s}: {v}")
    print(f"  Best CV MAE: {-search.best_score_:.4f}")

    return search.best_estimator_, best, -search.best_score_


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — PROPHET MODEL
# ═════════════════════════════════════════════════════════════════════════════

def train_prophet(prophet_train, prophet_test):
    """Fit Prophet with daily/weekly seasonality and holiday effects."""
    print("\n" + "=" * 72)
    print("  STEP 5: PROPHET TIME-SERIES MODEL")
    print("=" * 72)

    # Build holiday dataframe
    holidays_df = pd.DataFrame(US_HOLIDAYS)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1  # 1 day before
    holidays_df["upper_window"] = 1   # 1 day after

    print(f"  Holidays loaded    : {len(holidays_df)} entries")
    print(f"  Seasonality        : daily + weekly (auto) + yearly")
    print(f"  Fitting Prophet...")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holidays_df,
        changepoint_prior_scale=0.1,      # Regularize trend changes
        seasonality_prior_scale=10.0,     # Allow strong seasonality
        holidays_prior_scale=10.0,        # Allow strong holiday effects
        interval_width=0.95,
    )

    model.fit(prophet_train)
    print(f"  Prophet fitted on {len(prophet_train)} training days")

    # Predict on test dates
    future = prophet_test[["ds"]].copy()
    forecast = model.predict(future)

    # Ensure non-negative predictions (can't sell negative items)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    print(f"  Predictions generated for {len(forecast)} test days")

    return model, forecast


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — METRICS & COMPARATIVE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, name):
    """Compute MAE, RMSE, WAPE, and asymmetric business cost."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # WAPE = Weighted Absolute Percentage Error
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    # Business cost: stockout (1.5×) vs overstock (1.0×)
    errors = y_true - y_pred
    stockout_cost = np.sum(np.abs(errors[errors > 0]) * 1.5)
    overstock_cost = np.sum(np.abs(errors[errors < 0]) * 1.0)

    # Holiday performance
    return {
        "model": name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape_pct": round(wape, 2),
        "stockout_cost": round(stockout_cost, 2),
        "overstock_cost": round(overstock_cost, 2),
        "total_biz_cost": round(stockout_cost + overstock_cost, 2),
    }


def comparative_analysis(y_test, rf_pred, prophet_pred, test_df):
    """Side-by-side model comparison with business context."""
    print("\n" + "=" * 72)
    print("  STEP 6: COMPARATIVE ANALYSIS")
    print("=" * 72)

    y_true = y_test.values

    rf_metrics = compute_metrics(y_true, rf_pred, "Tuned Random Forest")
    prophet_metrics = compute_metrics(y_true, prophet_pred, "Prophet")

    # ── Metrics Table ──
    print("\n  ┌─────────────────────────┬────────────────────┬────────────────────┐")
    print("  │ Metric                  │ Tuned Random Forest│ Prophet            │")
    print("  ├─────────────────────────┼────────────────────┼────────────────────┤")
    print(f"  │ MAE                     │ {rf_metrics['mae']:<18} │ {prophet_metrics['mae']:<18} │")
    print(f"  │ RMSE                    │ {rf_metrics['rmse']:<18} │ {prophet_metrics['rmse']:<18} │")
    print(f"  │ WAPE (%)                │ {rf_metrics['wape_pct']:<18} │ {prophet_metrics['wape_pct']:<18} │")
    print(f"  │ Stockout Cost (1.5×)    │ {rf_metrics['stockout_cost']:<18} │ {prophet_metrics['stockout_cost']:<18} │")
    print(f"  │ Overstock Cost (1.0×)   │ {rf_metrics['overstock_cost']:<18} │ {prophet_metrics['overstock_cost']:<18} │")
    print(f"  │ Total Business Cost     │ {rf_metrics['total_biz_cost']:<18} │ {prophet_metrics['total_biz_cost']:<18} │")
    print("  └─────────────────────────┴────────────────────┴────────────────────┘")

    # ── Winner badges ──
    print("\n  Category Winners:")
    metrics_to_compare = [
        ("MAE", "mae", True),
        ("RMSE", "rmse", True),
        ("WAPE", "wape_pct", True),
        ("Business Cost", "total_biz_cost", True),
    ]
    for label, key, lower_is_better in metrics_to_compare:
        rf_val = rf_metrics[key]
        pr_val = prophet_metrics[key]
        if lower_is_better:
            winner = "🌲 RF" if rf_val < pr_val else "🔮 Prophet" if pr_val < rf_val else "🤝 Tie"
        else:
            winner = "🌲 RF" if rf_val > pr_val else "🔮 Prophet" if pr_val > rf_val else "🤝 Tie"
        print(f"    {label:20s} → {winner}")

    # ── Holiday Performance Trap Analysis ──
    print("\n  ── Performance Trap Analysis ──")
    holiday_dates = set(pd.to_datetime([h["ds"] for h in US_HOLIDAYS]))
    test_dates = test_df["date"].values

    holiday_mask = pd.Series(test_dates).apply(lambda d: pd.Timestamp(d) in holiday_dates).values
    non_holiday_mask = ~holiday_mask

    if holiday_mask.sum() > 0:
        rf_holiday_mae = mean_absolute_error(y_true[holiday_mask], rf_pred[holiday_mask])
        pr_holiday_mae = mean_absolute_error(y_true[holiday_mask], prophet_pred[holiday_mask])
        rf_holiday_bias = np.mean(rf_pred[holiday_mask] - y_true[holiday_mask])
        pr_holiday_bias = np.mean(prophet_pred[holiday_mask] - y_true[holiday_mask])

        print(f"    Holiday days in test set    : {holiday_mask.sum()}")
        print(f"    RF  holiday MAE / bias      : {rf_holiday_mae:.2f} / {rf_holiday_bias:+.2f} {'(overshoots)' if rf_holiday_bias > 0 else '(undershoots)'}")
        print(f"    Prophet holiday MAE / bias  : {pr_holiday_mae:.2f} / {pr_holiday_bias:+.2f} {'(overshoots)' if pr_holiday_bias > 0 else '(undershoots)'}")
    else:
        print("    No holidays in test period — holiday trap analysis skipped")

    if non_holiday_mask.sum() > 0:
        rf_normal_mae = mean_absolute_error(y_true[non_holiday_mask], rf_pred[non_holiday_mask])
        pr_normal_mae = mean_absolute_error(y_true[non_holiday_mask], prophet_pred[non_holiday_mask])
        print(f"    RF  normal-day MAE          : {rf_normal_mae:.2f}")
        print(f"    Prophet normal-day MAE      : {pr_normal_mae:.2f}")

    # ── Weekend vs Weekday Trap ──
    weekend_mask = test_df["is_weekend"].values.astype(bool)
    weekday_mask = ~weekend_mask
    if weekend_mask.sum() > 0 and weekday_mask.sum() > 0:
        rf_wend = mean_absolute_error(y_true[weekend_mask], rf_pred[weekend_mask])
        rf_wday = mean_absolute_error(y_true[weekday_mask], rf_pred[weekday_mask])
        pr_wend = mean_absolute_error(y_true[weekend_mask], prophet_pred[weekend_mask])
        pr_wday = mean_absolute_error(y_true[weekday_mask], prophet_pred[weekday_mask])
        print(f"\n    Weekend / Weekday MAE:")
        print(f"      RF     : Weekend {rf_wend:.2f} / Weekday {rf_wday:.2f}")
        print(f"      Prophet: Weekend {pr_wend:.2f} / Weekday {pr_wday:.2f}")

    return rf_metrics, prophet_metrics


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison(test_df, y_test, rf_pred, prophet_pred):
    """Actual vs Predicted for both models in a single chart."""
    print("\n" + "=" * 72)
    print("  STEP 7: VISUALIZATION")
    print("=" * 72)

    dates = test_df["date"].values

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: Actual vs Predicted ──
    ax1 = axes[0]
    ax1.plot(dates, y_test.values, color="#2C3E50", linewidth=1.8,
             label="Actual Demand", alpha=0.9, zorder=3)
    ax1.plot(dates, rf_pred, color="#E74C3C", linewidth=1.2,
             label="Tuned RF Prediction", alpha=0.75, linestyle="--")
    ax1.plot(dates, prophet_pred, color="#3498DB", linewidth=1.2,
             label="Prophet Prediction", alpha=0.75, linestyle="-.")

    # Highlight holidays
    holiday_dates = set(pd.to_datetime([h["ds"] for h in US_HOLIDAYS]))
    for hd in holiday_dates:
        if dates[0] <= np.datetime64(hd) <= dates[-1]:
            ax1.axvline(x=hd, color="#F39C12", alpha=0.3, linewidth=1)

    ax1.set_ylabel("Daily Quantity Sold", fontsize=12, fontweight="bold")
    ax1.set_title("Demand Forecasting: Tuned Random Forest vs Prophet",
                   fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # ── Bottom panel: Residual errors ──
    ax2 = axes[1]
    rf_residuals = y_test.values - rf_pred
    pr_residuals = y_test.values - prophet_pred

    ax2.bar(dates, rf_residuals, color="#E74C3C", alpha=0.5, width=1.0, label="RF Error")
    ax2.bar(dates, pr_residuals, color="#3498DB", alpha=0.5, width=1.0, label="Prophet Error")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_ylabel("Residual\n(+ = stockout)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#FAFAFA")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {CHART_PATH}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — RF FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def print_feature_importance(model, feature_cols, top_n=15):
    """Show top features from the tuned RF."""
    print("\n" + "=" * 72)
    print("  STEP 8: RF FEATURE IMPORTANCE (Top 15)")
    print("=" * 72)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<6}{'Feature':<25}{'Importance':<12}")
    print(f"  {'─' * 6}{'─' * 25}{'─' * 12}")
    for i in range(min(top_n, len(feature_cols))):
        idx = indices[i]
        print(f"  {i + 1:<6}{feature_cols[idx]:<25}{importances[idx]:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — FINAL VERDICT
# ═════════════════════════════════════════════════════════════════════════════

def final_verdict(rf_metrics, prophet_metrics):
    """Deployment recommendation based on business context."""
    print("\n" + "=" * 72)
    print("  STEP 9: FINAL VERDICT & DEPLOYMENT RECOMMENDATION")
    print("=" * 72)

    rf_wins = 0
    pr_wins = 0
    for key in ["mae", "rmse", "wape_pct", "total_biz_cost"]:
        if rf_metrics[key] < prophet_metrics[key]:
            rf_wins += 1
        elif prophet_metrics[key] < rf_metrics[key]:
            pr_wins += 1

    print(f"\n  Scorecard: RF {rf_wins} — Prophet {pr_wins}")

    if rf_wins > pr_wins:
        winner = "Tuned Random Forest"
        reason = "RF"
    elif pr_wins > rf_wins:
        winner = "Prophet"
        reason = "Prophet"
    else:
        # Tie-break by business cost
        if rf_metrics["total_biz_cost"] <= prophet_metrics["total_biz_cost"]:
            winner = "Tuned Random Forest"
            reason = "RF (tie-broken by lower business cost)"
        else:
            winner = "Prophet"
            reason = "Prophet (tie-broken by lower business cost)"

    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  🏆 RECOMMENDED MODEL: {winner:<38}║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  Reasoning:")
    print(f"  ─────────")

    # RF strengths
    print(f"  🌲 Random Forest Strengths:")
    print(f"     • Captures feature interactions (price × margin, lag × season)")
    print(f"     • Handles external features (holidays, price, transactions)")
    print(f"     • Lower latency at inference (no MCMC sampling)")
    print(f"     • More controllable via feature engineering")

    # Prophet strengths
    print(f"  🔮 Prophet Strengths:")
    print(f"     • Native trend decomposition (level + trend + seasonality)")
    print(f"     • Built-in uncertainty intervals for safety stock")
    print(f"     • Handles missing dates gracefully")
    print(f"     • Better for long-horizon (30+ day) forecasts")

    # Performance traps
    print(f"\n  ⚠️  Performance Traps to Watch:")
    print(f"     • RF can overfit to lag features if data regime changes (e.g., COVID)")
    print(f"     • Prophet may overshoot on holidays if historical holiday patterns are noisy")
    print(f"     • RF requires retraining when new stores/categories are added")
    print(f"     • Prophet can underreact to sudden demand spikes (smooth trend assumption)")

    # Hybrid suggestion
    print(f"\n  💡 Production Recommendation:")
    print(f"     Consider a HYBRID approach:")
    print(f"     • Use RF for short-horizon (1-7 day) tactical inventory ordering")
    print(f"     • Use Prophet for long-horizon (30-90 day) strategic planning")
    print(f"     • Ensemble: avg(RF, Prophet) often outperforms either alone")

    return winner


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def save_results(rf_metrics, prophet_metrics, best_params, winner):
    """Save benchmark results to JSON."""
    results = {
        "benchmark_date": datetime.now().isoformat(),
        "winner": winner,
        "rf_best_params": {k: str(v) for k, v in best_params.items()},
        "rf_metrics": rf_metrics,
        "prophet_metrics": prophet_metrics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║   HYBRID MODEL BENCHMARK — Tuned Random Forest vs Prophet              ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # Step 1
    daily = load_and_aggregate()

    # Step 2
    daily = engineer_features(daily)

    # Step 3
    (X_train, X_test, y_train, y_test, feature_cols,
     prophet_train, prophet_test, test_df) = temporal_split(daily)

    # Step 4: Tune RF
    rf_model, best_params, cv_mae = tune_random_forest(X_train, y_train)

    # Step 5: Train Prophet
    prophet_model, forecast = train_prophet(prophet_train, prophet_test)

    # Predictions
    rf_pred = rf_model.predict(X_test)
    prophet_pred = forecast["yhat"].values

    # Step 6: Compare
    rf_metrics, prophet_metrics = comparative_analysis(
        y_test, rf_pred, prophet_pred, test_df
    )

    # Step 7: Visualize
    plot_comparison(test_df, y_test, rf_pred, prophet_pred)

    # Step 8: Feature importance
    print_feature_importance(rf_model, feature_cols)

    # Step 9: Verdict
    winner = final_verdict(rf_metrics, prophet_metrics)

    # Step 10: Save
    save_results(rf_metrics, prophet_metrics, best_params, winner)

    print("\n" + "=" * 72)
    print("  ✅ BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
