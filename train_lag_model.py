"""
train_lag_model.py
------------------
Trains a LightGBM model for Unique Purchased Cards with:
  1. Holiday features  (IsHoliday, IsHolidayWeekend, DaysToNearestHoliday)
  2. Lag / rolling features  (Lag7, Lag14, Roll7Mean, Roll14Mean)
  3. Post-prediction bias calibration via Isotonic Regression

Usage:
    python train_lag_model.py

Produces:
    lgbm_cards_lag_model.txt        (LightGBM native text format)
    lgbm_cards_lag_calibrator.joblib (Isotonic Regression calibrator)
    lgbm_cards_lag_history.parquet   (last 28 days of sales for inference)
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import holidays as hol_lib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE       = os.path.join(BASE_DIR, "merged_data_model.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "lgbm_cards_lag_model.txt")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "lgbm_cards_lag_calibrator.joblib")
HISTORY_PATH    = os.path.join(BASE_DIR, "lgbm_cards_lag_history.parquet")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "purchased_cards"

FEATURES_V3 = [
    "Precipitation",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "WeekOfYear",
    "IsWeekend",
    "Quarter",
    "Loc Number",
    "IsHoliday",
    "IsHolidayWeekend",
    "DaysToNearestHoliday",
    "Lag7",
    "Lag14",
    "Roll7Mean",
    "Roll14Mean",
]

VALID_LOC_NUMBERS = [
    4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 47,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66,
    67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 150, 151, 152, 153, 156, 157, 158, 159, 160, 161, 162,
    163, 164, 165, 166, 167, 168, 169, 170, 177, 179, 189,
]

# ===========================================================================
# 1. Load data
# ===========================================================================

print("=" * 70)
print("LAG MODEL (V3) TRAINING")
print("  Improvements: Holiday + Lag/Rolling features + Isotonic calibration")
print("=" * 70)

print("\nLoading merged data...")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df):,} rows")

df = df.rename(columns={
    "date":               "Date",
    "loc_number":         "Loc Number",
    "new_precipitation":  "Precipitation",
    "purchased_cards":    TARGET,
})

df = df.dropna(subset=[TARGET])
df["Date"] = pd.to_datetime(df["Date"])
print(f"  After dropping nulls: {len(df):,} rows, "
      f"{df['Loc Number'].nunique()} unique locations")

# ===========================================================================
# 2. Feature engineering
# ===========================================================================

print("\n" + "=" * 70)
print("Engineering features (V3 -- holidays + lags)...")

# -- Calendar features --
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["Month"]      = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["Quarter"]    = df["Date"].dt.quarter

# -- Holiday features --
years = sorted(df["Date"].dt.year.unique())
us_holidays = hol_lib.US(years=range(min(years) - 1, max(years) + 2))
holiday_dates = np.array(sorted(us_holidays.keys()), dtype="datetime64[D]")
holiday_set = set(us_holidays.keys())

df["IsHoliday"] = df["Date"].dt.date.map(lambda d: 1 if d in holiday_set else 0)

date_vals = df["Date"].values.astype("datetime64[D]")
date_int = date_vals.astype(np.int64)
hol_int = holiday_dates.astype(np.int64)
idx = np.searchsorted(hol_int, date_int, side="left")
idx = np.clip(idx, 1, len(hol_int) - 1)
dist_left = np.abs(date_int - hol_int[idx - 1])
dist_right = np.abs(date_int - hol_int[idx])
df["DaysToNearestHoliday"] = np.minimum(dist_left, dist_right).astype(int)
df["IsHolidayWeekend"] = (df["DaysToNearestHoliday"] <= 1).astype(int)

# -- Lag / Rolling features (merge-based for exact calendar-day semantics) --
df = df.sort_values(["Loc Number", "Date"]).reset_index(drop=True)

print("  Computing lag features...")

# Lag7 and Lag14: merge-based (exact 7/14 calendar days ago)
for lag_days, col_name in [(7, "Lag7"), (14, "Lag14")]:
    lookup = df[["Loc Number", "Date", TARGET]].copy()
    lookup["Date"] = lookup["Date"] + pd.Timedelta(days=lag_days)
    lookup = lookup.rename(columns={TARGET: col_name})
    df = df.merge(
        lookup[["Loc Number", "Date", col_name]],
        on=["Loc Number", "Date"],
        how="left",
    )

# Roll7Mean and Roll14Mean: shift-based rolling per location
grouped = df.groupby("Loc Number", observed=True)[TARGET]
df["Roll7Mean"] = grouped.transform(
    lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
)
df["Roll14Mean"] = grouped.transform(
    lambda s: s.shift(1).rolling(window=14, min_periods=1).mean()
)

# Fallback: per-location day-of-week median
df["_dow"] = df["Date"].dt.dayofweek
fallback = (
    df.dropna(subset=[TARGET])
    .groupby(["Loc Number", "_dow"], observed=True)[TARGET]
    .median()
    .rename("_fallback")
)
df = df.merge(fallback, on=["Loc Number", "_dow"], how="left")

lag_cols = ["Lag7", "Lag14", "Roll7Mean", "Roll14Mean"]
global_median = df[TARGET].median()
for col in lag_cols:
    n_before = df[col].isna().sum()
    df[col] = df[col].fillna(df["_fallback"])
    df[col] = df[col].fillna(global_median)
    n_filled = n_before - df[col].isna().sum()
    print(f"    {col}: {n_before:,} NaN filled ({n_before/len(df)*100:.1f}%)")

df = df.drop(columns=["_dow", "_fallback"])

# Encode location as LightGBM category
df["Loc Number"] = pd.Categorical(
    df["Loc Number"], categories=VALID_LOC_NUMBERS
)

print(f"\n  Features: {FEATURES_V3}")
print(f"  Target:   {TARGET}")

# Show lag feature stats
print("\n  Lag feature correlations with target:")
for col in lag_cols:
    corr = df[TARGET].corr(df[col])
    print(f"    {col:15s}  r = {corr:.4f}")

# ===========================================================================
# 3. Time-based train / calibration / test split (70/10/20)
# ===========================================================================

print("\n" + "=" * 70)
print("Splitting data...")

q70 = df["Date"].quantile(0.70)
q80 = df["Date"].quantile(0.80)

train_mask = df["Date"] <= q70
cal_mask   = (df["Date"] > q70) & (df["Date"] <= q80)
test_mask  = df["Date"] > q80

X = df[FEATURES_V3]
y = df[TARGET]

X_train, y_train = X[train_mask], y[train_mask]
X_cal,   y_cal   = X[cal_mask],   y[cal_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"  Train:       {len(X_train):>7,} rows  "
      f"({df.loc[train_mask, 'Date'].min().date()} to "
      f"{df.loc[train_mask, 'Date'].max().date()})")
print(f"  Calibration: {len(X_cal):>7,} rows  "
      f"({df.loc[cal_mask, 'Date'].min().date()} to "
      f"{df.loc[cal_mask, 'Date'].max().date()})")
print(f"  Test:        {len(X_test):>7,} rows  "
      f"({df.loc[test_mask, 'Date'].min().date()} to "
      f"{df.loc[test_mask, 'Date'].max().date()})")

# ===========================================================================
# 4. LightGBM datasets
# ===========================================================================

train_ds = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=["Loc Number"],
    free_raw_data=False,
)
cal_ds = lgb.Dataset(
    X_cal, label=y_cal,
    categorical_feature=["Loc Number"],
    reference=train_ds,
    free_raw_data=False,
)

# ===========================================================================
# 5. Model parameters
# ===========================================================================

params = {
    "objective":         "regression",
    "metric":            ["mae", "rmse"],
    "boosting_type":     "gbdt",
    "num_leaves":        63,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              42,
}

# ===========================================================================
# 6. Train
# ===========================================================================

print("\n" + "=" * 70)
print("Training LightGBM (V3 -- lag features)...")

model = lgb.train(
    params,
    train_ds,
    num_boost_round=1000,
    valid_sets=[train_ds, cal_ds],
    valid_names=["train", "calibration"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"\n  Best iteration: {model.best_iteration}")

# ===========================================================================
# 7. Fit isotonic calibration on the calibration set
# ===========================================================================

print("\n" + "=" * 70)
print("Fitting isotonic bias calibration...")

raw_cal_preds = model.predict(X_cal, num_iteration=model.best_iteration)

calibrator = IsotonicRegression(
    y_min=0,
    out_of_bounds="clip",
    increasing=True,
)
calibrator.fit(raw_cal_preds, y_cal.values)

cal_preds_calibrated = calibrator.predict(raw_cal_preds)

mae_before = mean_absolute_error(y_cal, raw_cal_preds)
mae_after  = mean_absolute_error(y_cal, cal_preds_calibrated)
bias_before = np.mean(raw_cal_preds - y_cal.values)
bias_after  = np.mean(cal_preds_calibrated - y_cal.values)

print(f"  Cal set -- Before:  MAE={mae_before:.2f},  Bias={bias_before:+.2f}")
print(f"  Cal set -- After:   MAE={mae_after:.2f},  Bias={bias_after:+.2f}")

# ===========================================================================
# 8. Evaluate on test set
# ===========================================================================

print("\n" + "=" * 70)
print("Evaluating on held-out test set...")

raw_test_preds = model.predict(X_test, num_iteration=model.best_iteration)
cal_test_preds = calibrator.predict(raw_test_preds)
cal_test_preds = np.maximum(cal_test_preds, 0)

def report_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    median_ape = np.median(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    bias = np.mean(y_pred - y_true.values)
    print(f"\n  --- {name} ---")
    print(f"    MAE       : {mae:>10,.2f} cards")
    print(f"    RMSE      : {rmse:>10,.2f} cards")
    print(f"    R2        : {r2:>11.4f}")
    print(f"    MAPE      : {mape:>10.2f}%")
    print(f"    Median APE: {median_ape:>10.2f}%")
    print(f"    Bias      : {bias:>+10.2f} cards")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "MedianAPE": median_ape, "Bias": bias}

raw_metrics = report_metrics("Raw V3 (no calibration)", y_test, raw_test_preds)
cal_metrics = report_metrics("Calibrated V3 (lag + holiday + isotonic)", y_test, cal_test_preds)

# ===========================================================================
# 9. Feature importance
# ===========================================================================

print("\n" + "=" * 70)
importance = pd.DataFrame({
    "feature":    model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("Feature importance (gain):")
print(importance.to_string(index=False))

# ===========================================================================
# 10. Save model, calibrator, and history artifact
# ===========================================================================

print("\n" + "=" * 70)

# Model and calibrator
model.save_model(MODEL_PATH)
print(f"Model saved to:      {MODEL_PATH}")

joblib.dump(calibrator, CALIBRATOR_PATH)
print(f"Calibrator saved to: {CALIBRATOR_PATH}")

# History artifact: last 28 days of ALL available data
max_date = df["Date"].max()
history_cutoff = max_date - pd.Timedelta(days=28)
history_df = (
    df.loc[df["Date"] >= history_cutoff, ["Loc Number", "Date", TARGET]]
    .copy()
)
# Convert Loc Number back to int for storage (from Categorical)
history_df["Loc Number"] = history_df["Loc Number"].astype(int)
history_df = history_df.rename(columns={TARGET: "Unique Purchased Cards"})
history_df.to_parquet(HISTORY_PATH, index=False)
print(f"History saved to:    {HISTORY_PATH}")
print(f"  History range: {history_df['Date'].min().date()} to "
      f"{history_df['Date'].max().date()}")
print(f"  History rows:  {len(history_df):,}")

# ===========================================================================
# 11. Comparison summary
# ===========================================================================

print("\n" + "=" * 70)
print("COMPARISON -- Lag V3 vs Calibrated V2 vs Original")
print("=" * 70)
print(f"  {'Metric':<12}  {'Lag V3 (cal)':>14}  {'V2 (ref)':>14}  {'Original':>14}")
print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}")
print(f"  {'MAE':<12}  {cal_metrics['MAE']:>13,.2f}   {'~62':>13}   {'~66':>13}")
print(f"  {'RMSE':<12}  {cal_metrics['RMSE']:>13,.2f}   {'~97':>13}   {'~99':>13}")
print(f"  {'R2':<12}  {cal_metrics['R2']:>14.4f}   {'~0.809':>13}   {'~0.799':>13}")
print(f"  {'Bias':<12}  {cal_metrics['Bias']:>+13.2f}   {'~+10':>13}   {'N/A':>13}")
print(f"  {'Median APE':<12}  {cal_metrics['MedianAPE']:>13.2f}%  {'~19%':>13}   {'N/A':>13}")
print()
print("=" * 70)
print("Done.")
