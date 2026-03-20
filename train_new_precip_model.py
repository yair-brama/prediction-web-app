"""
train_new_precip_model.py
-------------------------
Standalone script to train a LightGBM model predicting Unique Purchased Cards
using the NEW precipitation data from merged_data_model.csv.

Usage:
    python train_new_precip_model.py

Produces:
    lgbm_cards_new_precip_model.txt   (LightGBM native text format)

This model is identical in architecture to the existing lgbm_cards_model.txt
but trained on the new precipitation values from historical_precipitation_model_data.csv
instead of the original precipitation values from Data_model_cards_updated.xlsx.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "merged_data_model.csv")
MODEL_PATH = os.path.join(BASE_DIR, "lgbm_cards_new_precip_model.txt")

# ---------------------------------------------------------------------------
# Constants (matching the existing model for apples-to-apples comparison)
# ---------------------------------------------------------------------------

TARGET   = "purchased_cards"
FEATURES = [
    "Precipitation",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "WeekOfYear",
    "IsWeekend",
    "Quarter",
    "Loc Number",
]

# All unique Loc Numbers present in the training data (from features.py)
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

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("Loading merged data...")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df):,} rows from {DATA_FILE}")

# Rename columns to match the model's expected format
df = df.rename(columns={
    "date":               "Date",
    "loc_number":         "Loc Number",
    "new_precipitation":  "Precipitation",   # <-- using the NEW precipitation
    "purchased_cards":    TARGET,
})

df = df.dropna(subset=[TARGET])
print(f"  After dropping nulls: {len(df):,} rows, "
      f"{df['Loc Number'].nunique()} unique locations")

# Show precipitation stats for context
print(f"\n  New Precipitation stats:")
print(f"    Min:    {df['Precipitation'].min():.4f}")
print(f"    Mean:   {df['Precipitation'].mean():.4f}")
print(f"    Median: {df['Precipitation'].median():.4f}")
print(f"    Max:    {df['Precipitation'].max():.4f}")

# Also report old_precipitation for comparison
if "old_precipitation" in df.columns:
    print(f"\n  Old Precipitation stats (for reference):")
    print(f"    Min:    {df['old_precipitation'].min():.4f}")
    print(f"    Mean:   {df['old_precipitation'].mean():.4f}")
    print(f"    Median: {df['old_precipitation'].median():.4f}")
    print(f"    Max:    {df['old_precipitation'].max():.4f}")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Engineering features...")

df["Date"]       = pd.to_datetime(df["Date"])
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["Month"]      = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["Quarter"]    = df["Date"].dt.quarter

# Encode location as native LightGBM category
df["Loc Number"] = pd.Categorical(
    df["Loc Number"], categories=VALID_LOC_NUMBERS
)

print(f"  Features: {FEATURES}")
print(f"  Target:   {TARGET}")

# ---------------------------------------------------------------------------
# 3. Time-based train / test split (80/20, same as existing model)
# ---------------------------------------------------------------------------

cutoff     = df["Date"].quantile(0.80)
train_mask = df["Date"] <= cutoff
test_mask  = df["Date"] >  cutoff

X = df[FEATURES]
y = df[TARGET]

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\n  Train: {len(X_train):,} rows  "
      f"({df.loc[train_mask, 'Date'].min().date()} to "
      f"{df.loc[train_mask, 'Date'].max().date()})")
print(f"  Test:  {len(X_test):,} rows  "
      f"({df.loc[test_mask, 'Date'].min().date()} to "
      f"{df.loc[test_mask, 'Date'].max().date()})")

# ---------------------------------------------------------------------------
# 4. LightGBM datasets
# ---------------------------------------------------------------------------

train_ds = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=["Loc Number"],
    free_raw_data=False,
)
test_ds = lgb.Dataset(
    X_test, label=y_test,
    categorical_feature=["Loc Number"],
    reference=train_ds,
    free_raw_data=False,
)

# ---------------------------------------------------------------------------
# 5. Model parameters (identical to existing model for fair comparison)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training LightGBM (Purchased Cards — NEW precipitation)...")

model = lgb.train(
    params,
    train_ds,
    num_boost_round=1000,
    valid_sets=[train_ds, test_ds],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"\n  Best iteration: {model.best_iteration}")

# ---------------------------------------------------------------------------
# 7. Evaluate on test set
# ---------------------------------------------------------------------------

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

print("\n--- Test-set metrics ---")
print(f"  MAE  : {mae:>12,.2f} cards")
print(f"  RMSE : {rmse:>12,.2f} cards")
print(f"  R2   : {r2:>13.4f}")
print(f"  MAPE : {mape:>12.2f}%")

# ---------------------------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------------------------

importance = pd.DataFrame({
    "feature":    model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("\n--- Feature importance (gain) ---")
print(importance.to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Per-location metrics (top 10 best + worst)
# ---------------------------------------------------------------------------

test_df = df[test_mask].copy()
test_df["predicted"] = y_pred

loc_metrics = (
    test_df.groupby("Loc Number", observed=True)
    .apply(lambda g: pd.Series({
        "n":    len(g),
        "MAE":  mean_absolute_error(g[TARGET], g["predicted"]),
        "RMSE": np.sqrt(mean_squared_error(g[TARGET], g["predicted"])),
        "R2":   r2_score(g[TARGET], g["predicted"]) if len(g) > 1 else float("nan"),
    }))
    .sort_values("MAE")
)

print("\n--- Per-location test metrics (top 10 best MAE) ---")
print(loc_metrics.head(10).to_string())

print("\n--- Per-location test metrics (top 10 worst MAE) ---")
print(loc_metrics.tail(10).to_string())

# ---------------------------------------------------------------------------
# 10. Save model
# ---------------------------------------------------------------------------

model.save_model(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 11. Comparison summary (reference values from existing model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("COMPARISON — New precipitation model vs existing model")
print("=" * 60)
print(f"  {'Metric':<8}  {'New Precip':>14}  {'Old (reference)':>16}")
print(f"  {'-'*8}  {'-'*14}  {'-'*16}")
print(f"  {'MAE':<8}  {mae:>13,.2f}   {'~68':>15}")
print(f"  {'RMSE':<8}  {rmse:>13,.2f}   {'~100':>15}")
print(f"  {'R2':<8}  {r2:>14.4f}   {'~0.7940':>15}")
print()
print("  (Old model reference values are from the lgbm_cards_model.txt")
print("   trained on the original precipitation data.)")
print("=" * 60)
print("Done.")
