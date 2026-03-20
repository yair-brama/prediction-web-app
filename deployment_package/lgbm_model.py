# -*- coding: utf-8 -*-
"""
LightGBM model to predict GL Rev per location based on Precipitation.
Data source: "20260202 Walk in sales - v1400.xlsx", sheet "Model Data"
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(_BASE_DIR, "20260202 Walk in sales - v1400.xlsx")

print("Loading data...")
df = pd.read_excel(FILE_PATH, sheet_name="Model Data")
print(f"  Loaded {len(df):,} rows, {df['Loc Number'].nunique()} unique locations")

# --------------------------------------------------------------------------- #
# 2. Clean data
# --------------------------------------------------------------------------- #
df = df.dropna(subset=["GL Rev"])
print(f"  After dropping nulls: {len(df):,} rows")

# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
df["DayOfWeek"]  = df["Date"].dt.dayofweek        # 0=Mon, 6=Sun
df["Month"]      = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["Quarter"]    = df["Date"].dt.quarter

# Encode location as category so LightGBM handles it natively
df["Loc Number"] = df["Loc Number"].astype("category")

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
TARGET = "GL Rev"

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------------------------------- #
# 4. Time-based train / test split (last ~20% of dates held out)
# --------------------------------------------------------------------------- #
cutoff     = df["Date"].quantile(0.80)
train_mask = df["Date"] <= cutoff
test_mask  = df["Date"] >  cutoff

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain rows: {len(X_train):,}  ({train_mask.sum() / len(df)*100:.0f}%)")
print(f"Test  rows: {len(X_test):,}  ({test_mask.sum()  / len(df)*100:.0f}%)")
print(f"Train date range: {df.loc[train_mask, 'Date'].min().date()} to "
      f"{df.loc[train_mask, 'Date'].max().date()}")
print(f"Test  date range: {df.loc[test_mask,  'Date'].min().date()} to "
      f"{df.loc[test_mask,  'Date'].max().date()}")

# --------------------------------------------------------------------------- #
# 5. LightGBM datasets
# --------------------------------------------------------------------------- #
cat_features = ["Loc Number"]

train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=cat_features,
    free_raw_data=False,
)
test_data = lgb.Dataset(
    X_test, label=y_test,
    categorical_feature=cat_features,
    reference=train_data,
    free_raw_data=False,
)

# --------------------------------------------------------------------------- #
# 6. Model parameters
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 7. Train
# --------------------------------------------------------------------------- #
print("\nTraining LightGBM model...")
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=callbacks,
)

print(f"\nBest iteration: {model.best_iteration}")

# --------------------------------------------------------------------------- #
# 8. Evaluate on test set
# --------------------------------------------------------------------------- #
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

print("\n--- Test-set metrics ---")
print(f"  MAE  : ${mae:>12,.2f}")
print(f"  RMSE : ${rmse:>12,.2f}")
print(f"  R2   : {r2:>13.4f}")
print(f"  MAPE : {mape:>12.2f}%")

# --------------------------------------------------------------------------- #
# 9. Feature importance
# --------------------------------------------------------------------------- #
importance = pd.DataFrame({
    "feature":    model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("\n--- Feature importance (gain) ---")
print(importance.to_string(index=False))

# --------------------------------------------------------------------------- #
# 10. Per-location metrics
# --------------------------------------------------------------------------- #
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

print("\n--- Per-location test metrics (sorted by MAE, top 10) ---")
print(loc_metrics.head(10).to_string())

# --------------------------------------------------------------------------- #
# 11. Save model
# --------------------------------------------------------------------------- #
model_path = os.path.join(_BASE_DIR, "lgbm_gl_rev_model.txt")
model.save_model(model_path)
print(f"\nModel saved to: {model_path}")

# --------------------------------------------------------------------------- #
# 12. Feature importance plot (optional)
# --------------------------------------------------------------------------- #
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(_BASE_DIR, "lgbm_feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.show()
except ImportError:
    print("matplotlib not installed - skipping plot.")

print("\nDone.")
