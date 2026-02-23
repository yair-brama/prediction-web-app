"""
train_models.py
---------------
Train all models on the same data split and save artifacts to disk.
Run once before launching app.py:

    python train_models.py

Produces:
    lgbm_gl_rev_model.txt              (LightGBM native text format)
    linreg_gl_rev_model.joblib         (sklearn Pipeline: OHE + LinearRegression)
    lgbm_cards_model.txt               (LightGBM for Unique Purchased Cards)
    linreg_cards_model.joblib          (sklearn Pipeline for Unique Purchased Cards)
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Import shared constants (single source of truth)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from features import VALID_LOC_NUMBERS, FEATURES, TARGET, TARGET_CARDS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(__file__)
FILE_PATH  = os.path.join(BASE_DIR, "20260202 Walk in sales - v1400.xlsx")
LGBM_PATH  = os.path.join(BASE_DIR, "lgbm_gl_rev_model.txt")
LINREG_PATH = os.path.join(BASE_DIR, "linreg_gl_rev_model.joblib")
LGBM_CARDS_PATH   = os.path.join(BASE_DIR, "lgbm_cards_model.txt")
LINREG_CARDS_PATH = os.path.join(BASE_DIR, "linreg_cards_model.joblib")

# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------

print("=" * 60)
print("Loading data...")
# If the file is locked (e.g. open in Excel), copy it to a temp path first
import shutil, tempfile
try:
    df = pd.read_excel(FILE_PATH, sheet_name="Model Data")
except PermissionError:
    tmp = os.path.join(tempfile.gettempdir(), "capstone_data_copy.xlsx")
    shutil.copy2(FILE_PATH, tmp)
    print(f"  (File locked — using temp copy at {tmp})")
    df = pd.read_excel(tmp, sheet_name="Model Data")
df = df.dropna(subset=[TARGET])
print(f"  {len(df):,} rows, {df['Loc Number'].nunique()} unique locations")

# ---------------------------------------------------------------------------
# 2. Shared calendar features
# ---------------------------------------------------------------------------

df["Date"]       = pd.to_datetime(df["Date"])
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["Month"]      = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["Quarter"]    = df["Date"].dt.quarter

# ---------------------------------------------------------------------------
# 3. Time-based train / test split (identical for both models)
# ---------------------------------------------------------------------------

cutoff     = df["Date"].quantile(0.80)
train_mask = df["Date"] <= cutoff
test_mask  = df["Date"] >  cutoff

print(f"\nTrain: {train_mask.sum():,} rows  "
      f"({df.loc[train_mask, 'Date'].min().date()} to {df.loc[train_mask, 'Date'].max().date()})")
print(f"Test:  {test_mask.sum():,} rows  "
      f"({df.loc[test_mask, 'Date'].min().date()} to {df.loc[test_mask, 'Date'].max().date()})")

y      = df[TARGET]
y_tr   = y[train_mask]
y_te   = y[test_mask]

# ---------------------------------------------------------------------------
# 4. LightGBM
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training LightGBM...")

df_lgbm = df.copy()
df_lgbm["Loc Number"] = pd.Categorical(
    df_lgbm["Loc Number"], categories=VALID_LOC_NUMBERS
)

X_lgbm    = df_lgbm[FEATURES]
X_tr_lgbm = X_lgbm[train_mask]
X_te_lgbm = X_lgbm[test_mask]

lgbm_params = {
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

train_ds = lgb.Dataset(
    X_tr_lgbm, label=y_tr,
    categorical_feature=["Loc Number"],
    free_raw_data=False,
)
test_ds = lgb.Dataset(
    X_te_lgbm, label=y_te,
    reference=train_ds,
    free_raw_data=False,
)

lgbm_model = lgb.train(
    lgbm_params,
    train_ds,
    num_boost_round=1000,
    valid_sets=[train_ds, test_ds],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

lgbm_preds = lgbm_model.predict(X_te_lgbm, num_iteration=lgbm_model.best_iteration)
lgbm_mae   = mean_absolute_error(y_te, lgbm_preds)
lgbm_rmse  = np.sqrt(mean_squared_error(y_te, lgbm_preds))
lgbm_r2    = r2_score(y_te, lgbm_preds)

print(f"  Best iteration : {lgbm_model.best_iteration}")
print(f"  MAE            : ${lgbm_mae:>12,.2f}")
print(f"  RMSE           : ${lgbm_rmse:>12,.2f}")
print(f"  R2             :  {lgbm_r2:>12.4f}")

lgbm_model.save_model(LGBM_PATH)
print(f"  Saved: {LGBM_PATH}")

# ---------------------------------------------------------------------------
# 5. Linear Regression (Pipeline: OHE + LinearRegression)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training Linear Regression...")

df_lr = df.copy()
df_lr["Loc Number"] = df_lr["Loc Number"].astype(int)

X_lr    = df_lr[FEATURES]
X_tr_lr = X_lr[train_mask]
X_te_lr = X_lr[test_mask]

numeric_cols = [f for f in FEATURES if f != "Loc Number"]
cat_cols     = ["Loc Number"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "ohe",
            OneHotEncoder(
                categories=[VALID_LOC_NUMBERS],
                sparse_output=False,
                handle_unknown="ignore",   # unseen loc -> all-zero row (safety net)
            ),
            cat_cols,
        ),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
)

linreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    LinearRegression(n_jobs=-1)),
])

linreg_pipeline.fit(X_tr_lr, y_tr)

lr_preds = linreg_pipeline.predict(X_te_lr)
lr_mae   = mean_absolute_error(y_te, lr_preds)
lr_rmse  = np.sqrt(mean_squared_error(y_te, lr_preds))
lr_r2    = r2_score(y_te, lr_preds)

print(f"  MAE  : ${lr_mae:>12,.2f}")
print(f"  RMSE : ${lr_rmse:>12,.2f}")
print(f"  R2   :  {lr_r2:>12.4f}")

joblib.dump(linreg_pipeline, LINREG_PATH)
print(f"  Saved: {LINREG_PATH}")

# ---------------------------------------------------------------------------
# 6. Side-by-side comparison (GL Rev)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("GL Rev model comparison on test set:")
print(f"  {'Metric':<8}  {'LightGBM':>14}  {'Linear Reg':>14}")
print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
print(f"  {'MAE':<8}  ${lgbm_mae:>13,.2f}  ${lr_mae:>13,.2f}")
print(f"  {'RMSE':<8}  ${lgbm_rmse:>13,.2f}  ${lr_rmse:>13,.2f}")
print(f"  {'R2':<8}  {lgbm_r2:>14.4f}  {lr_r2:>14.4f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 7. Load Purchased Cards data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Loading Purchased Cards data (sheet: 'Data Model Cards')...")
try:
    df_cards = pd.read_excel(FILE_PATH, sheet_name="Data Model Cards")
except PermissionError:
    tmp = os.path.join(tempfile.gettempdir(), "capstone_data_copy.xlsx")
    shutil.copy2(FILE_PATH, tmp)
    print(f"  (File locked — using temp copy at {tmp})")
    df_cards = pd.read_excel(tmp, sheet_name="Data Model Cards")

df_cards = df_cards.dropna(subset=[TARGET_CARDS])
print(f"  {len(df_cards):,} rows, {df_cards['Loc Number'].nunique()} unique locations")

df_cards["Date"]       = pd.to_datetime(df_cards["Date"])
df_cards["DayOfWeek"]  = df_cards["Date"].dt.dayofweek
df_cards["Month"]      = df_cards["Date"].dt.month
df_cards["DayOfMonth"] = df_cards["Date"].dt.day
df_cards["WeekOfYear"] = df_cards["Date"].dt.isocalendar().week.astype(int)
df_cards["IsWeekend"]  = (df_cards["DayOfWeek"] >= 5).astype(int)
df_cards["Quarter"]    = df_cards["Date"].dt.quarter

cutoff_c     = df_cards["Date"].quantile(0.80)
train_mask_c = df_cards["Date"] <= cutoff_c
test_mask_c  = df_cards["Date"] >  cutoff_c

print(f"\nCards Train: {train_mask_c.sum():,} rows  "
      f"({df_cards.loc[train_mask_c, 'Date'].min().date()} to "
      f"{df_cards.loc[train_mask_c, 'Date'].max().date()})")
print(f"Cards Test:  {test_mask_c.sum():,} rows  "
      f"({df_cards.loc[test_mask_c, 'Date'].min().date()} to "
      f"{df_cards.loc[test_mask_c, 'Date'].max().date()})")

y_c    = df_cards[TARGET_CARDS]
y_tr_c = y_c[train_mask_c]
y_te_c = y_c[test_mask_c]

# ---------------------------------------------------------------------------
# 8. LightGBM — Purchased Cards
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training LightGBM (Purchased Cards)...")

df_cards_lgbm = df_cards.copy()
df_cards_lgbm["Loc Number"] = pd.Categorical(
    df_cards_lgbm["Loc Number"], categories=VALID_LOC_NUMBERS
)

X_cards_lgbm    = df_cards_lgbm[FEATURES]
X_tr_cards_lgbm = X_cards_lgbm[train_mask_c]
X_te_cards_lgbm = X_cards_lgbm[test_mask_c]

cards_lgbm_params = {
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

train_ds_c = lgb.Dataset(
    X_tr_cards_lgbm, label=y_tr_c,
    categorical_feature=["Loc Number"],
    free_raw_data=False,
)
test_ds_c = lgb.Dataset(
    X_te_cards_lgbm, label=y_te_c,
    reference=train_ds_c,
    free_raw_data=False,
)

lgbm_cards_model = lgb.train(
    cards_lgbm_params,
    train_ds_c,
    num_boost_round=1000,
    valid_sets=[train_ds_c, test_ds_c],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

cards_lgbm_preds = lgbm_cards_model.predict(X_te_cards_lgbm, num_iteration=lgbm_cards_model.best_iteration)
cards_lgbm_mae   = mean_absolute_error(y_te_c, cards_lgbm_preds)
cards_lgbm_rmse  = np.sqrt(mean_squared_error(y_te_c, cards_lgbm_preds))
cards_lgbm_r2    = r2_score(y_te_c, cards_lgbm_preds)

print(f"  Best iteration : {lgbm_cards_model.best_iteration}")
print(f"  MAE            :  {cards_lgbm_mae:>12,.2f}")
print(f"  RMSE           :  {cards_lgbm_rmse:>12,.2f}")
print(f"  R2             :  {cards_lgbm_r2:>12.4f}")

lgbm_cards_model.save_model(LGBM_CARDS_PATH)
print(f"  Saved: {LGBM_CARDS_PATH}")

# ---------------------------------------------------------------------------
# 9. Linear Regression — Purchased Cards
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training Linear Regression (Purchased Cards)...")

df_cards_lr = df_cards.copy()
df_cards_lr["Loc Number"] = df_cards_lr["Loc Number"].astype(int)

X_cards_lr    = df_cards_lr[FEATURES]
X_tr_cards_lr = X_cards_lr[train_mask_c]
X_te_cards_lr = X_cards_lr[test_mask_c]

numeric_cols_c = [f for f in FEATURES if f != "Loc Number"]
cat_cols_c     = ["Loc Number"]

preprocessor_c = ColumnTransformer(
    transformers=[
        (
            "ohe",
            OneHotEncoder(
                categories=[VALID_LOC_NUMBERS],
                sparse_output=False,
                handle_unknown="ignore",
            ),
            cat_cols_c,
        ),
        ("num", "passthrough", numeric_cols_c),
    ],
    remainder="drop",
)

linreg_cards_pipeline = Pipeline([
    ("preprocessor", preprocessor_c),
    ("regressor",    LinearRegression(n_jobs=-1)),
])

linreg_cards_pipeline.fit(X_tr_cards_lr, y_tr_c)

cards_lr_preds = linreg_cards_pipeline.predict(X_te_cards_lr)
cards_lr_mae   = mean_absolute_error(y_te_c, cards_lr_preds)
cards_lr_rmse  = np.sqrt(mean_squared_error(y_te_c, cards_lr_preds))
cards_lr_r2    = r2_score(y_te_c, cards_lr_preds)

print(f"  MAE  :  {cards_lr_mae:>12,.2f}")
print(f"  RMSE :  {cards_lr_rmse:>12,.2f}")
print(f"  R2   :  {cards_lr_r2:>12.4f}")

joblib.dump(linreg_cards_pipeline, LINREG_CARDS_PATH)
print(f"  Saved: {LINREG_CARDS_PATH}")

# ---------------------------------------------------------------------------
# 10. Side-by-side comparison (Purchased Cards)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Purchased Cards model comparison on test set:")
print(f"  {'Metric':<8}  {'LightGBM':>14}  {'Linear Reg':>14}")
print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
print(f"  {'MAE':<8}   {cards_lgbm_mae:>13,.2f}   {cards_lr_mae:>13,.2f}")
print(f"  {'RMSE':<8}   {cards_lgbm_rmse:>13,.2f}   {cards_lr_rmse:>13,.2f}")
print(f"  {'R2':<8}   {cards_lgbm_r2:>13.4f}   {cards_lr_r2:>13.4f}")
print("=" * 60)
print("All models trained and saved.")
