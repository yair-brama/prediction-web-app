"""
generate_test_comparison.py
---------------------------
Loads all four trained models, runs predictions on the held-out test sets,
and writes a single CSV:

    test_predictions_comparison.csv

Columns:
    Date, Loc Number, Precipitation,
    Actual GL Rev,
    LGBM GL Rev Predicted, LGBM GL Rev Error, LGBM GL Rev Abs Error,
    LinReg GL Rev Predicted, LinReg GL Rev Error, LinReg GL Rev Abs Error,
    Actual Purchased Cards,
    LGBM Cards Predicted, LGBM Cards Error, LGBM Cards Abs Error,
    LinReg Cards Predicted, LinReg Cards Error, LinReg Cards Abs Error

Run from the Capstone directory:
    python generate_test_comparison.py
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shutil, tempfile, os, sys, warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from features import VALID_LOC_NUMBERS, FEATURES, TARGET, TARGET_CARDS

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
FILE_PATH         = os.path.join(BASE_DIR, "20260202 Walk in sales - v1400.xlsx")
LGBM_PATH         = os.path.join(BASE_DIR, "lgbm_gl_rev_model.txt")
LINREG_PATH       = os.path.join(BASE_DIR, "linreg_gl_rev_model.joblib")
LGBM_CARDS_PATH   = os.path.join(BASE_DIR, "lgbm_cards_model.txt")
LINREG_CARDS_PATH = os.path.join(BASE_DIR, "linreg_cards_model.joblib")
OUT_PATH          = os.path.join(BASE_DIR, "test_predictions_comparison.csv")


# ---------------------------------------------------------------------------
# Helper: safe Excel read (handles locked files)
# ---------------------------------------------------------------------------

def read_excel_safe(path, sheet_name):
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except PermissionError:
        tmp = os.path.join(tempfile.gettempdir(), "capstone_data_copy.xlsx")
        shutil.copy2(path, tmp)
        print(f"  (File locked — using temp copy)")
        return pd.read_excel(tmp, sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Helper: add calendar features
# ---------------------------------------------------------------------------

def add_calendar(df):
    out = df.copy()
    out["Date"]       = pd.to_datetime(out["Date"])
    out["DayOfWeek"]  = out["Date"].dt.dayofweek
    out["Month"]      = out["Date"].dt.month
    out["DayOfMonth"] = out["Date"].dt.day
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["IsWeekend"]  = (out["DayOfWeek"] >= 5).astype(int)
    out["Quarter"]    = out["Date"].dt.quarter
    return out


# ---------------------------------------------------------------------------
# 1. Load models
# ---------------------------------------------------------------------------

print("Loading models...")
lgbm_gl   = lgb.Booster(model_file=LGBM_PATH)
linreg_gl = joblib.load(LINREG_PATH)
lgbm_cards   = lgb.Booster(model_file=LGBM_CARDS_PATH)
linreg_cards = joblib.load(LINREG_CARDS_PATH)
print("  All four models loaded.")


# ---------------------------------------------------------------------------
# 2. GL Rev data & test split
# ---------------------------------------------------------------------------

print("\nLoading GL Rev data (sheet: 'Model Data')...")
df_gl = read_excel_safe(FILE_PATH, "Model Data")
df_gl = df_gl.dropna(subset=[TARGET])
df_gl = add_calendar(df_gl)

cutoff_gl    = df_gl["Date"].quantile(0.80)
test_mask_gl = df_gl["Date"] > cutoff_gl
df_gl_test   = df_gl[test_mask_gl].copy()
print(f"  GL Rev test rows: {len(df_gl_test):,}")

# Feature matrices
df_gl_test_lgbm = df_gl_test.copy()
df_gl_test_lgbm["Loc Number"] = pd.Categorical(
    df_gl_test_lgbm["Loc Number"], categories=VALID_LOC_NUMBERS
)
X_gl_lgbm = df_gl_test_lgbm[FEATURES]

df_gl_test_lr = df_gl_test.copy()
df_gl_test_lr["Loc Number"] = df_gl_test_lr["Loc Number"].astype(int)
X_gl_lr = df_gl_test_lr[FEATURES]

# Predictions
gl_lgbm_preds   = lgbm_gl.predict(X_gl_lgbm, num_iteration=lgbm_gl.best_iteration)
gl_linreg_preds = linreg_gl.predict(X_gl_lr).astype(float)


# ---------------------------------------------------------------------------
# 3. Cards data & test split
# ---------------------------------------------------------------------------

print("Loading Cards data (sheet: 'Data Model Cards')...")
df_cards = read_excel_safe(FILE_PATH, "Data Model Cards")
df_cards = df_cards.dropna(subset=[TARGET_CARDS])
df_cards = add_calendar(df_cards)

cutoff_c     = df_cards["Date"].quantile(0.80)
test_mask_c  = df_cards["Date"] > cutoff_c
df_cards_test = df_cards[test_mask_c].copy()
print(f"  Cards test rows: {len(df_cards_test):,}")

# Feature matrices
df_cards_test_lgbm = df_cards_test.copy()
df_cards_test_lgbm["Loc Number"] = pd.Categorical(
    df_cards_test_lgbm["Loc Number"], categories=VALID_LOC_NUMBERS
)
X_cards_lgbm = df_cards_test_lgbm[FEATURES]

df_cards_test_lr = df_cards_test.copy()
df_cards_test_lr["Loc Number"] = df_cards_test_lr["Loc Number"].astype(int)
X_cards_lr = df_cards_test_lr[FEATURES]

# Predictions
cards_lgbm_preds   = lgbm_cards.predict(X_cards_lgbm, num_iteration=lgbm_cards.best_iteration)
cards_linreg_preds = linreg_cards.predict(X_cards_lr).astype(float)


# ---------------------------------------------------------------------------
# 4. Build GL Rev comparison DataFrame
# ---------------------------------------------------------------------------

gl_out = df_gl_test[["Date", "Loc Number", "Precipitation", TARGET]].copy()
gl_out = gl_out.rename(columns={TARGET: "Actual GL Rev"})
gl_out["LGBM GL Rev Predicted"]      = gl_lgbm_preds
gl_out["LGBM GL Rev Error"]          = gl_out["LGBM GL Rev Predicted"] - gl_out["Actual GL Rev"]
gl_out["LGBM GL Rev Abs Error"]      = gl_out["LGBM GL Rev Error"].abs()
gl_out["LinReg GL Rev Predicted"]    = gl_linreg_preds
gl_out["LinReg GL Rev Error"]        = gl_out["LinReg GL Rev Predicted"] - gl_out["Actual GL Rev"]
gl_out["LinReg GL Rev Abs Error"]    = gl_out["LinReg GL Rev Error"].abs()
gl_out["Date"] = pd.to_datetime(gl_out["Date"]).dt.date


# ---------------------------------------------------------------------------
# 5. Build Cards comparison DataFrame
# ---------------------------------------------------------------------------

cards_out = df_cards_test[["Date", "Loc Number", "Precipitation", TARGET_CARDS]].copy()
cards_out = cards_out.rename(columns={TARGET_CARDS: "Actual Purchased Cards"})
cards_out["LGBM Cards Predicted"]    = cards_lgbm_preds
cards_out["LGBM Cards Error"]        = cards_out["LGBM Cards Predicted"] - cards_out["Actual Purchased Cards"]
cards_out["LGBM Cards Abs Error"]    = cards_out["LGBM Cards Error"].abs()
cards_out["LinReg Cards Predicted"]  = cards_linreg_preds
cards_out["LinReg Cards Error"]      = cards_out["LinReg Cards Predicted"] - cards_out["Actual Purchased Cards"]
cards_out["LinReg Cards Abs Error"]  = cards_out["LinReg Cards Error"].abs()
cards_out["Date"] = pd.to_datetime(cards_out["Date"]).dt.date


# ---------------------------------------------------------------------------
# 6. Merge on Date + Loc Number and export
# ---------------------------------------------------------------------------

print("\nMerging GL Rev and Cards test sets...")
merged = pd.merge(
    gl_out,
    cards_out,
    on=["Date", "Loc Number", "Precipitation"],
    how="outer",
)
merged = merged.sort_values(["Date", "Loc Number"]).reset_index(drop=True)
print(f"  Merged rows: {len(merged):,}")

merged.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")

# ---------------------------------------------------------------------------
# 7. Quick summary
# ---------------------------------------------------------------------------

print("\nSummary on test set:")
print(f"  {'Model':<30}  {'MAE':>12}  {'RMSE':>12}  {'R2':>8}")
print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*8}")

for col_actual, col_pred, label in [
    ("Actual GL Rev",         "LGBM GL Rev Predicted",   "LightGBM GL Rev"),
    ("Actual GL Rev",         "LinReg GL Rev Predicted", "LinReg GL Rev"),
    ("Actual Purchased Cards","LGBM Cards Predicted",    "LightGBM Cards"),
    ("Actual Purchased Cards","LinReg Cards Predicted",  "LinReg Cards"),
]:
    sub = merged[[col_actual, col_pred]].dropna()
    actual = sub[col_actual].values
    pred   = sub[col_pred].values
    mae  = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    unit = "$" if "GL Rev" in label else " cards"
    print(f"  {label:<30}  {mae:>11,.2f}{unit[0]}  {rmse:>11,.2f}{unit[0]}  {r2:>8.4f}")
