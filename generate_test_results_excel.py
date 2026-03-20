# -*- coding: utf-8 -*-
"""
generate_test_results_excel.py
-------------------------------
Generates an Excel workbook with two sheets showing test-set results
for the LightGBM new-precipitation model:

  Sheet 1 – "Test Predictions"   : row-level actual vs predicted
  Sheet 2 – "Location Summary"   : per-location aggregated metrics

Usage:
    python generate_test_results_excel.py

Produces:
    new_precip_model_test_results.xlsx
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "merged_data_model.csv")
MODEL_FILE  = os.path.join(BASE_DIR, "lgbm_cards_new_precip_model.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "new_precip_model_test_results.xlsx")

# ---------------------------------------------------------------------------
# Constants (must match training exactly)
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

# Location labels for display
LOC_LABELS = {
    4: "Atlanta", 5: "Philadelphia", 8: "Hollywood", 10: "Ontario",
    11: "Cincinnati", 12: "Denver", 13: "Utica", 14: "Irvine",
    15: "Palisades", 16: "Orange", 17: "Hilliard", 18: "San Antonio",
    20: "St. Louis", 21: "Austin", 22: "Jacksonville", 23: "Providence",
    24: "San Jose", 25: "Westminster", 26: "Pittsburgh", 27: "San Diego",
    28: "Miami", 29: "Frisco", 31: "Cleveland", 32: "Islandia",
    33: "Toronto", 34: "Santa Anita", 35: "Arundel", 36: "Concord",
    38: "Franklin", 39: "Houston II", 42: "Nashville", 43: "Scottsdale",
    44: "Westbury", 45: "Lawrenceville", 47: "Omaha", 49: "Kansas City",
    50: "TimesSquare", 51: "Maple Grove", 52: "Tempe",
    53: "Plymouth Meeting", 54: "Arlington", 55: "Richmond", 56: "Tulsa",
    57: "Indianapolis", 58: "Polaris", 59: "Wauwatosa", 60: "Roseville",
    61: "Braintree", 62: "Dallas", 63: "Clackamas", 64: "Orlando",
    66: "Oklahoma City", 67: "Orland Park", 68: "Boise",
    69: "Virginia Beach", 70: "Albany", 71: "Syracuse", 72: "Greenville",
    74: "Livonia", 75: "Westchester", 76: "Vernon Hills",
    77: "Panama City", 78: "Los Angeles", 79: "Albuquerque",
    80: "Manchester", 81: "Euless", 82: "Pelham", 83: "Rivercenter",
    84: "Woburn", 85: "Kentwood", 86: "Buffalo-Walden", 87: "Edina",
    88: "Fresno", 89: "Friendswood", 90: "Glendale", 91: "Springfield",
    92: "El Paso", 93: "Rochester", 94: "Summerlin",
    95: "Capitol Heights", 96: "Florence", 97: "Little Rock",
    98: "Oakville", 99: "Silver Spring", 100: "Toledo",
    101: "Overland Park", 102: "Daly City", 103: "Carlsbad",
    104: "Columbia", 105: "Tucson", 106: "New Orleans",
    107: "Myrtle Beach", 108: "Alpharetta", 109: "McAllen",
    110: "Anchorage", 111: "Pineville", 112: "Northridge",
    113: "Bayamon", 114: "Wayne", 115: "Auburn", 116: "Baltimore",
    117: "Rogers", 118: "Woodbridge", 119: "Memphis", 120: "Tampa",
    121: "Madison", 122: "Torrance", 124: "Salt Lake City",
    125: "Milford", 126: "Rosemont", 127: "Thousand Oaks",
    128: "Birmingham", 129: "Fairfax", 130: "Staten Island",
    131: "Louisville", 132: "Harrisburg", 133: "Corpus Christi",
    134: "North Hills", 135: "Daytona Beach", 136: "Fort Myers",
    137: "Sevierville", 138: "Winston-Salem", 140: "McDonough",
    141: "Gaithersburg", 142: "Shenandoah", 143: "Natick",
    144: "Huntsville", 145: "Wichita", 146: "Canton",
    147: "New Hampshire", 148: "Modesto", 150: "Bellevue",
    151: "Gloucester", 152: "Fairfield", 153: "Long Beach",
    156: "Greenwood", 157: "Concord CA", 158: "Lynnwood",
    159: "Chattanooga", 160: "Lehigh Valley", 161: "Green Bay",
    162: "Sioux Falls", 163: "Brooklyn Gateway", 164: "Bakersfield",
    165: "Gainesville", 166: "Brooklyn Atlantic", 167: "San Juan",
    168: "Augusta", 169: "Cary II", 170: "Lubbock",
    177: "Des Moines", 179: "Queen Creek", 189: "Unknown",
}


def main():
    # ------------------------------------------------------------------
    # 1. Load data (same steps as training script)
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df.rename(columns={
        "date":               "Date",
        "loc_number":         "Loc Number",
        "new_precipitation":  "Precipitation",
        "purchased_cards":    TARGET,
    })
    df = df.dropna(subset=[TARGET])

    # Feature engineering
    df["Date"]       = pd.to_datetime(df["Date"])
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["Month"]      = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"]    = df["Date"].dt.quarter

    df["Loc Number"] = pd.Categorical(
        df["Loc Number"], categories=VALID_LOC_NUMBERS
    )

    # ------------------------------------------------------------------
    # 2. Reproduce the EXACT same train/test split
    # ------------------------------------------------------------------
    cutoff    = df["Date"].quantile(0.80)
    test_mask = df["Date"] > cutoff

    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    print(f"Test set: {len(X_test):,} rows  "
          f"({df.loc[test_mask, 'Date'].min().date()} to "
          f"{df.loc[test_mask, 'Date'].max().date()})")

    # ------------------------------------------------------------------
    # 3. Load model and predict
    # ------------------------------------------------------------------
    print("Loading model and predicting...")
    booster = lgb.Booster(model_file=MODEL_FILE)
    y_pred  = booster.predict(X_test, num_iteration=booster.best_iteration)

    # ------------------------------------------------------------------
    # 4. Build the row-level results DataFrame
    # ------------------------------------------------------------------
    results = df.loc[test_mask, ["Date", "Loc Number", "Precipitation", TARGET]].copy()
    results["Loc Number"] = results["Loc Number"].astype(int)
    results["Location"]   = results["Loc Number"].map(
        lambda n: f"{n} - {LOC_LABELS.get(n, 'Unknown')}"
    )
    results["Predicted"]  = np.round(y_pred, 1)
    results["Error"]      = np.round(y_pred - results[TARGET].values, 1)
    results["Abs Error"]  = np.abs(results["Error"])
    results["Pct Error"]  = np.where(
        results[TARGET] != 0,
        np.round(results["Abs Error"] / results[TARGET] * 100, 1),
        np.nan,
    )

    # Rename for cleaner column headers
    results = results.rename(columns={TARGET: "Actual"})

    # Reorder columns
    results = results[[
        "Date", "Loc Number", "Location", "Precipitation",
        "Actual", "Predicted", "Error", "Abs Error", "Pct Error",
    ]]

    results = results.sort_values(["Loc Number", "Date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Build the per-location summary
    # ------------------------------------------------------------------
    def loc_summary(g):
        actual = g["Actual"]
        pred   = g["Predicted"]
        n      = len(g)
        mae    = mean_absolute_error(actual, pred)
        rmse   = np.sqrt(mean_squared_error(actual, pred))
        r2     = r2_score(actual, pred) if n > 1 else np.nan
        avg_actual = actual.mean()
        avg_pred   = pred.mean()
        return pd.Series({
            "Rows":           n,
            "Avg Actual":     round(avg_actual, 1),
            "Avg Predicted":  round(avg_pred, 1),
            "MAE":            round(mae, 2),
            "RMSE":           round(rmse, 2),
            "R2":             round(r2, 4) if not np.isnan(r2) else np.nan,
            "MAPE (%)":       round(
                np.mean(np.abs((actual - pred) / actual.replace(0, np.nan))) * 100, 1
            ),
        })

    loc_stats = (
        results.groupby(["Loc Number", "Location"], observed=True)
        .apply(loc_summary, include_groups=False)
        .reset_index()
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 6. Overall metrics row
    # ------------------------------------------------------------------
    overall_mae  = mean_absolute_error(results["Actual"], results["Predicted"])
    overall_rmse = np.sqrt(mean_squared_error(results["Actual"], results["Predicted"]))
    overall_r2   = r2_score(results["Actual"], results["Predicted"])
    overall_mape = np.mean(
        np.abs((results["Actual"] - results["Predicted"])
               / results["Actual"].replace(0, np.nan))
    ) * 100

    overall_row = pd.DataFrame([{
        "Loc Number":    "",
        "Location":      "** OVERALL **",
        "Rows":          len(results),
        "Avg Actual":    round(results["Actual"].mean(), 1),
        "Avg Predicted": round(results["Predicted"].mean(), 1),
        "MAE":           round(overall_mae, 2),
        "RMSE":          round(overall_rmse, 2),
        "R2":            round(overall_r2, 4),
        "MAPE (%)":      round(overall_mape, 1),
    }])

    loc_stats = pd.concat([overall_row, loc_stats], ignore_index=True)

    # ------------------------------------------------------------------
    # 7. Write to Excel with formatting
    # ------------------------------------------------------------------
    print(f"Writing to {OUTPUT_FILE}...")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Test Predictions", index=False)
        loc_stats.to_excel(writer, sheet_name="Location Summary", index=False)

        # Auto-size columns for both sheets
        for sheet_name in ["Test Predictions", "Location Summary"]:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max_len + 3, 30)

            # Freeze the header row
            ws.freeze_panes = "A2"

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print(f"\nDone! Wrote {OUTPUT_FILE}")
    print(f"  Sheet 'Test Predictions':  {len(results):,} rows")
    print(f"  Sheet 'Location Summary':  {len(loc_stats):,} rows (incl. overall)")
    print(f"\n  Overall Test Metrics:")
    print(f"    MAE  : {overall_mae:>10,.2f} cards")
    print(f"    RMSE : {overall_rmse:>10,.2f} cards")
    print(f"    R2   : {overall_r2:>11.4f}")
    print(f"    MAPE : {overall_mape:>10.1f}%")


if __name__ == "__main__":
    main()
