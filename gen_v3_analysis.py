"""
gen_v3_analysis.py
------------------
Reads predictions_v3.csv, computes comprehensive metrics comparing
V3 Lag Model predictions vs actual Unique Purchased Cards, maps
location numbers to city names, and generates a standalone HTML dashboard.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Location labels (from features.py LOC_LABELS)
# ---------------------------------------------------------------------------

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
    50: "TimesSquare", 51: "Maple Grove", 52: "Tempe", 53: "Plymouth Meeting",
    54: "Arlington", 55: "Richmond", 56: "Tulsa", 57: "Indianapolis",
    58: "Polaris", 59: "Wauwatosa", 60: "Roseville", 61: "Braintree",
    62: "Dallas", 63: "Clackamas", 64: "Orlando", 66: "Oklahoma City",
    67: "Orland Park", 68: "Boise", 69: "Virginia Beach", 70: "Albany",
    71: "Syracuse", 72: "Greenville", 74: "Livonia", 75: "Westchester",
    76: "Vernon Hills", 77: "Panama City", 78: "Los Angeles",
    79: "Albuquerque", 80: "Manchester", 81: "Euless", 82: "Pelham",
    83: "Rivercenter", 84: "Woburn", 85: "Kentwood", 86: "Buffalo-Walden",
    87: "Edina", 88: "Fresno", 89: "Friendswood", 90: "Glendale",
    91: "Springfield", 92: "El Paso", 93: "Rochester", 94: "Summerlin",
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

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "predictions_v3.csv")

df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])
df["Predicted"] = df["Predicted Purchased Cards"]
df["Actual"] = df["Unique Purchased Cards"]

# Drop rows with missing actuals
n_before = len(df)
df = df.dropna(subset=["Actual"]).copy()
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"  Dropped {n_dropped} rows with missing Actual values")

# Add location names
df["Location"] = df["Loc Number"].map(lambda x: LOC_LABELS.get(x, f"Loc {x}"))

print(f"Loaded {len(df):,} rows, {df['Loc Number'].nunique()} locations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 3. Overall metrics
# ---------------------------------------------------------------------------

pred = df["Predicted"].values
actual = df["Actual"].values
error = pred - actual

mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error**2))
r2 = 1 - np.sum(error**2) / np.sum((actual - np.mean(actual))**2)
bias = np.mean(error)
mape = np.mean(np.abs(error / np.where(actual == 0, np.nan, actual))) * 100
median_ape = np.median(np.abs(error / np.where(actual == 0, np.nan, actual))) * 100
over_pct = np.mean(error > 0) * 100
under_pct = np.mean(error < 0) * 100
exact_pct = np.mean(error == 0) * 100

total_excess = np.sum(error)

overall = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4),
    "Bias": round(bias, 2),
    "MAPE": round(mape, 2),
    "MedianAPE": round(median_ape, 2),
    "OverPredictPct": round(over_pct, 1),
    "UnderPredictPct": round(under_pct, 1),
    "TotalExcess": round(total_excess, 0),
    "NumRows": len(df),
    "NumLocations": int(df["Loc Number"].nunique()),
    "DateMin": str(df["Date"].min().date()),
    "DateMax": str(df["Date"].max().date()),
}

print(f"\n--- Overall Metrics ---")
for k, v in overall.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 4. Day-of-week analysis
# ---------------------------------------------------------------------------

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["DayOfWeek"] = df["Date"].dt.dayofweek

dow_data = []
for dow in range(7):
    sub = df[df["DayOfWeek"] == dow]
    e = sub["Predicted"].values - sub["Actual"].values
    a = sub["Actual"].values
    dow_data.append({
        "day": DOW_NAMES[dow],
        "count": len(sub),
        "mae": round(np.mean(np.abs(e)), 2),
        "rmse": round(np.sqrt(np.mean(e**2)), 2),
        "bias": round(np.mean(e), 2),
        "mape": round(np.mean(np.abs(e / np.where(a == 0, np.nan, a))) * 100, 2),
        "meanActual": round(np.mean(a), 1),
        "meanPredicted": round(np.mean(sub["Predicted"].values), 1),
    })

# ---------------------------------------------------------------------------
# 5. Weekly analysis
# ---------------------------------------------------------------------------

df["Week"] = df["Date"].dt.to_period("W").apply(lambda r: str(r.start_time.date()))

week_groups = df.groupby("Week")
week_data = []
for wk, grp in week_groups:
    e = grp["Predicted"].values - grp["Actual"].values
    a = grp["Actual"].values
    week_data.append({
        "week": wk,
        "count": len(grp),
        "mae": round(np.mean(np.abs(e)), 2),
        "bias": round(np.mean(e), 2),
        "totalActual": round(np.sum(a), 0),
        "totalPredicted": round(np.sum(grp["Predicted"].values), 0),
        "r2": round(1 - np.sum(e**2) / max(np.sum((a - np.mean(a))**2), 1e-10), 4),
    })

# ---------------------------------------------------------------------------
# 6. Location-level analysis (top 20 best + worst by MAE)
# ---------------------------------------------------------------------------

loc_groups = df.groupby("Loc Number")
loc_data_all = []
for loc_num, grp in loc_groups:
    e = grp["Predicted"].values - grp["Actual"].values
    a = grp["Actual"].values
    loc_mae = np.mean(np.abs(e))
    loc_data_all.append({
        "locNum": int(loc_num),
        "location": LOC_LABELS.get(int(loc_num), f"Loc {loc_num}"),
        "count": len(grp),
        "mae": round(loc_mae, 2),
        "rmse": round(np.sqrt(np.mean(e**2)), 2),
        "bias": round(np.mean(e), 2),
        "r2": round(1 - np.sum(e**2) / max(np.sum((a - np.mean(a))**2), 1e-10), 4),
        "meanActual": round(np.mean(a), 1),
        "mape": round(np.mean(np.abs(e / np.where(a == 0, np.nan, a))) * 100, 2),
    })

loc_data_all.sort(key=lambda x: x["mae"])
best_locs = loc_data_all[:15]
worst_locs = list(reversed(loc_data_all[-15:]))

# ---------------------------------------------------------------------------
# 7. Biggest single-row errors
# ---------------------------------------------------------------------------

df["Error"] = df["Predicted"] - df["Actual"]
df["AbsError"] = np.abs(df["Error"])
df["APE"] = np.abs(df["Error"] / df["Actual"].replace(0, np.nan)) * 100

worst_rows = df.nlargest(20, "AbsError")
worst_errors = []
for _, row in worst_rows.iterrows():
    worst_errors.append({
        "location": row["Location"],
        "locNum": int(row["Loc Number"]),
        "date": str(row["Date"].date()),
        "dayName": row["Date"].strftime("%A"),
        "predicted": round(row["Predicted"], 1),
        "actual": int(row["Actual"]),
        "error": round(row["Error"], 1),
        "ape": round(row["APE"], 1) if not np.isnan(row["APE"]) else None,
    })

# ---------------------------------------------------------------------------
# 8. Scatter data (sample up to 2000 points for chart)
# ---------------------------------------------------------------------------

if len(df) > 2000:
    scatter_df = df.sample(2000, random_state=42)
else:
    scatter_df = df

scatter_data = [
    {"x": round(row["Actual"], 1), "y": round(row["Predicted"], 1)}
    for _, row in scatter_df.iterrows()
]

# ---------------------------------------------------------------------------
# 9. Daily time series (aggregated across all locations)
# ---------------------------------------------------------------------------

daily = df.groupby("Date").agg(
    totalActual=("Actual", "sum"),
    totalPredicted=("Predicted", "sum"),
).reset_index()
daily = daily.sort_values("Date")

timeseries = [
    {
        "date": str(row["Date"].date()),
        "actual": round(row["totalActual"], 0),
        "predicted": round(row["totalPredicted"], 0),
    }
    for _, row in daily.iterrows()
]

# ---------------------------------------------------------------------------
# 10. Error distribution (histogram bins)
# ---------------------------------------------------------------------------

errors = df["Error"].values
hist_bins = np.linspace(-500, 500, 51)
hist_counts, hist_edges = np.histogram(np.clip(errors, -500, 500), bins=hist_bins)
error_dist = [
    {"binStart": round(hist_edges[i], 1), "binEnd": round(hist_edges[i+1], 1),
     "count": int(hist_counts[i])}
    for i in range(len(hist_counts))
]

# ---------------------------------------------------------------------------
# 11. Comparison with prior models
# ---------------------------------------------------------------------------

model_comparison = [
    {"model": "Original (V1)", "mae": 66, "rmse": 99, "r2": 0.799, "bias": None},
    {"model": "Calibrated (V2)", "mae": 62.19, "rmse": 96.52, "r2": 0.8088, "bias": 10.31},
    {"model": "Lag V3", "mae": overall["MAE"], "rmse": overall["RMSE"], "r2": overall["R2"], "bias": overall["Bias"]},
]

# ---------------------------------------------------------------------------
# 12. Build JSON payload
# ---------------------------------------------------------------------------

payload = {
    "overall": overall,
    "dow": dow_data,
    "weeks": week_data,
    "bestLocs": best_locs,
    "worstLocs": worst_locs,
    "worstErrors": worst_errors,
    "scatter": scatter_data,
    "timeseries": timeseries,
    "errorDist": error_dist,
    "modelComparison": model_comparison,
}

# ---------------------------------------------------------------------------
# 13. Generate HTML
# ---------------------------------------------------------------------------

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V3 Lag Model — Prediction Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --green: #4ade80; --red: #f87171; --amber: #fbbf24; --purple: #a78bfa;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }}
  h1 {{ text-align:center; font-size:1.8rem; margin-bottom:4px; color:var(--accent); }}
  .subtitle {{ text-align:center; color:var(--muted); margin-bottom:32px; font-size:0.95rem; }}
  .grid {{ display:grid; gap:24px; margin-bottom:32px; }}
  .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); }}
  .grid-1 {{ grid-template-columns: 1fr; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  .card h2 {{ font-size:1.1rem; margin-bottom:16px; color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:8px; }}
  .kpi {{ text-align:center; }}
  .kpi .value {{ font-size:2rem; font-weight:700; }}
  .kpi .label {{ color:var(--muted); font-size:0.85rem; margin-top:4px; }}
  .kpi .sub {{ color:var(--muted); font-size:0.75rem; margin-top:2px; }}
  .good {{ color: var(--green); }}
  .bad {{ color: var(--red); }}
  .neutral {{ color: var(--amber); }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ text-align:left; padding:8px 10px; border-bottom:2px solid var(--border); color:var(--accent); font-weight:600; }}
  td {{ padding:7px 10px; border-bottom:1px solid var(--border); }}
  tr:hover {{ background: rgba(56,189,248,0.06); }}
  .chart-wrap {{ position:relative; height:350px; }}
  .section-title {{ font-size:1.3rem; color:var(--accent); margin:32px 0 16px; border-bottom:2px solid var(--border); padding-bottom:8px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }}
  .badge-green {{ background:rgba(74,222,128,0.15); color:var(--green); }}
  .badge-red {{ background:rgba(248,113,113,0.15); color:var(--red); }}
  .model-evolution {{ display:flex; align-items:center; justify-content:center; gap:16px; flex-wrap:wrap; margin:24px 0; }}
  .evo-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 24px; text-align:center; min-width:180px; }}
  .evo-card.active {{ border-color:var(--accent); box-shadow:0 0 20px rgba(56,189,248,0.2); }}
  .evo-card .model-name {{ font-weight:700; margin-bottom:8px; }}
  .evo-card .metric {{ font-size:0.85rem; color:var(--muted); }}
  .evo-card .metric span {{ color:var(--text); font-weight:600; }}
  .evo-arrow {{ font-size:1.5rem; color:var(--muted); }}
  .improvement {{ color:var(--green); font-weight:600; font-size:0.9rem; }}
  @media(max-width:768px) {{
    .grid-2 {{ grid-template-columns:1fr; }}
    .grid-4 {{ grid-template-columns:repeat(2,1fr); }}
  }}
</style>
</head>
<body>

<h1>LightGBM V3 Lag Model — Prediction vs Actual Analysis</h1>
<p class="subtitle">
  Holdout period: {overall["DateMin"]} to {overall["DateMax"]} &bull;
  {overall["NumRows"]:,} predictions across {overall["NumLocations"]} locations &bull;
  Model: LightGBM + Holidays + Lag Features + Isotonic Calibration
</p>

<!-- ==================== MODEL EVOLUTION ==================== -->
<h3 class="section-title">Model Evolution</h3>
<div class="model-evolution">
  <div class="evo-card">
    <div class="model-name">V1 — Original</div>
    <div class="metric">MAE: <span>66</span></div>
    <div class="metric">RMSE: <span>99</span></div>
    <div class="metric">R²: <span>0.799</span></div>
  </div>
  <div class="evo-arrow">→</div>
  <div class="evo-card">
    <div class="model-name">V2 — Calibrated</div>
    <div class="metric">MAE: <span>62.19</span></div>
    <div class="metric">RMSE: <span>96.52</span></div>
    <div class="metric">R²: <span>0.809</span></div>
  </div>
  <div class="evo-arrow">→</div>
  <div class="evo-card active">
    <div class="model-name" style="color:var(--accent)">V3 — Lag</div>
    <div class="metric">MAE: <span style="color:var(--green)">{overall["MAE"]}</span></div>
    <div class="metric">RMSE: <span style="color:var(--green)">{overall["RMSE"]}</span></div>
    <div class="metric">R²: <span style="color:var(--green)">{overall["R2"]}</span></div>
  </div>
</div>
<p style="text-align:center" class="improvement">
  Total MAE improvement from V1 → V3:
  {round((1 - overall["MAE"]/66)*100, 1)}% reduction
  &nbsp;|&nbsp;
  R² improvement: {round((overall["R2"] - 0.799)*100, 2)} points
</p>

<!-- ==================== KPI CARDS ==================== -->
<h3 class="section-title">Overall Performance</h3>
<div class="grid grid-4">
  <div class="card kpi">
    <div class="value good">{overall["MAE"]}</div>
    <div class="label">MAE (cards)</div>
    <div class="sub">Mean Absolute Error</div>
  </div>
  <div class="card kpi">
    <div class="value" style="color:var(--accent)">{overall["RMSE"]}</div>
    <div class="label">RMSE (cards)</div>
    <div class="sub">Root Mean Squared Error</div>
  </div>
  <div class="card kpi">
    <div class="value good">{overall["R2"]}</div>
    <div class="label">R²</div>
    <div class="sub">Coefficient of Determination</div>
  </div>
  <div class="card kpi">
    <div class="value {'good' if abs(overall['Bias']) < 10 else 'neutral'}">{overall["Bias"]:+.2f}</div>
    <div class="label">Bias (cards)</div>
    <div class="sub">Mean Prediction Error</div>
  </div>
  <div class="card kpi">
    <div class="value" style="color:var(--purple)">{overall["MAPE"]}%</div>
    <div class="label">MAPE</div>
    <div class="sub">Mean Abs % Error</div>
  </div>
  <div class="card kpi">
    <div class="value" style="color:var(--purple)">{overall["MedianAPE"]}%</div>
    <div class="label">Median APE</div>
    <div class="sub">Median Abs % Error</div>
  </div>
  <div class="card kpi">
    <div class="value {'bad' if overall['OverPredictPct'] > 55 else 'neutral'}">{overall["OverPredictPct"]}%</div>
    <div class="label">Over-predict Rate</div>
    <div class="sub">{100-overall["OverPredictPct"]:.1f}% under-predict</div>
  </div>
  <div class="card kpi">
    <div class="value {'bad' if abs(overall['TotalExcess']) > 50000 else 'neutral'}">{overall["TotalExcess"]:+,.0f}</div>
    <div class="label">Total Excess Cards</div>
    <div class="sub">Sum of all errors</div>
  </div>
</div>

<!-- ==================== CHARTS ROW 1 ==================== -->
<div class="grid grid-2">
  <div class="card">
    <h2>Predicted vs Actual (Scatter)</h2>
    <div class="chart-wrap"><canvas id="scatterChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Error Distribution</h2>
    <div class="chart-wrap"><canvas id="errorHistChart"></canvas></div>
  </div>
</div>

<!-- ==================== TIME SERIES ==================== -->
<div class="grid grid-1">
  <div class="card">
    <h2>Daily Total: Predicted vs Actual (All Locations Combined)</h2>
    <div class="chart-wrap" style="height:300px"><canvas id="timeseriesChart"></canvas></div>
  </div>
</div>

<!-- ==================== DAY OF WEEK ==================== -->
<h3 class="section-title">Day-of-Week Breakdown</h3>
<div class="grid grid-2">
  <div class="card">
    <h2>MAE &amp; Bias by Day of Week</h2>
    <div class="chart-wrap"><canvas id="dowChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Day-of-Week Details</h2>
    <table>
      <thead><tr><th>Day</th><th>Count</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>MAPE</th><th>Avg Actual</th></tr></thead>
      <tbody>
"""

for d in dow_data:
    bias_cls = "bad" if d["bias"] > 10 else ("good" if d["bias"] < -10 else "")
    html += f"""        <tr>
          <td><strong>{d["day"]}</strong></td><td>{d["count"]:,}</td>
          <td>{d["mae"]:.1f}</td><td>{d["rmse"]:.1f}</td>
          <td class="{bias_cls}">{d["bias"]:+.1f}</td>
          <td>{d["mape"]:.1f}%</td><td>{d["meanActual"]:.0f}</td>
        </tr>
"""

html += """      </tbody>
    </table>
  </div>
</div>

<!-- ==================== WEEKLY ==================== -->
<h3 class="section-title">Weekly Breakdown</h3>
<div class="grid grid-1">
  <div class="card">
    <h2>Weekly MAE &amp; Bias</h2>
    <div class="chart-wrap" style="height:300px"><canvas id="weeklyChart"></canvas></div>
  </div>
</div>
<div class="grid grid-1">
  <div class="card">
    <h2>Weekly Details</h2>
    <table>
      <thead><tr><th>Week Starting</th><th>Rows</th><th>MAE</th><th>Bias</th><th>Total Actual</th><th>Total Predicted</th><th>R²</th></tr></thead>
      <tbody>
"""

for w in week_data:
    diff = w["totalPredicted"] - w["totalActual"]
    bias_cls = "bad" if w["bias"] > 15 else ("good" if abs(w["bias"]) < 5 else "")
    html += f"""        <tr>
          <td>{w["week"]}</td><td>{w["count"]:,}</td>
          <td>{w["mae"]:.1f}</td><td class="{bias_cls}">{w["bias"]:+.1f}</td>
          <td>{w["totalActual"]:,.0f}</td><td>{w["totalPredicted"]:,.0f}</td>
          <td>{w["r2"]:.3f}</td>
        </tr>
"""

html += """      </tbody>
    </table>
  </div>
</div>

<!-- ==================== LOCATIONS ==================== -->
<h3 class="section-title">Location Analysis</h3>
<div class="grid grid-2">
  <div class="card">
    <h2>🏆 Top 15 Best Locations (Lowest MAE)</h2>
    <table>
      <thead><tr><th>#</th><th>Location</th><th>MAE</th><th>Bias</th><th>R²</th><th>MAPE</th><th>Avg Actual</th></tr></thead>
      <tbody>
"""

for i, loc in enumerate(best_locs):
    html += f"""        <tr>
          <td>{i+1}</td><td><strong>{loc["location"]}</strong></td>
          <td class="good">{loc["mae"]:.1f}</td>
          <td>{loc["bias"]:+.1f}</td><td>{loc["r2"]:.3f}</td>
          <td>{loc["mape"]:.1f}%</td><td>{loc["meanActual"]:.0f}</td>
        </tr>
"""

html += """      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>⚠️ Top 15 Worst Locations (Highest MAE)</h2>
    <table>
      <thead><tr><th>#</th><th>Location</th><th>MAE</th><th>Bias</th><th>R²</th><th>MAPE</th><th>Avg Actual</th></tr></thead>
      <tbody>
"""

for i, loc in enumerate(worst_locs):
    html += f"""        <tr>
          <td>{i+1}</td><td><strong>{loc["location"]}</strong></td>
          <td class="bad">{loc["mae"]:.1f}</td>
          <td>{loc["bias"]:+.1f}</td><td>{loc["r2"]:.3f}</td>
          <td>{loc["mape"]:.1f}%</td><td>{loc["meanActual"]:.0f}</td>
        </tr>
"""

html += """      </tbody>
    </table>
  </div>
</div>

<!-- ==================== WORST INDIVIDUAL ERRORS ==================== -->
<h3 class="section-title">Largest Individual Errors</h3>
<div class="grid grid-1">
  <div class="card">
    <h2>Top 20 Worst Predictions</h2>
    <table>
      <thead><tr><th>#</th><th>Location</th><th>Date</th><th>Day</th><th>Predicted</th><th>Actual</th><th>Error</th><th>APE</th></tr></thead>
      <tbody>
"""

for i, e in enumerate(worst_errors):
    err_cls = "bad" if e["error"] > 0 else "good"
    ape_str = f'{e["ape"]:.0f}%' if e["ape"] is not None else "N/A"
    html += f"""        <tr>
          <td>{i+1}</td><td><strong>{e["location"]}</strong></td>
          <td>{e["date"]}</td><td>{e["dayName"]}</td>
          <td>{e["predicted"]:,.1f}</td><td>{e["actual"]:,}</td>
          <td class="{err_cls}">{e["error"]:+,.1f}</td>
          <td>{ape_str}</td>
        </tr>
"""

html += """      </tbody>
    </table>
  </div>
</div>

<!-- ==================== KEY FINDINGS ==================== -->
<h3 class="section-title">Key Findings &amp; Insights</h3>
<div class="grid grid-1">
  <div class="card" id="findings">
  </div>
</div>

<p style="text-align:center; color:var(--muted); margin-top:32px; font-size:0.8rem;">
  Generated on """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """ &bull;
  LightGBM V3 Lag Model — Capstone Project
</p>

<script>
// ===== Data =====
const DATA = """ + json.dumps(payload) + """;

// ===== Chart defaults =====
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ===== 1. Scatter =====
new Chart(document.getElementById('scatterChart'), {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Predictions',
      data: DATA.scatter,
      backgroundColor: 'rgba(56,189,248,0.3)',
      borderColor: 'rgba(56,189,248,0.6)',
      pointRadius: 2.5,
    }, {
      label: 'Perfect prediction',
      data: [{x:0,y:0},{x:Math.max(...DATA.scatter.map(d=>d.x)),y:Math.max(...DATA.scatter.map(d=>d.x))}],
      type: 'line',
      borderColor: '#4ade80',
      borderWidth: 2,
      borderDash: [6,3],
      pointRadius: 0,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top' } },
    scales: {
      x: { title: { display: true, text: 'Actual Cards' } },
      y: { title: { display: true, text: 'Predicted Cards' } },
    }
  }
});

// ===== 2. Error histogram =====
new Chart(document.getElementById('errorHistChart'), {
  type: 'bar',
  data: {
    labels: DATA.errorDist.map(b => b.binStart),
    datasets: [{
      label: 'Count',
      data: DATA.errorDist.map(b => b.count),
      backgroundColor: DATA.errorDist.map(b => b.binStart >= 0 ? 'rgba(248,113,113,0.6)' : 'rgba(56,189,248,0.6)'),
      borderWidth: 0,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: 'Prediction Error (cards)' } },
      y: { title: { display: true, text: 'Frequency' } },
    }
  }
});

// ===== 3. Time series =====
new Chart(document.getElementById('timeseriesChart'), {
  type: 'line',
  data: {
    labels: DATA.timeseries.map(d => d.date),
    datasets: [
      { label: 'Actual', data: DATA.timeseries.map(d => d.actual), borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', borderWidth: 2, pointRadius: 2, fill: true },
      { label: 'Predicted', data: DATA.timeseries.map(d => d.predicted), borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', borderWidth: 2, pointRadius: 2, fill: true },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { ticks: { maxTicksAuto: 10 } },
      y: { title: { display: true, text: 'Total Cards (all locations)' } },
    }
  }
});

// ===== 4. Day of week =====
new Chart(document.getElementById('dowChart'), {
  type: 'bar',
  data: {
    labels: DATA.dow.map(d => d.day.substring(0,3)),
    datasets: [
      { label: 'MAE', data: DATA.dow.map(d => d.mae), backgroundColor: 'rgba(56,189,248,0.7)', borderRadius: 4 },
      { label: 'Bias', data: DATA.dow.map(d => d.bias), backgroundColor: DATA.dow.map(d => d.bias > 0 ? 'rgba(248,113,113,0.7)' : 'rgba(74,222,128,0.7)'), borderRadius: 4 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top' } },
    scales: { y: { title: { display: true, text: 'Cards' } } }
  }
});

// ===== 5. Weekly chart =====
new Chart(document.getElementById('weeklyChart'), {
  type: 'bar',
  data: {
    labels: DATA.weeks.map(w => w.week),
    datasets: [
      { label: 'MAE', data: DATA.weeks.map(w => w.mae), backgroundColor: 'rgba(56,189,248,0.7)', borderRadius: 4 },
      { label: 'Bias', data: DATA.weeks.map(w => w.bias), backgroundColor: DATA.weeks.map(w => w.bias > 0 ? 'rgba(248,113,113,0.7)' : 'rgba(74,222,128,0.7)'), borderRadius: 4 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top' } },
    scales: { y: { title: { display: true, text: 'Cards' } } }
  }
});

// ===== 6. Key Findings (dynamic) =====
(function() {
  const o = DATA.overall;
  const dow = DATA.dow;
  const bestDay = dow.reduce((a,b) => a.mae < b.mae ? a : b);
  const worstDay = dow.reduce((a,b) => a.mae > b.mae ? a : b);
  const bestLoc = DATA.bestLocs[0];
  const worstLoc = DATA.worstLocs[0];

  // Calculate improvement percentages
  const maeImpV1 = ((1 - o.MAE / 66) * 100).toFixed(1);
  const maeImpV2 = ((1 - o.MAE / 62.19) * 100).toFixed(1);
  const r2ImpV1 = ((o.R2 - 0.799) * 100).toFixed(1);

  const findings = [
    {
      title: "Significant Improvement Over Prior Models",
      text: `V3 achieves MAE of ${o.MAE} cards — a <strong>${maeImpV1}% reduction</strong> from V1 (66) and <strong>${maeImpV2}% reduction</strong> from V2 (62.19). R² improved by ${r2ImpV1} percentage points over V1.`,
      type: "good"
    },
    {
      title: "Bias Characteristics",
      text: `Average bias is <strong>${o.Bias > 0 ? '+' : ''}${o.Bias.toFixed(1)} cards</strong> — the model ${o.Bias > 0 ? 'slightly over-predicts' : 'slightly under-predicts'} on average. Over-prediction rate: ${o.OverPredictPct}%. Total excess across all predictions: ${o.TotalExcess.toLocaleString()} cards.`,
      type: o.Bias > 10 ? "warning" : "good"
    },
    {
      title: "Best Day: " + bestDay.day,
      text: `${bestDay.day} has the lowest MAE at <strong>${bestDay.mae}</strong> cards (avg actual: ${bestDay.meanActual.toFixed(0)} cards). Bias: ${bestDay.bias > 0 ? '+' : ''}${bestDay.bias.toFixed(1)} cards.`,
      type: "good"
    },
    {
      title: "Most Challenging Day: " + worstDay.day,
      text: `${worstDay.day} has the highest MAE at <strong>${worstDay.mae}</strong> cards (avg actual: ${worstDay.meanActual.toFixed(0)} cards). This is a ${((worstDay.mae / bestDay.mae - 1) * 100).toFixed(0)}% higher error rate than ${bestDay.day}.`,
      type: "warning"
    },
    {
      title: "Best Location: " + bestLoc.location,
      text: `MAE of only <strong>${bestLoc.mae}</strong> cards with R² of ${bestLoc.r2} (avg actual: ${bestLoc.meanActual} cards/day).`,
      type: "good"
    },
    {
      title: "Most Challenging Location: " + worstLoc.location,
      text: `MAE of <strong>${worstLoc.mae}</strong> cards (avg actual: ${worstLoc.meanActual} cards/day). Bias: ${worstLoc.bias > 0 ? '+' : ''}${worstLoc.bias.toFixed(1)} cards.`,
      type: "warning"
    },
    {
      title: "Percentage Error Performance",
      text: `Median APE is <strong>${o.MedianAPE}%</strong> — meaning for the typical prediction, the model is off by about ${o.MedianAPE}% of actual sales. MAPE of ${o.MAPE}% is pulled higher by low-volume days.`,
      type: o.MedianAPE < 20 ? "good" : "warning"
    },
  ];

  let html = '<div style="display:grid;gap:16px">';
  findings.forEach(f => {
    const borderColor = f.type === 'good' ? 'var(--green)' : f.type === 'warning' ? 'var(--amber)' : 'var(--red)';
    const icon = f.type === 'good' ? '✅' : f.type === 'warning' ? '⚠️' : '❌';
    html += `<div style="border-left:3px solid ${borderColor};padding:12px 16px;background:rgba(0,0,0,0.2);border-radius:0 8px 8px 0">
      <div style="font-weight:700;margin-bottom:4px">${icon} ${f.title}</div>
      <div style="color:var(--muted);font-size:0.9rem">${f.text}</div>
    </div>`;
  });
  html += '</div>';
  document.getElementById('findings').innerHTML = html;
})();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 14. Write HTML
# ---------------------------------------------------------------------------

output_path = os.path.join(SCRIPT_DIR, "v3_lag_analysis.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nHTML dashboard saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")
