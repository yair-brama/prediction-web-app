import json, os

import tempfile
tmp_path = os.path.join(tempfile.gettempdir(), 'analysis_data.json')
with open(tmp_path) as f:
    data = json.load(f)

s = data['summary']
scatter = data['scatter']
daily = data['daily']
dow = data['dow']
weeks = data['weeks']
locs_best = data['locsBest']
locs_worst = data['locsWorst']
err_dist = data['errDist']
worst = data['worst']

# Build table rows
err_rows = ""
for e in err_dist:
    err_rows += f'<tr><td>&le; {e["threshold"]}%</td><td class="num">{e["count"]:,}</td><td class="num">{e["pct"]}%</td><td><div class="bar-container"><div class="bar-bg"><div class="bar" style="width:{e["pct"]}%"></div></div></div></td></tr>\n'

dow_rows = ""
for d in dow:
    badge = "badge-green" if d["mape"] < 30 else ("badge-orange" if d["mape"] < 60 else "badge-red")
    bias_color = "var(--orange)" if d["bias"] > 0 else "var(--accent)"
    sign = "+" if d["bias"] > 0 else ""
    dow_rows += f'<tr><td>{d["day"]}</td><td class="num">{d["mae"]}</td><td class="num"><span class="badge {badge}">{d["mape"]}%</span></td><td class="num" style="color:{bias_color}">{sign}{d["bias"]}</td><td class="num">{d["avgActual"]}</td><td class="num">{d["avgPred"]}</td></tr>\n'

week_rows = ""
for w in weeks:
    badge = "badge-green" if w["mape"] < 30 else ("badge-orange" if w["mape"] < 60 else "badge-red")
    week_rows += f'<tr><td>Wk {w["week"]}</td><td class="num">{w["mae"]}</td><td class="num"><span class="badge {badge}">{w["mape"]}%</span></td><td class="num" style="color:var(--orange)">+{w["bias"]}</td><td class="num">{w["avgActual"]}</td><td class="num">{w["avgPred"]}</td><td class="num">{w["count"]}</td></tr>\n'

best_rows = ""
for l in locs_best:
    bias_color = "var(--orange)" if l["bias"] > 0 else "var(--accent)"
    sign = "+" if l["bias"] > 0 else ""
    best_rows += f'<tr><td>Loc {l["loc"]}</td><td class="num">{l["mae"]}</td><td class="num"><span class="badge badge-green">{l["mape"]}%</span></td><td class="num" style="color:{bias_color}">{sign}{l["bias"]}</td></tr>\n'

worst_loc_rows = ""
for l in locs_worst:
    bias_color = "var(--orange)" if l["bias"] > 0 else "var(--accent)"
    sign = "+" if l["bias"] > 0 else ""
    worst_loc_rows += f'<tr><td>Loc {l["loc"]}</td><td class="num">{l["mae"]}</td><td class="num"><span class="badge badge-red">{l["mape"]}%</span></td><td class="num" style="color:{bias_color}">{sign}{l["bias"]}</td></tr>\n'

worst_rows = ""
for w in worst:
    worst_rows += f'<tr><td>Loc {w["loc"]}</td><td>{w["date"]}</td><td class="num">{w["pred"]:,}</td><td class="num">{w["actual"]:,}</td><td class="num" style="color:var(--red)">+{w["error"]:,}</td><td class="num"><span class="badge badge-red">+{w["pctErr"]}%</span></td></tr>\n'

over_pct = round(s['overPred']/s['totalRows']*100, 1)
under_pct = round(s['underPred']/s['totalRows']*100, 1)
exact_pct = round(s['exactMatch']/s['totalRows']*100, 1)
excess = s['predSum'] - s['actualSum']
excess_pct = round(excess / s['actualSum'] * 100, 1)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predicted vs Actual Cards - Model Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #3b82f6; --accent2: #22d3ee;
    --green: #22c55e; --red: #ef4444; --orange: #f59e0b;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,-apple-system,sans-serif; padding:24px; line-height:1.6; }}
  h1 {{ font-size:28px; font-weight:700; margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:14px; margin-bottom:32px; }}
  h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; color:var(--accent2); }}
  h3 {{ font-size:15px; font-weight:600; margin-bottom:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }}
  .grid {{ display:grid; gap:20px; margin-bottom:24px; }}
  .grid-4 {{ grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); }}
  .grid-2 {{ grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); }}
  .grid-3 {{ grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  .metric {{ text-align:center; }}
  .metric .value {{ font-size:32px; font-weight:700; }}
  .metric .label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-top:4px; }}
  .metric .sub {{ font-size:13px; color:var(--muted); margin-top:2px; }}
  .value.green {{ color:var(--green); }}
  .value.red {{ color:var(--red); }}
  .value.blue {{ color:var(--accent); }}
  .value.orange {{ color:var(--orange); }}
  .value.cyan {{ color:var(--accent2); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; padding:10px 12px; border-bottom:2px solid var(--border); color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:0.5px; }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
  tr:hover td {{ background:rgba(59,130,246,0.05); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  th.num {{ text-align:right; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }}
  .badge-green {{ background:rgba(34,197,94,0.15); color:var(--green); }}
  .badge-red {{ background:rgba(239,68,68,0.15); color:var(--red); }}
  .badge-orange {{ background:rgba(245,158,11,0.15); color:var(--orange); }}
  .bar-container {{ display:flex; align-items:center; gap:8px; }}
  .bar {{ height:8px; border-radius:4px; background:var(--accent); transition:width 0.3s; }}
  .bar-bg {{ flex:1; height:8px; border-radius:4px; background:var(--border); }}
  .chart-container {{ position:relative; height:300px; }}
  .takeaway {{ padding:16px 20px; border-left:3px solid var(--accent); background:rgba(59,130,246,0.08); border-radius:0 8px 8px 0; margin-bottom:12px; font-size:14px; }}
  .takeaway strong {{ color:var(--accent2); }}
  .section {{ margin-bottom:40px; }}
  .over-under {{ display:flex; gap:4px; height:24px; border-radius:6px; overflow:hidden; margin-top:8px; }}
  .over-under .seg {{ display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; }}
</style>
</head>
<body>

<h1>Predicted vs Actual Purchased Cards</h1>
<p class="subtitle">Model holdout set analysis &middot; {s['totalRows']:,} observations &middot; Feb 4 &ndash; Mar 6, 2026</p>

<div class="section">
  <div class="grid grid-4">
    <div class="card metric"><div class="value blue">{s['mae']}</div><div class="label">Mean Absolute Error</div><div class="sub">Avg cards off per prediction</div></div>
    <div class="card metric"><div class="value cyan">{s['medianApe']}%</div><div class="label">Median Abs % Error</div><div class="sub">MAPE: {s['mape']}%</div></div>
    <div class="card metric"><div class="value orange">+{s['bias']}</div><div class="label">Mean Bias</div><div class="sub">Systematic over-prediction</div></div>
    <div class="card metric"><div class="value green">{s['corr']}</div><div class="label">Correlation</div><div class="sub">R&sup2; = {s['r2']}</div></div>
  </div>
</div>

<div class="section">
  <h2>Summary Statistics</h2>
  <div class="grid grid-3">
    <div class="card">
      <h3>Descriptive</h3>
      <table>
        <tr><th></th><th class="num">Predicted</th><th class="num">Actual</th></tr>
        <tr><td>Mean</td><td class="num">{s['predMean']}</td><td class="num">{s['actualMean']}</td></tr>
        <tr><td>Median</td><td class="num">{s['predMedian']}</td><td class="num">{s['actualMedian']}</td></tr>
        <tr><td>Total</td><td class="num">{s['predSum']:,}</td><td class="num">{s['actualSum']:,}</td></tr>
      </table>
      <p style="margin-top:12px;font-size:13px;color:var(--muted)">Total over-prediction: <strong style="color:var(--orange)">+{excess:,}</strong> cards ({excess_pct}%)</p>
    </div>
    <div class="card">
      <h3>Error Metrics</h3>
      <table>
        <tr><td>MAE</td><td class="num">{s['mae']}</td></tr>
        <tr><td>RMSE</td><td class="num">{s['rmse']}</td></tr>
        <tr><td>MAPE</td><td class="num">{s['mape']}%</td></tr>
        <tr><td>Median APE</td><td class="num">{s['medianApe']}%</td></tr>
        <tr><td>Bias</td><td class="num">+{s['bias']}</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>Prediction Direction</h3>
      <div class="over-under">
        <div class="seg" style="width:{over_pct}%;background:var(--orange);">{over_pct:.0f}% Over</div>
        <div class="seg" style="width:{under_pct}%;background:var(--accent);">{under_pct:.0f}% Under</div>
        <div class="seg" style="width:{exact_pct}%;background:var(--green);"></div>
      </div>
      <table style="margin-top:12px">
        <tr><td>Over-predicted</td><td class="num" style="color:var(--orange)">{s['overPred']:,} ({over_pct}%)</td></tr>
        <tr><td>Under-predicted</td><td class="num" style="color:var(--accent)">{s['underPred']:,} ({under_pct}%)</td></tr>
        <tr><td>Exact match</td><td class="num" style="color:var(--green)">{s['exactMatch']} ({exact_pct}%)</td></tr>
      </table>
    </div>
  </div>
</div>

<div class="section">
  <h2>Visual Analysis</h2>
  <div class="grid grid-2">
    <div class="card"><h3>Predicted vs Actual (Scatter)</h3><div class="chart-container"><canvas id="scatterChart"></canvas></div></div>
    <div class="card"><h3>Daily Totals Over Time</h3><div class="chart-container"><canvas id="timeChart"></canvas></div></div>
  </div>
</div>

<div class="section">
  <h2>Error Distribution</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Cumulative Accuracy</h3>
      <table>
        <tr><th>Within Threshold</th><th class="num">Rows</th><th class="num">%</th><th style="width:40%">Coverage</th></tr>
        {err_rows}
      </table>
    </div>
    <div class="card"><h3>Error Distribution</h3><div class="chart-container"><canvas id="errDistChart"></canvas></div></div>
  </div>
</div>

<div class="section">
  <h2>Performance by Day of Week</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Metrics Table</h3>
      <table>
        <tr><th>Day</th><th class="num">MAE</th><th class="num">MAPE</th><th class="num">Bias</th><th class="num">Avg Actual</th><th class="num">Avg Pred</th></tr>
        {dow_rows}
      </table>
    </div>
    <div class="card"><h3>MAE &amp; Bias by Day</h3><div class="chart-container"><canvas id="dowChart"></canvas></div></div>
  </div>
</div>

<div class="section">
  <h2>Performance by Week</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Weekly Metrics</h3>
      <table>
        <tr><th>Week</th><th class="num">MAE</th><th class="num">MAPE</th><th class="num">Bias</th><th class="num">Avg Actual</th><th class="num">Avg Pred</th><th class="num">Count</th></tr>
        {week_rows}
      </table>
    </div>
    <div class="card"><h3>Weekly Comparison</h3><div class="chart-container"><canvas id="weekChart"></canvas></div></div>
  </div>
</div>

<div class="section">
  <h2>Performance by Location</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Top 10 Best (Lowest MAPE)</h3>
      <table>
        <tr><th>Location</th><th class="num">MAE</th><th class="num">MAPE</th><th class="num">Bias</th></tr>
        {best_rows}
      </table>
    </div>
    <div class="card">
      <h3>Top 10 Worst (Highest MAPE)</h3>
      <table>
        <tr><th>Location</th><th class="num">MAE</th><th class="num">MAPE</th><th class="num">Bias</th></tr>
        {worst_loc_rows}
      </table>
    </div>
  </div>
</div>

<div class="section">
  <h2>Largest Individual Errors</h2>
  <div class="card">
    <table>
      <tr><th>Location</th><th>Date</th><th class="num">Predicted</th><th class="num">Actual</th><th class="num">Error</th><th class="num">% Error</th></tr>
      {worst_rows}
    </table>
  </div>
</div>

<div class="section">
  <h2>Key Takeaways</h2>
  <div class="takeaway"><strong>1. Systematic over-prediction bias</strong> &mdash; The model over-predicts 73.5% of the time, totaling +{excess:,} excess cards predicted ({excess_pct}% above actual).</div>
  <div class="takeaway"><strong>2. Feb 22 was a catastrophic miss</strong> &mdash; 8 of the top 15 largest errors occurred on this date. The model likely missed a holiday/event that suppressed actual sales (Presidents' Day weekend).</div>
  <div class="takeaway"><strong>3. Weekend predictions are weakest</strong> &mdash; Saturday has the highest MAE (175) with a large upward bias (+151). Sunday is well-calibrated (bias near 0).</div>
  <div class="takeaway"><strong>4. Strong correlation (0.948)</strong> &mdash; The model captures the right patterns with R&sup2; = 0.899, but needs downward calibration to correct the systematic bias.</div>
  <div class="takeaway"><strong>5. Low-volume locations inflate MAPE</strong> &mdash; The worst MAPE locations (Loc 143: 660%, Loc 80: 485%) are low-volume, where small absolute errors produce large percentage errors.</div>
</div>

<script>
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = "'Segoe UI',system-ui,sans-serif";

var scatterData = {json.dumps(scatter)};
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      data: scatterData.map(function(d) {{ return {{x: d[0], y: d[1]}}; }}),
      backgroundColor: 'rgba(59,130,246,0.3)',
      borderColor: 'rgba(59,130,246,0.6)',
      pointRadius: 2,
    }}, {{
      type: 'line',
      data: [{{x:0,y:0}},{{x:2000,y:2000}}],
      borderColor: 'rgba(239,68,68,0.5)',
      borderDash: [5,5],
      borderWidth: 2,
      pointRadius: 0,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display:true, text:'Actual Cards', color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }},
      y: {{ title: {{ display:true, text:'Predicted Cards', color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }}
    }}
  }}
}});

var dailyData = {json.dumps(daily)};
new Chart(document.getElementById('timeChart'), {{
  type: 'line',
  data: {{
    labels: dailyData.map(function(d) {{ return d[0]; }}),
    datasets: [{{
      label: 'Predicted',
      data: dailyData.map(function(d) {{ return d[1]; }}),
      borderColor: '#3b82f6',
      borderWidth: 2, pointRadius: 2, fill: false,
    }}, {{
      label: 'Actual',
      data: dailyData.map(function(d) {{ return d[2]; }}),
      borderColor: '#22c55e',
      borderWidth: 2, pointRadius: 2, fill: false,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ usePointStyle: true }} }} }},
    scales: {{
      x: {{ grid: {{ color:'#1e293b' }}, ticks: {{ maxTicksLimit:10 }} }},
      y: {{ grid: {{ color:'#1e293b' }}, title: {{ display:true, text:'Total Cards', color:'#94a3b8' }} }}
    }}
  }}
}});

var errDistData = {json.dumps(err_dist)};
new Chart(document.getElementById('errDistChart'), {{
  type: 'bar',
  data: {{
    labels: errDistData.map(function(d) {{ return '\u2264'+d.threshold+'%'; }}),
    datasets: [{{
      data: errDistData.map(function(d) {{ return d.pct; }}),
      backgroundColor: errDistData.map(function(d) {{ return d.threshold<=25 ? 'rgba(34,197,94,0.6)' : 'rgba(59,130,246,0.6)'; }}),
      borderColor: errDistData.map(function(d) {{ return d.threshold<=25 ? '#22c55e' : '#3b82f6'; }}),
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ display:false }} }},
      y: {{ grid: {{ color:'#1e293b' }}, title: {{ display:true, text:'% of Predictions', color:'#94a3b8' }}, max:100 }}
    }}
  }}
}});

var dowLabels = {json.dumps([d['day'][:3] for d in dow])};
var dowMAE = {json.dumps([d['mae'] for d in dow])};
var dowBias = {json.dumps([d['bias'] for d in dow])};
new Chart(document.getElementById('dowChart'), {{
  type: 'bar',
  data: {{
    labels: dowLabels,
    datasets: [{{
      label: 'MAE', data: dowMAE,
      backgroundColor: 'rgba(59,130,246,0.6)', borderColor: '#3b82f6', borderWidth: 1,
    }}, {{
      label: 'Bias', data: dowBias,
      backgroundColor: 'rgba(245,158,11,0.6)', borderColor: '#f59e0b', borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ usePointStyle: true }} }} }},
    scales: {{
      x: {{ grid: {{ display:false }} }},
      y: {{ grid: {{ color:'#1e293b' }}, title: {{ display:true, text:'Cards', color:'#94a3b8' }} }}
    }}
  }}
}});

var weekLabels = {json.dumps(['Wk '+str(w['week']) for w in weeks])};
var weekPred = {json.dumps([w['avgPred'] for w in weeks])};
var weekActual = {json.dumps([w['avgActual'] for w in weeks])};
new Chart(document.getElementById('weekChart'), {{
  type: 'bar',
  data: {{
    labels: weekLabels,
    datasets: [{{
      label: 'Avg Predicted', data: weekPred,
      backgroundColor: 'rgba(59,130,246,0.6)', borderColor: '#3b82f6', borderWidth: 1,
    }}, {{
      label: 'Avg Actual', data: weekActual,
      backgroundColor: 'rgba(34,197,94,0.6)', borderColor: '#22c55e', borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ usePointStyle: true }} }} }},
    scales: {{
      x: {{ grid: {{ display:false }} }},
      y: {{ grid: {{ color:'#1e293b' }}, title: {{ display:true, text:'Avg Cards', color:'#94a3b8' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

with open(os.path.join('C:/Users/yairb/Downloads', 'model_analysis.html'), 'w', encoding='utf-8') as f:
    f.write(html)
print("Done! Wrote model_analysis.html")
