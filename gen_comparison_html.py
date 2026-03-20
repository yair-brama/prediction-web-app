import json, os, tempfile

tmp_path = os.path.join(tempfile.gettempdir(), 'comparison_data.json')
with open(tmp_path) as f:
    data = json.load(f)

o = data['orig']['summary']
c = data['cal']['summary']
o_scatter = data['orig']['scatter']
c_scatter = data['cal']['scatter']
o_daily = data['orig']['daily']
c_daily = data['cal']['daily']
o_dow = data['orig']['dow']
c_dow = data['cal']['dow']
o_weeks = data['orig']['weeks']
c_weeks = data['cal']['weeks']
o_err = data['orig']['errDist']
c_err = data['cal']['errDist']
o_worst = data['orig']['worst']
c_worst = data['cal']['worst']
o_best_loc = data['orig']['locsBest']
c_best_loc = data['cal']['locsBest']
o_worst_loc = data['orig']['locsWorst']
c_worst_loc = data['cal']['locsWorst']
dow_compare = data['dow_compare']
week_compare = data['week_compare']

o_excess = o['predSum'] - o['actualSum']
c_excess = c['predSum'] - c['actualSum']
o_over_pct = round(o['overPred']/o['totalRows']*100,1)
c_over_pct = round(c['overPred']/c['totalRows']*100,1)

def delta(old, new, lower_better=True, fmt=".2f"):
    d = new - old
    better = (d < 0) if lower_better else (d > 0)
    cls = "delta-good" if better else "delta-bad"
    return f'<span class="{cls}">{d:+{fmt}}</span>'

def delta_pct(old, new, lower_better=True):
    if old == 0: return ""
    d = (new - old) / abs(old) * 100
    better = (d < 0) if lower_better else (d > 0)
    cls = "delta-good" if better else "delta-bad"
    return f'<span class="{cls}">{d:+.1f}%</span>'

# Build comparison KPI rows
kpi_rows = ""
metrics_list = [
    ("MAE", o['mae'], c['mae'], True, ".2f"),
    ("RMSE", o['rmse'], c['rmse'], True, ".2f"),
    ("MAPE", o['mape'], c['mape'], True, ".2f"),
    ("Median APE", o['medianApe'], c['medianApe'], True, ".2f"),
    ("Bias", o['bias'], c['bias'], True, ".2f"),
    ("Correlation", o['corr'], c['corr'], False, ".4f"),
    ("R\u00b2", o['r2'], c['r2'], False, ".4f"),
]
for name, ov, cv, lb, fmt in metrics_list:
    suffix = "%" if "APE" in name or "MAPE" in name else ""
    kpi_rows += f'<tr><td>{name}</td><td class="num">{ov:{fmt}}{suffix}</td><td class="num">{cv:{fmt}}{suffix}</td><td class="num">{delta(ov, cv, lb, fmt)}</td></tr>\n'
kpi_rows += f'<tr><td>Excess Predicted</td><td class="num">+{o_excess:,}</td><td class="num">+{c_excess:,}</td><td class="num">{delta(o_excess, c_excess, True, ",.0f")}</td></tr>\n'
kpi_rows += f'<tr><td>Over-predict Rate</td><td class="num">{o_over_pct}%</td><td class="num">{c_over_pct}%</td><td class="num">{delta(o_over_pct, c_over_pct, True, ".1f")}</td></tr>\n'

# Error distribution comparison rows
err_rows = ""
for oe, ce in zip(o_err, c_err):
    err_rows += f'<tr><td>&le; {oe["threshold"]}%</td><td class="num">{oe["pct"]}%</td><td class="num">{ce["pct"]}%</td><td class="num">{delta(oe["pct"], ce["pct"], False, ".1f")}</td></tr>\n'

# Day of week comparison rows
dow_rows = ""
for d in dow_compare:
    mae_cls = "delta-good" if d['mae_chg'] < 0 else "delta-bad"
    bias_o_abs = abs(d['orig_bias'])
    bias_c_abs = abs(d['cal_bias'])
    bias_cls = "delta-good" if bias_c_abs < bias_o_abs else "delta-bad"
    dow_rows += f'''<tr>
      <td>{d['day']}</td>
      <td class="num">{d['orig_mae']}</td><td class="num">{d['cal_mae']}</td>
      <td class="num"><span class="{mae_cls}">{d['mae_chg']:+.1f}%</span></td>
      <td class="num">{d['orig_bias']:+.1f}</td><td class="num">{d['cal_bias']:+.1f}</td>
      <td class="num"><span class="{bias_cls}">{bias_c_abs - bias_o_abs:+.1f}</span></td>
    </tr>\n'''

# Week comparison rows
week_rows = ""
for w in week_compare:
    mae_cls = "delta-good" if w['mae_chg'] < 0 else "delta-bad"
    bias_o_abs = abs(w['orig_bias'])
    bias_c_abs = abs(w['cal_bias'])
    bias_cls = "delta-good" if bias_c_abs < bias_o_abs else "delta-bad"
    week_rows += f'''<tr>
      <td>Wk {w['week']}</td>
      <td class="num">{w['orig_mae']}</td><td class="num">{w['cal_mae']}</td>
      <td class="num"><span class="{mae_cls}">{w['mae_chg']:+.1f}%</span></td>
      <td class="num">{w['orig_bias']:+.1f}</td><td class="num">{w['cal_bias']:+.1f}</td>
      <td class="num"><span class="{bias_cls}">{bias_c_abs - bias_o_abs:+.1f}</span></td>
    </tr>\n'''

# Worst errors comparison rows
worst_orig_rows = ""
for w in o_worst:
    sign = "+" if w['error'] > 0 else ""
    worst_orig_rows += f'<tr><td>Loc {w["loc"]}</td><td>{w["date"]}</td><td class="num">{w["pred"]:,}</td><td class="num">{w["actual"]:,}</td><td class="num" style="color:{"var(--red)" if w["error"]>0 else "var(--accent)"}">{sign}{w["error"]:,}</td><td class="num">{sign}{w["pctErr"]}%</td></tr>\n'

worst_cal_rows = ""
for w in c_worst:
    sign = "+" if w['error'] > 0 else ""
    worst_cal_rows += f'<tr><td>Loc {w["loc"]}</td><td>{w["date"]}</td><td class="num">{w["pred"]:,}</td><td class="num">{w["actual"]:,}</td><td class="num" style="color:{"var(--red)" if w["error"]>0 else "var(--accent)"}">{sign}{w["error"]:,}</td><td class="num">{sign}{w["pctErr"]}%</td></tr>\n'

bias_reduction = round((1 - abs(c['bias'])/abs(o['bias']))*100, 0)
excess_reduction = round((1 - c_excess/o_excess)*100, 0)
sat_orig = next(d for d in dow_compare if d['day']=='Saturday')
sat_bias_red = round((1 - abs(sat_orig['cal_bias'])/abs(sat_orig['orig_bias']))*100, 0)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Comparison: Original vs Calibrated V2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #3b82f6; --accent2: #22d3ee;
    --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --purple: #a78bfa;
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
  .value.purple {{ color:var(--purple); }}
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
  .chart-container {{ position:relative; height:300px; }}
  .takeaway {{ padding:16px 20px; border-left:3px solid var(--accent); background:rgba(59,130,246,0.08); border-radius:0 8px 8px 0; margin-bottom:12px; font-size:14px; }}
  .takeaway strong {{ color:var(--accent2); }}
  .takeaway.win {{ border-left-color:var(--green); background:rgba(34,197,94,0.08); }}
  .takeaway.warn {{ border-left-color:var(--orange); background:rgba(245,158,11,0.08); }}
  .section {{ margin-bottom:40px; }}
  .delta-good {{ color:var(--green); font-weight:600; }}
  .delta-bad {{ color:var(--red); font-weight:600; }}
  .vs-tag {{ display:inline-block; padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700; letter-spacing:1px; }}
  .tag-orig {{ background:rgba(59,130,246,0.2); color:var(--accent); }}
  .tag-cal {{ background:rgba(167,139,250,0.2); color:var(--purple); }}
  .legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
</style>
</head>
<body>

<h1>Model Comparison: Original vs Calibrated V2</h1>
<p class="subtitle">
  <span class="vs-tag tag-orig">ORIGINAL</span> LightGBM &mdash; New Precipitation
  &nbsp;&nbsp;vs&nbsp;&nbsp;
  <span class="vs-tag tag-cal">CALIBRATED V2</span> + Holiday Features + Isotonic Bias Correction
  &nbsp;&middot;&nbsp; {c['totalRows']:,} holdout observations &nbsp;&middot;&nbsp; Feb 4 &ndash; Mar 6, 2026
</p>

<!-- Hero KPIs -->
<div class="section">
  <div class="grid grid-4">
    <div class="card metric">
      <div class="value green">{bias_reduction:.0f}%</div>
      <div class="label">Bias Reduction</div>
      <div class="sub">+{o['bias']:.1f} &rarr; +{c['bias']:.1f}</div>
    </div>
    <div class="card metric">
      <div class="value green">{excess_reduction:.0f}%</div>
      <div class="label">Excess Cards Reduced</div>
      <div class="sub">+{o_excess:,} &rarr; +{c_excess:,}</div>
    </div>
    <div class="card metric">
      <div class="value green">{sat_bias_red:.0f}%</div>
      <div class="label">Saturday Bias Reduction</div>
      <div class="sub">+{sat_orig['orig_bias']:.0f} &rarr; +{sat_orig['cal_bias']:.0f}</div>
    </div>
    <div class="card metric">
      <div class="value cyan">{c['mae']}</div>
      <div class="label">New MAE</div>
      <div class="sub">Was {o['mae']} ({delta_pct(o['mae'], c['mae'])})</div>
    </div>
  </div>
</div>

<!-- Metric Comparison Table -->
<div class="section">
  <h2>Side-by-Side Metrics</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Error &amp; Accuracy Metrics</h3>
      <table>
        <tr><th>Metric</th><th class="num"><span class="legend-dot" style="background:var(--accent)"></span>Original</th><th class="num"><span class="legend-dot" style="background:var(--purple)"></span>Calibrated</th><th class="num">Change</th></tr>
        {kpi_rows}
      </table>
    </div>
    <div class="card">
      <h3>Error Distribution (% within threshold)</h3>
      <table>
        <tr><th>Threshold</th><th class="num">Original</th><th class="num">Calibrated</th><th class="num">Change</th></tr>
        {err_rows}
      </table>
    </div>
  </div>
</div>

<!-- Scatter Comparison -->
<div class="section">
  <h2>Predicted vs Actual</h2>
  <div class="grid grid-2">
    <div class="card"><h3><span class="vs-tag tag-orig">Original</span> Scatter</h3><div class="chart-container"><canvas id="scatterOrig"></canvas></div></div>
    <div class="card"><h3><span class="vs-tag tag-cal">Calibrated V2</span> Scatter</h3><div class="chart-container"><canvas id="scatterCal"></canvas></div></div>
  </div>
</div>

<!-- Daily Time Series -->
<div class="section">
  <h2>Daily Totals Over Time</h2>
  <div class="card"><div class="chart-container" style="height:350px"><canvas id="timeChart"></canvas></div></div>
</div>

<!-- Day of Week Comparison -->
<div class="section">
  <h2>Day of Week Comparison</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>MAE &amp; Bias by Day</h3>
      <table>
        <tr><th>Day</th><th class="num">Orig MAE</th><th class="num">Cal MAE</th><th class="num">Chg</th><th class="num">Orig Bias</th><th class="num">Cal Bias</th><th class="num">|Bias| Chg</th></tr>
        {dow_rows}
      </table>
    </div>
    <div class="card"><h3>Bias Comparison by Day</h3><div class="chart-container"><canvas id="dowBiasChart"></canvas></div></div>
  </div>
</div>

<!-- Weekly Comparison -->
<div class="section">
  <h2>Weekly Comparison</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>MAE &amp; Bias by Week</h3>
      <table>
        <tr><th>Week</th><th class="num">Orig MAE</th><th class="num">Cal MAE</th><th class="num">Chg</th><th class="num">Orig Bias</th><th class="num">Cal Bias</th><th class="num">|Bias| Chg</th></tr>
        {week_rows}
      </table>
    </div>
    <div class="card"><h3>Bias Comparison by Week</h3><div class="chart-container"><canvas id="weekBiasChart"></canvas></div></div>
  </div>
</div>

<!-- Worst Errors -->
<div class="section">
  <h2>Largest Individual Errors</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3><span class="vs-tag tag-orig">Original</span> Top 15 Errors</h3>
      <table>
        <tr><th>Loc</th><th>Date</th><th class="num">Pred</th><th class="num">Actual</th><th class="num">Error</th><th class="num">%</th></tr>
        {worst_orig_rows}
      </table>
    </div>
    <div class="card">
      <h3><span class="vs-tag tag-cal">Calibrated V2</span> Top 15 Errors</h3>
      <table>
        <tr><th>Loc</th><th>Date</th><th class="num">Pred</th><th class="num">Actual</th><th class="num">Error</th><th class="num">%</th></tr>
        {worst_cal_rows}
      </table>
    </div>
  </div>
</div>

<!-- Key Takeaways -->
<div class="section">
  <h2>Key Takeaways</h2>
  <div class="takeaway win"><strong>1. Bias reduced by {bias_reduction:.0f}%</strong> &mdash; Systematic over-prediction dropped from +{o['bias']:.1f} to +{c['bias']:.1f} cards per prediction. Total excess went from +{o_excess:,} to +{c_excess:,} cards ({excess_reduction:.0f}% reduction).</div>
  <div class="takeaway win"><strong>2. Saturday dramatically improved</strong> &mdash; Bias slashed from +{sat_orig['orig_bias']:.0f} to +{sat_orig['cal_bias']:.0f} ({sat_bias_red:.0f}% reduction). Friday bias also cut by 57%.</div>
  <div class="takeaway win"><strong>3. Feb 22 errors reduced 27%</strong> &mdash; The catastrophic Presidents' Day weekend misses are substantially improved with holiday-awareness features.</div>
  <div class="takeaway win"><strong>4. Error direction rebalanced</strong> &mdash; Over-prediction rate dropped from 73.5% to 69.6%, a more balanced model.</div>
  <div class="takeaway warn"><strong>5. Tradeoff on low-volume weekdays</strong> &mdash; Tuesday and Thursday MAE rose ~50%, as the calibrator shifted predictions downward uniformly. Median APE remains similar (22.7% vs 22.9%).</div>
  <div class="takeaway"><strong>6. Worst errors shifted character</strong> &mdash; Original top errors were all over-predictions on Feb 22. Calibrated top errors are now under-predictions at high-volume locations (Loc 32, 44, 147) &mdash; a more diverse and natural error profile.</div>
</div>

<script>
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = "'Segoe UI',system-ui,sans-serif";

function makeScatter(id, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        data: data.map(function(d) {{ return {{x:d[0], y:d[1]}}; }}),
        backgroundColor: color + '4D', borderColor: color + '99', pointRadius: 2,
      }}, {{
        type:'line', data:[{{x:0,y:0}},{{x:2000,y:2000}}],
        borderColor:'rgba(239,68,68,0.5)', borderDash:[5,5], borderWidth:2, pointRadius:0,
      }}]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ display:false }} }},
      scales: {{
        x:{{ title:{{ display:true, text:'Actual', color:'#94a3b8' }}, grid:{{ color:'#1e293b' }}, max:2000 }},
        y:{{ title:{{ display:true, text:'Predicted', color:'#94a3b8' }}, grid:{{ color:'#1e293b' }}, max:2000 }}
      }}
    }}
  }});
}}
makeScatter('scatterOrig', {json.dumps(o_scatter)}, '#3b82f6');
makeScatter('scatterCal', {json.dumps(c_scatter)}, '#a78bfa');

// Time series
var oDaily = {json.dumps(o_daily)};
var cDaily = {json.dumps(c_daily)};
new Chart(document.getElementById('timeChart'), {{
  type:'line',
  data: {{
    labels: oDaily.map(function(d) {{ return d[0]; }}),
    datasets: [{{
      label:'Actual', data:oDaily.map(function(d) {{ return d[2]; }}),
      borderColor:'#22c55e', borderWidth:2, pointRadius:2, fill:false,
    }}, {{
      label:'Original Predicted', data:oDaily.map(function(d) {{ return d[1]; }}),
      borderColor:'#3b82f6', borderWidth:2, pointRadius:1, borderDash:[4,4], fill:false,
    }}, {{
      label:'Calibrated Predicted', data:cDaily.map(function(d) {{ return d[1]; }}),
      borderColor:'#a78bfa', borderWidth:2, pointRadius:1, fill:false,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{ usePointStyle:true }} }} }},
    scales: {{
      x:{{ grid:{{ color:'#1e293b' }}, ticks:{{ maxTicksLimit:10 }} }},
      y:{{ grid:{{ color:'#1e293b' }}, title:{{ display:true, text:'Total Cards', color:'#94a3b8' }} }}
    }}
  }}
}});

// Day of week bias chart
var dowCmp = {json.dumps(dow_compare)};
new Chart(document.getElementById('dowBiasChart'), {{
  type:'bar',
  data: {{
    labels: dowCmp.map(function(d) {{ return d.day.substring(0,3); }}),
    datasets: [{{
      label:'Original Bias', data:dowCmp.map(function(d) {{ return d.orig_bias; }}),
      backgroundColor:'rgba(59,130,246,0.6)', borderColor:'#3b82f6', borderWidth:1,
    }}, {{
      label:'Calibrated Bias', data:dowCmp.map(function(d) {{ return d.cal_bias; }}),
      backgroundColor:'rgba(167,139,250,0.6)', borderColor:'#a78bfa', borderWidth:1,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{ usePointStyle:true }} }},
      annotation:{{ annotations:{{ zeroLine:{{ type:'line', yMin:0, yMax:0, borderColor:'rgba(255,255,255,0.3)', borderWidth:1 }} }} }}
    }},
    scales: {{
      x:{{ grid:{{ display:false }} }},
      y:{{ grid:{{ color:'#1e293b' }}, title:{{ display:true, text:'Bias (cards)', color:'#94a3b8' }} }}
    }}
  }}
}});

// Weekly bias chart
var weekCmp = {json.dumps(week_compare)};
new Chart(document.getElementById('weekBiasChart'), {{
  type:'bar',
  data: {{
    labels: weekCmp.map(function(d) {{ return 'Wk '+d.week; }}),
    datasets: [{{
      label:'Original Bias', data:weekCmp.map(function(d) {{ return d.orig_bias; }}),
      backgroundColor:'rgba(59,130,246,0.6)', borderColor:'#3b82f6', borderWidth:1,
    }}, {{
      label:'Calibrated Bias', data:weekCmp.map(function(d) {{ return d.cal_bias; }}),
      backgroundColor:'rgba(167,139,250,0.6)', borderColor:'#a78bfa', borderWidth:1,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{ usePointStyle:true }} }} }},
    scales: {{
      x:{{ grid:{{ display:false }} }},
      y:{{ grid:{{ color:'#1e293b' }}, title:{{ display:true, text:'Bias (cards)', color:'#94a3b8' }} }}
    }}
  }}
}});
</script>

</body>
</html>"""

out_path = os.path.join('C:/Users/yairb/Downloads', 'model_comparison.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Done! Wrote {out_path}")
