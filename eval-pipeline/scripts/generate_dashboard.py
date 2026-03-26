#!/usr/bin/env python3
"""
Generate a static dashboard (dashboard.html) powered by Chart.js
using existing pipeline outputs:
  - eval-pipeline/reports/task_timeline.json (flat task list)
  - eval-pipeline/reports/sprint_summary.json (aggregated sprint metrics)

Features:
  * Row 1: Burndown (remaining_main) + Cumulative Done (cumulative_done_main)
  * Row 2: Sprint main vs sub distribution + Engineer task distribution (current snapshot)
  * Row 3: Status distribution (all tasks) + Sprint completion ratio bar
  * Velocity & forecast (estimated sprints to finish based on average done_main per sprint > 0)

Usage:
  python scripts/generate_dashboard.py \
      --task-json eval-pipeline/reports/task_timeline.json \
      --sprint-json eval-pipeline/reports/sprint_summary.json \
      --out-file eval-pipeline/reports/dashboard.html

If paths omitted, defaults are used (assuming execution from repo root or eval-pipeline/).

Extensibility notes:
  - Add per-status burn-up: derive from task statuses over time once historical snapshots exist.
  - Add forecast ranges: incorporate standard deviation of velocity.
  - Add filtering: convert to a small React/Alpine.js progressive enhancement if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import math
from datetime import datetime
from typing import Any, Dict, List
from string import Template

DEFAULT_TASK_JSON = "eval-pipeline/reports/task_timeline.json"
DEFAULT_SPRINT_JSON = "eval-pipeline/reports/sprint_summary.json"
DEFAULT_OUT_FILE = "eval-pipeline/reports/dashboard.html"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_velocity_and_forecast(sprint_summary: Dict[str, Any]) -> Dict[str, Any]:
  sprints = sprint_summary.get("sprints", [])
  velocities: List[int] = []
  for s in sprints:
    done_main = s.get("done_main", 0)
    if done_main > 0:
      velocities.append(done_main)
  avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
  if len(velocities) >= 2:
    last_two_velocity = sum(velocities[-2:]) / 2
  elif len(velocities) == 1:
    last_two_velocity = velocities[0]
  else:
    last_two_velocity = 0.0

  total_main = sprint_summary.get("totals", {}).get("all_main", 0)
  initial_main = sprints[0].get("main_tasks", 0) if sprints else 0
  last_remaining_main = sprints[-1].get("remaining_main", 0) if sprints else 0

  effective_velocity = last_two_velocity if last_two_velocity > 0 else avg_velocity
  if effective_velocity > 0:
    raw_forecast = last_remaining_main / effective_velocity
    forecast_sprints_ceil = math.ceil(raw_forecast)
  else:
    raw_forecast = None
    forecast_sprints_ceil = None

  burn_pct = 1 - (last_remaining_main / total_main) if total_main > 0 else 0.0

  remaining_sprints_window = 1 if last_remaining_main > 0 else 0
  if effective_velocity > 0 and remaining_sprints_window > 0:
    risk = last_remaining_main > (effective_velocity * remaining_sprints_window)
  else:
    risk = last_remaining_main > 0 and effective_velocity == 0

  # Risk level based on average velocity capacity window
  if last_remaining_main == 0:
    risk_level = "None"
  else:
    if avg_velocity <= 0:
      risk_level = "High"
    else:
      ratio = last_remaining_main / avg_velocity
      if ratio <= 1:
        risk_level = "Low"
      elif ratio <= 2:
        risk_level = "Medium"
      else:
        risk_level = "High"

  return {
    "average_velocity": avg_velocity,
    "last_two_velocity": last_two_velocity,
    "effective_velocity": effective_velocity,
    "forecast_sprints_remaining_raw": raw_forecast,
    "forecast_sprints_remaining": forecast_sprints_ceil,
    "last_remaining_main": last_remaining_main,
    "total_main": total_main,
    "burn_percentage": burn_pct,
    "risk_flag": risk,
    "risk_level": risk_level,
    "initial_main": initial_main,
  }


def extract_status_distribution(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for t in tasks:
        status = t.get("status", "Unknown")
        dist[status] = dist.get(status, 0) + 1
    return dist


def read_flat_tasks(task_json: str) -> List[Dict[str, Any]]:
    if not os.path.exists(task_json):
        raise FileNotFoundError(f"Task JSON not found: {task_json}")
    data = load_json(task_json)
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]  # future proofing if wrapper used
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported task JSON format")


def build_html(tasks: List[Dict[str, Any]], sprint_summary: Dict[str, Any], velocity: Dict[str, Any]) -> str:
    sprints = sprint_summary.get("sprints", [])
    sprint_labels = [s.get("sprint") for s in sprints]
    burndown_remaining = [s.get("remaining_main", 0) for s in sprints]
    cumulative_done = [s.get("cumulative_done_main", 0) for s in sprints]
    completion_ratios = [round(s.get("completion_ratio", 0) * 100, 2) for s in sprints]
    main_vs_sub = [[s.get("main_tasks", 0), s.get("sub_tasks", 0)] for s in sprints]

    engineer_keys: List[str] = []
    for s in sprints:
        for eng in s.get("engineers", {}).keys():
            if eng not in engineer_keys:
                engineer_keys.append(eng)
    engineer_distribution_series = {eng: [] for eng in engineer_keys}
    for s in sprints:
        eng_map = s.get("engineers", {})
        for eng in engineer_keys:
            engineer_distribution_series[eng].append(eng_map.get(eng, 0))

    status_dist = extract_status_distribution(tasks)

    def js(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)

    gen_time = datetime.utcnow().isoformat() + 'Z'
    last_two_vel = velocity["last_two_velocity"]
    effective_velocity = velocity.get("effective_velocity", 0.0)
    avg_velocity = velocity.get("average_velocity", 0.0)
    forecast = velocity.get("forecast_sprints_remaining")
    forecast_text = f"{forecast} sprint(s)" if forecast is not None else "N/A"
    burn_pct = velocity.get("burn_percentage", 0.0) * 100
    remaining_main = velocity.get("last_remaining_main", 0)
    risk_flag = velocity.get("risk_flag", False)
    risk_level = velocity.get("risk_level", "None")
    risk_label = {
        "None": "✅ None",
        "Low": "🟢 Low",
        "Medium": "🟠 Medium",
        "High": "🔴 High",
    }.get(risk_level, risk_level)

    template = Template("""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8" />
<title>LLM Eval Pipeline Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:16px; background:#0f1115; color:#e5e7eb; }
 h1 { margin:0 0 4px; font-size:20px; }
 h2 { font-size:16px; margin:24px 0 8px; }
 .grid { display:grid; gap:20px; }
 .row { display:grid; gap:20px; }
 .row-2cols { grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); }
 .row-3cols { grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); }
 .card { background:#1e2530; padding:16px 18px 14px; border-radius:12px; box-shadow:0 2px 4px rgba(0,0,0,.35); position:relative; }
 .kpi-grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin:12px 0 4px; }
 .kpi { background:#1b2029; border:1px solid #2c3745; border-radius:10px; padding:10px 12px; }
 .kpi h3 { margin:0 0 4px; font-size:12px; font-weight:600; letter-spacing:.5px; text-transform:uppercase; color:#8fa3bf; }
 .kpi p { margin:0; font-size:18px; font-weight:600; }
 canvas { max-width:100%; height:260px !important; }
 a { color:#4ea1ff; }
 footer { margin-top:40px; font-size:12px; color:#64748b; }
</style>
</head>
<body>
<header>
  <h1>LLM Evaluation Pipeline Dashboard</h1>
  <div style="font-size:12px;color:#94a3b8;">Generated at $gen_time</div>
</header>
<section class="kpi-grid">
  <div class="kpi"><h3>Velocity(Last2)</h3><p>$last_two_velocity</p></div>
  <div class="kpi"><h3>Effective Velocity</h3><p>$effective_velocity</p></div>
  <div class="kpi"><h3>Forecast Sprint(ceil)</h3><p>$forecast_text</p></div>
  <div class="kpi"><h3>Burn %</h3><p>$burn_pct%</p></div>
  <div class="kpi"><h3>Risk Level</h3><p>$risk_label</p></div>
  <div class="kpi"><h3>尚餘 Main Tasks</h3><p>$last_remaining_main</p></div>
  <div class="kpi"><h3>總 Main Tasks</h3><p>$total_main</p></div>
</section>
<div class="grid">
  <div class="row row-2cols">
    <div class="card"><h2>Burndown (Remaining Main)</h2><canvas id="burndownChart"></canvas></div>
    <div class="card"><h2>Cumulative Done (Main)</h2><canvas id="cumulativeChart"></canvas></div>
  </div>
  <div class="row row-2cols">
    <div class="card"><h2>Main vs Sub / Sprint</h2><canvas id="mainSubChart"></canvas></div>
    <div class="card"><h2>Engineer Distribution</h2><canvas id="engineerChart"></canvas></div>
  </div>
  <div class="row row-2cols">
    <div class="card"><h2>Status Distribution (All Tasks)</h2><canvas id="statusChart"></canvas></div>
    <div class="card"><h2>Completion % per Sprint</h2><canvas id="completionChart"></canvas></div>
  </div>
</div>
<footer>Static dashboard generated from pipeline artifacts. Extend via scripts/generate_dashboard.py.</footer>
<script>
 const sprintLabels = $sprint_labels;
 const remainingData = $burndown_remaining;
 const cumulativeData = $cumulative_done;
 const completionData = $completion_ratios;
 const mainSub = $main_vs_sub; // [[main, sub], ...]
 const engineerSeries = $engineer_distribution;
 const statusDist = $status_dist;

 function makeChart(id, config) { new Chart(document.getElementById(id), config); }

 const palette = ['#60a5fa','#34d399','#fbbf24','#f87171','#a78bfa','#4ade80','#f472b6','#22d3ee','#c084fc','#fb7185'];

 makeChart('burndownChart', {
   type: 'line',
   data: { labels: sprintLabels, datasets: [{ label: 'Remaining Main', data: remainingData, tension:.25, borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.2)', fill:true, pointRadius:5, pointHoverRadius:7 }]},
   options: { responsive:true, plugins:{ legend:{labels:{color:'#cbd5e1'}}, tooltip:{mode:'index'}}, scales:{ x:{ticks:{color:'#94a3b8'}}, y:{ticks:{color:'#94a3b8'}, beginAtZero:true } } }
 });
 makeChart('cumulativeChart', {
   type: 'line',
   data: { labels: sprintLabels, datasets: [{ label: 'Cumulative Done (Main)', data: cumulativeData, tension:.25, borderColor:'#34d399', backgroundColor:'rgba(52,211,153,0.25)', fill:true, pointRadius:5}]},
   options: { responsive:true, plugins:{ legend:{labels:{color:'#cbd5e1'}}}, scales:{ x:{ticks:{color:'#94a3b8'}}, y:{ticks:{color:'#94a3b8'}, beginAtZero:true } } }
 });
 makeChart('mainSubChart', {
   type: 'bar',
   data: { labels: sprintLabels, datasets: [ { label:'Main', data: mainSub.map(x=>x[0]), backgroundColor:'#3b82f6' }, { label:'Sub', data: mainSub.map(x=>x[1]), backgroundColor:'#64748b' } ] },
   options:{ responsive:true, plugins:{ legend:{labels:{color:'#cbd5e1'}}}, scales:{ x:{stacked:true, ticks:{color:'#94a3b8'}}, y:{stacked:true, ticks:{color:'#94a3b8'}, beginAtZero:true } } }
 });
 const engineerDatasets = Object.keys(engineerSeries).map((eng, idx) => ({ label: eng, data: engineerSeries[eng], backgroundColor: palette[idx % palette.length] }));
 makeChart('engineerChart', { type: 'bar', data:{ labels: sprintLabels, datasets: engineerDatasets }, options:{ responsive:true, plugins:{ legend:{labels:{color:'#cbd5e1'}, position:'bottom'}}, scales:{ x:{stacked:true, ticks:{color:'#94a3b8'}}, y:{stacked:true, ticks:{color:'#94a3b8'}, beginAtZero:true } } } });
 makeChart('statusChart', { type: 'pie', data:{ labels:Object.keys(statusDist), datasets:[{ data:Object.values(statusDist), backgroundColor:palette }] }, options:{ plugins:{ legend:{labels:{color:'#cbd5e1'}} } } });
 makeChart('completionChart', { type:'bar', data:{ labels:sprintLabels, datasets:[{ label:'Completion %', data:completionData, backgroundColor:'#a78bfa' }] }, options:{ responsive:true, plugins:{ legend:{labels:{color:'#cbd5e1'}}}, scales:{ x:{ticks:{color:'#94a3b8'}}, y:{ticks:{callback:v=>v+'%', color:'#94a3b8'}, beginAtZero:true, max:100 } } } });
</script>
</body>
</html>""")

    html = template.substitute(
        gen_time=gen_time,
        last_two_velocity=f"{last_two_vel:.2f}",
        effective_velocity=f"{effective_velocity:.2f}",
        forecast_text=forecast_text,
        last_remaining_main=velocity["last_remaining_main"],
        total_main=velocity["total_main"],
        burn_pct=f"{burn_pct:.1f}",
        risk_label=risk_label,
        sprint_labels=js(sprint_labels),
        burndown_remaining=js(burndown_remaining),
        cumulative_done=js(cumulative_done),
        completion_ratios=js(completion_ratios),
        main_vs_sub=js(main_vs_sub),
        engineer_distribution=js(engineer_distribution_series),
        status_dist=js(status_dist),
    )
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate static Chart.js dashboard from pipeline JSON outputs")
    parser.add_argument("--task-json", default=DEFAULT_TASK_JSON, help="Path to flat task JSON (task_timeline.json)")
    parser.add_argument("--sprint-json", default=DEFAULT_SPRINT_JSON, help="Path to sprint summary JSON (sprint_summary.json)")
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="Output HTML file path")
    args = parser.parse_args()

    tasks = read_flat_tasks(args.task_json)
    sprint_summary = load_json(args.sprint_json)
    velocity = compute_velocity_and_forecast(sprint_summary)

    html = build_html(tasks, sprint_summary, velocity)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard written to {args.out_file}")


if __name__ == "__main__":
    main()
