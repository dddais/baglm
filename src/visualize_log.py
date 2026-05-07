"""
Visualize BaGLM online session logs.

Generates an HTML report with:
  - Belief evolution timeline chart
  - Step transition timeline
  - Confidence & latency plots
  - Per-segment detail table

Usage:
    python src/visualize_log.py logs/online/session_20260507_*.json
    python src/visualize_log.py logs/online/session_20260507_*.json -o report.html
"""

import argparse
import json
import os
import sys
from datetime import datetime


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>BaGLM Session Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 15px 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 150px; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .stat-card .label {{ font-size: 13px; color: #7f8c8d; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 10px 15px; text-align: left; }}
  td {{ padding: 8px 15px; border-bottom: 1px solid #ecf0f1; }}
  tr:hover td {{ background: #f1f2f6; }}
  .step-tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
  .conf-high {{ background: #d4edda; color: #155724; }}
  .conf-mid  {{ background: #fff3cd; color: #856404; }}
  .conf-low  {{ background: #f8d7da; color: #721c24; }}
  .timeline {{ display: flex; align-items: center; gap: 2px; margin: 15px 0; flex-wrap: wrap; }}
  .timeline-block {{ padding: 6px 10px; border-radius: 4px; font-size: 11px; color: white; text-align: center; min-width: 30px; }}
  .meta {{ color: #7f8c8d; font-size: 14px; margin-bottom: 10px; }}
</style>
</head>
<body>

<h1>BaGLM Online Session Report</h1>
<div class="meta">
  Source: {source} &nbsp;|&nbsp; Server: {server} &nbsp;|&nbsp;
  Segments: {total_segments} &nbsp;|&nbsp;
  Time: {timestamp}
</div>

<div class="stats">
  <div class="stat-card"><div class="value">{avg_conf:.3f}</div><div class="label">Avg Confidence</div></div>
  <div class="stat-card"><div class="value">{max_conf:.3f}</div><div class="label">Max Confidence</div></div>
  <div class="stat-card"><div class="value">{min_conf:.3f}</div><div class="label">Min Confidence</div></div>
  <div class="stat-card"><div class="value">{avg_latency:.0f}ms</div><div class="label">Avg Inference</div></div>
  <div class="stat-card"><div class="value">{n_transitions}</div><div class="label">Step Transitions</div></div>
</div>

<h2>Step Timeline</h2>
<div class="timeline" id="timeline"></div>

<h2>Belief Evolution</h2>
<div class="chart-container">
  <canvas id="beliefChart" height="100"></canvas>
</div>

<h2>Confidence & Latency</h2>
<div class="chart-container">
  <canvas id="metricChart" height="80"></canvas>
</div>

<h2>Per-Segment Details</h2>
<table>
  <tr>
    <th>Seg</th><th>Step</th><th>Confidence</th><th>Belief (top)</th>
    <th>Inference</th><th>Round-trip</th>
  </tr>
  {table_rows}
</table>

<script>
// Step colors
const stepColors = {step_colors_json};
const allSteps = {all_steps_json};

// Timeline
const timelineEl = document.getElementById('timeline');
const segData = {seg_data_json};
segData.forEach(d => {{
  const block = document.createElement('div');
  block.className = 'timeline-block';
  block.style.backgroundColor = stepColors[d.step] || '#95a5a6';
  block.style.flex = '1';
  block.title = 'Seg ' + d.seg + ': ' + d.step + ' (' + (d.confidence*100).toFixed(1) + '%)';
  block.textContent = d.seg;
  timelineEl.appendChild(block);
}});

// Belief chart
new Chart(document.getElementById('beliefChart'), {{
  type: 'line',
  data: {{
    labels: segData.map(d => d.seg),
    datasets: allSteps.map(step => ({{
      label: step,
      data: segData.map(d => d.belief[step] || 0),
      borderColor: stepColors[step],
      backgroundColor: stepColors[step] + '33',
      fill: true,
      tension: 0.3,
      pointRadius: 3,
    }}))
  }},
  options: {{
    responsive: true,
    scales: {{
      y: {{ min: 0, max: 1, title: {{ display: true, text: 'Belief' }} }},
      x: {{ title: {{ display: true, text: 'Segment' }} }}
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});

// Metric chart
new Chart(document.getElementById('metricChart'), {{
  type: 'bar',
  data: {{
    labels: segData.map(d => d.seg),
    datasets: [
      {{
        label: 'Confidence',
        data: segData.map(d => d.confidence),
        backgroundColor: 'rgba(52, 152, 219, 0.7)',
        yAxisID: 'y',
      }},
      {{
        label: 'Inference (ms)',
        data: segData.map(d => d.inference_ms),
        type: 'line',
        borderColor: 'rgba(231, 76, 60, 0.9)',
        backgroundColor: 'transparent',
        yAxisID: 'y1',
        tension: 0.3,
      }}
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      y:  {{ min: 0, max: 1, position: 'left', title: {{ display: true, text: 'Confidence' }} }},
      y1: {{ min: 0, position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Latency (ms)' }} }},
      x:  {{ title: {{ display: true, text: 'Segment' }} }}
    }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});
</script>
</body>
</html>"""


def generate_step_colors(steps):
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    ]
    return {s: palette[i % len(palette)] for i, s in enumerate(steps)}


def load_session(path):
    with open(path) as f:
        return json.load(f)


def build_table_rows(results, step_colors):
    rows = []
    for r in results:
        conf = r["confidence"]
        conf_cls = "conf-high" if conf >= 0.5 else ("conf-mid" if conf >= 0.2 else "conf-low")
        # Top 3 belief
        belief = r.get("belief", {})
        top3 = sorted(belief.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = " ".join(f"{k}: {v:.2f}" for k, v in top3)
        step_color = step_colors.get(r["step"], "#95a5a6")
        rows.append(
            f'<tr>'
            f'<td>{r["seg"]}</td>'
            f'<td><span class="step-tag" style="background:{step_color}33;color:{step_color}">{r["step"]}</span></td>'
            f'<td><span class="step-tag {conf_cls}">{conf:.3f}</span></td>'
            f'<td style="font-size:12px">{top3_str}</td>'
            f'<td>{r["inference_ms"]:.0f}ms</td>'
            f'<td>{r["round_trip_ms"]:.0f}ms</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def visualize(session_paths, output_path=None):
    sessions = [load_session(p) for p in session_paths]

    # Use the first session as the main one (or merge if multiple)
    s = sessions[0]
    results = s["results"]

    # Collect all step names across sessions
    all_steps_set = set()
    for sess in sessions:
        for r in sess["results"]:
            all_steps_set.update(r.get("belief", {}).keys())
    all_steps = sorted(all_steps_set)
    step_colors = generate_step_colors(all_steps)

    # Build per-seg data for JS
    seg_data = []
    for r in results:
        seg_data.append({
            "seg": r["seg"],
            "step": r["step"],
            "confidence": r["confidence"],
            "inference_ms": r["inference_ms"],
            "belief": r.get("belief", {}),
        })

    html = HTML_TEMPLATE.format(
        source=s.get("source", "unknown"),
        server=s.get("server", "unknown"),
        timestamp=s.get("timestamp", "unknown"),
        total_segments=s.get("total_segments", len(results)),
        avg_conf=s.get("avg_confidence", 0),
        max_conf=s.get("max_confidence", 0),
        min_conf=s.get("min_confidence", 0),
        avg_latency=s.get("avg_inference_ms", 0),
        n_transitions=len(s.get("step_transitions", [])),
        table_rows=build_table_rows(results, step_colors),
        step_colors_json=json.dumps(step_colors),
        all_steps_json=json.dumps(all_steps),
        seg_data_json=json.dumps(seg_data),
    )

    if output_path is None:
        base = os.path.splitext(os.path.basename(session_paths[0]))[0]
        output_path = os.path.join(os.path.dirname(session_paths[0]), f"{base}_report.html")

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize BaGLM session logs")
    parser.add_argument("inputs", nargs="+", help="Session JSON log file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    visualize(args.inputs, args.output)


if __name__ == "__main__":
    main()
