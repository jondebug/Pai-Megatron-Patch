#!/usr/bin/env python3
"""Generate a Pareto frontier HTML chart from benchmark_results.csv.

Usage:
    python generate_pareto.py                          # uses default CSV path
    python generate_pareto.py --csv path/to/results.csv
    python generate_pareto.py --min-accuracy 50        # filter out runs below 50%
    python generate_pareto.py --output my_pareto.html
"""

import argparse
import csv
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
DEFAULT_CSV = SCRIPT_DIR / "benchmark_results.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent.parent.parent / "pareto_accuracy_vs_cp.html"
BASELINE_CP = 4780


def load_data(csv_path, min_accuracy=0):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    points = []
    for r in rows:
        avg = float(r.get("benchmark_avg") or "0")
        cp = float(r.get("eval_crit_path") or "0")
        if avg <= 0 or cp <= 0 or avg < min_accuracy:
            continue

        name = r.get("run_name", "")
        cat = r.get("category", "other")
        rl = r.get("rl_enabled") == "True"
        aux = r.get("aux_enabled") == "True"
        rlc = r.get("rl_loss_coeff", "") or "0"
        auxc = r.get("aux_loss_coeff", "") or "0"
        iters = r.get("train_iters", "") or "0"
        sweep = r.get("sweep_id", "") or ""
        lm_loss = float(r.get("eval_lm_loss") or "0")
        h = float(r.get("hellaswag") or "0")
        a = float(r.get("arc_challenge") or "0")
        w = float(r.get("winogrande") or "0")

        klc = r.get("kl_loss_coeff", "") or ""

        # Detect PPO epochs and special flags from run name
        ppo_k = ""
        if "k20" in name:
            ppo_k = "20"
        elif "k5" in name:
            ppo_k = "5"
        has_lm = "lm1.0" in name
        has_buf = "buf" in name

        # Auto-classify degraded runs
        display_cat = cat
        if avg < 55 and cat == "rl+aux":
            display_cat = "degraded"

        # Build short label
        if name == "pretrained_baseline":
            label = "Pretrained"
        elif cat == "aux_only":
            label = "Aux {}s (c={})".format(iters, auxc)
        elif cat == "rl_only":
            rtype = r.get("rl_reward_type", "").replace("per_token_", "").replace("_", " ")
            label = "RL {} rlc={}".format(rtype, rlc)
        elif display_cat == "degraded":
            label = "rlc={} k={}".format(rlc, ppo_k)
            if has_buf:
                label += " buf"
            if klc:
                label += " KL={}".format(klc)
        else:
            label = "RL+aux rlc={}".format(rlc)
            if has_lm:
                label = "RL+aux+LM rlc={}".format(rlc)
            if ppo_k:
                label += " k={}".format(ppo_k)
            if klc:
                label += " KL={}".format(klc)

        cp_red = (BASELINE_CP - cp) / BASELINE_CP * 100

        points.append({
            "label": label,
            "cat": display_cat,
            "sweep": sweep,
            "rl": rl,
            "aux": aux,
            "iters": int(iters) if iters != "0" else 0,
            "rlc": float(rlc),
            "cp": cp,
            "lm": lm_loss,
            "h": h if h else None,
            "a": a if a else None,
            "w": w if w else None,
            "acc": avg,
            "x": round(cp_red, 2),
            "y": avg,
            "ppo_k": ppo_k,
            "lm_reward": has_lm,
            "klc": klc,
        })

    return points


def compute_pareto_frontier(points):
    dominated = set()
    for i, pi in enumerate(points):
        for j, pj in enumerate(points):
            if i == j:
                continue
            if pj["x"] >= pi["x"] and pj["y"] >= pi["y"] and (
                pj["x"] > pi["x"] or pj["y"] > pi["y"]
            ):
                dominated.add(i)
                break
    frontier = [p for i, p in enumerate(points) if i not in dominated]
    return sorted(frontier, key=lambda p: p["x"])


def generate_html(points, output_path):
    cats_for_frontier = ["aux_only", "rl_only", "rl+aux"]
    frontiers = {}
    for cat in cats_for_frontier:
        cat_points = [p for p in points if p["cat"] == cat]
        if cat_points:
            frontiers[cat] = compute_pareto_frontier(cat_points)

    min_acc = min(p["y"] for p in points)
    y_min = max(30, int(min_acc) - 2)

    frontier_colors = {
        "aux_only": "rgba(46,125,50,0.45)",
        "rl_only": "rgba(198,40,40,0.45)",
        "rl+aux": "rgba(21,101,192,0.45)",
    }
    frontier_labels = {
        "aux_only": "Aux-only Frontier",
        "rl_only": "RL-only Frontier",
        "rl+aux": "RL+Aux Frontier",
    }

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pareto: Accuracy vs Critical Path Reduction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }}
  .container {{ max-width: 1000px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }}
  h1 {{ text-align: center; font-size: 1.3em; color: #333; margin: 0 0 4px 0; }}
  .subtitle {{ text-align: center; color: #888; font-size: 0.85em; margin-bottom: 20px; }}
  .chart-wrap {{ position: relative; width: 100%; aspect-ratio: 4/3; }}
  .notes {{ margin-top: 16px; font-size: 0.78em; color: #666; line-height: 1.6; }}
  .notes b {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.72em; }}
  th, td {{ padding: 4px 6px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; position: sticky; top: 0; }}
  .c-pre {{ color: #757575; }} .c-aux {{ color: #2e7d32; }} .c-rl {{ color: #c62828; }} .c-rla {{ color: #1565c0; }} .c-bad {{ color: #999; }}
</style>
</head>
<body>
<div class="container">
  <h1>Accuracy vs Critical Path Reduction</h1>
  <p class="subtitle">Top-right = best. Hover for details. 3 Pareto frontiers: aux-only, RL-only, RL+aux.</p>
  <div class="chart-wrap"><canvas id="pareto"></canvas></div>
  <div class="notes">
    <b>Baseline CP = {baseline_cp}</b> (pretrained Qwen3-30B-A3B, no training).<br>
    <b>Dashed lines</b> = per-category Pareto frontiers (non-dominated points within each method).<br>
    <b>Gray star</b> = pretrained baseline (no training, no CP reduction).
    <b>{n_total}</b> total runs plotted.
  </div>
  <table id="data-table">
    <thead><tr><th>Label</th><th>Cat</th><th>Sweep</th><th>RL</th><th>Aux</th><th>Iters</th><th>RLC</th><th>KL</th><th>CP</th><th>CP%</th><th>LM Loss</th><th>Hella</th><th>ARC-C</th><th>Wino</th><th>Avg</th></tr></thead>
    <tbody></tbody>
  </table>
</div>
<script>
const BASELINE_CP = {baseline_cp};
const DATA = {data_json};
const FRONTIERS = {frontiers_json};

const STYLES = {{
  pretrained: {{ bg: '#9e9e9e', border: '#616161', shape: 'star', r: 14 }},
  aux_only:   {{ bg: '#4caf50', border: '#2e7d32', shape: 'rectRot', r: 10 }},
  rl_only:    {{ bg: 'rgba(229,57,53,0.85)', border: '#c62828', shape: 'triangle', r: 9 }},
  'rl+aux':   {{ bg: 'rgba(33,150,243,0.85)', border: '#1565c0', shape: 'circle', r: 9 }},
  degraded:   {{ bg: 'rgba(180,180,180,0.4)', border: 'rgba(150,150,150,0.5)', shape: 'crossRot', r: 7 }},
}};
const LABELS = {{
  pretrained: 'Pretrained Baseline', aux_only: 'Aux Loss Only',
  rl_only: 'RL Only (no aux)', 'rl+aux': 'RL + Aux Combined',
  degraded: 'Multi-epoch PPO (degraded)',
}};
const FRONTIER_COLORS = {frontier_colors_json};
const FRONTIER_LABELS = {frontier_labels_json};

const groups = {{}};
DATA.forEach(d => {{ (groups[d.cat] = groups[d.cat] || []).push(d); }});

const chartDatasets = [];

// Per-category Pareto frontier lines
Object.entries(FRONTIERS).forEach(([cat, pts]) => {{
  chartDatasets.push({{
    label: FRONTIER_LABELS[cat] || cat + ' Frontier',
    data: pts,
    borderColor: FRONTIER_COLORS[cat] || 'rgba(100,100,100,0.3)',
    borderWidth: 2, borderDash: [8, 4],
    showLine: true, pointRadius: 0, pointHoverRadius: 0, tension: 0, order: 10,
  }});
}});

// Scatter points per category
Object.entries(groups).forEach(([cat, pts]) => {{
  const s = STYLES[cat] || STYLES.degraded;
  chartDatasets.push({{
    label: LABELS[cat] || cat, data: pts.sort((a,b) => a.x - b.x),
    backgroundColor: s.bg, borderColor: s.border, pointStyle: s.shape,
    pointRadius: s.r, pointHoverRadius: s.r + 4, borderWidth: 2,
    showLine: false, order: cat === 'degraded' ? 5 : 1,
  }});
}});

const catClass = {{ pretrained:'c-pre', aux_only:'c-aux', rl_only:'c-rl', 'rl+aux':'c-rla', degraded:'c-bad' }};
const tbody = document.querySelector('#data-table tbody');
[...DATA].sort((a,b) => b.acc - a.acc).forEach(d => {{
  const tr = document.createElement('tr');
  tr.className = catClass[d.cat] || '';
  tr.innerHTML =
    '<td>'+d.label+'</td><td>'+d.cat+'</td><td>'+(d.sweep||'--')+'</td>'
    +'<td>'+(d.rl?'Y':'N')+'</td><td>'+(d.aux?'Y':'N')+'</td>'
    +'<td>'+(d.iters||'--')+'</td><td>'+(d.rlc||'--')+'</td><td>'+(d.klc||'--')+'</td>'
    +'<td>'+d.cp.toFixed(0)+'</td><td>'+d.x.toFixed(1)+'%</td>'
    +'<td>'+d.lm.toFixed(4)+'</td>'
    +'<td>'+(d.h!=null?d.h.toFixed(1):'--')+'</td>'
    +'<td>'+(d.a!=null?d.a.toFixed(1):'--')+'</td>'
    +'<td>'+(d.w!=null?d.w.toFixed(1):'--')+'</td>'
    +'<td><b>'+d.acc.toFixed(1)+'%</b></td>';
  tbody.appendChild(tr);
}});

new Chart(document.getElementById('pareto').getContext('2d'), {{
  type: 'scatter',
  data: {{ datasets: chartDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: true, aspectRatio: 4/3,
    layout: {{ padding: {{ top: 10, right: 20, bottom: 10, left: 10 }} }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Critical Path Reduction (%)', font: {{ size: 14, weight: 'bold' }} }},
        min: -3, max: 45, ticks: {{ callback: v => v + '%' }}, grid: {{ color: '#f0f0f0' }},
      }},
      y: {{
        title: {{ display: true, text: 'Benchmark Accuracy (%)', font: {{ size: 14, weight: 'bold' }} }},
        min: {y_min}, max: 66, ticks: {{ callback: v => v.toFixed(0) + '%' }}, grid: {{ color: '#f0f0f0' }},
      }}
    }},
    plugins: {{
      legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 16, font: {{ size: 11 }} }} }},
      tooltip: {{
        backgroundColor: 'rgba(30,30,30,0.92)', titleFont: {{ size: 13, weight: 'bold' }},
        bodyFont: {{ size: 11 }}, padding: 12, cornerRadius: 8,
        callbacks: {{
          title: items => items[0].raw.label,
          afterTitle: ctx => {{
            const p = ctx[0].raw;
            const parts = [p.cat];
            if (p.sweep) parts.push('sweep: ' + p.sweep);
            return parts.join(' | ');
          }},
          label: ctx => {{
            const p = ctx.raw;
            return [
              '', 'Benchmark Avg:  ' + p.acc.toFixed(1) + '%',
              '  HellaSwag:    ' + (p.h != null ? p.h.toFixed(1)+'%' : '--'),
              '  ARC-Challenge:' + (p.a != null ? p.a.toFixed(1)+'%' : '--'),
              '  WinoGrande:   ' + (p.w != null ? p.w.toFixed(1)+'%' : '--'),
              '', 'Eval CP:        ' + p.cp.toFixed(0) + '  (' + p.x.toFixed(1) + '% reduction)',
              'Eval LM Loss:   ' + p.lm.toFixed(4),
              'Train iters:    ' + (p.iters || '--'),
              ...(p.rl ? ['RL coeff:       ' + p.rlc] : []),
              ...(p.ppo_k ? ['PPO epochs:     ' + p.ppo_k] : []),
              ...(p.lm_reward ? ['LM reward:      yes'] : []),
              ...(p.klc ? ['KL coeff:       ' + p.klc] : []),
            ];
          }}
        }}
      }},
    }}
  }},
  plugins: [{{
    id: 'labelPoints',
    afterDatasetsDraw(chart) {{
      const c = chart.ctx;
      c.save(); c.font = '9px -apple-system, sans-serif'; c.textAlign = 'center';
      chart.data.datasets.forEach((ds, di) => {{
        if (ds.label.includes('degraded') || ds.label.includes('Frontier')) return;
        const meta = chart.getDatasetMeta(di);
        meta.data.forEach((pt, pi) => {{
          const d = ds.data[pi];
          if (d.acc < 55) return;
          let ox = 0, oy = -14;
          if (d.label === 'Pretrained') {{ ox = 45; oy = 4; }}
          c.fillStyle = ds.borderColor || '#333';
          c.fillText(d.label, pt.x + ox, pt.y + oy);
        }});
      }});
      c.restore();
    }}
  }}]
}});
</script>
</body>

</html>"""

    html = html.format(
        baseline_cp=BASELINE_CP,
        n_total=len(points),
        data_json=json.dumps(points),
        frontiers_json=json.dumps(frontiers),
        frontier_colors_json=json.dumps(frontier_colors),
        frontier_labels_json=json.dumps(frontier_labels),
        y_min=y_min,
    )

    with open(output_path, "w") as f:
        f.write(html)
    print("Pareto chart written to: {}".format(output_path))
    n_frontier = sum(len(v) for v in frontiers.values())
    print("  {} data points, {} on frontiers (aux:{}, rl:{}, rl+aux:{})".format(
        len(points), n_frontier,
        len(frontiers.get("aux_only", [])),
        len(frontiers.get("rl_only", [])),
        len(frontiers.get("rl+aux", [])),
    ))


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto chart from benchmark CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-accuracy", type=float, default=0,
                        help="Filter out runs below this accuracy %%")
    args = parser.parse_args()

    points = load_data(args.csv, args.min_accuracy)
    if not points:
        print("No data points found in {}".format(args.csv))
        return

    generate_html(points, args.output)


if __name__ == "__main__":
    main()
