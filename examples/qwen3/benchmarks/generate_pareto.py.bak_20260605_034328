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
BASELINE_LABEL = "Qwen3-30B-A3B"


def load_data(csv_path, min_accuracy=0, limit_filter=None, max_train_iters=None, run_name_prefix=None):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    points = []
    for r in rows:
        avg = float(r.get("benchmark_avg") or "0")
        cp = float(r.get("eval_crit_path") or "0")
        if run_name_prefix:
            prefixes = [p.strip() for p in run_name_prefix.split(",")]
            if not any(r.get("run_name","").startswith(p) for p in prefixes):
                continue
        if avg <= 0 or cp <= 0 or avg < min_accuracy:
            continue
        if limit_filter is not None and limit_filter != "":
            row_limit = r.get("limit", "")
            if row_limit != str(limit_filter):
                continue
        if max_train_iters is not None:
            bench_iter = r.get("bench_iteration", "")
            train_iters_val = r.get("train_iters", "")
            try:
                if bench_iter and int(bench_iter) > max_train_iters:
                    continue
            except ValueError:
                pass
            try:
                if train_iters_val and int(train_iters_val) > max_train_iters:
                    continue
            except ValueError:
                pass

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
        elif "k3" in name:
            ppo_k = "3"
        has_lm = "lm1.0" in name
        has_buf = "buf" in name
        has_gumbel = "gumbel" in name
        has_cosine = "_cos_" in name or "_cos_" in name
        has_gae = "gae" in name

        # Auto-classify degraded runs
        display_cat = cat
        if avg < 55 and cat == "rl+aux":
            display_cat = "degraded"

        # Build short label
        if name.startswith("pretrained_baseline") or cat == "pretrained":
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
            if has_gumbel:
                label += " gumbel"
            if has_cosine:
                label += " cos"
            if has_gae:
                label += " gae"
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
            "gumbel": has_gumbel,
            "cosine": has_cosine,
            "gae": has_gae,
        })

    # Deduplicate: keep one entry per (label, cat, cp) — prefer higher accuracy
    seen = {}
    deduped = []
    for p in points:
        key = (p["label"], p["cat"], round(p["cp"], 0))
        if key not in seen or p["acc"] > seen[key]["acc"]:
            seen[key] = p
    deduped = list(seen.values())

    return deduped


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


def generate_html(points, output_path, y_min_override=None, y_max_override=None):
    cats_for_frontier = ["aux_only", "rl_only", "rl+aux"]
    frontiers = {}
    for cat in cats_for_frontier:
        cat_points = [p for p in points if p["cat"] == cat]
        if cat_points:
            frontiers[cat] = compute_pareto_frontier(cat_points)

    min_acc = min(p["y"] for p in points)
    y_min = y_min_override if y_min_override is not None else max(30, int(min_acc) - 2)
    y_max = y_max_override if y_max_override is not None else 66

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
    <b>Baseline CP = {baseline_cp}</b> (pretrained {baseline_label}, no training).<br>
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
  pretrained: {{ bg: '#000000', border: '#000', shape: 'star', r: 22 }},
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
        min: -10, max: 45, ticks: {{ callback: v => v + '%' }}, grid: {{ color: '#f0f0f0' }},
      }},
      y: {{
        title: {{ display: true, text: 'Benchmark Accuracy (%)', font: {{ size: 14, weight: 'bold' }} }},
        min: {y_min}, max: {y_max}, ticks: {{ callback: v => v.toFixed(0) + '%' }}, grid: {{ color: '#f0f0f0' }},
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
              ...(p.gumbel ? ['Gumbel routing: yes'] : []),
              ...(p.cosine ? ['Cosine sched:   yes'] : []),
              ...(p.gae ? ['GAE:            yes'] : []),
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
        baseline_label=BASELINE_LABEL,
        n_total=len(points),
        data_json=json.dumps(points),
        frontiers_json=json.dumps(frontiers),
        frontier_colors_json=json.dumps(frontier_colors),
        frontier_labels_json=json.dumps(frontier_labels),
        y_min=y_min,
        y_max=y_max,
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



def ensure_baseline_point(points, csv_path, run_name_prefix=None):
    """Guarantee the pretrained baseline is plotted, regardless of limit filter
    or whether its CSV row has an eval_crit_path. The baseline CP is supplied via
    --baseline-cp (BASELINE_CP); its accuracy is read from the highest-accuracy
    pretrained row in the CSV. Any pre-existing pretrained points are replaced."""
    import csv as _csv
    # Find best pretrained accuracy in the CSV (full/inf eval wins via max).
    best = None
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            nm = r.get("run_name", "")
            cat = r.get("category", "")
            if not (nm.startswith("pretrained_235b") or (cat == "pretrained" and "235b" in nm.lower())):
                continue
            try:
                acc = float(r.get("benchmark_avg") or 0)
            except ValueError:
                continue
            if acc <= 0:
                continue
            h = r.get("hellaswag") or None
            a = r.get("arc_challenge") or None
            w = r.get("winogrande") or None
            if best is None or acc > best["acc"]:
                best = {"acc": acc, "h": float(h) if h else None,
                        "a": float(a) if a else None, "w": float(w) if w else None}
    if best is None:
        return points  # no pretrained row at all; leave as-is
    # Drop any existing pretrained points and inject one clean baseline point.
    points = [p for p in points if p.get("cat") != "pretrained"]
    points.append({
        "label": "Pretrained", "cat": "pretrained", "sweep": "",
        "rl": False, "aux": False, "iters": 0, "rlc": 0.0,
        "cp": BASELINE_CP, "lm": 0.0,
        "h": best["h"], "a": best["a"], "w": best["w"],
        "acc": best["acc"], "x": 0.0, "y": best["acc"],
        "ppo_k": "", "lm_reward": False, "klc": "",
        "gumbel": False, "cosine": False, "gae": False,
    })
    print("Injected baseline point: cp={} acc={:.2f}".format(BASELINE_CP, best["acc"]))
    return points


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto chart from benchmark CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-accuracy", type=float, default=0,
                        help="Filter out runs below this accuracy %%")
    parser.add_argument("--limit-filter", type=str, default=None,
                        help="Only include runs with this limit value (e.g. '1000')")
    parser.add_argument("--clean", action="store_true",
                        help="Only display points that are on a Pareto frontier (aux, rl, or rl+aux)")
    parser.add_argument("--max-train-iters", type=int, default=8000,
                        help="Only include runs benchmarked at or below this training iteration")
    parser.add_argument("--y-min", type=float, default=61,
                        help="Y-axis minimum (default: auto)")
    parser.add_argument("--run-name-prefix", type=str, default=None,
                        help="Only include rows whose run_name starts with this string (e.g. '235b' for 235B-only chart)")
    parser.add_argument("--baseline-cp", type=float, default=4780,
                        help="Reference baseline critical_path (default 4780 = 30B pretrained; use 8800 for 235B)")
    parser.add_argument("--baseline-label", type=str, default="Qwen3-30B-A3B",
                        help="Label for the baseline shown on the chart")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Y-axis maximum (default: 66)")
    args = parser.parse_args()
    global BASELINE_CP, BASELINE_LABEL
    BASELINE_CP = args.baseline_cp
    BASELINE_LABEL = args.baseline_label

    if args.output.suffix != '.html':
        args.output = args.output.with_suffix('.html')

    points = load_data(args.csv, args.min_accuracy, limit_filter=args.limit_filter, max_train_iters=args.max_train_iters, run_name_prefix=args.run_name_prefix)
    print(f"max train iters: {args.max_train_iters}")
    print(f"y-min: {args.y_min}")
    print(f"y-max: {args.y_max}")
    if not points:
        print("No data points found in {}".format(args.csv))
        return

    if args.clean:
        frontier_set = set()
        for cat in ["aux_only", "rl_only", "rl+aux"]:
            cat_points = [p for p in points if p["cat"] == cat]
            if cat_points:
                for p in compute_pareto_frontier(cat_points):
                    frontier_set.add(id(p))
        points = [p for p in points if id(p) in frontier_set or p["cat"] == "pretrained"]
        print("Clean mode: {} Pareto-optimal points retained".format(len(points)))

    points = ensure_baseline_point(points, args.csv, run_name_prefix=args.run_name_prefix)
    generate_html(points, args.output, y_min_override=args.y_min, y_max_override=args.y_max)


if __name__ == "__main__":
    main()
