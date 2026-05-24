#!/usr/bin/env python3
"""Unified results collector for 235B v5/v6 sweeps.

For every cell directory in output_router_finetuning/235bv*/:
  - Parse hyperparameters from cell name + sweep config defaults
  - Pull eval_lm_loss + critical_path from W&B history at all eval points
  - Scrape lm-eval accuracy (HellaSwag / ARC-C / Winogrande acc_norm) from logs
  - Resolve checkpoint_path (cell's distcp root)
  - Write rows keyed by (run_name, bench_iteration) — append or update

Writes to benchmark_results.csv with the same schema as the 30B sweep.
Usage:  python collect_results_unified.py [--limit N] [--cells regex]
"""
import argparse, csv, json, re, sys
from pathlib import Path
from datetime import datetime

REPO = Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch")
OUTPUT_BASE = Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning")
BENCH_LOG_DIR = Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs")
CSV_PATH = REPO / "examples/qwen3/benchmarks/benchmark_results.csv"

COLS = [
    "run_name","bench_iteration","sweep_id","category","rl_enabled","aux_enabled","train_iters",
    "rl_reward_type","rl_loss_coeff","aux_loss_coeff","kl_loss_coeff","lm_reward_coeff",
    "eval_crit_path","eval_lm_loss",
    "hellaswag","arc_challenge","winogrande","benchmark_avg",
    "benchmark_time_sec","limit","timestamp","comments","checkpoint_path",
]

def parse_cell_hyperparams(name):
    """Derive hyperparams from cell name."""
    cfg = dict(
        sweep_id="", category="other",
        rl_enabled=False, aux_enabled=False, train_iters=1500,
        rl_reward_type="", rl_loss_coeff="", aux_loss_coeff="",
        kl_loss_coeff="0", lm_reward_coeff="0",
    )
    if "v5a" in name: cfg["sweep_id"] = "so3oxhq0"
    elif "v5b" in name: cfg["sweep_id"] = "hxautflf"
    elif name.startswith("235bv6"): cfg["sweep_id"] = "60fe99g5"

    cfg["rl_enabled"] = ("_norl_" not in name and not name.startswith("norl_") and "pretrained" not in name)
    aux = re.search(r"aux([\d.]+)", name)
    if aux:
        cfg["aux_loss_coeff"] = aux.group(1)
        cfg["aux_enabled"] = float(aux.group(1)) > 0
    elif name.startswith("235bv5b") or name.startswith("235bv6"):
        cfg["aux_loss_coeff"] = "0.01"; cfg["aux_enabled"] = True
    rlc = re.search(r"rlc([\d.]+)", name)
    if rlc: cfg["rl_loss_coeff"] = rlc.group(1)
    kl = re.search(r"_kl([\d.]+)", name)
    if kl: cfg["kl_loss_coeff"] = kl.group(1)
    lm = re.search(r"_lm([\d.]+)", name)
    if lm: cfg["lm_reward_coeff"] = lm.group(1)
    if cfg["rl_enabled"]:
        cfg["rl_reward_type"] = "per_token_load_weighted"
    if cfg["rl_enabled"] and cfg["aux_enabled"]: cfg["category"] = "rl+aux"
    elif cfg["rl_enabled"]: cfg["category"] = "rl_only"
    elif cfg["aux_enabled"]: cfg["category"] = "aux_only"
    elif "pretrained" in name: cfg["category"] = "pretrained"
    return cfg

def parse_lmeval_log(path):
    """Return dict of (cell, iter) -> {arc_n, hella_n, wino} or None."""
    txt = path.read_text(errors="ignore")
    m = re.search(r"Run name:\s+(\S+)", txt)
    if not m: return None
    name = m.group(1)
    for suffix in ("_v3", "_v2"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    m = re.search(r"hf_converted_iter(\d+)_cp", txt)
    bench_iter = int(m.group(1)) if m else None
    r = {}
    m = re.search(r"\|arc_challenge\s*\|\s*\d+\s*\|none\s*\|\s*\d+\|acc\s*\|.*?\|\s*([\d.]+)\s*\|.*?\n\s*\|\s+\|\s+\|none\s*\|\s*\d+\|acc_norm\s*\|.*?\|\s*([\d.]+)\s*\|", txt)
    if m: r["arc_n"] = float(m.group(2))
    m = re.search(r"\|hellaswag\s*\|\s*\d+\s*\|none\s*\|\s*\d+\|acc\s*\|.*?\|\s*([\d.]+)\s*\|.*?\n\s*\|\s+\|\s+\|none\s*\|\s*\d+\|acc_norm\s*\|.*?\|\s*([\d.]+)\s*\|", txt)
    if m: r["hella_n"] = float(m.group(2))
    m = re.search(r"\|winogrande\s*\|\s*\d+\s*\|none\s*\|\s*\d+\|acc\s*\|.*?\|\s*([\d.]+)\s*\|", txt)
    if m: r["wino"] = float(m.group(1))
    if all(k in r for k in ["arc_n","hella_n","wino"]):
        return name, bench_iter, r
    return None

def pull_wandb_metrics():
    """Return {cell_name: [(iter, lm_loss, cp), ...]} dict from wandb."""
    try:
        import wandb
    except ImportError:
        return {}
    api = wandb.Api()
    out = {}
    runs = list(api.runs("nvr-israel/qwen3-router-training", per_page=500))
    for r in runs:
        if not r.name or not (r.name.startswith("235bv5") or r.name.startswith("235bv6")):
            continue
        try:
            h = r.history(keys=["iteration","critical_eval/lm_loss","critical_eval/critical_path"], pandas=False)
            for e in h:
                it = e.get("iteration") or e.get("_step")
                lm = e.get("critical_eval/lm_loss")
                cp = e.get("critical_eval/critical_path")
                if it is None or (lm is None and cp is None): continue
                out.setdefault(r.name, []).append((int(it), lm, cp))
        except Exception:
            continue
    # Dedup per (cell, iter) — keep last
    for cell, pts in out.items():
        seen = {}
        for it, lm, cp in pts:
            seen[it] = (lm if lm is not None else seen.get(it,(None,None))[0],
                        cp if cp is not None else seen.get(it,(None,None))[1])
        out[cell] = sorted([(it, lm, cp) for it,(lm,cp) in seen.items()])
    return out

def find_checkpoint_path(cell, iter_=None, is_hf=False):
    """Return iter-specific checkpoint path. is_hf=True for benchmark rows (HF dir).
    Same (cell, iter, kind) → same path. Different iters/kinds → different paths."""
    d = OUTPUT_BASE / cell
    if not d.is_dir(): return ""
    cks = sorted(d.glob("checkpoint/pretrain-mcore-*"))
    if not cks: return ""
    if iter_ is None:
        return str(cks[0])
    # Search across all checkpoint subdirs (gbs-8 v5 dir AND gbs-16 v6 dir)
    for ck in cks:
        if is_hf:
            hf = ck / f"hf_converted_iter{iter_}_cp"
            if hf.is_dir(): return str(hf)
        else:
            it_dir = ck / f"iter_{int(iter_):07d}"
            if it_dir.is_dir(): return str(it_dir)
    # Fallback: cell-level dir if specific iter dir not found
    return str(cks[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--cells", default=r"^(235bv[56]|pretrained_235b)", help="cell name regex")
    ap.add_argument("--out", default=str(CSV_PATH))
    args = ap.parse_args()
    cell_re = re.compile(args.cells)

    # 1) Scrape all lm-eval logs
    print(f"Scraping {len(list(BENCH_LOG_DIR.glob('baseline_vllm_*.out')))} lm-eval logs ...")
    bench_results = {}  # (cell, iter) -> {arc_n, hella_n, wino}
    for log in sorted(BENCH_LOG_DIR.glob("baseline_vllm_*.out")):
        parsed = parse_lmeval_log(log)
        if parsed is None: continue
        cell, it, r = parsed
        if not cell_re.match(cell): continue
        bench_results[(cell, it)] = r

    # 2) Pull W&B history
    print("Pulling W&B history (lm_loss, critical_path)...")
    wb = pull_wandb_metrics()
    print(f"  {len(wb)} cells with wandb metrics")

    # 3) Build output rows
    rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 3a. Rows for each lm-eval'd cell at the eval iter
    for (cell, it), r in bench_results.items():
        cfg = parse_cell_hyperparams(cell)
        # Find closest wandb metric to bench_iter
        eval_lm, eval_cp = None, None
        for pt_it, pt_lm, pt_cp in wb.get(cell, []):
            if it is None or abs(pt_it - it) <= 50:  # within 50 iters
                eval_lm = pt_lm if pt_lm is not None else eval_lm
                eval_cp = pt_cp if pt_cp is not None else eval_cp
        bavg = round((r["arc_n"]+r["hella_n"]+r["wino"])/3, 4)
        rows.append({
            "run_name": cell, "bench_iteration": it,
            **cfg,
            "eval_crit_path": round(eval_cp,1) if eval_cp is not None else "",
            "eval_lm_loss": round(eval_lm,4) if eval_lm is not None else "",
            "hellaswag": r["hella_n"], "arc_challenge": r["arc_n"], "winogrande": r["wino"],
            "benchmark_avg": bavg, "limit": args.limit, "timestamp": timestamp,
            "comments": "vLLM TP=8, acc_norm for arc/hella",
            "checkpoint_path": find_checkpoint_path(cell, it, is_hf=True),
        })

    # 3b. Rows for each cell's LATEST W&B metric snapshot (training-only, no benchmark)
    for cell, pts in wb.items():
        if not cell_re.match(cell): continue
        if not pts: continue
        latest = pts[-1]
        it, lm, cp = latest
        # Skip if we already have a benchmark row at this iter
        if (cell, it) in bench_results: continue
        cfg = parse_cell_hyperparams(cell)
        rows.append({
            "run_name": cell, "bench_iteration": it,
            **cfg,
            "eval_crit_path": round(cp,1) if cp is not None else "",
            "eval_lm_loss": round(lm,4) if lm is not None else "",
            "hellaswag": "", "arc_challenge": "", "winogrande": "",
            "benchmark_avg": "", "limit": "", "timestamp": timestamp,
            "comments": "training-only (no lm-eval yet)",
            "checkpoint_path": find_checkpoint_path(cell, it, is_hf=False),
        })

    # 4) Dedupe by (run_name, bench_iteration)
    seen = {}
    for row in rows:
        key = (row["run_name"], row["bench_iteration"])
        # Prefer rows with lm-eval results
        if key not in seen or (row.get("benchmark_avg") and not seen[key].get("benchmark_avg")):
            seen[key] = row
    final_rows = list(seen.values())

    # 5) Write CSV (preserve existing 30B rows, append 235B rows)
    out_path = Path(args.out)
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            for r in csv.DictReader(f):
                # Drop existing 235B rows; keep 30B + pretrained 30B baseline
                if cell_re.match(r.get("run_name","")): continue
                existing.append(r)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        # 30B + other rows first (preserve order they had)
        for r in existing:
            w.writerow({k: r.get(k, "") for k in COLS})
        # Then 235B rows, sorted
        for r in sorted(final_rows, key=lambda x:(x["run_name"], x.get("bench_iteration") or 0)):
            w.writerow({k: r.get(k, "") for k in COLS})

    n_235b = len(final_rows)
    print(f"\nWrote {len(existing)+n_235b} rows total to {out_path}")
    print(f"  - {len(existing)} preserved (30B + non-235B)")
    print(f"  - {n_235b} 235B rows ({sum(1 for r in final_rows if r.get('benchmark_avg'))} with lm-eval, {sum(1 for r in final_rows if not r.get('benchmark_avg'))} training-only)")

if __name__ == "__main__":
    main()
