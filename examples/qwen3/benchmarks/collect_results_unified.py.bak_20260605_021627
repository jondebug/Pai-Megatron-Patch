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
    "wt_decode_tps","wt_ttft_ms","wt_e2e_ms",
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
    # Strip trailing _iter<N> (used in limit=inf run names) so the cell maps to
    # the same hyperparams + wandb CP as the base cell.
    name = re.sub(r"_iter\d+$", "", name)
    # Parse lm-eval limit from header ("Limit:   1000" or "Limit:   inf").
    ml = re.search(r"Limit:\s+(\S+)", txt)
    limit = ml.group(1) if ml else "1000"
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
        return name, bench_iter, limit, r
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
        if not r.name or not r.name.startswith("235b"):
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


def scan_all_distcp():
    """Every distcp checkpoint on disk: (cell, iter) -> checkpoint_path. This makes
    the CSV a complete registry — even cells never benchmarked appear with their path."""
    out = {}
    if not OUTPUT_BASE.is_dir():
        return out
    for cell_dir in sorted(OUTPUT_BASE.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("235b"):
            continue
        ck = cell_dir / "checkpoint"
        if not ck.is_dir():
            continue
        for sub in ck.iterdir():
            if not sub.is_dir():
                continue
            for d in sub.iterdir():
                m = re.match(r"iter_(\d+)$", d.name)
                if m and any(d.glob("*.distcp")):
                    out[(cell_dir.name, int(m.group(1)))] = str(d)
    return out


def pull_walltime():
    """(cell, iter) -> {wt_decode_tps, wt_ttft_ms, wt_e2e_ms} from cp_latency_results JSONs."""
    wtdir = Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results")
    out = {}
    if not wtdir.is_dir():
        return out
    for f in sorted(wtdir.glob("vllm_pretrained_235b_vs_*.json")):
        m = re.search(r"_vs_(.+?)_iter(\d+)", f.name)
        if not m:
            continue
        cell, it = m.group(1), int(m.group(2))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for mdl in d.get("models", []):
            if mdl.get("name", "").startswith("pretrained"):
                continue
            cells = mdl.get("cells", {})
            c = cells.get("plen256_bs8") or (next(iter(cells.values())) if cells else None)
            if c:
                out[(cell, it)] = {
                    "wt_decode_tps": round(c.get("decode_tps", 0), 2),
                    "wt_ttft_ms": round(c.get("ttft_ms_mean", 0), 1),
                    "wt_e2e_ms": round(c.get("end_to_end_ms_mean", 0), 1),
                }
    return out



import glob as _glob
_SWEEPLOGS="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/sweep_logs"
_LOGMAP=None
def scrape_cp_from_logs(cell, target_iter):
    """Recover num_tokens_on_critical_path from training logs for cells whose
    wandb run lacks critical_eval/critical_path (e.g. old 235b-rladv/pareto/klv)."""
    global _LOGMAP
    if _LOGMAP is None:
        _LOGMAP={}
        for lg in _glob.glob(f"{_SWEEPLOGS}/*/logs/*.log"):
            import os as _os
            _LOGMAP.setdefault(_os.path.basename(lg)[:-4], []).append(lg)
    best=None; bestd=1e9
    for lg in _LOGMAP.get(cell, []):
        try: txt=open(lg, errors="ignore").read()
        except Exception: continue
        for m in re.finditer(r"iteration\s+(\d+)/.*?num_tokens_on_critical_path:\s*([0-9.E+]+)", txt):
            it=int(m.group(1)); d=abs(it-target_iter)
            if d<bestd: bestd=d; best=float(m.group(2))
    return (best, bestd) if best is not None else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--cells", default=r"^(235b)", help="cell name regex")
    ap.add_argument("--out", default=str(CSV_PATH))
    args = ap.parse_args()
    cell_re = re.compile(args.cells)

    # 1) Scrape all lm-eval logs
    print(f"Scraping {len(list(BENCH_LOG_DIR.glob('baseline_vllm_*.out')))} lm-eval logs ...")
    bench_results = {}  # (cell, iter, limit) -> {arc_n, hella_n, wino}
    for log in sorted(BENCH_LOG_DIR.glob("baseline_vllm_*.out")):
        parsed = parse_lmeval_log(log)
        if parsed is None: continue
        cell, it, limit, r = parsed
        if not cell_re.match(cell): continue
        bench_results[(cell, it, limit)] = r

    # 2) Pull W&B history
    print("Pulling W&B history (lm_loss, critical_path)...")
    wb = pull_wandb_metrics()
    print(f"  {len(wb)} cells with wandb metrics")
    all_distcp = scan_all_distcp()
    print(f"  {len(all_distcp)} distcp checkpoints on disk")
    walltime = pull_walltime()
    print(f"  {len(walltime)} checkpoints with vLLM walltime")

    # 3) Build output rows
    rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 3a. Rows for each lm-eval'd cell at the eval iter
    for (cell, it, limit), r in bench_results.items():
        cfg = parse_cell_hyperparams(cell)
        # Find closest wandb metric to bench_iter
        eval_lm, eval_cp = None, None
        for pt_it, pt_lm, pt_cp in wb.get(cell, []):
            if it is None or abs(pt_it - it) <= 50:  # within 50 iters
                eval_lm = pt_lm if pt_lm is not None else eval_lm
                eval_cp = pt_cp if pt_cp is not None else eval_cp
        # Pretrained baseline has no wandb run; pin its critical path to the
        # established 235B baseline value so it always plots.
        if eval_cp is None and cell.startswith("pretrained_235b"):
            eval_cp = 8800.0
        # Fallback: recover CP from training logs for cells lacking wandb critical_path
        if eval_cp is None and it is not None:
            lcp, ld = scrape_cp_from_logs(cell, it)
            if lcp is not None and ld <= 100:
                eval_cp = lcp
        # Store as percentages (0-100) to match the existing 30B format used by generate_pareto.py
        bavg_pct = round((r["arc_n"]+r["hella_n"]+r["wino"])/3 * 100, 2)
        rows.append({
            "run_name": cell, "bench_iteration": it,
            **cfg,
            "eval_crit_path": round(eval_cp,1) if eval_cp is not None else "",
            "eval_lm_loss": round(eval_lm,4) if eval_lm is not None else "",
            "hellaswag": round(r["hella_n"]*100, 2), "arc_challenge": round(r["arc_n"]*100, 2),
            "winogrande": round(r["wino"]*100, 2),
            "benchmark_avg": bavg_pct, "limit": limit, "timestamp": timestamp,
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
        if any((cell, it, L) in bench_results for L in ("1000", "inf")): continue
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

    # 3c) Registry rows: ensure EVERY distcp checkpoint has a row (path + hyperparams),
    #     even if never benchmarked and no wandb metric.
    have = {(r["run_name"], r["bench_iteration"]) for r in rows}
    for (cell, it), path in all_distcp.items():
        if not cell_re.match(cell): continue
        if (cell, it) in have: continue
        cfg = parse_cell_hyperparams(cell)
        # attach wandb metric if available near this iter
        lm_v = cp_v = None
        for pt_it, pt_lm, pt_cp in wb.get(cell, []):
            if abs(pt_it - it) <= 50:
                lm_v = pt_lm if pt_lm is not None else lm_v
                cp_v = pt_cp if pt_cp is not None else cp_v
        rows.append({
            "run_name": cell, "bench_iteration": it, **cfg,
            "eval_crit_path": round(cp_v,1) if cp_v is not None else "",
            "eval_lm_loss": round(lm_v,4) if lm_v is not None else "",
            "hellaswag": "", "arc_challenge": "", "winogrande": "",
            "benchmark_avg": "", "limit": "", "timestamp": timestamp,
            "comments": "registry (distcp on disk, not yet benchmarked)",
            "checkpoint_path": path,
        })

    # 3d) Attach vLLM walltime to every row that has a matching checkpoint
    for r in rows:
        try: it = int(r["bench_iteration"])
        except (ValueError, TypeError): continue
        wt = walltime.get((r["run_name"], it))
        if wt:
            r.update(wt)

    # 4) Dedupe by (run_name, bench_iteration)
    seen = {}
    for row in rows:
        key = (row["run_name"], row["bench_iteration"], row.get("limit", ""))
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
