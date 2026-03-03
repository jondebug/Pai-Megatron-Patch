#!/usr/bin/env python3
"""Collect benchmark results into a single CSV with training config metadata.

Usage:
    # Auto-collect from all benchmark_results dirs:
    python collect_benchmark_results.py

    # Add a comment to a specific run:
    python collect_benchmark_results.py --comment "ptbin_n6_rlc0.1_g0.2_noaux_norm_PPO_ent0.01_c256_r09=best RL-only run"

    # Specify custom paths:
    python collect_benchmark_results.py --output-dir /path/to/output --sweep-logs-dir /path/to/sweep_logs
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_OUTPUT_DIR = Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning")
DEFAULT_SWEEP_LOGS = SCRIPT_DIR.parent / "sweep_logs"
DEFAULT_CSV = SCRIPT_DIR / "benchmark_results.csv"

CSV_COLUMNS = [
    "run_name",
    "timestamp",
    "train_iters",
    "rl_reward_type",
    "rl_loss_coeff",
    "rl_discount_factor",
    "rl_normalize_rewards",
    "rl_reward_topn",
    "moe_aux_loss_coeff",
    "use_rl_loss",
    "rl_algorithm",
    "rl_ppo_baseline_type",
    "rl_critic_hidden_dims",
    "rl_ppo_entropy_coeff",
    "kl_loss_coeff",
    "eval_crit_path",
    "eval_lm_loss",
    "hellaswag",
    "arc_challenge",
    "winogrande",
    "benchmark_avg",
    "benchmark_time_sec",
    "hellaswag_time_sec",
    "arc_challenge_time_sec",
    "winogrande_time_sec",
    "limit",
    "comments",
]


def find_benchmark_results(output_dir: Path):
    """Find all accuracy_summary.json files under the output directory."""
    pattern = str(output_dir / "*/checkpoint/*/benchmark_results/accuracy_summary.json")
    return glob.glob(pattern)


def extract_run_name(summary_path: str) -> str:
    """Extract run name from the path."""
    parts = Path(summary_path).parts
    for i, p in enumerate(parts):
        if p == "output_router_finetuning" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def extract_run_index(run_name: str) -> int:
    """Extract run index from name like 'ptbin_n6_rlc0.1_..._r09' -> 9."""
    m = re.search(r'_r(\d+)$', run_name)
    return int(m.group(1)) if m else -1


def load_sweep_configs(sweep_logs_dir: Path) -> dict:
    """Load all sweep_combinations.json files and build a lookup.

    Returns dict: run_index -> (sweep_params, fixed_params)
    """
    configs = {}
    combo_files = glob.glob(str(sweep_logs_dir / "*/sweep_combinations.json"))

    for combo_file in combo_files:
        try:
            with open(combo_file) as f:
                data = json.load(f)
            combinations = data.get("combinations", [])
            fixed = data.get("fixed_params", {})
            for idx, combo in enumerate(combinations):
                merged = {**fixed, **combo}
                configs[idx] = (combo, fixed, merged)
        except Exception:
            continue

    return configs


def extract_train_iters(run_name: str) -> str:
    """Extract train_iters from checkpoint path name."""
    m = re.search(r'ti-(\d+)', run_name)
    return m.group(1) if m else ""


def load_timing(benchmark_dir: str) -> dict:
    """Load timing.json if it exists."""
    timing_path = os.path.join(benchmark_dir, "timing.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            return json.load(f)
    return {}


def load_eval_metrics_from_log(output_dir: Path, run_name: str) -> dict:
    """Try to extract final eval metrics from training log files."""
    metrics = {"eval_crit_path": "", "eval_lm_loss": ""}

    sweep_log_dirs = glob.glob(str(output_dir.parent / "Pai-Megatron-Patch/examples/qwen3/sweep_logs/*/logs"))
    for log_dir in sweep_log_dirs:
        log_file = os.path.join(log_dir, f"{run_name}.log")
        if not os.path.exists(log_file):
            continue

        try:
            with open(log_file) as f:
                content = f.read()

            # Find last eval line
            eval_lines = re.findall(r'validation loss at iteration.*', content)
            if eval_lines:
                last_eval = eval_lines[-1]
                lm = re.search(r'lm loss value:\s+([\d.E+-]+)', last_eval)
                cp = re.search(r'num_tokens_on_critical_path value:\s+([\d.E+-]+)', last_eval)
                if lm:
                    metrics["eval_lm_loss"] = f"{float(lm.group(1)):.4f}"
                if cp:
                    v = float(cp.group(1))
                    if v > 0:
                        metrics["eval_crit_path"] = f"{v:.1f}"

            # If eval critical path is 0, try getting from training metrics
            if not metrics["eval_crit_path"]:
                cp_matches = re.findall(r'num_tokens_on_critical_path:\s+([\d.E+-]+)', content)
                if cp_matches:
                    last_values = [float(v) for v in cp_matches[-50:] if float(v) > 0]
                    if last_values:
                        import statistics
                        metrics["eval_crit_path"] = f"{statistics.mean(last_values):.1f}"
        except Exception:
            continue

    return metrics


def collect_results(output_dir: Path, sweep_logs_dir: Path, comments: dict = None) -> list:
    """Collect all benchmark results into a list of row dicts."""
    if comments is None:
        comments = {}

    summary_files = find_benchmark_results(output_dir)
    if not summary_files:
        print("No benchmark results found.", file=sys.stderr)
        return []

    sweep_configs = load_sweep_configs(sweep_logs_dir)

    rows = []
    for summary_path in sorted(summary_files):
        run_name = extract_run_name(summary_path)
        run_index = extract_run_index(run_name)
        benchmark_dir = os.path.dirname(summary_path)

        with open(summary_path) as f:
            summary = json.load(f)

        scores = summary.get("scores", {})
        avg = summary.get("average", 0)

        # Load timing
        timing = load_timing(benchmark_dir)

        # Load training config from sweep combinations
        config = {}
        if run_index >= 0 and run_index in sweep_configs:
            _, _, merged = sweep_configs[run_index]
            config = merged

        # Load eval metrics from training logs
        eval_metrics = load_eval_metrics_from_log(output_dir, run_name)

        # Extract limit from lm-eval results.json
        limit = ""
        results_json = os.path.join(benchmark_dir, "results.json")
        if os.path.exists(results_json):
            try:
                with open(results_json) as f:
                    rj = json.load(f)
                lim = rj.get("config", {}).get("limit")
                if lim is not None:
                    limit = str(int(lim)) if isinstance(lim, float) and lim == int(lim) else str(lim)
            except Exception:
                pass

        # Get checkpoint path timestamp as benchmark timestamp
        ts = ""
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(summary_path)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        # Extract train_iters from checkpoint subdir name
        ckpt_parts = Path(summary_path).parts
        train_iters = ""
        for part in ckpt_parts:
            m = re.search(r'ti-(\d+)', part)
            if m:
                train_iters = m.group(1)
                break

        row = {
            "run_name": run_name,
            "timestamp": ts,
            "train_iters": train_iters or config.get("train_iters", ""),
            "rl_reward_type": config.get("rl_reward_type", ""),
            "rl_loss_coeff": config.get("rl_loss_coeff", ""),
            "rl_discount_factor": config.get("rl_discount_factor", ""),
            "rl_normalize_rewards": config.get("rl_normalize_rewards", ""),
            "rl_reward_topn": config.get("rl_reward_topn", ""),
            "moe_aux_loss_coeff": config.get("moe_aux_loss_coeff", ""),
            "use_rl_loss": config.get("use_rl_loss", ""),
            "rl_algorithm": config.get("rl_algorithm", ""),
            "rl_ppo_baseline_type": config.get("rl_ppo_baseline_type", ""),
            "rl_critic_hidden_dims": str(config.get("rl_critic_hidden_dims", "")),
            "rl_ppo_entropy_coeff": config.get("rl_ppo_entropy_coeff", ""),
            "kl_loss_coeff": config.get("kl_loss_coeff", ""),
            "eval_crit_path": eval_metrics.get("eval_crit_path", ""),
            "eval_lm_loss": eval_metrics.get("eval_lm_loss", ""),
            "hellaswag": f"{scores.get('hellaswag', '')}" if scores.get('hellaswag') is not None else "",
            "arc_challenge": f"{scores.get('arc_challenge', '')}" if scores.get('arc_challenge') is not None else "",
            "winogrande": f"{scores.get('winogrande', '')}" if scores.get('winogrande') is not None else "",
            "benchmark_avg": f"{avg:.2f}" if avg else "",
            "benchmark_time_sec": timing.get("total_seconds", ""),
            "hellaswag_time_sec": timing.get("per_task", {}).get("hellaswag", ""),
            "arc_challenge_time_sec": timing.get("per_task", {}).get("arc_challenge", ""),
            "winogrande_time_sec": timing.get("per_task", {}).get("winogrande", ""),
            "limit": limit,
            "comments": comments.get(run_name, ""),
        }

        rows.append(row)
        print(f"  Collected: {run_name} (avg={avg:.1f}%)")

    return rows


def write_csv(rows: list, csv_path: Path):
    """Write rows to CSV, merging with existing data."""
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["run_name"]] = row

    # Merge: new data overwrites existing (except comments which are preserved)
    for row in rows:
        name = row["run_name"]
        if name in existing and existing[name].get("comments") and not row.get("comments"):
            row["comments"] = existing[name]["comments"]
        existing[name] = row

    # Write
    all_rows = sorted(existing.values(), key=lambda r: r.get("run_name", ""))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV written: {csv_path} ({len(all_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results into CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing run output dirs")
    parser.add_argument("--sweep-logs-dir", type=Path, default=DEFAULT_SWEEP_LOGS,
                        help="Directory containing sweep_logs subdirs")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Output CSV path")
    parser.add_argument("--comment", type=str, action="append", default=[],
                        help="Add comment: 'run_name=comment text' (can repeat)")
    args = parser.parse_args()

    # Parse comments
    comments = {}
    for c in args.comment:
        if "=" in c:
            name, text = c.split("=", 1)
            comments[name.strip()] = text.strip()

    print(f"Scanning: {args.output_dir}")
    print(f"Sweep logs: {args.sweep_logs_dir}")
    print(f"Output CSV: {args.csv}")
    print()

    rows = collect_results(args.output_dir, args.sweep_logs_dir, comments)

    if rows:
        write_csv(rows, args.csv)
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
