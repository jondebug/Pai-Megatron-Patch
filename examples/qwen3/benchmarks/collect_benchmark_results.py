#!/usr/bin/env python3
"""Collect benchmark results into a single CSV with training config metadata.

All config fields (rl_enabled, aux_enabled, rl_loss_coeff, etc.) are derived
directly from the run name and checkpoint path.

Usage:
    python collect_benchmark_results.py
    python collect_benchmark_results.py --comment "run_name=some note"
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(
    "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
)
DEFAULT_CSV = SCRIPT_DIR / "benchmark_results.csv"

CSV_COLUMNS = [
    "run_name",
    "bench_iteration",
    "sweep_id",
    "category",
    "rl_enabled",
    "aux_enabled",
    "train_iters",
    "rl_reward_type",
    "rl_loss_coeff",
    "aux_loss_coeff",
    "kl_loss_coeff",
    "eval_crit_path",
    "eval_lm_loss",
    "hellaswag",
    "arc_challenge",
    "winogrande",
    "benchmark_avg",
    "benchmark_time_sec",
    "limit",
    "timestamp",
    "comments",
    "checkpoint_path",
]

SWEEP_ID_MAP = {
    "per_token_rl_sweep": "11oi7yzk",
    "pertoken_rl_push_sweep": "ic7dzzax",
    "pertoken_rl_fixed_sweep": "5bmzj7mn",
    "ppo_multiepoch_v2": "ch2if0ee",
    "ppo_replay_buffer_sweep": "tkumso27",
    "improved_rl_sweep": "8zbkapcs",
    "kl_ab_test_sweep": "1ykmwe84",
    "focused_rl_sweep": "9iafzzke",
}


def _find_sweep_dir_for_run(run_name):
    """Find the sweep_logs directory containing this run's log file."""
    sweep_base = SCRIPT_DIR.parent / "sweep_logs"
    if not sweep_base.exists():
        return None
    for d in sweep_base.iterdir():
        if not d.is_dir():
            continue
        log_file = d / "logs" / "{}.log".format(run_name)
        if log_file.exists():
            return str(d)
    return None


def parse_run_config(run_name):
    """Derive all config fields from the run name string."""
    config = {
        "rl_enabled": False,
        "aux_enabled": False,
        "rl_reward_type": "",
        "rl_loss_coeff": "",
        "aux_loss_coeff": "",
        "kl_loss_coeff": "",
    }

    if run_name == "pretrained_baseline":
        return config

    has_norl = "_norl" in run_name or run_name.startswith("norl_")
    config["rl_enabled"] = not has_norl

    has_noaux = "noaux" in run_name
    aux_match = re.search(r"aux([\d.]+)", run_name)
    if aux_match and not has_noaux:
        coeff = float(aux_match.group(1))
        config["aux_enabled"] = coeff > 0
        config["aux_loss_coeff"] = str(coeff)
    elif has_noaux:
        config["aux_enabled"] = False
        config["aux_loss_coeff"] = "0"
    elif not has_noaux and not aux_match:
        # Name doesn't encode aux info — look up sweep config for the default
        sweep_dir = _find_sweep_dir_for_run(run_name)
        if sweep_dir:
            try:
                combo_path = os.path.join(sweep_dir, "sweep_combinations.json")
                if os.path.exists(combo_path):
                    with open(combo_path) as f:
                        data = json.load(f)
                    fixed = data.get("fixed_params", {})
                    coeff = fixed.get("moe_aux_loss_coeff", 0)
                    if coeff and float(coeff) > 0:
                        config["aux_enabled"] = True
                        config["aux_loss_coeff"] = str(coeff)
            except Exception:
                pass

    rlc_match = re.search(r"rlc([\d.]+)", run_name)
    if rlc_match and config["rl_enabled"]:
        config["rl_loss_coeff"] = rlc_match.group(1)

    kl_match = re.search(r"kl([\d.]+)", run_name)
    if kl_match:
        config["kl_loss_coeff"] = kl_match.group(1)

    if config["rl_enabled"]:
        if "ptload" in run_name:
            config["rl_reward_type"] = "per_token_load_weighted"
        elif "ptbin" in run_name:
            config["rl_reward_type"] = "per_token_topn_binary"
        elif "crit" in run_name:
            config["rl_reward_type"] = "critical_path"

    # For KL sweep runs without rlc in name, look up from sweep config
    if config["rl_enabled"] and not config["rl_loss_coeff"]:
        sweep_dir = _find_sweep_dir_for_run(run_name)
        if sweep_dir:
            try:
                combo_path = os.path.join(sweep_dir, "sweep_combinations.json")
                if os.path.exists(combo_path):
                    with open(combo_path) as f:
                        data = json.load(f)
                    fixed = data.get("fixed_params", {})
                    rlc = fixed.get("rl_loss_coeff")
                    if rlc:
                        config["rl_loss_coeff"] = str(rlc)
            except Exception:
                pass

    return config


def derive_category(config, run_name):
    if run_name == "pretrained_baseline":
        return "pretrained"
    rl = config["rl_enabled"]
    aux = config["aux_enabled"]
    if rl and aux:
        return "rl+aux"
    elif rl:
        return "rl_only"
    elif aux:
        return "aux_only"
    return "other"


def find_benchmark_results(output_dir):
    results = []
    results.extend(
        glob.glob(
            str(output_dir / "*/checkpoint/*/benchmark_results/accuracy_summary.json")
        )
    )
    results.extend(
        glob.glob(str(output_dir / "*/benchmark_results/accuracy_summary.json"))
    )
    results.extend(
        glob.glob(
            str(output_dir / "*/checkpoint/*/benchmark_iter*/accuracy_summary.json")
        )
    )
    return results


def extract_run_name(summary_path):
    parts = Path(summary_path).parts
    for i, p in enumerate(parts):
        if p == "output_router_finetuning" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def extract_checkpoint_path(summary_path):
    p = Path(summary_path)
    bench_idx = None
    for i, part in enumerate(p.parts):
        if part == "benchmark_results" or part.startswith("benchmark_iter"):
            bench_idx = i
            break
    if bench_idx is not None:
        return str(Path(*p.parts[: bench_idx]))
    return ""


def extract_bench_iteration(summary_path):
    """Extract benchmark iteration from path like .../benchmark_iter2000/accuracy_summary.json.
    For benchmark_results/ (non-iteration-specific), reads latest_checkpointed_iteration.txt."""
    p = Path(summary_path)
    for part in p.parts:
        if part.startswith("benchmark_iter"):
            return part.replace("benchmark_iter", "")
    ckpt_dir = extract_checkpoint_path(summary_path)
    if ckpt_dir:
        latest_file = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_file):
            try:
                return open(latest_file).read().strip()
            except Exception:
                pass
    return ""


def load_timing(benchmark_dir):
    timing_path = os.path.join(benchmark_dir, "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_eval_metrics_from_log(output_dir, run_name):
    metrics = {"eval_crit_path": "", "eval_lm_loss": "", "sweep_id": ""}
    sweep_log_base = output_dir.parent / "Pai-Megatron-Patch/examples/qwen3/sweep_logs"
    sweep_log_dirs = glob.glob(str(sweep_log_base / "*/logs"))

    for log_dir in sweep_log_dirs:
        log_file = os.path.join(log_dir, "{}.log".format(run_name))
        if not os.path.exists(log_file):
            continue

        sweep_dir_path = Path(log_dir).parent
        sweep_dir_name = sweep_dir_path.name

        # Try reading sweep_id.txt (written by wandb_sweep_config.py)
        sweep_id_file = sweep_dir_path / "sweep_id.txt"
        if sweep_id_file.exists():
            try:
                metrics["sweep_id"] = sweep_id_file.read_text().strip()
            except Exception:
                pass

        # Fallback to hardcoded map for older sweeps
        if not metrics["sweep_id"]:
            for prefix, sid in SWEEP_ID_MAP.items():
                if sweep_dir_name.startswith(prefix):
                    metrics["sweep_id"] = sid
                    break

        try:
            with open(log_file) as f:
                content = f.read()
            eval_lines = re.findall(r"validation loss at iteration.*", content)
            if eval_lines:
                last_eval = eval_lines[-1]
                lm = re.search(r"lm loss value:\s+([\d.E+-]+)", last_eval)
                cp = re.search(
                    r"num_tokens_on_critical_path value:\s+([\d.E+-]+)", last_eval
                )
                if lm:
                    metrics["eval_lm_loss"] = "{:.4f}".format(float(lm.group(1)))
                if cp:
                    v = float(cp.group(1))
                    if v > 0:
                        metrics["eval_crit_path"] = "{:.1f}".format(v)
            if not metrics["eval_crit_path"]:
                cp_matches = re.findall(
                    r"num_tokens_on_critical_path:\s+([\d.E+-]+)", content
                )
                if cp_matches:
                    last_values = [
                        float(v) for v in cp_matches[-50:] if float(v) > 0
                    ]
                    if last_values:
                        import statistics

                        metrics["eval_crit_path"] = "{:.1f}".format(
                            statistics.mean(last_values)
                        )
        except Exception:
            continue
    return metrics


def collect_results(output_dir, comments=None):
    if comments is None:
        comments = {}

    summary_files = find_benchmark_results(output_dir)
    if not summary_files:
        print("No benchmark results found.", file=sys.stderr)
        return []

    rows = []
    for summary_path in sorted(summary_files):
        run_name = extract_run_name(summary_path)
        bench_iteration = extract_bench_iteration(summary_path)
        benchmark_dir = os.path.dirname(summary_path)

        with open(summary_path) as f:
            summary = json.load(f)

        scores = summary.get("scores", {})
        avg = summary.get("average", 0)
        timing = load_timing(benchmark_dir)
        eval_metrics = load_eval_metrics_from_log(output_dir, run_name)
        config = parse_run_config(run_name)
        category = derive_category(config, run_name)

        train_iters = ""
        m = re.search(r"ti-(\d+)", summary_path)
        if m:
            train_iters = m.group(1)

        ckpt_path = extract_checkpoint_path(summary_path)

        limit = ""
        results_json = os.path.join(benchmark_dir, "results.json")
        if os.path.exists(results_json):
            try:
                with open(results_json) as f:
                    rj = json.load(f)
                lim = rj.get("config", {}).get("limit")
                if lim is not None:
                    limit = (
                        str(int(lim))
                        if isinstance(lim, float) and lim == int(lim)
                        else str(lim)
                    )
            except Exception:
                pass

        ts = ""
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(summary_path)).strftime(
                "%Y-%m-%d %H:%M"
            )
        except Exception:
            pass

        def fmt_score(val):
            if val is None:
                return ""
            return "{:.1f}".format(val) if isinstance(val, float) else str(val)

        row = {
            "run_name": run_name,
            "bench_iteration": bench_iteration,
            "sweep_id": eval_metrics.get("sweep_id", ""),
            "category": category,
            "rl_enabled": str(config["rl_enabled"]),
            "aux_enabled": str(config["aux_enabled"]),
            "train_iters": train_iters,
            "rl_reward_type": config["rl_reward_type"],
            "rl_loss_coeff": config["rl_loss_coeff"],
            "aux_loss_coeff": config["aux_loss_coeff"],
            "kl_loss_coeff": config["kl_loss_coeff"],
            "checkpoint_path": ckpt_path,
            "eval_crit_path": eval_metrics.get("eval_crit_path", ""),
            "eval_lm_loss": eval_metrics.get("eval_lm_loss", ""),
            "hellaswag": fmt_score(scores.get("hellaswag")),
            "arc_challenge": fmt_score(scores.get("arc_challenge")),
            "winogrande": fmt_score(scores.get("winogrande")),
            "benchmark_avg": "{:.2f}".format(avg) if avg else "",
            "benchmark_time_sec": (
                str(timing.get("total_seconds", ""))
                if timing.get("total_seconds")
                else ""
            ),
            "limit": limit,
            "timestamp": ts,
            "comments": comments.get(run_name, ""),
        }

        rows.append(row)
        print("  Collected: {} [{}] iter={} (avg={:.1f}%)".format(run_name, category, bench_iteration, avg))

    return rows


def _row_key(row):
    return (row.get("run_name", ""), row.get("bench_iteration", "latest"))


def write_csv(rows, csv_path):
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[_row_key(row)] = row

    for row in rows:
        key = _row_key(row)
        if key in existing:
            old = existing[key]
            if old.get("comments") and not row.get("comments"):
                row["comments"] = old["comments"]
            if not row.get("eval_crit_path") and old.get("eval_crit_path"):
                row["eval_crit_path"] = old["eval_crit_path"]
            if not row.get("eval_lm_loss") and old.get("eval_lm_loss"):
                row["eval_lm_loss"] = old["eval_lm_loss"]
            if not row.get("sweep_id") and old.get("sweep_id"):
                row["sweep_id"] = old["sweep_id"]
        existing[key] = row

    all_rows = sorted(existing.values(), key=lambda r: (r.get("run_name", ""), r.get("bench_iteration", "")))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nCSV written: {} ({} rows)".format(csv_path, len(all_rows)))


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results into CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--comment", type=str, action="append", default=[])
    args = parser.parse_args()

    comments = {}
    for c in args.comment:
        if "=" in c:
            name, text = c.split("=", 1)
            comments[name.strip()] = text.strip()

    print("Scanning: {}".format(args.output_dir))
    print("Output CSV: {}".format(args.csv))
    print()

    rows = collect_results(args.output_dir, comments)
    if rows:
        write_csv(rows, args.csv)
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
