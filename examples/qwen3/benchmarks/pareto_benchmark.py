#!/usr/bin/env python3
"""
Find Pareto-optimal checkpoints from a W&B sweep and optionally benchmark them.

Usage:
    # List Pareto-optimal points at eval step 3000
    python pareto_benchmark.py --sweep-id qho6ccev --step 3000

    # Also submit benchmark jobs for Pareto-optimal checkpoints
    python pareto_benchmark.py --sweep-id qho6ccev --step 3000 --benchmark

    # Use final eval (last logged step per run)
    python pareto_benchmark.py --sweep-id qho6ccev --step final

    # Custom output base path
    python pareto_benchmark.py --sweep-id qho6ccev --step 3000 --output-base /path/to/checkpoints
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_sweep_data(sweep_id, entity="nvr-israel", project="qwen3-router-training"):
    import wandb
    api = wandb.Api(timeout=120)
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return sweep


def get_eval_at_step(run, target_step):
    """Get eval metrics at a specific training step from W&B history.

    Uses W&B's history() with pandas=False for faster bulk retrieval
    instead of scan_history which pages through every logged step.
    """
    cp_key = "critical_eval/critical_path"
    lm_key = "critical_eval/lm_loss"
    fallback_cp = "eval/num_tokens_on_critical_path"
    fallback_lm = "eval/lm loss"

    if target_step == "final":
        s = dict(run.summary)
        cp = s.get(cp_key) or s.get(fallback_cp)
        lm = s.get(lm_key) or s.get(fallback_lm)
        iteration = s.get("iteration", "?")
        if cp is not None and lm is not None:
            return {"cp": float(cp), "lm": float(lm), "step": iteration}
        return None

    target_step = int(target_step)

    try:
        keys = [cp_key, lm_key, fallback_cp, fallback_lm, "iteration", "_step"]
        hist = run.history(keys=keys, pandas=False, samples=500)
        best = None
        for row in hist:
            step = row.get("_step") or row.get("iteration")
            if step is None:
                continue
            step = int(step)
            cp = row.get(cp_key) or row.get(fallback_cp)
            lm = row.get(lm_key) or row.get(fallback_lm)
            if cp is None or lm is None:
                continue
            cp, lm = float(cp), float(lm)
            if math.isnan(cp) or math.isnan(lm):
                continue
            if step == target_step:
                return {"cp": cp, "lm": lm, "step": step}
            if step <= target_step:
                if best is None or step > best["step"]:
                    best = {"cp": cp, "lm": lm, "step": step}
        return best
    except Exception:
        pass

    best = None
    for row in run.scan_history(
        keys=[cp_key, lm_key, fallback_cp, fallback_lm, "iteration", "_step"],
        page_size=500,
    ):
        step = row.get("_step") or row.get("iteration")
        if step is None:
            continue
        step = int(step)
        cp = row.get(cp_key) or row.get(fallback_cp)
        lm = row.get(lm_key) or row.get(fallback_lm)
        if cp is None or lm is None:
            continue
        cp, lm = float(cp), float(lm)
        if math.isnan(cp) or math.isnan(lm):
            continue
        if step == target_step:
            return {"cp": cp, "lm": lm, "step": step}
        if step <= target_step:
            if best is None or step > best["step"]:
                best = {"cp": cp, "lm": lm, "step": step}

    return best


def compute_pareto_front(points):
    """Compute Pareto front minimizing both CP and LM loss."""
    sorted_pts = sorted(points, key=lambda p: p["cp"])
    pareto = []
    best_lm = float("inf")
    for p in sorted_pts:
        if p["lm"] < best_lm:
            pareto.append(p)
            best_lm = p["lm"]
    return pareto


def find_checkpoint_dir(run_name, output_base):
    """Find the Megatron checkpoint directory for a run."""
    run_dir = os.path.join(output_base, run_name)
    if not os.path.isdir(run_dir):
        return None

    ckpt_dir = os.path.join(run_dir, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return None

    subdirs = os.listdir(ckpt_dir)
    if not subdirs:
        return None

    inner = os.path.join(ckpt_dir, subdirs[0])
    latest_file = os.path.join(inner, "latest_checkpointed_iteration.txt")
    if os.path.isfile(latest_file):
        return inner

    return None


def submit_benchmark(checkpoint_dir, run_name, wandb_run_id, wandb_project, limit=None):
    """Submit a SLURM benchmark job for a single checkpoint."""
    script_dir = Path(__file__).parent
    submit_script = script_dir / "submit_benchmark.sh"

    if not submit_script.exists():
        print(f"  ERROR: {submit_script} not found")
        return None

    cmd = [
        "sbatch",
        str(submit_script),
        "--checkpoint-dir", checkpoint_dir,
        "--run-name", run_name,
        "--wandb-run-id", wandb_run_id,
        "--wandb-project", wandb_project,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"  ERROR: sbatch failed: {result.stderr.strip()}")
        return None


def submit_batch_benchmark(entries, wandb_project, limit=None):
    """Submit a single SLURM job that benchmarks multiple checkpoints sequentially.

    Args:
        entries: list of dicts with checkpoint_dir, run_name, wandb_run_id
        wandb_project: W&B project name
        limit: optional sample limit per task

    Returns:
        job_id or None
    """
    script_dir = Path(__file__).parent
    batch_script = script_dir / "submit_batch_benchmark.sh"

    if not batch_script.exists():
        print(f"  ERROR: {batch_script} not found")
        return None

    import time
    manifest_path = script_dir / f"_manifest_{os.getpid()}_{int(time.time()*1000)}.json"
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    cmd = [
        "sbatch",
        str(batch_script),
        "--manifest", str(manifest_path),
        "--wandb-project", wandb_project,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"  ERROR: sbatch failed: {result.stderr.strip()}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Find Pareto-optimal checkpoints from a W&B sweep and optionally benchmark them."
    )
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument(
        "--step",
        required=True,
        help='Training step to evaluate at (integer or "final")',
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Submit SLURM benchmark jobs for Pareto-optimal checkpoints",
    )
    parser.add_argument(
        "--output-base",
        default="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning",
        help="Base directory containing run output directories",
    )
    parser.add_argument(
        "--entity", default="nvr-israel", help="W&B entity"
    )
    parser.add_argument(
        "--project", default="qwen3-router-training", help="W&B project"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Save all results to CSV (default: <sweep_id>_pareto_step<step>.csv)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark ALL runs, not just Pareto-optimal ones",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per benchmark task (e.g. 1000 for faster runs). "
             "Default: no limit (full dataset).",
    )
    parser.add_argument(
        "--per-job",
        type=int,
        default=1,
        help="Number of checkpoints to benchmark per SLURM job (default: 1). "
             "Use higher values to avoid QOS job submission limits. "
             "E.g. --per-job 5 puts 5 checkpoints in each job.",
    )
    args = parser.parse_args()

    print(f"Fetching sweep {args.sweep_id}...")
    sweep = get_sweep_data(args.sweep_id, args.entity, args.project)
    runs = [r for r in sweep.runs if r.state in ("finished", "crashed")]
    print(f"Found {len(runs)} finished runs")

    print(f"\nCollecting eval metrics at step={args.step}...")
    points = []
    missing = []
    for i, run in enumerate(runs):
        sys.stdout.write(f"\r  Processing {i+1}/{len(runs)}: {run.name[:50]}...")
        sys.stdout.flush()
        metrics = get_eval_at_step(run, args.step)
        if metrics is None:
            missing.append(run.name)
            continue
        points.append({
            "name": run.name,
            "run_id": run.id,
            "cp": metrics["cp"],
            "lm": metrics["lm"],
            "eval_step": metrics["step"],
        })
    print(f"\r  Collected metrics for {len(points)} runs" + " " * 40)

    if missing:
        print(f"  Missing eval at step {args.step}: {len(missing)} runs")

    if not points:
        print("No data points found. Exiting.")
        return

    pareto = compute_pareto_front(points)

    for p in points:
        p["pareto"] = p in pareto

    points.sort(key=lambda p: p["cp"])

    print(f"\n{'='*100}")
    print(f"ALL RUNS AT STEP {args.step} (sorted by CP)")
    print(f"{'='*100}")
    print(f"{'#':>3s}  {'Run Name':55s} {'CP':>8s} {'LM':>8s} {'Step':>5s} {'Pareto':>7s}")
    print("-" * 90)
    for i, p in enumerate(points):
        marker = "  <<<" if p["pareto"] else ""
        print(
            f"{i+1:3d}  {p['name'][:55]:55s} {p['cp']:8.1f} {p['lm']:8.4f} {str(p['eval_step']):>5s} {marker}"
        )

    print(f"\n{'='*100}")
    print(f"PARETO FRONT ({len(pareto)} points)")
    print(f"{'='*100}")
    print(f"{'#':>3s}  {'Run Name':55s} {'CP':>8s} {'LM':>8s}")
    print("-" * 80)
    for i, p in enumerate(pareto):
        print(f"{i+1:3d}  {p['name'][:55]:55s} {p['cp']:8.1f} {p['lm']:8.4f}")

    csv_path = args.csv or os.path.join(
        Path(__file__).parent,
        f"{args.sweep_id}_pareto_step{args.step}.csv",
    )
    fieldnames = ["name", "run_id", "cp", "lm", "eval_step", "pareto", "checkpoint_dir"]
    for p in points:
        ckpt = find_checkpoint_dir(p["name"], args.output_base)
        p["checkpoint_dir"] = ckpt or ""

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    print(f"\nSaved to {csv_path}")

    if args.benchmark:
        to_benchmark = points if args.all else pareto
        limit_str = f" (limit={args.limit})" if args.limit else " (full dataset)"
        print(f"\n{'='*100}")
        print(f"SUBMITTING BENCHMARK JOBS ({len(to_benchmark)} runs){limit_str}")
        print(f"{'='*100}")

        bench_step = args.step
        pending = []
        for p in to_benchmark:
            ckpt_dir = find_checkpoint_dir(p["name"], args.output_base)
            if not ckpt_dir:
                print(f"  SKIP {p['name']}: checkpoint not found")
                continue

            actual_iter = p.get("eval_step")
            if bench_step == "final" or actual_iter is None:
                results_subdir = "benchmark_results"
                bench_iter = None
            else:
                results_subdir = f"benchmark_iter{actual_iter}"
                bench_iter = actual_iter

            already_done = os.path.join(ckpt_dir, results_subdir, "accuracy_summary.json")
            if os.path.exists(already_done):
                print(f"  SKIP {p['name']}: already benchmarked")
                continue

            # Check that the iteration checkpoint actually exists on disk
            if bench_iter is not None:
                iter_dir = os.path.join(ckpt_dir, f"iter_{int(bench_iter):07d}")
                if not os.path.isdir(iter_dir):
                    print(f"  SKIP {p['name']}: iter {bench_iter} checkpoint not on disk")
                    continue

            pending.append({
                "checkpoint_dir": ckpt_dir,
                "run_name": p["name"],
                "wandb_run_id": p["run_id"],
                "iteration": bench_iter,
            })

        if not pending:
            print("  All checkpoints already benchmarked.")
        elif args.per_job <= 1:
            submitted = []
            for entry in pending:
                job_id = submit_benchmark(
                    entry["checkpoint_dir"], entry["run_name"],
                    entry["wandb_run_id"], args.project, limit=args.limit,
                )
                if job_id:
                    print(f"  SUBMITTED {entry['run_name']}: job {job_id}")
                    submitted.append((entry["run_name"], job_id))
                else:
                    print(f"  FAILED {entry['run_name']}")
            if submitted:
                print(f"\n{len(submitted)} benchmark jobs submitted.")
                print("Monitor with: squeue -u $USER")
                print("Cancel all:   scancel " + " ".join(j for _, j in submitted))
        else:
            batches = []
            for i in range(0, len(pending), args.per_job):
                batches.append(pending[i : i + args.per_job])

            submitted = []
            for batch_idx, batch in enumerate(batches):
                names = [e["run_name"] for e in batch]
                print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} checkpoints")
                for n in names:
                    print(f"    - {n}")
                job_id = submit_batch_benchmark(batch, args.project, limit=args.limit)
                if job_id:
                    print(f"    -> job {job_id}")
                    submitted.append(job_id)
                else:
                    print(f"    -> FAILED to submit")

            if submitted:
                print(f"\n{len(submitted)} batch jobs submitted ({len(pending)} checkpoints total).")
                print("Monitor with: squeue -u $USER")
                print("Cancel all:   scancel " + " ".join(submitted))

        if pending:
            print(f"\nAfter all jobs finish, collect results into CSV:")
            collect_script = Path(__file__).parent / "collect_benchmark_results.py"
            print(f"  python3 {collect_script}")
    else:
        print(f"\nTo benchmark Pareto-optimal points, run:")
        limit_hint = f" --limit 1000" if not args.limit else ""
        print(f"  python {__file__} --sweep-id {args.sweep_id} --step {args.step} --benchmark{limit_hint}")


if __name__ == "__main__":
    main()
