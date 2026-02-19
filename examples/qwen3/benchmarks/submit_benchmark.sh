#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmark_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmark_%j.err

# =============================================================================
# Post-Training Benchmark Script
#
# Converts a Megatron checkpoint to HF format, runs lm-evaluation-harness,
# and logs results back to the same WandB training run.
#
# Called automatically by wandb_agent_runner.py after training completes.
# Can also be run manually:
#   sbatch submit_benchmark.sh \
#     --checkpoint-dir /path/to/megatron/ckpt \
#     --wandb-run-id <run_id> \
#     --wandb-project qwen3-router-training \
#     --run-name my_run_name
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"

CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
BENCHMARK_LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs"

# Defaults
CHECKPOINT_DIR=""
WANDB_RUN_ID=""
WANDB_PROJECT="qwen3-router-training"
RUN_NAME=""
MODEL_SIZE="A3B"
BATCH_SIZE=8
TASKS="mmlu,hellaswag,arc_challenge,winogrande"

# Parse arguments (passed after -- by sbatch, or directly)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)   CHECKPOINT_DIR="$2"; shift 2 ;;
        --wandb-run-id)     WANDB_RUN_ID="$2"; shift 2 ;;
        --wandb-project)    WANDB_PROJECT="$2"; shift 2 ;;
        --run-name)         RUN_NAME="$2"; shift 2 ;;
        --model-size)       MODEL_SIZE="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --tasks)            TASKS="$2"; shift 2 ;;
        *)                  shift ;;  # Skip unknown args (SLURM may pass extras)
    esac
done

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: --checkpoint-dir is required"
    exit 1
fi

mkdir -p "${BENCHMARK_LOG_DIR}"

# Derive output paths
HF_OUTPUT_DIR="${CHECKPOINT_DIR}/hf_converted"
RESULTS_DIR="${CHECKPOINT_DIR}/benchmark_results"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "POST-TRAINING BENCHMARK"
echo "============================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Checkpoint:      ${CHECKPOINT_DIR}"
echo "HF Output:       ${HF_OUTPUT_DIR}"
echo "Results Dir:     ${RESULTS_DIR}"
echo "WandB Run ID:    ${WANDB_RUN_ID:-none}"
echo "WandB Project:   ${WANDB_PROJECT}"
echo "Run Name:        ${RUN_NAME:-unknown}"
echo "Tasks:           ${TASKS}"
echo "Start Time:      $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Convert Megatron checkpoint to HuggingFace format
# =============================================================================
if [ -d "${HF_OUTPUT_DIR}" ] && [ -f "${HF_OUTPUT_DIR}/config.json" ]; then
    echo "Step 1: HF checkpoint already exists at ${HF_OUTPUT_DIR}, skipping conversion"
else
    echo "Step 1: Converting Megatron checkpoint -> HuggingFace"

    # The converter expects ep=8 for A3B by default, but our training uses ep=4.
    # Override MODEL_PARALLEL_ARGS to match training config.
    export MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4"

    cd "${CONVERTOR_DIR}"
    bash scripts/qwen3/run_8xH20.sh \
        "${MODEL_SIZE}" \
        "${CHECKPOINT_DIR}" \
        "${HF_OUTPUT_DIR}" \
        true \
        true \
        bf16 \
        "${ORIGINAL_HF}"
    cd "${SCRIPT_DIR}"

    echo "Conversion complete: ${HF_OUTPUT_DIR}"
fi

# =============================================================================
# Step 2: Run lm-evaluation-harness
# =============================================================================
echo ""
echo "Step 2: Running lm-evaluation-harness"
echo "  Model:  ${HF_OUTPUT_DIR}"
echo "  Tasks:  ${TASKS}"
echo "  Batch:  ${BATCH_SIZE}"

accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=${HF_OUTPUT_DIR},trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RESULTS_DIR}" \
    --log_samples

echo "lm-eval complete. Results in: ${RESULTS_DIR}"

# =============================================================================
# Step 3: Parse results
# =============================================================================
echo ""
echo "Step 3: Parsing results"

python3 "${SCRIPT_DIR}/_parse_lm_eval_results.py" "${RESULTS_DIR}"

# =============================================================================
# Step 4: Log to WandB (same training run)
# =============================================================================
echo ""
echo "Step 4: Logging results to WandB"

python3 - "${RESULTS_DIR}" "${WANDB_RUN_ID}" "${WANDB_PROJECT}" "${RUN_NAME}" << 'PYEOF'
import json
import os
import sys
import glob

results_dir = sys.argv[1]
wandb_run_id = sys.argv[2] if len(sys.argv) > 2 else ""
wandb_project = sys.argv[3] if len(sys.argv) > 3 else "qwen3-router-training"
run_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

# Load lm-eval results
result_files = glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True)
if not result_files:
    print("No results.json found, skipping WandB logging")
    sys.exit(0)

result_file = max(result_files, key=os.path.getmtime)
with open(result_file) as f:
    data = json.load(f)

results = data.get("results", {})

BENCHMARKS = {
    "mmlu": ("acc,none", "acc"),
    "hellaswag": ("acc_norm,none", "acc_norm"),
    "arc_challenge": ("acc_norm,none", "acc_norm"),
    "winogrande": ("acc,none", "acc"),
    "gsm8k": ("exact_match,strict-match", "exact_match"),
    "truthfulqa_mc2": ("acc,none", "acc"),
}

scores = {}
for task_key, (metric_key, fallback_prefix) in BENCHMARKS.items():
    task_result = results.get(task_key, {})
    if not task_result:
        for key in results:
            if task_key in key:
                task_result = results[key]
                break
    score = task_result.get(metric_key)
    if score is None:
        for k, v in task_result.items():
            if fallback_prefix in k and isinstance(v, (int, float)):
                score = v
                break
    if score is not None:
        scores[task_key] = score * 100

valid_scores = [s for s in scores.values()]
avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

print(f"\nBenchmark scores:")
for task, score in scores.items():
    print(f"  {task}: {score:.2f}%")
print(f"  Average: {avg:.2f}%")

# Log to WandB
if not wandb_run_id:
    print("No WandB run ID provided, skipping WandB logging")
    sys.exit(0)

try:
    import wandb

    run = wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        resume="must",
    )

    benchmark_metrics = {}
    for task, score in scores.items():
        benchmark_metrics[f"benchmark/{task}"] = score
    benchmark_metrics["benchmark/average"] = avg

    wandb.log(benchmark_metrics)

    # Also set as run summary so they appear in sweep comparison tables
    for task, score in scores.items():
        wandb.run.summary[f"benchmark/{task}"] = score
    wandb.run.summary["benchmark/average"] = avg

    wandb.finish()
    print(f"Logged benchmark results to WandB run {wandb_run_id}")

except ImportError:
    print("wandb not installed, skipping WandB logging")
except Exception as e:
    print(f"WandB logging failed: {e}")

PYEOF

echo ""
echo "============================================================"
echo "Benchmark complete: ${RUN_NAME}"
echo "End Time: $(date)"
echo "============================================================"
