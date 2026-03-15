#!/bin/bash
#SBATCH --job-name=benchmarks_all
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmarks_all_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmarks_all_%j.err

# =============================================================================
# Run lm-eval benchmarks on multiple checkpoints sequentially in one SLURM job.
# Each checkpoint uses all GPUs via accelerate for faster evaluation.
# =============================================================================
set -euo pipefail

SCRIPT_PATH_FOR_RESUBMIT="${0}"
resubmit_if_needed() {
    local remaining=0
    for run in "${RUNS[@]}"; do
        local actual_sub="${CKPT_SUB}"
        if [ ! -d "${BASE}/${run}/checkpoint/${CKPT_SUB}" ] 2>/dev/null; then
            actual_sub=$(ls "${BASE}/${run}/checkpoint/" 2>/dev/null | head -1)
        fi
        local summary="${BASE}/${run}/checkpoint/${actual_sub}/benchmark_results/accuracy_summary.json"
        [ ! -f "${summary}" ] && remaining=$((remaining + 1))
    done
    if [ "${remaining}" -gt 0 ]; then
        echo "RESUBMIT: ${remaining} runs remaining, submitting continuation..."
        sbatch "${SCRIPT_PATH_FOR_RESUBMIT}" 2>/dev/null || true
    else
        echo "ALL RUNS COMPLETE"
    fi
}
trap resubmit_if_needed EXIT

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
TASKS="hellaswag,arc_challenge,winogrande"
BATCH_SIZE=8
BASE="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CKPT_SUB="pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10"
PRETRAIN_CKPT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore"

RUNS=(
    "norl_aux0.001_r16"
    "norl_aux0.005_r17"
    "norl_aux0.01_r18"
    "norl_aux0.01_cpb_n1_a0.01_r19"
    "norl_aux0.01_topo0.01_r20"
    "norl_aux0.01_topo0.1_r21"
)

COMMENTS=(
    "compv2: aux-only coeff=0.001 5k iters"
    "compv2: aux-only coeff=0.005 5k iters"
    "compv2: aux-only coeff=0.01 5k iters"
    "compv2: aux+CPB n=1 alpha=0.01 5k iters"
    "compv2: aux+topo lambda=0.01 5k iters"
    "compv2: aux+topo lambda=0.1 5k iters"
)

# --- 3000-iter checkpoints (set latest_checkpointed_iteration.txt to 3000 before conversion) ---
RUNS_3K=(
    "norl_aux0.01_r18"
    "norl_aux0.01_cpb_n1_a0.01_r19"
    "norl_aux0.01_topo0.01_r20"
    "norl_aux0.01_topo0.1_r21"
)
TARGET_ITER=3000

echo "============================================================"
echo "BATCH BENCHMARK - ${#RUNS[@]} checkpoints"
echo "SLURM Job ID: ${SLURM_JOB_ID:-manual}"
echo "Tasks: ${TASKS}"
echo "Start: $(date)"
echo "============================================================"

ALL_RUNS=("${RUNS[@]}")

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${REPO_ROOT}/examples/qwen3/benchmarks" \
     bash "${REPO_ROOT}/examples/qwen3/benchmarks/_run_lm_eval_batch.sh" \
         "${REPO_ROOT}" "${BASE}" "${CKPT_SUB}" "${BATCH_SIZE}" "${ALL_RUNS[@]}" || true

echo ""
echo "============================================================"
echo "PASS 2: 3000-iter checkpoints"
echo "============================================================"

# For 3000-iter benchmarks: temporarily set latest_checkpointed_iteration.txt to 3000,
# convert & benchmark, store results under benchmark_results_3k, then restore.
for run in "${RUNS_3K[@]}"; do
    ACTUAL_SUB="${CKPT_SUB}"
    if [ ! -d "${BASE}/${run}/checkpoint/${CKPT_SUB}" ]; then
        ACTUAL_SUB=$(ls "${BASE}/${run}/checkpoint/" 2>/dev/null | head -1)
    fi
    CKPT_DIR="${BASE}/${run}/checkpoint/${ACTUAL_SUB}"
    ITER_FILE="${CKPT_DIR}/latest_checkpointed_iteration.txt"
    RESULTS_3K="${CKPT_DIR}/benchmark_results_3k"

    if [ -f "${RESULTS_3K}/accuracy_summary.json" ]; then
        echo "SKIP 3k (already done): ${run}"
        continue
    fi

    if [ ! -d "${CKPT_DIR}/iter_0003000" ]; then
        echo "SKIP 3k (no iter_0003000): ${run}"
        continue
    fi

    echo "Setting up 3k benchmark for: ${run}"
    # Save original latest iteration
    ORIG_ITER=$(cat "${ITER_FILE}" 2>/dev/null || echo "5000")
    echo "3000" > "${ITER_FILE}"
    # Remove old HF conversion so it re-converts with iter 3000
    rm -rf "${CKPT_DIR}/hf_converted" 2>/dev/null

    srun --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
         --container-workdir="${REPO_ROOT}/examples/qwen3/benchmarks" \
         bash "${REPO_ROOT}/examples/qwen3/benchmarks/_run_lm_eval_batch.sh" \
             "${REPO_ROOT}" "${BASE}" "${ACTUAL_SUB}" "${BATCH_SIZE}" "${run}" || true

    # Move results to _3k directory and restore original state
    if [ -d "${CKPT_DIR}/benchmark_results" ]; then
        mv "${CKPT_DIR}/benchmark_results" "${RESULTS_3K}" 2>/dev/null || true
    fi
    echo "${ORIG_ITER}" > "${ITER_FILE}"
    rm -rf "${CKPT_DIR}/hf_converted" 2>/dev/null
    echo "Restored latest iteration to ${ORIG_ITER} for: ${run}"
done

echo "Batch benchmark finished: $(date)"
