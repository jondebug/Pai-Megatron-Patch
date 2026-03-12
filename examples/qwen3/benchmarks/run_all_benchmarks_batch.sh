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
CKPT_SUB="pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-10"
PRETRAIN_CKPT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore"

RUNS=(
    "c256_ppo_k5_buf8_r03"
    "c256_ppo_kl1.0_r02"
    "c256_ppo_k5_buf8_kl0.1_r04"
)

COMMENTS=(
    "1ykmwe84: KL AB test k=5 buf=8 with KL divergence + PPO fixes"
    "1ykmwe84: KL AB test k=1 kl=1.0"
    "1ykmwe84: KL AB test k=5 buf=8 kl=0.1"
)

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

echo "Batch benchmark finished: $(date)"
