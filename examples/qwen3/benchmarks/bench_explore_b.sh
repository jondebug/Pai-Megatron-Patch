#!/bin/bash
#SBATCH --job-name=bench_explore_b
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/bench_explore_b_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/bench_explore_b_%j.err

set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
BASE="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CKPT_SUB="pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-10"

RUNS=(
    "explore_c256_ppo_cos_aux0.003_kl0.0003_r26"
    "explore_c256_ppo_cos_aux0.003_r10"
    "explore_c256_ppo_cos_gae0.95_aux0.003_r11"
    "explore_c256_ppo_gumbel_t0.3_cos_aux0.003_r14"
)

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${REPO_ROOT}/examples/qwen3/benchmarks" \
     bash "${REPO_ROOT}/examples/qwen3/benchmarks/_run_lm_eval_batch.sh" \
         "${REPO_ROOT}" "${BASE}" "${CKPT_SUB}" 8 "${RUNS[@]}"
