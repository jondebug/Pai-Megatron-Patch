#!/bin/bash
#SBATCH --job-name=cp_vllm_ngc
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_ngc_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_ngc_%j.err

# =============================================================================
# Real end-to-end CP→latency benchmark using vLLM (in NGC/Docker container).
#
# Uses vllm/vllm-openai Docker image (cleanly avoiding the PyTorch 2.7
# conflict in our pai-megatron-patch container). Compares N models on
# matched prompts at multiple (prompt_len, batch_size) cells.
#
# Default: pretrained baseline vs B1 (RL+aux) vs aux_only (pure aux loss).
# All three are confirmed (by cp_microbench) to differ in measured CP.
#
# Configurable via env: MODELS (comma-separated name=path), PROMPT_LENGTHS,
# BATCH_SIZES, MAX_TOKENS, NUM_TRIALS, TP_SIZE.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
VLLM_CONTAINER="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# Default 3-model comparison
PRETRAINED="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
B1="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/rl-discovery_rlc0.5_c256_ppo_aux0.001_r00/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10/hf_converted"
AUX_ONLY="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/norl_aux0.005_r36/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-1500-wi-10/hf_converted_iter1500"

MODELS="${MODELS:-pretrained=${PRETRAINED},B1_rlaux_iter5000=${B1},aux_only_iter1500=${AUX_ONLY}}"
TP_SIZE="${TP_SIZE:-8}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,1024,4096}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32}"
MAX_TOKENS="${MAX_TOKENS:-256}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
OUTPUT_PATH="${RESULTS_DIR}/vllm_ngc_${TAG}.json"

echo "============================================================"
echo "vLLM (NGC container) END-TO-END BENCHMARK"
echo "============================================================"
echo "SLURM Job ID:   ${SLURM_JOB_ID:-manual}"
echo "Container:      ${VLLM_CONTAINER}"
echo "Models:         ${MODELS}"
echo "TP/EP size:     ${TP_SIZE}"
echo "Prompt lengths: ${PROMPT_LENGTHS}"
echo "Batch sizes:    ${BATCH_SIZES}"
echo "Max tokens:     ${MAX_TOKENS}"
echo "Trials:         ${NUM_TRIALS}"
echo "Output:         ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "============================================================"

if [ ! -f "${VLLM_CONTAINER}" ]; then
    echo "ERROR: vLLM container not found at ${VLLM_CONTAINER}"
    echo "       Run the pull job first (sbatch ... enroot import ...)"
    exit 1
fi

srun --container-image="${VLLM_CONTAINER}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         set -euo pipefail
         echo '=== vLLM container baseline ==='
         python3 -c 'import sys, vllm, torch; print(\"python:\", sys.version.split()[0]); print(\"torch:\", torch.__version__); print(\"vllm:\", vllm.__version__)'

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         export VLLM_WORKER_MULTIPROC_METHOD=spawn
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         python3 ${SCRIPT_DIR}/cp_vllm_bench.py \
             --models          '${MODELS}' \
             --output          '${OUTPUT_PATH}' \
             --tp-size         ${TP_SIZE} \
             --prompt-lengths  '${PROMPT_LENGTHS}' \
             --batch-sizes     '${BATCH_SIZES}' \
             --max-tokens      ${MAX_TOKENS} \
             --num-trials      ${NUM_TRIALS}
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Result JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
