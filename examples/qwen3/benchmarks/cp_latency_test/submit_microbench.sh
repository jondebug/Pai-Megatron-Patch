#!/bin/bash
#SBATCH --job-name=cp_microbench
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_%j.err

# =============================================================================
# CP→Latency Microbenchmark
#
# Captures per-layer routing distributions from two HF models, then simulates
# expert-parallel step latency at EP ∈ {1,2,4,8,16,32,64,128}. Two GPUs is
# enough — the 30B-A3B model fits across two H100 80GB with device_map="auto".
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# Defaults (Alibaba pretrained vs B4 = pareto_g0_..._r80 iter 2000, CP=2659)
BASELINE_MODEL="${BASELINE_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete}"
TRAINED_MODEL="${TRAINED_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/pareto_g0_c256_ppo_aux0.01_cpb_n1_a0.01_r80/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10/hf_converted_iter2000}"
BASELINE_NAME="${BASELINE_NAME:-pretrained_cp4780}"
TRAINED_NAME="${TRAINED_NAME:-r80_iter2000_cp2659}"
NUM_BATCHES="${NUM_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
OUTPUT_PATH="${RESULTS_DIR}/microbench_${BASELINE_NAME}_vs_${TRAINED_NAME}_${TAG}.json"

echo "============================================================"
echo "CP→LATENCY MICROBENCHMARK"
echo "============================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Baseline model:  ${BASELINE_MODEL}"
echo "Trained model:   ${TRAINED_MODEL}"
echo "Output:          ${OUTPUT_PATH}"
echo "num_batches=${NUM_BATCHES} batch_size=${BATCH_SIZE} seq_length=${SEQ_LENGTH}"
echo "Start: $(date)"
echo "============================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         set -euo pipefail
         pip install --quiet datasets 'transformers>=4.51' accelerate 2>/dev/null || true

         # Use persistent HF cache on Lustre to avoid re-downloading WikiText.
         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         WIKITEXT_LOCAL=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow

         python3 ${SCRIPT_DIR}/cp_microbench.py \
             --baseline-model '${BASELINE_MODEL}' \
             --trained-model  '${TRAINED_MODEL}' \
             --baseline-name  '${BASELINE_NAME}' \
             --trained-name   '${TRAINED_NAME}' \
             --output         '${OUTPUT_PATH}' \
             --dataset-name   \"\${WIKITEXT_LOCAL}\" \
             --num-batches    ${NUM_BATCHES} \
             --batch-size     ${BATCH_SIZE} \
             --seq-length     ${SEQ_LENGTH}
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Results JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
