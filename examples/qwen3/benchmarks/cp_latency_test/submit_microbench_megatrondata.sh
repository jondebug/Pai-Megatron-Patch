#!/bin/bash
#SBATCH --job-name=cp_micro_megdata
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_megdata_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_megdata_%j.err

# =============================================================================
# CP→Latency Microbenchmark — on TRAINING distribution (Megatron mmap dataset).
#
# Sanity check for the WikiText regression result. If the trained model's
# CP advantage (CSV says CP=2659 vs baseline 4780) only shows up on the
# training distribution, that's an out-of-distribution generalization
# problem. If it doesn't reproduce here either, the CSV's CP=2659 is
# itself suspect (different aggregation / batch composition than what we
# measure here with HF inference).
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

BASELINE_MODEL="${BASELINE_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete}"
TRAINED_MODEL="${TRAINED_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/pareto_g0_c256_ppo_aux0.01_cpb_n1_a0.01_r80/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10/hf_converted_iter2000}"
BASELINE_NAME="${BASELINE_NAME:-pretrained_megdata}"
TRAINED_NAME="${TRAINED_NAME:-r80_megdata}"
NUM_BATCHES="${NUM_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TAG="${TAG:-megdata_$(date +%H%M%S)}"

# Megatron mmap dataset prefix (no extension; reader auto-appends .idx/.bin).
MEGDATA_PREFIX="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
OUTPUT_PATH="${RESULTS_DIR}/microbench_${TAG}.json"

echo "============================================================"
echo "CP→LATENCY MICROBENCHMARK — TRAINING DISTRIBUTION"
echo "============================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Baseline model:  ${BASELINE_MODEL}"
echo "Trained model:   ${TRAINED_MODEL}"
echo "Megatron data:   ${MEGDATA_PREFIX}.{idx,bin}"
echo "num_batches=${NUM_BATCHES} batch_size=${BATCH_SIZE} seq_length=${SEQ_LENGTH}"
echo "Output:          ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "============================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         set -euo pipefail
         pip install --quiet 'transformers>=4.51' accelerate 2>/dev/null || true
         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:\${PYTHONPATH:-}
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         python3 ${SCRIPT_DIR}/cp_microbench.py \
             --baseline-model '${BASELINE_MODEL}' \
             --trained-model  '${TRAINED_MODEL}' \
             --baseline-name  '${BASELINE_NAME}' \
             --trained-name   '${TRAINED_NAME}' \
             --output         '${OUTPUT_PATH}' \
             --dataset-name   '${MEGDATA_PREFIX}' \
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
