#!/bin/bash
#SBATCH --job-name=cp_micro_dbg
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_dbg_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/microbench_dbg_%j.err

# =============================================================================
# CP→Latency Microbenchmark — DEBUG variant
#
# 1 GPU, tiny workload. End-to-end run in ~10–15 min including model load.
# Verifies the script flow: model loading, routing hook attachment,
# trace capture, FFN pre-timing, simulation, output writing.
#
# Once this succeeds, run submit_microbench.sh for the real measurement.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

BASELINE_MODEL="${BASELINE_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete}"
TRAINED_MODEL="${TRAINED_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/pareto_g0_c256_ppo_aux0.01_cpb_n1_a0.01_r80/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10/hf_converted_iter2000}"
BASELINE_NAME="${BASELINE_NAME:-pretrained_dbg}"
TRAINED_NAME="${TRAINED_NAME:-r80_dbg}"
NUM_BATCHES="${NUM_BATCHES:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
TAG="${TAG:-dbg_$(date +%H%M%S)}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
OUTPUT_PATH="${RESULTS_DIR}/microbench_${TAG}.json"

echo "============================================================"
echo "CP→LATENCY MICROBENCHMARK — DEBUG (1 GPU, tiny workload)"
echo "============================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "GPU node:        $(hostname)"
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
         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         # Smaller FFN-timing grid + fewer reps for debug speed.
         # Point at the locally-cached arrow file directly to bypass HF Hub
         # pattern resolution (which is broken in the container's older
         # datasets version on Python 3.12).
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
             --seq-length     ${SEQ_LENGTH} \
             --ffn-repeats    5 \
             --ffn-warmup     2
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Results JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
