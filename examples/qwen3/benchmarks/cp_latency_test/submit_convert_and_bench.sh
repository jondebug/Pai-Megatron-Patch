#!/bin/bash
#SBATCH --job-name=cp_conv_bench
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv_bench_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv_bench_%j.err

# =============================================================================
# Convert a Megatron checkpoint to HF, then run cp_microbench on it.
#
# Required env:
#   TRAINED_MEGATRON_CKPT  Megatron checkpoint dir (the iter_NNNNNNN's parent)
#   TRAINED_NAME           Short label, used in result filenames
#
# Optional env:
#   BASELINE_MODEL  HF baseline (default: Qwen3-30B-A3B-complete)
#   NUM_BATCHES, BATCH_SIZE, SEQ_LENGTH (default: 64, 4, 2048)
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"

# Required
TRAINED_MEGATRON_CKPT="${TRAINED_MEGATRON_CKPT:?Set TRAINED_MEGATRON_CKPT env var}"
TRAINED_NAME="${TRAINED_NAME:?Set TRAINED_NAME env var}"

BASELINE_MODEL="${BASELINE_MODEL:-${ORIGINAL_HF}}"
BASELINE_NAME="${BASELINE_NAME:-pretrained}"
NUM_BATCHES="${NUM_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

HF_OUTPUT_DIR="${TRAINED_MEGATRON_CKPT}/hf_converted_iter_cpbench_${TAG}"
OUTPUT_PATH="${RESULTS_DIR}/microbench_${BASELINE_NAME}_vs_${TRAINED_NAME}_${TAG}.json"
MEGDATA_PREFIX="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

echo "============================================================"
echo "CONVERT + CP-MICROBENCH"
echo "============================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Megatron ckpt:   ${TRAINED_MEGATRON_CKPT}"
echo "Trained name:    ${TRAINED_NAME}"
echo "HF output:       ${HF_OUTPUT_DIR}"
echo "Baseline:        ${BASELINE_MODEL}"
echo "Result JSON:     ${OUTPUT_PATH}"
echo "Settings:        num_batches=${NUM_BATCHES} bs=${BATCH_SIZE} seq=${SEQ_LENGTH}"
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
         export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         # ------------------------------------------------------------
         # Step 1: Convert Megatron -> HF (4 GPUs, EP=4 to match training)
         # ------------------------------------------------------------
         export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4'
         export KUBERNETES_CONTAINER_RESOURCE_GPU=4

         if ls '${HF_OUTPUT_DIR}'/*.safetensors 1>/dev/null 2>&1; then
             echo 'HF checkpoint exists, skipping conversion.'
         else
             echo '------------------------------------------------------------'
             echo 'Step 1: Converting Megatron -> HuggingFace'
             echo '------------------------------------------------------------'
             cd '${CONVERTOR_DIR}'
             bash scripts/qwen3/run_8xH20.sh \
                 A3B \
                 '${TRAINED_MEGATRON_CKPT}' \
                 '${HF_OUTPUT_DIR}' \
                 true \
                 true \
                 bf16 \
                 '${ORIGINAL_HF}'
             echo 'Conversion complete.'
         fi

         # ------------------------------------------------------------
         # Step 2: Microbench on Megatron mmap (training distribution)
         # ------------------------------------------------------------
         echo '------------------------------------------------------------'
         echo 'Step 2: cp_microbench on training distribution'
         echo '------------------------------------------------------------'
         cd '${SCRIPT_DIR}'
         python3 cp_microbench.py \
             --baseline-model '${BASELINE_MODEL}' \
             --trained-model  '${HF_OUTPUT_DIR}' \
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
echo "Result JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
