#!/bin/bash
#SBATCH --job-name=conv235
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=03:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv235_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv235_%j.err

# =============================================================================
# Convert a Qwen3-235B-A22B Megatron checkpoint to HF format.
#
# Required env:
#   TRAINED_MEGATRON_CKPT  Megatron checkpoint root (parent of iter_NNNNNNN/)
#   ITER_NUM               iteration to convert (e.g. 1500)
#
# Output: <TRAINED_MEGATRON_CKPT>/hf_converted_iter${ITER_NUM}_cp/
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B"

TRAINED_MEGATRON_CKPT="${TRAINED_MEGATRON_CKPT:?Set TRAINED_MEGATRON_CKPT env var}"
ITER_NUM="${ITER_NUM:?Set ITER_NUM env var}"
HF_OUTPUT_DIR="${TRAINED_MEGATRON_CKPT}/hf_converted_iter${ITER_NUM}_cp"

echo "============================================================"
echo "235B MEGATRON -> HF CONVERSION"
echo "============================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID:-manual}"
echo "Megatron:     ${TRAINED_MEGATRON_CKPT}"
echo "Iter:         ${ITER_NUM}"
echo "HF output:    ${HF_OUTPUT_DIR}"
echo "Start: $(date)"
echo "============================================================"

# Sanity checks
ITER_DIR="${TRAINED_MEGATRON_CKPT}/iter_$(printf '%07d' ${ITER_NUM})"
if [ ! -d "${ITER_DIR}" ]; then
    echo "ERROR: iter_$(printf '%07d' ${ITER_NUM}) not found at ${ITER_DIR}"
    exit 1
fi

# Skip if already converted
if ls "${HF_OUTPUT_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    echo "HF checkpoint already exists, skipping conversion."
    exit 0
fi

mkdir -p "${HF_OUTPUT_DIR}"

# Point latest_checkpointed_iteration.txt at the desired iter (the converter
# uses Megatron's latest-iter discovery rather than an explicit --iter flag).
LATEST_FILE="${TRAINED_MEGATRON_CKPT}/latest_checkpointed_iteration.txt"
ORIGINAL_LATEST=""
if [ -f "${LATEST_FILE}" ]; then
    ORIGINAL_LATEST=$(cat "${LATEST_FILE}")
fi
echo "${ITER_NUM}" > "${LATEST_FILE}"
trap "if [ -n '${ORIGINAL_LATEST}' ]; then echo '${ORIGINAL_LATEST}' > '${LATEST_FILE}'; fi" EXIT

# 2-node 16-GPU conversion. Single-node (TP=1 PP=1 EP=8 / 8 GPUs) OOMs at the
# load step because each GPU ends up holding all of weights/EP-shard plus the
# HF target buffer simultaneously (78 GB / 80 GB). PP=2 splits the layer stack
# across the 2 nodes, halving the per-GPU memory peak.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${CONVERTOR_DIR}" \
     --nodes=2 --ntasks-per-node=1 \
     bash -c "
         set -euo pipefail
         export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
         # Match the converter script's expected env: PP=2 across 2 nodes, EP=8 within each.
         export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --expert-model-parallel-size 8'
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8
         export WORLD_SIZE=2
         export RANK=\${SLURM_PROCID}
         export MASTER_ADDR='${MASTER_ADDR}'
         export MASTER_PORT='${MASTER_PORT}'

         echo \"node \$(hostname) RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE}\"
         cd ${CONVERTOR_DIR}
         bash scripts/qwen3/run_A22B_16xH20.sh \
             A22B \
             '${TRAINED_MEGATRON_CKPT}' \
             '${HF_OUTPUT_DIR}' \
             true \
             true \
             bf16 \
             '${ORIGINAL_HF}'
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "HF dir: ${HF_OUTPUT_DIR}"
ls "${HF_OUTPUT_DIR}"/*.safetensors 2>/dev/null | wc -l | xargs -I {} echo "  safetensors written: {}"
echo "============================================================"
exit $EXIT
