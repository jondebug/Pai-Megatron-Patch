#!/bin/bash
#SBATCH --job-name=conv235_1n
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=02:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv235_1n_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/conv235_1n_%j.err

# =============================================================================
# Single-node 235B Megatron->HF conversion attempt with TP=8 EP=8 PP=1.
#
# v1 (TP=1 PP=1 EP=8) OOMed because attention was REPLICATED on every GPU.
# This variant shards attention across the 8 GPUs (TP=8) so per-GPU memory
# footprint drops from ~89 GB to ~64 GB. Risky — may still OOM during the HF
# write step which adds ~30 GB on rank 0. If it survives we save the 16-GPU
# 2-node allocation.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B"

TRAINED_MEGATRON_CKPT="${TRAINED_MEGATRON_CKPT:?Set TRAINED_MEGATRON_CKPT env var}"
ITER_NUM="${ITER_NUM:?Set ITER_NUM env var}"
HF_OUTPUT_DIR="${TRAINED_MEGATRON_CKPT}/hf_converted_iter${ITER_NUM}_cp_1n"

echo "============================================================"
echo "235B MEGATRON -> HF (1-node TP=8 EP=8 attempt)"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Megatron: ${TRAINED_MEGATRON_CKPT}"
echo "Iter:     ${ITER_NUM}"
echo "HF out:   ${HF_OUTPUT_DIR}"
echo "Start: $(date)"
echo "============================================================"

ITER_DIR="${TRAINED_MEGATRON_CKPT}/iter_$(printf '%07d' ${ITER_NUM})"
[ -d "${ITER_DIR}" ] || { echo "ERROR: ${ITER_DIR} missing"; exit 1; }
ls "${HF_OUTPUT_DIR}"/*.safetensors 1>/dev/null 2>&1 && { echo "Already converted, exit"; exit 0; }
mkdir -p "${HF_OUTPUT_DIR}"

LATEST_FILE="${TRAINED_MEGATRON_CKPT}/latest_checkpointed_iteration.txt"
ORIGINAL_LATEST=""
[ -f "${LATEST_FILE}" ] && ORIGINAL_LATEST=$(cat "${LATEST_FILE}")
echo "${ITER_NUM}" > "${LATEST_FILE}"
trap "[ -n '${ORIGINAL_LATEST}' ] && echo '${ORIGINAL_LATEST}' > '${LATEST_FILE}' || true" EXIT

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${CONVERTOR_DIR}" \
     bash -c "
         set -euo pipefail
         export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
         export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --expert-model-parallel-size 8'
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8

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
ls "${HF_OUTPUT_DIR}"/*.safetensors 2>/dev/null | wc -l | xargs -I {} echo "  safetensors: {}"
echo "============================================================"
exit $EXIT
