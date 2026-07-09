#!/bin/bash
#SBATCH --job-name=validate_base_convert
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/validate_base_convert_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/validate_base_convert_%j.err

# Convert the new torch_dist 235B base (release/) -> HF into a SCRATCH dir.
# Does NOT touch the protected base dir. 2-node PP=2 EP=8 (proven memory-safe).
set -euo pipefail

PROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
CONVERTOR_DIR=${PROOT}/toolkits/distributed_checkpoints_convertor
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
ORIGINAL_HF=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore-dist
HF_OUTPUT_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/_validate_dist_base_hf

mkdir -p /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs

echo "============================================================"
echo "VALIDATE BASE CONVERT  base=$BASE"
echo "Job: ${SLURM_JOB_ID:-manual}  Nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "HF out: $HF_OUTPUT_DIR"
echo "Start: $(date)"
echo "============================================================"

if ls "${HF_OUTPUT_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    NS=$(ls ${HF_OUTPUT_DIR}/*.safetensors | wc -l)
    echo "HF already converted ($NS shards), skipping convert."
    exit 0
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT
mkdir -p "${HF_OUTPUT_DIR}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/lustre/fsw/portfolios/nvr/users/jonathanp:/lustre/fsw/portfolios/nvr/users/jonathanp" \
     --container-workdir="${CONVERTOR_DIR}" \
     --nodes=2 --ntasks-per-node=1 \
     bash -c "
         set -euo pipefail
         export PYTHONPATH=${PROOT}:${PROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
         export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --expert-model-parallel-size 8'
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8
         export WORLD_SIZE=2 RANK=\${SLURM_PROCID}
         export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
         echo \"node \$(hostname) RANK=\${RANK}\"
         bash scripts/qwen3/run_A22B_16xH20.sh A22B '${BASE}' '${HF_OUTPUT_DIR}' true true bf16 '${ORIGINAL_HF}'
     "
CONV_EXIT=$?
if [ ${CONV_EXIT} -ne 0 ]; then echo "Convert FAILED exit=${CONV_EXIT}"; exit 1; fi
NSHARDS=$(ls ${HF_OUTPUT_DIR}/*.safetensors 2>/dev/null | wc -l)
echo "Convert done: ${NSHARDS} shards (expect 118)"
if [ "${NSHARDS}" -lt 100 ]; then echo "ERROR: too few shards"; exit 1; fi
echo "Convert OK. End: $(date)"
