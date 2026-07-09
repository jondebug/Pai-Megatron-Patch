#!/bin/bash
#SBATCH --job-name=conv_eval_235b
#SBATCH --partition=polar4,polar3,polar,interactive,grizzly
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/conv_eval_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/conv_eval_%x_%j.err

# Required env:
#   TRAINED_MEGATRON_CKPT - path to checkpoint root (containing iter_NNNNNNN/)
#   ITER_NUM              - iteration number
#   LIMIT                 - lm-eval limit (1000 or empty for full)
#   RUN_NAME              - cell name for logging
# Uses 2-node PP=2 EP=8 convert (PP=2 halves per-GPU memory vs 1-node EP=8 which OOMs)

set -euo pipefail

PROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
REPO_ROOT=${PROOT}
CONVERTOR_DIR=${PROOT}/toolkits/distributed_checkpoints_convertor
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
ORIGINAL_HF=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B

TRAINED_MEGATRON_CKPT="${TRAINED_MEGATRON_CKPT:?Set TRAINED_MEGATRON_CKPT}"
ITER_NUM="${ITER_NUM:?Set ITER_NUM}"
LIMIT="${LIMIT:-}"
RUN_NAME="${RUN_NAME:-unknown}"
HF_OUTPUT_DIR="${TRAINED_MEGATRON_CKPT}/hf_converted_iter${ITER_NUM}_cp"
LIMIT_TAG="${LIMIT:-inf}"
BENCHMARK_DIR="${TRAINED_MEGATRON_CKPT}/benchmark_iter${ITER_NUM}_limit${LIMIT_TAG}"

mkdir -p /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs

echo "============================================================"
echo "CONVERT + EVAL: ${RUN_NAME} iter=${ITER_NUM} limit=${LIMIT_TAG}"
echo "Job: ${SLURM_JOB_ID:-manual}  Nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "HF: ${HF_OUTPUT_DIR}"
echo "============================================================"

ITER_DIR="${TRAINED_MEGATRON_CKPT}/iter_$(printf '%07d' ${ITER_NUM})"
if [ ! -d "${ITER_DIR}" ]; then
    echo "ERROR: ${ITER_DIR} not found"
    exit 1
fi

# Skip if accuracy already exists
if [ -f "${BENCHMARK_DIR}/accuracy_summary.json" ]; then
    echo "accuracy_summary.json already exists at ${BENCHMARK_DIR}, skipping."
    exit 0
fi

# STEP 1: Convert (skip if HF already has safetensors)
if ls "${HF_OUTPUT_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    echo "HF already converted ($(ls ${HF_OUTPUT_DIR}/*.safetensors | wc -l) shards), skipping convert."
else
    echo "Starting 2-node PP=2 EP=8 convert..."
    LATEST_FILE="${TRAINED_MEGATRON_CKPT}/latest_checkpointed_iteration.txt"
    ORIGINAL_LATEST=""
    [ -f "${LATEST_FILE}" ] && ORIGINAL_LATEST=$(cat "${LATEST_FILE}")
    echo "${ITER_NUM}" > "${LATEST_FILE}"

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
             export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
             export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --expert-model-parallel-size 8'
             export KUBERNETES_CONTAINER_RESOURCE_GPU=8
             export WORLD_SIZE=2 RANK=\${SLURM_PROCID}
             export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
             echo \"node \$(hostname) RANK=\${RANK}\"
             bash scripts/qwen3/run_A22B_16xH20.sh A22B '${TRAINED_MEGATRON_CKPT}' '${HF_OUTPUT_DIR}' true true bf16 '${ORIGINAL_HF}'
         "
    CONV_EXIT=$?
    [ -n "${ORIGINAL_LATEST}" ] && echo "${ORIGINAL_LATEST}" > "${LATEST_FILE}" || true
    if [ ${CONV_EXIT} -ne 0 ]; then echo "Convert FAILED exit=${CONV_EXIT}"; exit 1; fi
    NSHARDS=$(ls ${HF_OUTPUT_DIR}/*.safetensors 2>/dev/null | wc -l)
    if [ "${NSHARDS}" -lt 100 ]; then
        echo "ERROR: only ${NSHARDS} safetensors after convert (expected 118)"; exit 1
    fi
    echo "Convert OK: ${NSHARDS} shards"
fi

# CONVERT-ONLY (2026-06-06 waste fix): eval runs separately 1-node via run_lm_eval_ord,
# so we free the 2-node alloc here instead of idling node-1 through a ~10min eval.
echo "[convert-only] HF ready; eval handled separately (1-node). Releasing 2-node alloc."
exit 0

# STEP 2: lm-eval (1-node, uses only 1 task on node-0)
mkdir -p "${BENCHMARK_DIR}"
TASKS="hellaswag,arc_challenge,winogrande"
if [ -n "${LIMIT}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
else
    LIMIT_ARG=""
fi

srun --ntasks=1 --nodes=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/lustre/fsw/portfolios/nvr/users/jonathanp:/lustre/fsw/portfolios/nvr/users/jonathanp" \
     --container-workdir="${BENCHMARK_DIR}" \
     bash -c "
         set -uo pipefail
         # Install accelerate first (lm_eval depends on clear_device_cache from >=1.2.0)
         pip install 'accelerate>=1.2.0' --quiet 2>&1 | tail -2
         pip install 'lm_eval' --quiet 2>&1 | tail -2
         python3 -c 'from lm_eval import simple_evaluate; print(\"lm_eval import OK\")' || { echo 'lm_eval import FAILED'; exit 1; }
         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=\${HF_HOME}/datasets
         export TMPDIR=/tmp/lmeval_\${SLURM_JOB_ID:-$$}
         export VLLM_NO_USAGE_STATS=1
         export DO_NOT_TRACK=1
         export XDG_CONFIG_HOME=\${TMPDIR}/xdg
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE} \${TMPDIR} \${XDG_CONFIG_HOME}/vllm
         touch \${XDG_CONFIG_HOME}/vllm/do_not_track 2>/dev/null
         python3 -m lm_eval --model vllm \
             --model_args pretrained='${HF_OUTPUT_DIR}',tensor_parallel_size=8,dtype=bfloat16,enforce_eager=True,gpu_memory_utilization=0.85,max_model_len=2048,trust_remote_code=True \
             --tasks ${TASKS} ${LIMIT_ARG} \
             --output_path '${BENCHMARK_DIR}' \
             --log_samples
         python3 -c \"
import json, glob
from pathlib import Path
bd = Path('${BENCHMARK_DIR}')
files = sorted(bd.rglob('results*.json'))
if not files: print('No results file'); exit(1)
data = json.loads(files[0].read_text())
r = data.get('results', {})
acc = {
    'hellaswag': r.get('hellaswag', {}).get('acc_norm,none', 0),
    'arc_challenge': r.get('arc_challenge', {}).get('acc_norm,none', 0),
    'winogrande': r.get('winogrande', {}).get('acc,none', 0),
}
avg = sum(acc.values()) / 3 * 100
s = {'run_name': '${RUN_NAME}', 'iteration': ${ITER_NUM}, 'limit': '${LIMIT_TAG}',
     'benchmark_avg': round(avg, 2), **{k: round(v*100, 2) for k,v in acc.items()}}
Path('${BENCHMARK_DIR}/accuracy_summary.json').write_text(json.dumps(s, indent=2))
print(json.dumps(s, indent=2))
\"
     "
echo "lm-eval exit: $?"
