#!/bin/bash
#SBATCH --job-name=cp_vllm_smoke
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_smoke_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_smoke_%j.err

# =============================================================================
# vLLM smoke test: minimum viable run to verify vLLM boots Qwen3-30B-A3B
# with TP=8 + EP=8 in the NGC vllm-openai container.
#
#   - 1 model (pretrained baseline)
#   - 1 cell (prompt_len=256, batch_size=4)
#   - 2 trials, 1 warmup
#   - max_tokens=64
#
# Total target time: ~10 min including container start + model load. If this
# succeeds, submit the full submit_vllm_bench_ngc.sh.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
VLLM_CONTAINER="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"

PRETRAINED="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
TAG="${TAG:-smoke_$(date +%H%M%S)}"
OUTPUT_PATH="${RESULTS_DIR}/vllm_${TAG}.json"

echo "============================================================"
echo "vLLM SMOKE TEST"
echo "============================================================"
echo "SLURM Job ID:   ${SLURM_JOB_ID:-manual}"
echo "Container:      ${VLLM_CONTAINER}"
echo "Output:         ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "============================================================"

if [ ! -f "${VLLM_CONTAINER}" ]; then
    echo "ERROR: container not found: ${VLLM_CONTAINER}"
    exit 1
fi

srun --container-image="${VLLM_CONTAINER}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         set -euo pipefail
         echo '=== container baseline ==='
         python3 -c 'import sys, vllm, torch; print(\"py\", sys.version.split()[0]); print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__)'
         nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
         echo

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export VLLM_WORKER_MULTIPROC_METHOD=spawn
         mkdir -p \${HF_HOME}

         python3 ${SCRIPT_DIR}/cp_vllm_bench.py \
             --models          \"pretrained=${PRETRAINED}\" \
             --output          '${OUTPUT_PATH}' \
             --tp-size         8 \
             --prompt-lengths  '256' \
             --batch-sizes     '4' \
             --max-tokens      64 \
             --num-warmup      1 \
             --num-trials      2
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Result JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
