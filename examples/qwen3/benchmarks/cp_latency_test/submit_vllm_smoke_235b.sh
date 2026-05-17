#!/bin/bash
#SBATCH --job-name=cp_vllm_smk235
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_smk235_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_smk235_%j.err

# =============================================================================
# vLLM smoke for Qwen3-235B-A22B (pretrained baseline only).
#
# Verifies vLLM can boot the 235B (~470 GB bf16) on 8 × 80GB GPU. Tight memory
# (gpu_memory_utilization=0.92 to leave just enough for KV cache + cuda graphs).
# If this OOMs we'll fall back to 16 GPUs (2 nodes, TP=16).
#
# 1 model, 1 cell (plen=256 bs=4 max_tokens=64), 2 trials. ~30-60 min total
# including model load (~10-15 min for 470 GB bf16) and CUDA graph capture.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
VLLM_CONTAINER="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"

PRETRAINED_235B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
TAG="${TAG:-smk235_$(date +%H%M%S)}"
OUTPUT_PATH="${RESULTS_DIR}/vllm_${TAG}.json"

echo "============================================================"
echo "vLLM 235B SMOKE TEST"
echo "============================================================"
echo "SLURM Job ID:   ${SLURM_JOB_ID:-manual}"
echo "Container:      ${VLLM_CONTAINER}"
echo "Model:          ${PRETRAINED_235B}"
echo "Output:         ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "============================================================"

if [ ! -f "${VLLM_CONTAINER}" ]; then echo "ERROR: container missing: ${VLLM_CONTAINER}"; exit 1; fi

srun --container-image="${VLLM_CONTAINER}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         set -euo pipefail
         python3 -c 'import sys, vllm, torch; print(\"py\", sys.version.split()[0]); print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__)'
         nvidia-smi --query-gpu=index,name,memory.total --format=csv

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export VLLM_WORKER_MULTIPROC_METHOD=spawn
         mkdir -p \${HF_HOME}

         # Inline override of gpu_memory_utilization to 0.92 by patching the script's
         # default at the CLI: cp_vllm_bench.py uses the --models path so we just need
         # to bump utilization. The script doesn't currently expose that flag, so we
         # set it via a small monkeypatch.
         python3 -c \"
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
import cp_vllm_bench as m
_orig = m._build_engine
def _patched(p, args):
    from vllm import LLM
    common = {
        'model': p,
        'tensor_parallel_size': args.tp_size,
        'trust_remote_code': True,
        'dtype': 'bfloat16',
        'enforce_eager': False,
        'max_model_len': max(2048, max(int(s) for s in args.prompt_lengths.split(',')) + args.max_tokens + 64),
        'gpu_memory_utilization': 0.92,
        'seed': args.seed,
    }
    try:
        return LLM(enable_expert_parallel=True, **common)
    except (TypeError, ValueError) as e:
        print(f'enable_expert_parallel rejected: {e}; falling back')
        return LLM(**common)
m._build_engine = _patched
sys.argv = [
    'cp_vllm_bench.py',
    '--models', 'pretrained_235b=${PRETRAINED_235B}',
    '--output', '${OUTPUT_PATH}',
    '--tp-size', '8',
    '--prompt-lengths', '256',
    '--batch-sizes', '4',
    '--max-tokens', '64',
    '--num-warmup', '1',
    '--num-trials', '2',
]
m.main()
\"
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Result JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
