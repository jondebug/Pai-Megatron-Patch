#!/bin/bash
# Inline HellaSwag benchmark: converts checkpoint and runs lm-eval.
# Called as a subprocess during training (async on rank 0).
# Usage: bash run_inline_benchmark.sh <checkpoint_dir> <iteration> [hellaswag_limit]
set -euo pipefail

CKPT_DIR="$1"
ITERATION="$2"
BENCH_LIMIT="${3:-100}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"

HF_OUTPUT="${CKPT_DIR}/hf_converted_iter${ITERATION}"
RESULTS_DIR="${CKPT_DIR}/benchmark_iter${ITERATION}"

export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${RESULTS_DIR}"

# Skip if already done
if [ -f "${RESULTS_DIR}/accuracy_summary.json" ]; then
    echo "[BENCHMARK] iter ${ITERATION}: already done, skipping"
    exit 0
fi

echo "[BENCHMARK] iter ${ITERATION}: starting conversion + HellaSwag (limit=${BENCH_LIMIT})"

# Step 1: Convert Megatron checkpoint to HF
if ! ls "${HF_OUTPUT}"/*.safetensors 1>/dev/null 2>&1; then
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:${PYTHONPATH:-}"
    export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4'
    export KUBERNETES_CONTAINER_RESOURCE_GPU=4
    cd "${CONVERTOR_DIR}"
    bash scripts/qwen3/run_8xH20.sh A3B \
        "${CKPT_DIR}" "${HF_OUTPUT}" \
        true true bf16 "${ORIGINAL_HF}" 2>&1
    cd "${SCRIPT_DIR}"
fi

# Step 2: Run HellaSwag
if ls "${HF_OUTPUT}"/*.safetensors 1>/dev/null 2>&1; then
    pip install 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${HF_OUTPUT},trust_remote_code=True,dtype=bfloat16" \
        --tasks hellaswag \
        --batch_size 8 \
        --output_path "${RESULTS_DIR}" \
        --device cuda:0 \
        --limit "${BENCH_LIMIT}" 2>&1

    python3 "${SCRIPT_DIR}/benchmarks/_parse_lm_eval_results.py" "${RESULTS_DIR}" 2>&1 || true

    # Upload to WandB
    python3 -c "
import json, os, glob
results_files = glob.glob('${RESULTS_DIR}/**/results.json', recursive=True)
if results_files:
    with open(max(results_files, key=os.path.getmtime)) as f:
        data = json.load(f)
    hs = data.get('results',{}).get('hellaswag',{})
    acc = hs.get('acc_norm,none', hs.get('acc,none'))
    if acc is not None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({'benchmark/hellaswag_accuracy': acc * 100, 'benchmark/hellaswag_iter': int('${ITERATION}')})
                print(f'[BENCHMARK] Logged: hellaswag={acc*100:.1f}% at iter ${ITERATION}')
        except Exception as e:
            print(f'[BENCHMARK] WandB log failed: {e}')
        print(f'[BENCHMARK] HellaSwag: {acc*100:.1f}% at iter ${ITERATION}')
" 2>&1 || true
else
    echo "[BENCHMARK] WARNING: No HF checkpoint, skipping"
fi

echo "[BENCHMARK] iter ${ITERATION}: complete"
