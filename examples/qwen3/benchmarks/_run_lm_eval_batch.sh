#!/bin/bash
# Inner script called by run_all_benchmarks_batch.sh inside the container.
# Arguments: REPO_ROOT BASE CKPT_SUB BATCH_SIZE RUN1 RUN2 ... RUNN
set -euo pipefail

REPO_ROOT="$1"; shift
BASE="$1"; shift
CKPT_SUB="$1"; shift
BATCH_SIZE="$1"; shift
RUNS=("$@")

pip install wandb 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null

export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:${PYTHONPATH:-}"
export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4'
export KUBERNETES_CONTAINER_RESOURCE_GPU=4

TASKS="hellaswag,arc_challenge,winogrande"

for run in "${RUNS[@]}"; do
    if [ "$run" = "pretrained_baseline" ]; then
        HF_MODEL="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
        RESULTS_DIR="${BASE}/pretrained_baseline/benchmark_results"
        MEGATRON_CKPT=""
    else
        # Auto-detect checkpoint subdirectory if CKPT_SUB doesn't exist for this run
        ACTUAL_CKPT_SUB="${CKPT_SUB}"
        if [ ! -d "${BASE}/${run}/checkpoint/${CKPT_SUB}" ]; then
            ACTUAL_CKPT_SUB=$(ls "${BASE}/${run}/checkpoint/" 2>/dev/null | head -1)
            echo "  Auto-detected checkpoint subdir: ${ACTUAL_CKPT_SUB}"
        fi
        HF_MODEL="${BASE}/${run}/checkpoint/${ACTUAL_CKPT_SUB}/hf_converted"
        RESULTS_DIR="${BASE}/${run}/checkpoint/${ACTUAL_CKPT_SUB}/benchmark_results"
        MEGATRON_CKPT="${BASE}/${run}/checkpoint/${ACTUAL_CKPT_SUB}"
    fi
    mkdir -p "${RESULTS_DIR}"

    # Skip runs that already have complete benchmark results
    if [ -f "${RESULTS_DIR}/accuracy_summary.json" ]; then
        has_scores=$(python3 -c "
import json
with open('${RESULTS_DIR}/accuracy_summary.json') as f:
    s = json.load(f).get('scores',{})
vals = [v for v in [s.get('hellaswag'), s.get('arc_challenge'), s.get('winogrande')] if v is not None]
print(len(vals))
" 2>/dev/null || echo "0")
        if [ "${has_scores}" = "3" ]; then
            echo "SKIP (already benchmarked): ${run}"
            continue
        fi
    fi

    echo ""
    echo "============================================================"
    echo "BENCHMARKING: ${run}"
    echo "  Model: ${HF_MODEL}"
    echo "============================================================"

    if [ -n "${MEGATRON_CKPT}" ] && ! ls "${HF_MODEL}"/*.safetensors 1>/dev/null 2>&1; then
        echo "  Converting Megatron checkpoint -> HuggingFace..."
        rm -rf "${HF_MODEL}" 2>/dev/null
        cd "${CONVERTOR_DIR}"
        bash scripts/qwen3/run_8xH20.sh \
            A3B \
            "${MEGATRON_CKPT}" \
            "${HF_MODEL}" \
            true true bf16 \
            "${ORIGINAL_HF}"
        cd "${REPO_ROOT}/examples/qwen3/benchmarks"
        echo "  Conversion complete."
    fi

    if ! ls "${HF_MODEL}"/*.safetensors 1>/dev/null 2>&1 && ! ls "${HF_MODEL}"/*.bin 1>/dev/null 2>&1; then
        echo "ERROR: No HF checkpoint at ${HF_MODEL}, skipping"
        continue
    fi

    EVAL_START=$(date +%s)

    echo "  Running all tasks: ${TASKS}"
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${HF_MODEL},trust_remote_code=True,dtype=bfloat16" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${RESULTS_DIR}" \
        --device cuda:0 \
        --limit 1000

    EVAL_END=$(date +%s)
    EVAL_ELAPSED=$((EVAL_END - EVAL_START))

    echo "{\"total_seconds\": ${EVAL_ELAPSED}}" > "${RESULTS_DIR}/timing.json"
    echo "BENCHMARK_TIME: ${run} ${EVAL_ELAPSED} seconds"
    echo "lm-eval complete for: ${run}"

    python3 "${REPO_ROOT}/examples/qwen3/benchmarks/_parse_lm_eval_results.py" "${RESULTS_DIR}" || true
    python3 "${REPO_ROOT}/examples/qwen3/benchmarks/collect_benchmark_results.py" || true
done

echo ""
echo "============================================================"
echo "ALL LM-EVAL BENCHMARKS COMPLETE"
echo "============================================================"
