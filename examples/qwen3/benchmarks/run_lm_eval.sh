#!/bin/bash
# =============================================================================
# run_lm_eval.sh -- Accuracy Benchmarks via lm-evaluation-harness
#
# Converts a Megatron checkpoint to HuggingFace format, then runs
# lm-evaluation-harness with standard academic benchmarks:
#   MMLU (5-shot), HellaSwag (10-shot), ARC-Challenge (25-shot),
#   WinoGrande (5-shot), GSM8K (5-shot), TruthfulQA (0-shot)
#
# Usage:
#   bash run_lm_eval.sh \
#     --checkpoint-dir /path/to/megatron/ckpt \
#     --hf-output-dir /path/to/hf/output \
#     --original-hf-checkpoint /path/to/original/hf/weights \
#     --results-dir ./results \
#     [--model-size A3B] \
#     [--skip-conversion] \
#     [--batch-size 16]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CHECKPOINT_DIR=""
HF_OUTPUT_DIR=""
ORIGINAL_HF_CHECKPOINT=""
RESULTS_DIR="./lm_eval_results"
MODEL_SIZE="A3B"
SKIP_CONVERSION=false
BATCH_SIZE=16

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"; shift 2 ;;
        --hf-output-dir)
            HF_OUTPUT_DIR="$2"; shift 2 ;;
        --original-hf-checkpoint)
            ORIGINAL_HF_CHECKPOINT="$2"; shift 2 ;;
        --results-dir)
            RESULTS_DIR="$2"; shift 2 ;;
        --model-size)
            MODEL_SIZE="$2"; shift 2 ;;
        --skip-conversion)
            SKIP_CONVERSION=true; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate required arguments
# -----------------------------------------------------------------------------
if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: --checkpoint-dir is required"
    exit 1
fi
if [ -z "${HF_OUTPUT_DIR}" ]; then
    echo "Error: --hf-output-dir is required"
    exit 1
fi
if [ -z "${ORIGINAL_HF_CHECKPOINT}" ]; then
    echo "Error: --original-hf-checkpoint is required (for tokenizer/config)"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Step 1: Convert Megatron checkpoint to HuggingFace format
# =============================================================================
if [ "${SKIP_CONVERSION}" = false ]; then
    echo "============================================================"
    echo "Step 1: Converting Megatron checkpoint -> HuggingFace"
    echo "  Source:  ${CHECKPOINT_DIR}"
    echo "  Output:  ${HF_OUTPUT_DIR}"
    echo "  HF ref:  ${ORIGINAL_HF_CHECKPOINT}"
    echo "============================================================"

    cd "${CONVERTOR_DIR}"
    bash scripts/qwen3/run_8xH20.sh \
        "${MODEL_SIZE}" \
        "${CHECKPOINT_DIR}" \
        "${HF_OUTPUT_DIR}" \
        true \
        true \
        bf16 \
        "${ORIGINAL_HF_CHECKPOINT}"
    cd "${SCRIPT_DIR}"

    echo "Conversion complete. HF checkpoint at: ${HF_OUTPUT_DIR}"
else
    echo "Skipping checkpoint conversion (--skip-conversion)"
fi

# =============================================================================
# Step 2: Run lm-evaluation-harness
# =============================================================================
echo "============================================================"
echo "Step 2: Running lm-evaluation-harness"
echo "  Model:      ${HF_OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Results:    ${RESULTS_DIR}"
echo "============================================================"

TASKS="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa"

accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=${HF_OUTPUT_DIR},trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RESULTS_DIR}" \
    --log_samples

echo "lm-eval complete. Raw results saved to: ${RESULTS_DIR}"

# =============================================================================
# Step 3: Parse results and print LaTeX table
# =============================================================================
echo "============================================================"
echo "Step 3: Generating results summary"
echo "============================================================"

export RESULTS_DIR
python3 "${SCRIPT_DIR}/_parse_lm_eval_results.py" "${RESULTS_DIR}"

echo ""
echo "All accuracy benchmarks complete."
