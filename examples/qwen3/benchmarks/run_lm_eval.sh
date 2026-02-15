#!/bin/bash
# =============================================================================
# run_lm_eval.sh -- Convert Megatron checkpoint to HF and run lm-evaluation-harness
#
# Usage:
#   ./run_lm_eval.sh \
#     --checkpoint-dir /path/to/megatron/checkpoint \
#     --hf-output-dir /path/to/hf/output \
#     --original-hf /path/to/original/hf/weights \
#     --results-dir ./results \
#     [--skip-conversion]  # skip MG->HF conversion if already done
#     [--model-size A3B]   # model size for conversion script
#     [--batch-size 8]     # batch size for lm-eval
#     [--tasks "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa_mc2"]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults
MODEL_SIZE="A3B"
BATCH_SIZE=8
SKIP_CONVERSION=false
TASKS="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa_mc2"
CHECKPOINT_DIR=""
HF_OUTPUT_DIR=""
ORIGINAL_HF=""
RESULTS_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2;;
        --hf-output-dir) HF_OUTPUT_DIR="$2"; shift 2;;
        --original-hf) ORIGINAL_HF="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --skip-conversion) SKIP_CONVERSION=true; shift;;
        --model-size) MODEL_SIZE="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --tasks) TASKS="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Validate required arguments
if [ -z "$HF_OUTPUT_DIR" ]; then
    echo "ERROR: --hf-output-dir is required"
    exit 1
fi
if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="${HF_OUTPUT_DIR}/lm_eval_results"
fi

echo "============================================================"
echo "LM-Evaluation-Harness Benchmark"
echo "============================================================"
echo "Checkpoint:      ${CHECKPOINT_DIR:-'(skipping conversion)'}"
echo "HF Output:       ${HF_OUTPUT_DIR}"
echo "Original HF:     ${ORIGINAL_HF:-'(not provided)'}"
echo "Results:         ${RESULTS_DIR}"
echo "Tasks:           ${TASKS}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Skip Conversion: ${SKIP_CONVERSION}"
echo "============================================================"

# Step 1: Convert Megatron checkpoint to HuggingFace format
if [ "$SKIP_CONVERSION" = false ]; then
    if [ -z "$CHECKPOINT_DIR" ] || [ -z "$ORIGINAL_HF" ]; then
        echo "ERROR: --checkpoint-dir and --original-hf required for conversion"
        exit 1
    fi

    echo ""
    echo "[Step 1/3] Converting Megatron checkpoint to HuggingFace format..."
    echo "  Source: ${CHECKPOINT_DIR}"
    echo "  Target: ${HF_OUTPUT_DIR}"

    CONVERTER_SCRIPT="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor/scripts/qwen3/run_8xH20.sh"
    if [ ! -f "$CONVERTER_SCRIPT" ]; then
        echo "ERROR: Conversion script not found: $CONVERTER_SCRIPT"
        exit 1
    fi

    cd "${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
    bash scripts/qwen3/run_8xH20.sh \
        "${MODEL_SIZE}" \
        "${CHECKPOINT_DIR}" \
        "${HF_OUTPUT_DIR}" \
        true \
        true \
        bf16 \
        "${ORIGINAL_HF}"

    echo "  Conversion complete."
else
    echo ""
    echo "[Step 1/3] Skipping conversion (--skip-conversion)"
    if [ ! -d "$HF_OUTPUT_DIR" ]; then
        echo "ERROR: HF output directory does not exist: $HF_OUTPUT_DIR"
        exit 1
    fi
fi

# Step 2: Run lm-evaluation-harness
echo ""
echo "[Step 2/3] Running lm-evaluation-harness..."
echo "  Model: ${HF_OUTPUT_DIR}"
echo "  Tasks: ${TASKS}"

LM_EVAL_DIR="${REPO_ROOT}/backends/LM-Evaluation-Harness-240310"
if [ ! -d "$LM_EVAL_DIR" ]; then
    echo "WARNING: LM-Evaluation-Harness not found at ${LM_EVAL_DIR}"
    echo "Trying system lm_eval..."
    LM_EVAL_CMD="python -m lm_eval"
else
    cd "$LM_EVAL_DIR"
    LM_EVAL_CMD="python -m lm_eval"
fi

mkdir -p "${RESULTS_DIR}"

${LM_EVAL_CMD} \
    --model hf \
    --model_args "pretrained=${HF_OUTPUT_DIR},trust_remote_code=True,dtype=bfloat16" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RESULTS_DIR}" \
    --log_samples 2>&1 | tee "${RESULTS_DIR}/lm_eval_output.log"

echo "  Results saved to: ${RESULTS_DIR}"

# Step 3: Parse results and generate LaTeX table
echo ""
echo "[Step 3/3] Generating results summary..."

python3 << 'PYEOF'
import json
import os
import sys
import glob

results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir:
    sys.exit(0)

# Find the results JSON file (lm-eval saves in a subdirectory)
result_files = glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True)
if not result_files:
    print("WARNING: No results.json found in", results_dir)
    sys.exit(0)

results_file = result_files[0]
print(f"Reading results from: {results_file}")

with open(results_file) as f:
    data = json.load(f)

results = data.get("results", {})

# Extract key metrics
task_metrics = {}
for task_name, task_data in results.items():
    # lm-eval stores metrics with different keys depending on task
    acc = task_data.get("acc,none", task_data.get("acc_norm,none", task_data.get("exact_match,strict-match", None)))
    if acc is not None:
        task_metrics[task_name] = acc

if not task_metrics:
    print("WARNING: No accuracy metrics found in results")
    sys.exit(0)

# Print summary table
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"{'Task':<25} {'Accuracy':>10}")
print("-" * 60)
for task, acc in sorted(task_metrics.items()):
    print(f"{task:<25} {acc:>10.4f}")
print("-" * 60)
avg = sum(task_metrics.values()) / len(task_metrics)
print(f"{'Average':<25} {avg:>10.4f}")
print("=" * 60)

# Generate LaTeX table
latex_file = os.path.join(results_dir, "accuracy_table.tex")
with open(latex_file, "w") as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{LM-Evaluation-Harness Benchmark Results}\n")
    f.write("\\begin{tabular}{lc}\n")
    f.write("\\toprule\n")
    f.write("Task & Accuracy \\\\\n")
    f.write("\\midrule\n")
    for task, acc in sorted(task_metrics.items()):
        clean_name = task.replace("_", "\\_")
        f.write(f"{clean_name} & {acc:.4f} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"Average & {avg:.4f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nLaTeX table saved to: {latex_file}")

# Also save a clean JSON summary
summary_file = os.path.join(results_dir, "accuracy_summary.json")
with open(summary_file, "w") as f:
    json.dump({"tasks": task_metrics, "average": avg}, f, indent=2)
print(f"JSON summary saved to: {summary_file}")

PYEOF

echo ""
echo "============================================================"
echo "LM-Eval benchmark complete. Results in: ${RESULTS_DIR}"
echo "============================================================"
