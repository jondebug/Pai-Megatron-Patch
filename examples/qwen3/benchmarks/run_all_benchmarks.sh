#!/bin/bash
# =============================================================================
# run_all_benchmarks.sh -- Top-Level Benchmark Runner
#
# Takes two checkpoint paths (baseline + optimized), runs all three
# benchmarks, and produces a combined results summary.
#
# Usage:
#   ./run_all_benchmarks.sh \
#     --baseline-ckpt /path/to/original/megatron/ckpt \
#     --optimized-ckpt /path/to/router-trained/ckpt \
#     --original-hf /path/to/original/hf/weights \
#     --output-dir ./benchmark_results \
#     [--model-size A3B] \
#     [--num-batches 100] \
#     [--batch-size 4] \
#     [--seq-length 2048] \
#     [--skip-accuracy] \
#     [--skip-critical-path] \
#     [--skip-latency]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BASELINE_CKPT=""
OPTIMIZED_CKPT=""
ORIGINAL_HF=""
OUTPUT_DIR="./benchmark_results"
MODEL_SIZE="A3B"
NUM_BATCHES=100
BATCH_SIZE=4
SEQ_LENGTH=2048
LM_EVAL_BATCH_SIZE=16
SKIP_ACCURACY=false
SKIP_CRITICAL_PATH=false
SKIP_LATENCY=false

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-ckpt)
            BASELINE_CKPT="$2"; shift 2 ;;
        --optimized-ckpt)
            OPTIMIZED_CKPT="$2"; shift 2 ;;
        --original-hf)
            ORIGINAL_HF="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --model-size)
            MODEL_SIZE="$2"; shift 2 ;;
        --num-batches)
            NUM_BATCHES="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --seq-length)
            SEQ_LENGTH="$2"; shift 2 ;;
        --lm-eval-batch-size)
            LM_EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --skip-accuracy)
            SKIP_ACCURACY=true; shift ;;
        --skip-critical-path)
            SKIP_CRITICAL_PATH=true; shift ;;
        --skip-latency)
            SKIP_LATENCY=true; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate required arguments
# -----------------------------------------------------------------------------
if [ -z "${BASELINE_CKPT}" ]; then
    echo "Error: --baseline-ckpt is required"; exit 1
fi
if [ -z "${OPTIMIZED_CKPT}" ]; then
    echo "Error: --optimized-ckpt is required"; exit 1
fi
if [ -z "${ORIGINAL_HF}" ]; then
    echo "Error: --original-hf is required (for tokenizer/config and conversion)"; exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Derived paths
BASELINE_HF_DIR="${OUTPUT_DIR}/hf_baseline"
OPTIMIZED_HF_DIR="${OUTPUT_DIR}/hf_optimized"

echo "############################################################"
echo "# MoE Benchmark Suite                                      #"
echo "############################################################"
echo ""
echo "Baseline Megatron ckpt:   ${BASELINE_CKPT}"
echo "Optimized Megatron ckpt:  ${OPTIMIZED_CKPT}"
echo "Original HF checkpoint:   ${ORIGINAL_HF}"
echo "Output directory:         ${OUTPUT_DIR}"
echo ""

# =============================================================================
# Step 1: Accuracy Benchmarks (lm-evaluation-harness)
# =============================================================================
if [ "${SKIP_ACCURACY}" = false ]; then
    echo "============================================================"
    echo "BENCHMARK 1/3: Accuracy (lm-evaluation-harness)"
    echo "============================================================"
    echo ""

    echo "--- Baseline accuracy ---"
    bash "${SCRIPT_DIR}/run_lm_eval.sh" \
        --checkpoint-dir "${BASELINE_CKPT}" \
        --hf-output-dir "${BASELINE_HF_DIR}" \
        --original-hf-checkpoint "${ORIGINAL_HF}" \
        --results-dir "${OUTPUT_DIR}/accuracy_baseline" \
        --model-size "${MODEL_SIZE}" \
        --batch-size "${LM_EVAL_BATCH_SIZE}"

    echo ""
    echo "--- Optimized accuracy ---"
    bash "${SCRIPT_DIR}/run_lm_eval.sh" \
        --checkpoint-dir "${OPTIMIZED_CKPT}" \
        --hf-output-dir "${OPTIMIZED_HF_DIR}" \
        --original-hf-checkpoint "${ORIGINAL_HF}" \
        --results-dir "${OUTPUT_DIR}/accuracy_optimized" \
        --model-size "${MODEL_SIZE}" \
        --batch-size "${LM_EVAL_BATCH_SIZE}"

    echo ""
    echo "Accuracy benchmarks complete."
else
    echo "Skipping accuracy benchmarks (--skip-accuracy)"
    # If HF dirs don't exist yet, convert anyway for other benchmarks
    if [ ! -d "${BASELINE_HF_DIR}" ] || [ ! -d "${OPTIMIZED_HF_DIR}" ]; then
        echo "NOTE: HF checkpoints needed for critical-path/latency benchmarks."
        echo "Running conversion only..."
        REPO_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"
        CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"

        if [ ! -d "${BASELINE_HF_DIR}" ]; then
            cd "${CONVERTOR_DIR}"
            bash scripts/qwen3/run_8xH20.sh \
                "${MODEL_SIZE}" "${BASELINE_CKPT}" "${BASELINE_HF_DIR}" true true bf16 "${ORIGINAL_HF}"
            cd "${SCRIPT_DIR}"
        fi
        if [ ! -d "${OPTIMIZED_HF_DIR}" ]; then
            cd "${CONVERTOR_DIR}"
            bash scripts/qwen3/run_8xH20.sh \
                "${MODEL_SIZE}" "${OPTIMIZED_CKPT}" "${OPTIMIZED_HF_DIR}" true true bf16 "${ORIGINAL_HF}"
            cd "${SCRIPT_DIR}"
        fi
    fi
fi

echo ""

# =============================================================================
# Step 2: Critical Path & Load Metrics
# =============================================================================
if [ "${SKIP_CRITICAL_PATH}" = false ]; then
    echo "============================================================"
    echo "BENCHMARK 2/3: Critical Path & Load Metrics"
    echo "============================================================"
    echo ""

    echo "--- Baseline critical path ---"
    python3 "${SCRIPT_DIR}/measure_critical_path.py" \
        --model-path "${BASELINE_HF_DIR}" \
        --output-path "${OUTPUT_DIR}/critical_path_baseline.json" \
        --num-batches "${NUM_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}"

    # Extract baseline critical path for speedup computation
    BASELINE_CP=$(python3 -c "
import json
with open('${OUTPUT_DIR}/critical_path_baseline.json') as f:
    d = json.load(f)
print(d['summary']['avg_num_tokens_on_critical_path'])
")

    echo ""
    echo "--- Optimized critical path ---"
    python3 "${SCRIPT_DIR}/measure_critical_path.py" \
        --model-path "${OPTIMIZED_HF_DIR}" \
        --output-path "${OUTPUT_DIR}/critical_path_optimized.json" \
        --num-batches "${NUM_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}" \
        --baseline-critical-path "${BASELINE_CP}"

    # Generate comparison JSON
    python3 -c "
import json

with open('${OUTPUT_DIR}/critical_path_baseline.json') as f:
    baseline = json.load(f)
with open('${OUTPUT_DIR}/critical_path_optimized.json') as f:
    optimized = json.load(f)

comparison = {
    'baseline': baseline['summary'],
    'optimized': optimized['summary'],
    'speedup': baseline['summary']['avg_num_tokens_on_critical_path'] / max(optimized['summary']['avg_num_tokens_on_critical_path'], 1e-10),
    'critical_path_reduction_pct': (1.0 - optimized['summary']['avg_num_tokens_on_critical_path'] / max(baseline['summary']['avg_num_tokens_on_critical_path'], 1e-10)) * 100,
}

with open('${OUTPUT_DIR}/critical_path_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print('Critical path comparison saved.')
"

    echo ""
    echo "Critical path benchmarks complete."
else
    echo "Skipping critical path benchmarks (--skip-critical-path)"
fi

echo ""

# =============================================================================
# Step 3: Inference Latency
# =============================================================================
if [ "${SKIP_LATENCY}" = false ]; then
    echo "============================================================"
    echo "BENCHMARK 3/3: Inference Latency"
    echo "============================================================"
    echo ""

    python3 "${SCRIPT_DIR}/measure_latency.py" \
        --model-path "${OPTIMIZED_HF_DIR}" \
        --baseline-model-path "${BASELINE_HF_DIR}" \
        --output-path "${OUTPUT_DIR}/latency_comparison.json" \
        --num-batches "${NUM_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}"

    echo ""
    echo "Latency benchmarks complete."
else
    echo "Skipping latency benchmarks (--skip-latency)"
fi

echo ""

# =============================================================================
# Step 4: Generate Combined Summary
# =============================================================================
echo "============================================================"
echo "Generating combined summary"
echo "============================================================"

python3 - "${OUTPUT_DIR}" <<'PYEOF'
import json
import os
import sys

output_dir = sys.argv[1]
summary_lines = []
summary_lines.append("# MoE Benchmark Results Summary\n")
summary_lines.append(f"Output directory: `{output_dir}`\n")

# Accuracy
for label, subdir in [("Baseline", "accuracy_baseline"), ("Optimized", "accuracy_optimized")]:
    summary_path = os.path.join(output_dir, subdir, "accuracy_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        summary_lines.append(f"\n## Accuracy -- {label}\n")
        for task, score in data.get("scores", {}).items():
            if score is not None:
                summary_lines.append(f"- **{task}**: {score:.2f}%")
            else:
                summary_lines.append(f"- **{task}**: N/A")
        summary_lines.append(f"- **Average**: {data.get('average', 0):.2f}%")

# Critical path comparison
cp_path = os.path.join(output_dir, "critical_path_comparison.json")
if os.path.exists(cp_path):
    with open(cp_path) as f:
        data = json.load(f)
    summary_lines.append("\n## Critical Path Comparison\n")
    bl = data.get("baseline", {})
    opt = data.get("optimized", {})
    summary_lines.append(f"- **Baseline critical path**: {bl.get('avg_num_tokens_on_critical_path', 'N/A'):.1f} tokens")
    summary_lines.append(f"- **Optimized critical path**: {opt.get('avg_num_tokens_on_critical_path', 'N/A'):.1f} tokens")
    summary_lines.append(f"- **Speedup**: {data.get('speedup', 'N/A'):.4f}x")
    summary_lines.append(f"- **Critical path reduction**: {data.get('critical_path_reduction_pct', 'N/A'):.2f}%")
    summary_lines.append(f"- **Baseline perplexity**: {bl.get('avg_perplexity', 'N/A'):.2f}")
    summary_lines.append(f"- **Optimized perplexity**: {opt.get('avg_perplexity', 'N/A'):.2f}")

# Latency comparison
lat_path = os.path.join(output_dir, "latency_comparison.json")
if os.path.exists(lat_path):
    with open(lat_path) as f:
        data = json.load(f)
    summary_lines.append("\n## Latency Comparison\n")
    bl = data.get("baseline", {})
    opt = data.get("optimized", {})
    comp = data.get("comparison", {})
    if bl:
        summary_lines.append(f"- **Baseline mean latency**: {bl.get('mean_ms', 'N/A'):.2f} ms (p50={bl.get('p50_ms', 'N/A'):.2f}, p95={bl.get('p95_ms', 'N/A'):.2f})")
    summary_lines.append(f"- **Optimized mean latency**: {opt.get('mean_ms', 'N/A'):.2f} ms (p50={opt.get('p50_ms', 'N/A'):.2f}, p95={opt.get('p95_ms', 'N/A'):.2f})")
    if comp:
        summary_lines.append(f"- **Speedup**: {comp.get('speedup', 'N/A'):.4f}x")
        summary_lines.append(f"- **Latency reduction**: {comp.get('latency_reduction_pct', 'N/A'):.2f}%")

summary_text = "\n".join(summary_lines) + "\n"

summary_path = os.path.join(output_dir, "summary.md")
with open(summary_path, "w") as f:
    f.write(summary_text)

print(summary_text)
print(f"\nSummary saved to: {summary_path}")
PYEOF

echo ""
echo "############################################################"
echo "# All benchmarks complete!                                 #"
echo "############################################################"
echo ""
echo "Output files:"
echo "  ${OUTPUT_DIR}/accuracy_table.tex          (if accuracy ran)"
echo "  ${OUTPUT_DIR}/critical_path_comparison.json"
echo "  ${OUTPUT_DIR}/latency_comparison.json"
echo "  ${OUTPUT_DIR}/summary.md"
