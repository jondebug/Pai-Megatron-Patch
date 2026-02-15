#!/bin/bash
# =============================================================================
# run_all_benchmarks.sh -- Compare baseline vs optimized checkpoints
#
# Runs all benchmarks (accuracy, critical path, latency) on two checkpoints
# and produces a combined results summary suitable for an academic paper.
#
# Usage:
#   ./run_all_benchmarks.sh \
#     --baseline-ckpt /path/to/original/megatron/ckpt \
#     --optimized-ckpt /path/to/router-trained/megatron/ckpt \
#     --original-hf /path/to/original/hf/weights \
#     --output-dir ./benchmark_results \
#     [--skip-conversion]        # if HF checkpoints already exist
#     [--skip-lm-eval]           # skip accuracy benchmarks (slow)
#     [--num-batches 100]        # batches for critical path / latency
#     [--model-size A3B]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults
MODEL_SIZE="A3B"
NUM_BATCHES=100
WARMUP_BATCHES=10
BATCH_SIZE=1
SEQ_LENGTH=128
SKIP_CONVERSION=false
SKIP_LM_EVAL=false
BASELINE_CKPT=""
OPTIMIZED_CKPT=""
ORIGINAL_HF=""
OUTPUT_DIR="./benchmark_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-ckpt) BASELINE_CKPT="$2"; shift 2;;
        --optimized-ckpt) OPTIMIZED_CKPT="$2"; shift 2;;
        --original-hf) ORIGINAL_HF="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --skip-conversion) SKIP_CONVERSION=true; shift;;
        --skip-lm-eval) SKIP_LM_EVAL=true; shift;;
        --model-size) MODEL_SIZE="$2"; shift 2;;
        --num-batches) NUM_BATCHES="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --seq-length) SEQ_LENGTH="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Validate
if [ -z "$ORIGINAL_HF" ]; then
    echo "ERROR: --original-hf is required (path to original HuggingFace weights for tokenizer/config)"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

BASELINE_HF="${OUTPUT_DIR}/baseline_hf"
OPTIMIZED_HF="${OUTPUT_DIR}/optimized_hf"

echo "============================================================"
echo "MoE Router Optimization Benchmark Suite"
echo "============================================================"
echo "Baseline Ckpt:   ${BASELINE_CKPT:-'(using original HF directly)'}"
echo "Optimized Ckpt:  ${OPTIMIZED_CKPT:-'(not provided)'}"
echo "Original HF:     ${ORIGINAL_HF}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Batches:         ${NUM_BATCHES} (+ ${WARMUP_BATCHES} warmup)"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Seq Length:      ${SEQ_LENGTH}"
echo "============================================================"

# ============================================================
# Step 1: Checkpoint Conversion (if needed)
# ============================================================
if [ "$SKIP_CONVERSION" = false ]; then
    if [ -n "$BASELINE_CKPT" ]; then
        echo ""
        echo ">>> Converting BASELINE checkpoint to HuggingFace format..."
        bash "${SCRIPT_DIR}/run_lm_eval.sh" \
            --checkpoint-dir "${BASELINE_CKPT}" \
            --hf-output-dir "${BASELINE_HF}" \
            --original-hf "${ORIGINAL_HF}" \
            --model-size "${MODEL_SIZE}" \
            --results-dir /dev/null \
            --tasks "none" 2>/dev/null || true
        # If conversion fails or baseline ckpt not provided, use original HF
        if [ ! -d "${BASELINE_HF}" ] || [ -z "$(ls -A ${BASELINE_HF} 2>/dev/null)" ]; then
            echo "  Using original HF as baseline."
            BASELINE_HF="${ORIGINAL_HF}"
        fi
    else
        echo "  No baseline Megatron checkpoint provided, using original HF."
        BASELINE_HF="${ORIGINAL_HF}"
    fi

    if [ -n "$OPTIMIZED_CKPT" ]; then
        echo ""
        echo ">>> Converting OPTIMIZED checkpoint to HuggingFace format..."
        cd "${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
        bash scripts/qwen3/run_8xH20.sh \
            "${MODEL_SIZE}" \
            "${OPTIMIZED_CKPT}" \
            "${OPTIMIZED_HF}" \
            true \
            true \
            bf16 \
            "${ORIGINAL_HF}"
    fi
else
    echo "  Skipping conversion (--skip-conversion)"
    if [ ! -d "${BASELINE_HF}" ]; then
        BASELINE_HF="${ORIGINAL_HF}"
    fi
fi

echo ""
echo "Baseline HF:  ${BASELINE_HF}"
echo "Optimized HF: ${OPTIMIZED_HF}"

# ============================================================
# Step 2: LM-Eval Accuracy Benchmarks
# ============================================================
if [ "$SKIP_LM_EVAL" = false ]; then
    echo ""
    echo "============================================================"
    echo "RUNNING LM-EVAL ACCURACY BENCHMARKS"
    echo "============================================================"

    echo ">>> Baseline accuracy..."
    bash "${SCRIPT_DIR}/run_lm_eval.sh" \
        --hf-output-dir "${BASELINE_HF}" \
        --results-dir "${OUTPUT_DIR}/lm_eval_baseline" \
        --skip-conversion \
        --batch-size 8

    if [ -d "${OPTIMIZED_HF}" ]; then
        echo ">>> Optimized accuracy..."
        bash "${SCRIPT_DIR}/run_lm_eval.sh" \
            --hf-output-dir "${OPTIMIZED_HF}" \
            --results-dir "${OUTPUT_DIR}/lm_eval_optimized" \
            --skip-conversion \
            --batch-size 8
    fi
else
    echo ""
    echo "  Skipping LM-eval (--skip-lm-eval)"
fi

# ============================================================
# Step 3: Critical Path Metrics
# ============================================================
echo ""
echo "============================================================"
echo "MEASURING CRITICAL PATH METRICS"
echo "============================================================"

echo ">>> Baseline critical path..."
python3 "${SCRIPT_DIR}/measure_critical_path.py" \
    --model-path "${BASELINE_HF}" \
    --num-batches "${NUM_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --output-file "${OUTPUT_DIR}/critical_path_baseline.json" \
    --label "baseline"

if [ -d "${OPTIMIZED_HF}" ]; then
    echo ">>> Optimized critical path..."
    python3 "${SCRIPT_DIR}/measure_critical_path.py" \
        --model-path "${OPTIMIZED_HF}" \
        --num-batches "${NUM_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}" \
        --output-file "${OUTPUT_DIR}/critical_path_optimized.json" \
        --label "optimized"
fi

# ============================================================
# Step 4: Latency Measurement
# ============================================================
echo ""
echo "============================================================"
echo "MEASURING INFERENCE LATENCY"
echo "============================================================"

if [ -d "${OPTIMIZED_HF}" ]; then
    python3 "${SCRIPT_DIR}/measure_latency.py" \
        --model-path "${BASELINE_HF}" \
        --compare-path "${OPTIMIZED_HF}" \
        --num-batches "${NUM_BATCHES}" \
        --warmup-batches "${WARMUP_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}" \
        --output-file "${OUTPUT_DIR}/latency_comparison.json" \
        --label "baseline" \
        --compare-label "optimized"
else
    python3 "${SCRIPT_DIR}/measure_latency.py" \
        --model-path "${BASELINE_HF}" \
        --num-batches "${NUM_BATCHES}" \
        --warmup-batches "${WARMUP_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seq-length "${SEQ_LENGTH}" \
        --output-file "${OUTPUT_DIR}/latency_baseline.json" \
        --label "baseline"
fi

# ============================================================
# Step 5: Generate Combined Summary
# ============================================================
echo ""
echo "============================================================"
echo "GENERATING COMBINED SUMMARY"
echo "============================================================"

python3 << 'PYEOF'
import json
import os
import glob
import sys

output_dir = os.environ.get("OUTPUT_DIR", "./benchmark_results")

summary = {"benchmarks": {}}

# Load critical path results
for label in ["baseline", "optimized"]:
    cp_file = os.path.join(output_dir, f"critical_path_{label}.json")
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            summary["benchmarks"][f"critical_path_{label}"] = json.load(f)

# Load latency results
for fname in ["latency_comparison.json", "latency_baseline.json"]:
    lat_file = os.path.join(output_dir, fname)
    if os.path.exists(lat_file):
        with open(lat_file) as f:
            summary["benchmarks"]["latency"] = json.load(f)
        break

# Load accuracy results
for label in ["baseline", "optimized"]:
    acc_file = os.path.join(output_dir, f"lm_eval_{label}", "accuracy_summary.json")
    if os.path.exists(acc_file):
        with open(acc_file) as f:
            summary["benchmarks"][f"accuracy_{label}"] = json.load(f)

# Generate markdown summary
md_lines = ["# MoE Router Optimization Benchmark Results\n"]

# Critical path comparison
if "critical_path_baseline" in summary["benchmarks"]:
    bl = summary["benchmarks"]["critical_path_baseline"]
    md_lines.append("## Critical Path\n")
    md_lines.append("| Metric | Baseline | Optimized | Change |")
    md_lines.append("|--------|----------|-----------|--------|")

    bl_cp = bl["critical_path"]["mean"]
    if "critical_path_optimized" in summary["benchmarks"]:
        opt = summary["benchmarks"]["critical_path_optimized"]
        opt_cp = opt["critical_path"]["mean"]
        change = (opt_cp - bl_cp) / bl_cp * 100
        md_lines.append(f"| Critical Path (tokens) | {bl_cp:.0f} | {opt_cp:.0f} | {change:+.1f}% |")
        md_lines.append(f"| Imbalance Ratio | {bl['imbalance_ratio']['mean']:.3f}x | {opt['imbalance_ratio']['mean']:.3f}x | |")
        md_lines.append(f"| Entropy | {bl['normalized_entropy']['mean']:.4f} | {opt['normalized_entropy']['mean']:.4f} | |")

        theo_speedup = bl_cp / opt_cp
        md_lines.append(f"\n**Theoretical Speedup (expert-parallel):** {theo_speedup:.2f}x\n")
    else:
        md_lines.append(f"| Critical Path (tokens) | {bl_cp:.0f} | - | - |")

# Latency comparison
if "latency" in summary["benchmarks"]:
    lat = summary["benchmarks"]["latency"]
    md_lines.append("## Inference Latency\n")
    md_lines.append("| Metric | Baseline | Optimized |")
    md_lines.append("|--------|----------|-----------|")
    if "primary" in lat and "comparison" in lat:
        p = lat["primary"]["latency_ms"]
        c = lat["comparison"]["latency_ms"]
        md_lines.append(f"| Mean Latency (ms) | {p['mean']:.2f} | {c['mean']:.2f} |")
        md_lines.append(f"| P95 Latency (ms) | {p['p95']:.2f} | {c['p95']:.2f} |")
        md_lines.append(f"| Throughput (tok/s) | {lat['primary']['throughput_tokens_per_sec']:.0f} | {lat['comparison']['throughput_tokens_per_sec']:.0f} |")
        if "speedup" in lat:
            md_lines.append(f"\n**Measured Speedup:** {lat['speedup']['mean_latency_ratio']:.3f}x\n")

# Accuracy comparison
if "accuracy_baseline" in summary["benchmarks"]:
    bl_acc = summary["benchmarks"]["accuracy_baseline"]
    md_lines.append("## Accuracy (LM-Evaluation-Harness)\n")
    md_lines.append("| Task | Baseline | Optimized | Delta |")
    md_lines.append("|------|----------|-----------|-------|")

    opt_acc = summary["benchmarks"].get("accuracy_optimized", {})
    bl_tasks = bl_acc.get("tasks", {})
    opt_tasks = opt_acc.get("tasks", {})

    for task in sorted(bl_tasks.keys()):
        bl_v = bl_tasks[task]
        opt_v = opt_tasks.get(task)
        if opt_v is not None:
            delta = opt_v - bl_v
            md_lines.append(f"| {task} | {bl_v:.4f} | {opt_v:.4f} | {delta:+.4f} |")
        else:
            md_lines.append(f"| {task} | {bl_v:.4f} | - | - |")

    bl_avg = bl_acc.get("average", 0)
    opt_avg = opt_acc.get("average")
    if opt_avg is not None:
        delta = opt_avg - bl_avg
        md_lines.append(f"| **Average** | **{bl_avg:.4f}** | **{opt_avg:.4f}** | **{delta:+.4f}** |")

md_text = "\n".join(md_lines)

# Write files
summary_md = os.path.join(output_dir, "summary.md")
with open(summary_md, "w") as f:
    f.write(md_text)

summary_json = os.path.join(output_dir, "summary.json")
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(md_text)
print(f"\nSummary saved to: {summary_md}")
print(f"Full data saved to: {summary_json}")

PYEOF

echo ""
echo "============================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "Results in: ${OUTPUT_DIR}"
echo "  summary.md              -- Markdown report"
echo "  summary.json            -- Full structured data"
echo "  critical_path_*.json    -- Per-layer routing metrics"
echo "  latency_*.json          -- Inference timing data"
echo "  lm_eval_*/              -- Accuracy benchmark results"
echo "============================================================"
