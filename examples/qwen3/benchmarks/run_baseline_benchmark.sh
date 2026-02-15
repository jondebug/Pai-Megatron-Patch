#!/bin/bash
#SBATCH --job-name=baseline_benchmark
#SBATCH --account=nvr_israel_scne
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_results/baseline_benchmark_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_results/baseline_benchmark_%j.err

# =============================================================================
# Baseline benchmark: measure critical path on pretrained Qwen3-30B-A3B
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
OUTPUT_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_results"
NUM_BATCHES=100
BATCH_SIZE=1
SEQ_LENGTH=128

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Baseline Benchmark: Qwen3-30B-A3B (pretrained)"
echo "============================================================"
echo "Model:       ${MODEL_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Batches:     ${NUM_BATCHES}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Seq Length:  ${SEQ_LENGTH}"
echo "GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================================"

# Install dependencies if needed
pip install -q wandb 2>/dev/null || true

# Run critical path measurement
echo ""
echo ">>> Measuring critical path metrics..."
python3 "${SCRIPT_DIR}/measure_critical_path.py" \
    --model-path "${MODEL_PATH}" \
    --num-batches "${NUM_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --output-file "${OUTPUT_DIR}/critical_path_baseline.json" \
    --label "Qwen3-30B-A3B-pretrained"

# Run latency measurement
echo ""
echo ">>> Measuring inference latency..."
python3 "${SCRIPT_DIR}/measure_latency.py" \
    --model-path "${MODEL_PATH}" \
    --num-batches "${NUM_BATCHES}" \
    --warmup-batches 10 \
    --batch-size "${BATCH_SIZE}" \
    --seq-length "${SEQ_LENGTH}" \
    --output-file "${OUTPUT_DIR}/latency_baseline.json" \
    --label "Qwen3-30B-A3B-pretrained"

# Log results to WandB
echo ""
echo ">>> Logging results to WandB..."
python3 << 'PYEOF'
import json
import os

output_dir = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_results"

# Load results
cp_file = os.path.join(output_dir, "critical_path_baseline.json")
lat_file = os.path.join(output_dir, "latency_baseline.json")

cp_data = {}
lat_data = {}

if os.path.exists(cp_file):
    with open(cp_file) as f:
        cp_data = json.load(f)

if os.path.exists(lat_file):
    with open(lat_file) as f:
        lat_data = json.load(f)

# Log to WandB
try:
    import wandb

    run = wandb.init(
        project="qwen3-router-training",
        name="baseline-benchmark-pretrained",
        tags=["benchmark", "baseline", "pretrained"],
        config={
            "model": "Qwen3-30B-A3B",
            "checkpoint": "pretrained (no training)",
            "num_batches": cp_data.get("config", {}).get("num_batches", 100),
            "batch_size": cp_data.get("config", {}).get("batch_size", 1),
            "seq_length": cp_data.get("config", {}).get("seq_length", 128),
        }
    )

    # Log critical path metrics
    if cp_data:
        cp = cp_data.get("critical_path", {})
        wandb.log({
            "benchmark/critical_path_mean": cp.get("mean", 0),
            "benchmark/critical_path_std": cp.get("std", 0),
            "benchmark/critical_path_p50": cp.get("p50", 0),
            "benchmark/critical_path_p95": cp.get("p95", 0),
            "benchmark/ideal_critical_path": cp_data.get("ideal_critical_path", 0),
            "benchmark/imbalance_ratio": cp_data.get("imbalance_ratio", {}).get("mean", 0),
            "benchmark/normalized_entropy": cp_data.get("normalized_entropy", {}).get("mean", 0),
            "benchmark/num_moe_layers": cp_data.get("num_moe_layers", 0),
        })

        # Log per-layer max tokens
        per_layer = cp_data.get("per_layer_avg_max_tokens", [])
        for i, val in enumerate(per_layer):
            wandb.log({f"per_layer/layer_{i}_max_tokens": val})

        # Log LM loss if available
        lm = cp_data.get("lm_loss", {})
        if lm.get("mean") is not None:
            wandb.log({
                "benchmark/lm_loss": lm["mean"],
                "benchmark/perplexity": lm.get("perplexity", 0),
            })

    # Log latency metrics
    if lat_data:
        primary = lat_data.get("primary", lat_data)
        lat = primary.get("latency_ms", {})
        wandb.log({
            "benchmark/latency_mean_ms": lat.get("mean", 0),
            "benchmark/latency_p50_ms": lat.get("p50", 0),
            "benchmark/latency_p95_ms": lat.get("p95", 0),
            "benchmark/latency_p99_ms": lat.get("p99", 0),
            "benchmark/throughput_tokens_per_sec": primary.get("throughput_tokens_per_sec", 0),
        })

    # Log summary table
    if cp_data:
        cp = cp_data.get("critical_path", {})
        table = wandb.Table(columns=["Metric", "Value"])
        table.add_data("Critical Path (mean)", f"{cp.get('mean', 0):.0f}")
        table.add_data("Critical Path (p95)", f"{cp.get('p95', 0):.0f}")
        table.add_data("Ideal Critical Path", f"{cp_data.get('ideal_critical_path', 0):.0f}")
        table.add_data("Imbalance Ratio", f"{cp_data.get('imbalance_ratio', {}).get('mean', 0):.3f}x")
        if lat_data:
            primary = lat_data.get("primary", lat_data)
            lat = primary.get("latency_ms", {})
            table.add_data("Latency Mean (ms)", f"{lat.get('mean', 0):.2f}")
            table.add_data("Throughput (tok/s)", f"{primary.get('throughput_tokens_per_sec', 0):.0f}")
        wandb.log({"benchmark/summary_table": table})

    wandb.finish()
    print("WandB logging complete.")

except ImportError:
    print("WARNING: wandb not installed, skipping WandB logging")
except Exception as e:
    print(f"WARNING: WandB logging failed: {e}")

PYEOF

echo ""
echo "============================================================"
echo "Baseline benchmark complete. Results in: ${OUTPUT_DIR}"
echo "============================================================"
