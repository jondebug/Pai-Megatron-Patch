#!/bin/bash
#SBATCH --job-name=baseline_vllm
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/baseline_vllm_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/baseline_vllm_%j.err

# vLLM TP=8 accuracy benchmark over hellaswag/arc_challenge/winogrande.
# Usage: sbatch submit_baseline_benchmark_vllm.sh [hf_dir] [run_name] [limit]
set -uo pipefail

HF_DIR="${1:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B}"
RUN_NAME="${2:-pretrained_235b}"
LIMIT="${3:-1000}"
RESULTS_DIR="${HF_DIR}/benchmark_results_vllm_limit${LIMIT}"

CONTAINER=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
INNER=$REPO_ROOT/examples/qwen3/benchmarks/_run_lm_eval_vllm_inner.sh

echo "============================================================"
echo "vLLM lm-eval (TP=8)  job=${SLURM_JOB_ID:-manual}  node=${SLURM_NODELIST:-local}"
echo "HF dir:      $HF_DIR"
echo "Run name:    $RUN_NAME"
echo "Limit:       $LIMIT"
echo "Results dir: $RESULTS_DIR"
echo "Start:       $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

# Export env so the inner helper sees them (avoids embedded-quote hell)
export HF_DIR RESULTS_DIR LIMIT

srun --container-image="$CONTAINER" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$(dirname "$HF_DIR")" \
     --export=ALL,HF_DIR="$HF_DIR",RESULTS_DIR="$RESULTS_DIR",LIMIT="$LIMIT" \
     "$INNER"

RC=$?
echo "============================================================"
echo "Finished at $(date) (exit $RC). Result files:"
ls -la "$RESULTS_DIR" 2>&1 | head -20
exit $RC
