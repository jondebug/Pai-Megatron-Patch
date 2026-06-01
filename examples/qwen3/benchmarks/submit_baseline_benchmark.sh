#!/bin/bash
#SBATCH --job-name=baseline_bench
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/baseline_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/baseline_%j.err

# Minimal baseline benchmark for an already-HF checkpoint. Skips conversion.
# Usage: sbatch submit_baseline_benchmark.sh <hf_checkpoint_dir> <run_name> [limit]

set -uo pipefail

HF_DIR="${1:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B}"
RUN_NAME="${2:-pretrained_235b}"
LIMIT="${3:-1000}"
RESULTS_DIR="${HF_DIR}/benchmark_results_limit${LIMIT}"

CONTAINER=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh

echo "============================================================"
echo "Baseline benchmark for HF checkpoint"
echo "HF dir:      $HF_DIR"
echo "Run name:    $RUN_NAME"
echo "Limit:       $LIMIT"
echo "Results dir: $RESULTS_DIR"
echo "SLURM job:   ${SLURM_JOB_ID:-manual}  Node: ${SLURM_NODELIST:-local}"
echo "Start:       $(date)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

srun --container-image="$CONTAINER" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$(dirname $HF_DIR)" \
     bash -c "
       set -uo pipefail
       pip install 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null
       export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
       export HF_DATASETS_CACHE=\$HF_HOME/datasets
       mkdir -p \$HF_HOME \$HF_DATASETS_CACHE

       echo 'Running lm_eval directly on HF checkpoint'
       echo '  hf_dir:  $HF_DIR'
       echo '  tasks:   hellaswag,arc_challenge,winogrande'
       echo '  limit:   $LIMIT'

       python3 -m lm_eval \
         --model hf \
         --model_args pretrained=$HF_DIR,trust_remote_code=True,dtype=bfloat16,device_map=auto \
         --tasks hellaswag,arc_challenge,winogrande \
         --batch_size 8 \
         --output_path $RESULTS_DIR \
         --limit $LIMIT
     "

echo "============================================================"
echo "lm_eval finished at $(date)"
echo "Results in $RESULTS_DIR"
ls -la $RESULTS_DIR 2>&1 | head -10
echo "============================================================"
