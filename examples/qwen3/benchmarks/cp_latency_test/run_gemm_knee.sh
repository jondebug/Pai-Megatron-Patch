#!/bin/bash
#SBATCH --job-name=gemm_knee
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/gemm_knee_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
srun --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  python3 $DIR/gemm_knee.py
echo "=== done rc=$? $(date) ==="
