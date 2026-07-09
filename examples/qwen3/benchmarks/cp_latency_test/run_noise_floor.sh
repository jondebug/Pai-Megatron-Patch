#!/bin/bash
#SBATCH --job-name=noise_floor
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/noise_floor_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
# NOISE-FLOOR CONTROL.
# Load the SAME pretrained model twice, under two distinct cache roots (no sharing).
# Whatever delta we see between pre_first and pre_second is the pure compile-noise floor.
# If pre_first vs pre_second already differs by ~+9%, then "fine-tuned cells are +9% slower"
# is a noise artifact, not a fine-tuning artifact. If pre_first vs pre_second is ~0%, then
# the +9% delta is genuinely cell-dependent (fine-tuned-induced) and needs explaining.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/nf_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   # Same pretrained loaded as two cells — naming differs so vllm treats them independently
   python3 '$SCRIPT' --models pre_first='$PRE',pre_second='$PRE' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/noise_floor_${SLURM_JOB_ID}.json
  '
echo "done rc=$?"
