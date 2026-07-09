#!/bin/bash
#SBATCH --job-name=above_knee
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/above_knee_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# Above-knee test: tokens_per_forward in {65536 (M=4096 knee), 131072 (M=8192 saturated)}
# Configs lowering plen and raising bs to keep memory bounded while pushing tokens per forward.
# Set VLLM_MAX_NUM_BATCHED_TOKENS to disable chunked prefill.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/ak_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   # Run 1: M=4096 (at knee). bs=64 plen=1024 = 65536 tokens/forward.
   export VLLM_MAX_NUM_BATCHED_TOKENS=65536
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 64 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/above_knee_M4096_${SLURM_JOB_ID}.json
   # Run 2: M=8192 (saturated). bs=128 plen=1024 = 131072 tokens/forward.
   export VLLM_MAX_NUM_BATCHED_TOKENS=131072
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 128 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/above_knee_M8192_${SLURM_JOB_ID}.json
  '
echo "done rc=$?"
