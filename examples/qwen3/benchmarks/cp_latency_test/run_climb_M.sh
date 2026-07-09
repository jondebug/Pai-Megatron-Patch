#!/bin/bash
#SBATCH --job-name=climb_M
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/climb_M_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# Climb the GEMM-throughput curve in single-forward mode by overriding max_num_batched_tokens.
# Targets per-expert M of 1024 (69%) and 2048 (85%). Knee M=4096 OOMs at 8 GPUs.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/climb_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR

   # Run 1: M=1024 (69% asymptote). bs=16 plen=1024 = 16384 tokens/forward.
   export VLLM_MAX_NUM_BATCHED_TOKENS=16384
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 16 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/climb_M1024_${SLURM_JOB_ID}.json || echo "M=1024 RC=$?"

   # Run 2: M=2048 (85% asymptote). bs=32 plen=1024 = 32768 tokens/forward.
   export VLLM_MAX_NUM_BATCHED_TOKENS=32768
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 32 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/climb_M2048_${SLURM_JOB_ID}.json || echo "M=2048 RC=$?"
  '
echo "done rc=$?"
