#!/bin/bash
#SBATCH --job-name=climb_M3
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/climb_M3_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# Climb M curve: high gpu_mem_util + small max_num_seqs → minimize KV reservation, max activation headroom.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/cm3_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   # default gpu_memory_utilization = 0.85; max_num_seqs tightened to just what we need.
   export VLLM_GPU_MEMORY_UTILIZATION=0.90

   # M=1024 (69%). bs=16 plen=1024 = 16384 tokens. max_num_seqs only needs to cover bs.
   export VLLM_MAX_NUM_BATCHED_TOKENS=16384 VLLM_MAX_NUM_SEQS=32
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 16 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/climb3_M1024_${SLURM_JOB_ID}.json || echo "M=1024 RC=$?"

   # M=2048 (85%). bs=32 plen=1024 = 32768 tokens.
   export VLLM_MAX_NUM_BATCHED_TOKENS=32768 VLLM_MAX_NUM_SEQS=64
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 32 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/climb3_M2048_${SLURM_JOB_ID}.json || echo "M=2048 RC=$?"

   # M=4096 (94% knee). bs=64 plen=1024 = 65536 tokens.
   export VLLM_MAX_NUM_BATCHED_TOKENS=65536 VLLM_MAX_NUM_SEQS=128
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 1024 --batch-sizes 64 --max-tokens 4 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/climb3_M4096_${SLURM_JOB_ID}.json || echo "M=4096 RC=$?"
  '
echo "done rc=$?"
