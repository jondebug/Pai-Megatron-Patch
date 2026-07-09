#!/bin/bash
#SBATCH --job-name=bs_sweep2
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/bs_sweep2_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
R05=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc0.1_ppo_aux0.02_r05/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2500_cp
# Wider bs sweep at EP8 plen=8192. Tracks per-expert M up the GEMM-saturation curve.
#   plen=8192, top_k=8, EP8, 16 experts/GPU → avg per-local-expert M = bs*plen*top_k/(EP*n_experts/EP) = bs*plen*top_k/n_experts = bs*512
#   bs=2  → ~1024 avg M  (near knee)
#   bs=4  → ~2048 avg M  (just above knee — predicted r15 win is here)
#   bs=8  → ~4096 avg M
#   bs=16 → ~8192 avg M  (deeply saturated)
#   bs=32 → ~16384 avg M (very deep)
# Include r05 (4th-lowest-CP cell) to widen the cell coverage.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/bs2_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15',r05_ep8='$R05' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 2,4,8,16,32 --max-tokens 8 \
     --num-warmup 3 --num-trials 50 --cuda-graphs \
     --output '$RES'/bs_sweep2_${SLURM_JOB_ID}.json
  '
echo "done rc=$?"
