#!/bin/bash
#SBATCH --job-name=reverse_cache
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/reverse_cache_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# REVERSE-DIRECTION CONTROL.
# Step 1: load r15 first to populate Triton+Inductor caches with r15's tuned kernels.
# Step 2: load pretrained second under the SAME cache root, no separate isolation.
# If the +9% delta is purely a compile-cache artifact, pretrained-second should now exhibit
# the slowness (i.e. the "bad" cell is whichever loads second under another cell's cache),
# not whichever cell is fine-tuned. If pretrained-second is FAST despite using r15's cache,
# the artifact is intrinsic to the fine-tuned model state, not the cache.
SHARED=$BASE/rl_token_routing/cp_latency_results/shared_caches/rev_$SLURM_JOB_ID
mkdir -p $SHARED/triton $SHARED/inductor
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/rev_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TRITON_CACHE_DIR='$SHARED'/triton
   export TORCHINDUCTOR_CACHE_DIR='$SHARED'/inductor
   mkdir -p $TMPDIR
   # ORDER FLIPPED: r15 first, pretrained second
   python3 '$SCRIPT' --models r15_first='$R15',pre_second='$PRE' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/reverse_cache_${SLURM_JOB_ID}.json
  '
echo "done rc=$?"
