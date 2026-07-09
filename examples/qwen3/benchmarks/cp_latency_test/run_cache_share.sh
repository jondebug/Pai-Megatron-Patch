#!/bin/bash
#SBATCH --job-name=cache_share
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cache_share_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
R05=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc0.1_ppo_aux0.02_r05/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2500_cp
STAGED_DIR=$BASE/rl_token_routing/staged_models
STAGED=$STAGED_DIR/m
mkdir -p $STAGED_DIR
echo "=== cache_share Layer-1 test on $(hostname) $(date) ==="
echo "STAGED path: $STAGED"

# Run pretrained at STAGED path first to populate the cache slot keyed on $STAGED
ln -sfn $PRE $STAGED
echo "=== STAGED -> pretrained ==="
ls -la $STAGED

srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c '
   export TMPDIR=/tmp/cs_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   echo "=== Run A: pretrained via STAGED path (populates cache) ==="
   python3 '$SCRIPT' --models preA='$STAGED' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/cs_runA_${SLURM_JOB_ID}.json
   echo "=== Cache directories after Run A ==="
   ls -lt ~/.cache/vllm/torch_compile_cache/ | head -5
  '

# Swap symlink to r15
ln -sfn $R15 $STAGED
echo "=== STAGED -> r15 ==="
ls -la $STAGED

srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c '
   export TMPDIR=/tmp/cs_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   echo "=== Run B: r15 via SAME STAGED path (should hit Run A cache) ==="
   python3 '$SCRIPT' --models r15B='$STAGED' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/cs_runB_${SLURM_JOB_ID}.json
   echo "=== Cache directories after Run B ==="
   ls -lt ~/.cache/vllm/torch_compile_cache/ | head -5
  '

# Swap to r05
ln -sfn $R05 $STAGED
echo "=== STAGED -> r05 ==="

srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c '
   export TMPDIR=/tmp/cs_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   echo "=== Run C: r05 via SAME STAGED path (should hit Run A cache) ==="
   python3 '$SCRIPT' --models r05C='$STAGED' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/cs_runC_${SLURM_JOB_ID}.json
  '
echo "=== done rc=$? $(date) ==="
