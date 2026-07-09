#!/bin/bash
#SBATCH --job-name=diag_extended
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/diag_extended_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
R05=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc0.1_ppo_aux0.02_r05/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2500_cp
R60=$BASE/rl_token_routing/output_router_finetuning/235bv5b_rlc0.5_aux0.015_kl0.0003_r60/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2661_cp
echo "=== diag extended: 5-cell same-node, graphs-ON, n=50 on $(hostname) $(date) ==="
# Cells loaded in order: pre, pre_dup (same path - tests load-order effect), r15, r05, r60
# n=50 trials gives tight CI. graphs-on (--cuda-graphs) tests the production path.
# TORCH_LOGS=inductor dumps compile decisions to stderr; we tee to file.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c '
   export TMPDIR=/tmp/diag_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TORCH_LOGS="inductor,recompiles" TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_cache
   mkdir -p $TMPDIR/inductor_cache
   LOG='$RES'/diag_extended_${SLURM_JOB_ID}.torchlog
   python3 '$SCRIPT' --models pre_ep8='$PRE',pre_dup_ep8='$PRE',r15_ep8='$R15',r05_ep8='$R05',r60_ep8='$R60' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 50 --cuda-graphs \
     --output '$RES'/diag_extended_${SLURM_JOB_ID}.json 2> $LOG | tail -200
   echo RC=$?
   echo === inductor cache size ===
   du -sh $TMPDIR/inductor_cache
   ls -la $TMPDIR/inductor_cache | head -20
   echo === torchlog size ===
   du -sh $LOG
  '
echo "=== done rc=$? $(date) ==="
