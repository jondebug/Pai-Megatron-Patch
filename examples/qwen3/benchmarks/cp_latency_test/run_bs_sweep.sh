#!/bin/bash
#SBATCH --job-name=bs_sweep
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/bs_sweep_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# BS-SWEEP test: per-expert M is the regime parameter. Climb the curve.
#   bs=1: ~512 tok/expert  (below knee — pretrained faster, r15 +9%, observed)
#   bs=2: ~1024 tok/expert (near knee — predicted shrinkage)
#   bs=4: ~2048 tok/expert (above knee — predicted sign flip near zero)
#   bs=8: ~4096 tok/expert (well above knee — predicted r15 FASTER than pretrained)
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/bs_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1,2,4,8 --max-tokens 8 \
     --num-warmup 3 --num-trials 20 --cuda-graphs \
     --output '$RES'/bs_sweep_${SLURM_JOB_ID}.json
  '
echo "done rc=$?"
