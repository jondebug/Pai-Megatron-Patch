#!/bin/bash
#SBATCH --job-name=decode_routing
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/decode_routing_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
DUMP=$DIR/cp_routing_dump.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# Decode-regime routing: seq_length=1 (1 token per request), batch_size = various.
# At each bs, captures per-layer per-expert token counts during single-step "decode" forward.
# Tests the §17 idleness prediction at the actual decode operating point.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/dr_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   for bs in 64 256 512 1024 2048; do
     for mname in pre r15; do
       case $mname in
         pre) M='$PRE';;
         r15) M='$R15';;
       esac
       python3 '$DUMP' --model $M --name ${mname}_decode_bs${bs} --seq-length 1 --batch-size $bs \
         --num-batches 4 --out '$RES'/routing_decode_${mname}_bs${bs}_${SLURM_JOB_ID}.json
     done
   done
  '
echo "done rc=$?"
