#!/bin/bash
#SBATCH --job-name=perexpert_M
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/perexpert_M_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
DUMP=$DIR/cp_routing_dump.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# Routing-dump per-expert-M distribution at the bs values where TTFT crosses the knee.
# Generates per-layer, per-expert token counts so we can directly read off:
#   - the distribution of M_i across the 128 experts at each bs
#   - the busiest-expert M (the canonical CP metric)
#   - the median expert M  (what most experts see)
#   - the count of experts below the GEMM-saturation knee (~1000 M)
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/pem_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   for bs in 1 2 4 8 16; do
     for mname in pre r15; do
       case $mname in
         pre) M='$PRE';;
         r15) M='$R15';;
       esac
       python3 '$DUMP' --model $M --name ${mname}_bs${bs} --seq-length 8192 --batch-size $bs \
         --num-batches 4 --out '$RES'/routing_${mname}_bs${bs}_${SLURM_JOB_ID}.json
     done
   done
  '
echo "done rc=$?"
