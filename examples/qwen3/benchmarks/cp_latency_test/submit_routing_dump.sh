#!/bin/bash
#SBATCH --job-name=cp_routing_dump
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_routing_dump_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_routing_dump_%j.err
# Resolve CP-semantics paradox + measure real 235B per-layer hot-expert imbalance.
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
RES=$BASE/rl_token_routing/cp_latency_results
SCRIPT=$DIR/cp_routing_dump.py
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
echo "=== routing dump node=$(hostname) $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c "
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 MAX_MEM_PER_GPU=62GiB HF_HUB_OFFLINE=1
    export MEGDATA_PREFIX=$BASE/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
    export PYTHONPATH=$BASE/rl_token_routing/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624:\${PYTHONPATH:-}
    python3 $SCRIPT --model $PRE --name pretrained_235b --seq-length 2048 --batch-size 4 --num-batches 8 \
      --out $RES/routing_dump_pretrained_\${SLURM_JOB_ID}.json
    echo R_PRE=\$?
    python3 $SCRIPT --model $R15 --name r15_cp4682 --seq-length 2048 --batch-size 4 --num-batches 8 \
      --out $RES/routing_dump_r15_\${SLURM_JOB_ID}.json
    echo R_R15=\$?
  "
echo "=== done rc=$? $(date) ==="
