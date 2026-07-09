#!/bin/bash
#SBATCH --job-name=ar_compare
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ar_compare_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
NSYS=$BASE/tools/nsys/NsightSystems-cli-2025.2.1/target-linux-x64/nsys
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
R05=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc0.1_ppo_aux0.02_r05/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2500_cp
echo "=== same-node AR comparison on $(hostname) $(date) ==="
# This launches vLLM ONCE with three models registered; cp_vllm_bench loads
# them sequentially in the same engine context, so NCCL/topology/IPC state
# is identical between cells.
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c '
   export TMPDIR=/tmp/arcomp_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p $TMPDIR
   OUT='$RES'/ar_compare_samenode_${SLURM_JOB_ID}
   '$NSYS' profile -t cuda,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node --sample=none \
     --force-overwrite=true -o $OUT \
     python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15',r05_ep8='$R05' --tp-size 8 \
       --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
       --num-warmup 3 --num-trials 20 --cuda-graphs \
       --output '$RES'/ar_compare_samenode_${SLURM_JOB_ID}.json
   echo RC=$?
   ls -la $OUT.nsys-rep
  '
echo "=== done rc=$? $(date) ==="
