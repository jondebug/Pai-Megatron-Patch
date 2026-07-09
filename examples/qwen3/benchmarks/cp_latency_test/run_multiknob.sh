#!/bin/bash
#SBATCH --job-name=multiknob
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/multiknob_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
# MULTI-KNOB combo: combo_kernels=False + cudagraph_mode=PIECEWISE + shared Triton/Inductor sub-caches.
# Each single knob closed at most ~2-3% of the +9% delta. This stacks all three to see if combined
# effect crosses the noise threshold.
COMPILE_JSON='{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false},"cudagraph_mode":"PIECEWISE"}'
SHARED=$BASE/rl_token_routing/cp_latency_results/shared_caches/mk_$SLURM_JOB_ID
mkdir -p $SHARED/triton $SHARED/inductor
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image=$CONTAINER --container-mounts=$BASE:$BASE --container-workdir=$DIR \
  bash -c '
   export TMPDIR=/tmp/mk_$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export VLLM_COMPILATION_CONFIG_JSON='"'"$COMPILE_JSON"'"'
   export TRITON_CACHE_DIR='$SHARED'/triton
   export TORCHINDUCTOR_CACHE_DIR='$SHARED'/inductor
   mkdir -p $TMPDIR
   python3 '$SCRIPT' --models pre_ep8='$PRE',r15_ep8='$R15' --tp-size 8 \
     --prompt-lengths 8192 --batch-sizes 1 --max-tokens 8 \
     --num-warmup 3 --num-trials 30 --cuda-graphs \
     --output '$RES'/multiknob_${SLURM_JOB_ID}.json
   echo "=== triton cache:"; du -sh '$SHARED'/triton 2>&1 | tail -1
   echo "=== inductor cache:"; du -sh '$SHARED'/inductor 2>&1 | tail -1
  '
echo "done rc=$?"
