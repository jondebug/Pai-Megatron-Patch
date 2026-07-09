#!/bin/bash
#SBATCH --job-name=cp_ep8_trace
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_trace_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_trace_%x_%j.err
# Profiled decode burst (PATCHED profiler_config path) for stage decomposition + 3rd stress leg.
# One model; profile at the PROFBS batch (default 64) and also bs8 if PROFBS2 set.
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
MODEL="${MODEL:?}"; MODELNAME="${MODELNAME:?}"; PROFBS="${PROFBS:-64}"
echo "=== EP8 trace model=$MODELNAME profbs=$PROFBS node=$(hostname) $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c "
   export TMPDIR=/tmp/ep8tr_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1; mkdir -p \$TMPDIR
   PDIR=$RES/trace2_${MODELNAME}_ep8_bs${PROFBS}_\${SLURM_JOB_ID}; mkdir -p \$PDIR
   export VLLM_TORCH_PROFILER_DIR=\$PDIR
   python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
     --prompt-lengths 256 --batch-sizes $PROFBS --max-tokens 16 \
     --num-warmup 3 --num-trials 2 --profile --profile-steps 16 \
     --output $RES/vllm_ep8_traceprof_${MODELNAME}_bs${PROFBS}_\${SLURM_JOB_ID}.json
   RC=\$?; echo \"PROF_RC=\$RC\"; echo \"TRACE_DIR=\$PDIR\"; ls -la \$PDIR | head -20
   exit \$RC
  "
echo "=== done rc=$? $(date) ==="
