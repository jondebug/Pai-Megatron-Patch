#!/bin/bash
#SBATCH --job-name=cp_ep8_swprof
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_swprof_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_swprof_%x_%j.err

# Phase B: single-node EP=8 (all-NVLink). For ONE model ($MODEL/$MODELNAME):
#   1) batch sweep bs=1,8,32,64,128 at plen256 (decode_tps vs batch -> exposes CP/compute knee)
#   2) a profiled decode burst at bs=$PROFBS (per-rank GPU kernel traces -> stage decomposition)
# Run once per checkpoint (pretrained, r15). Compare cross-run.
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
MODEL="${MODEL:?set MODEL}"
MODELNAME="${MODELNAME:?set MODELNAME}"
PROFBS="${PROFBS:-64}"
SWEEP_BS="${SWEEP_BS:-1,8,32,64,128}"
MNT="$BASE:$BASE"

echo "=== Phase B EP=8 sweep+profile  model=$MODELNAME  node=$(hostname)  $(date) ==="
echo "    sweep bs=$SWEEP_BS  profbs=$PROFBS"

srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" \
     --container-workdir="$DIR" \
  bash -c "
   export TMPDIR=/tmp/ep8sp_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   mkdir -p \$TMPDIR
   PDIR=$RES/trace_${MODELNAME}_ep8_\${SLURM_JOB_ID}
   mkdir -p \$PDIR
   # 1) batch sweep (no profiler) -> aggregate decode_tps vs batch
   python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
     --prompt-lengths 256 --batch-sizes $SWEEP_BS --max-tokens 128 \
     --num-warmup 3 --num-trials 8 \
     --output $RES/vllm_ep8_sweep_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC1=\$?
   echo \"SWEEP_RC=\$RC1\"
   # 2) profiled decode burst at bs=$PROFBS -> per-rank kernel traces in \$PDIR
   export VLLM_TORCH_PROFILER_DIR=\$PDIR
   python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
     --prompt-lengths 256 --batch-sizes $PROFBS --max-tokens 32 \
     --num-warmup 3 --num-trials 3 \
     --profile --profile-steps 32 \
     --output $RES/vllm_ep8_profcfg_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC2=\$?
   echo \"PROF_RC=\$RC2\"
   echo \"TRACE_DIR=\$PDIR\"; ls -la \$PDIR | head
   exit \$(( RC1 || RC2 ))
  "
RC=$?
echo "=== done rc=$RC  $(date) ==="
exit $RC
