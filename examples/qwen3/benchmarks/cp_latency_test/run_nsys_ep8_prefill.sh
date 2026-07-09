#!/bin/bash
#SBATCH --job-name=nsys_ep8_pre
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=interactive,polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/nsys_ep8_pre_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/nsys_ep8_pre_%j.err
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
MODEL="${MODEL:?set MODEL}"
MODELNAME="${MODELNAME:?set MODELNAME}"
NSYS=$BASE/tools/nsys/NsightSystems-cli-2025.2.1/target-linux-x64/nsys
echo "=== nsys EP8 prefill model=$MODELNAME $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" --container-workdir="$DIR" \
  bash -c "
   export TMPDIR=/tmp/nsysep8_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
   mkdir -p \$TMPDIR
   OUT=$RES/nsys_ep8_prefill_${MODELNAME}_\${SLURM_JOB_ID}
   # nsys-traced prefill (graphs ON, max_tokens=1, bs=1, 2 trials with 1 warmup)
   $NSYS profile -t cuda,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node --sample=none \
     --force-overwrite=true -o \$OUT \
     python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
       --prompt-lengths 8192 --batch-sizes 1 --max-tokens 1 \
       --num-warmup 1 --num-trials 2 --cuda-graphs \
       --output $RES/nsys_ep8_prefill_${MODELNAME}_\${SLURM_JOB_ID}.json 2>&1 | tail -40
   echo RC=\$?
   ls -la \$OUT.nsys-rep
  "
echo "=== done rc=$? $(date) ==="
