#!/bin/bash
#SBATCH --job-name=cp_ep8_prefprof
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_prefprof_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ep8_prefprof_%x_%j.err
# Phase B STEP 2 (prefill): profile a PREFILL forward at long prompts (CP's regime).
# For ONE model: sweep prefill plen {2048,4096,8192} bs1 (TTFT walltime, n=8) + a PROFILED
# prefill (max_tokens=1, plen=8192) -> per-rank kernel trace -> isolate expert_ffn prefill compute.
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
MODEL="${MODEL:?set MODEL}"; MODELNAME="${MODELNAME:?set MODELNAME}"
PROFPLEN="${PROFPLEN:-8192}"
MNT="$BASE:$BASE"
echo "=== Phase B EP=8 PREFILL profile model=$MODELNAME node=$(hostname) $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   export TMPDIR=/tmp/ep8pp_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   mkdir -p \$TMPDIR
   PDIR=$RES/trace_pref_${MODELNAME}_ep8_\${SLURM_JOB_ID}; mkdir -p \$PDIR
   # 1) prefill TTFT walltime sweep (max_tokens=1 -> e2e ~= prefill+1; n=8)
   python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
     --prompt-lengths 2048,4096,8192 --batch-sizes 1 --max-tokens 1 \
     --num-warmup 3 --num-trials 8 \
     --output $RES/vllm_ep8_prefill_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC1=\$?; echo \"PREFILL_RC=\$RC1\"
   # 2) profiled prefill (one long prompt, max_tokens=1) -> kernel trace
   export VLLM_TORCH_PROFILER_DIR=\$PDIR
   python3 $SCRIPT --models ${MODELNAME}_ep8=$MODEL --tp-size 8 \
     --prompt-lengths $PROFPLEN --batch-sizes 1 --max-tokens 1 \
     --num-warmup 2 --num-trials 2 --profile --profile-steps 1 \
     --output $RES/vllm_ep8_prefprofcfg_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC2=\$?; echo \"PROF_RC=\$RC2\"; echo \"TRACE_DIR=\$PDIR\"; ls -la \$PDIR|head
   exit \$(( RC1 || RC2 ))
  "
echo "=== done rc=$? $(date) ==="
