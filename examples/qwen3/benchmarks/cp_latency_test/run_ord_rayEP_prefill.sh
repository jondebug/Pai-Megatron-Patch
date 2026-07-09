#!/bin/bash
#SBATCH --job-name=ord_rayEP_prefill
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ord_rayEP_prefill_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/ord_rayEP_prefill_%x_%j.err

# CLEAN EXPERIMENT (reviewer-requested): EP=NODES*8 multi-node Ray vLLM, COMPUTE-BOUND
# PREFILL per-rank FFN trace, CUDA GRAPHS ON (enforce_eager=False), to test whether the
# busiest-GPU prefill expert-FFN drops ~38% (pretrained->r15) above the FFN knee, and
# whether the per-step straggler/TTFT tracks that drop.
#   - prompt_lengths 8192 (>>knee), batch sweep 1,2 -> 8192/16384 prefill tok (busiest ~3789/~7600 pre)
#   - --cuda-graphs : strips per-layer decode launch jitter so the straggler test is clean
#   - --prefill-profile : profiles a SINGLE max_tokens=1 generation = pure prefill forward
#   - high num-trials for tight TTFT CI to beat the ~1900ms EP64 variance
# Env: MODEL, MODELNAME, set --nodes=N. Reuses cp_vllm_bench.py + parse_decode_trace.py (per-rank).
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
MODEL="${MODEL:?set MODEL}"
MODELNAME="${MODELNAME:?set MODELNAME}"
RAY_VER="${RAY_VER:-2.48.0}"
PLEN="${PLEN:-8192}"
SWEEP_BS="${SWEEP_BS:-1,2}"
PROFBS="${PROFBS:-1}"
NTRIALS="${NTRIALS:-20}"
MNT="$BASE:$BASE"

NN=${SLURM_NNODES}
TP=$(( NN * 8 ))
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk "{print \$1}")
PORT=6379
RAYPORTS="--include-dashboard=false"
WORKERPORTS=""
PIP="pip install --quiet ray==${RAY_VER} 2>/dev/null || pip install --quiet ray 2>/dev/null || true"
echo "=== Ray vLLM PREFILL nodes=$NN TP/EP=$TP model=$MODELNAME plen=$PLEN graphs=ON head=$HEAD $(date) ==="

# Workers
if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "$PIP
      for i in \$(seq 1 40); do ray start --address=$HEAD_IP:$PORT --num-gpus=8 --disable-usage-stats $WORKERPORTS && break; echo 'worker waiting...'; sleep 10; done
      sleep 7200" &
fi

# Head + driver
srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   $PIP
   export TMPDIR=/tmp/rayEPpre_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0
   export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
   mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm
   ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=8 --disable-usage-stats $RAYPORTS
   export RAY_ADDRESS=$HEAD_IP:$PORT
   echo 'waiting for ray cluster to reach $TP GPUs...'
   formed=0; for i in \$(seq 1 36); do
     TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf \"%d\",\$1}');
     echo \"  ray GPUs=\${TOT:-0}/$TP (poll \$i)\"; [ \"\${TOT:-0}\" -ge \"$TP\" ] && { formed=1; break; }; sleep 10;
   done
   [ \"\$formed\" != 1 ] && { echo \"CLUSTER FAILED TO FORM (\${TOT:-0}/$TP)\"; ray stop 2>/dev/null; exit 2; }
   ray status 2>&1 | head -25

   PDIR=$RES/trace_prefill_${MODELNAME}_ep${TP}_\${SLURM_JOB_ID}; mkdir -p \$PDIR
   # 1) Unprofiled prefill TTFT sweep with CUDA graphs ON, high trials for tight CI.
   timeout 1500 python3 $SCRIPT --models ${MODELNAME}_ep${TP}=$MODEL --tp-size $TP --distributed-executor-backend ray \
     --prompt-lengths $PLEN --batch-sizes $SWEEP_BS --max-tokens 8 --num-warmup 3 --num-trials $NTRIALS \
     --cuda-graphs \
     --output $RES/vllm_ep${TP}_prefill_sweep_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC1=\$?; echo SWEEP_RC=\$RC1
   # 2) Profiled PURE-PREFILL forward (graphs on); per-rank trace -> busiest-GPU FFN.
   export VLLM_TORCH_PROFILER_DIR=\$PDIR
   timeout 1500 python3 $SCRIPT --models ${MODELNAME}_ep${TP}=$MODEL --tp-size $TP --distributed-executor-backend ray \
     --prompt-lengths $PLEN --batch-sizes $PROFBS --max-tokens 1 --num-warmup 4 --num-trials 4 \
     --cuda-graphs --profile --prefill-profile \
     --output $RES/vllm_ep${TP}_prefill_profcfg_${MODELNAME}_\${SLURM_JOB_ID}.json
   RC2=\$?; echo PROF_RC=\$RC2; echo TRACE_DIR=\$PDIR; ls -la \$PDIR | head
   ray stop 2>/dev/null || true
   exit \$(( RC1 || RC2 ))
  "
EXIT=$?
echo "=== done Exit=$EXIT $(date) ==="
exit $EXIT
