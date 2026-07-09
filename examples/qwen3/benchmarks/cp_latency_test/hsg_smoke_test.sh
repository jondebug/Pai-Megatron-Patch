#!/bin/bash
# Smoke test: run our bench against Qwen3.5-397B-A17B-FP8 to validate pipeline on GB200.
# Not our target model but same family — proves vLLM+Ray+bench pipeline works.
#SBATCH --job-name=hsg_smoke
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hsg_smoke_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-arm.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
mkdir -p $RES
MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_infbench/common/models/Qwen3.5-397B-A17B-FP8
RAY_VER="${RAY_VER:-2.48.0}"
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NN=${SLURM_NNODES}
TP=$(( NN * 4 ))
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk "{print \$1}")
PORT=6379
PIP="pip install --quiet ray==${RAY_VER} 2>/dev/null || pip install --quiet ray 2>/dev/null || true"
echo "=== SMOKE TEST: Qwen3.5-397B-A17B-FP8 on GB200 EP=$TP $(date) ==="

if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "$PIP
      for i in \$(seq 1 40); do ray start --address=$HEAD_IP:$PORT --num-gpus=4 --disable-usage-stats && break; echo 'worker waiting...'; sleep 10; done
      sleep 5400" &
fi

srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   $PIP
   export TMPDIR=/tmp/smoke_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
   mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm
   pip install --quiet msgpack 2>/dev/null || true
   ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=4 --disable-usage-stats
   export RAY_ADDRESS=$HEAD_IP:$PORT
   for i in \$(seq 1 36); do
     TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf \"%d\",\$1}');
     echo \"  ray GPUs=\${TOT:-0}/$TP\"; [ \"\${TOT:-0}\" -ge \"$TP\" ] && break; sleep 10;
   done
   ray status
   # single tiny bench to verify pipeline
   python3 $SCRIPT --models qwen3p5_397b=$MODEL --tp-size $TP \
     --distributed-executor-backend ray \
     --prompt-lengths 1024 --batch-sizes 1,4 --max-tokens 8 \
     --num-warmup 2 --num-trials 5 --cuda-graphs \
     --output $RES/hsg_smoke_\$SLURM_JOB_ID.json
   echo === smoke DONE ===
   ray stop 2>/dev/null || true
  "
EXIT=$?
echo "=== done EXIT=$EXIT $(date) ==="
exit $EXIT
