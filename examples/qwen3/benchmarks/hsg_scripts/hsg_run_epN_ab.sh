#!/bin/bash
# Configurable EP={16,32,64} A/B: pretrained vs r15_router_swap decode sweep matching ORD Phase C.
# FIXED port strategy: keep default GCS port 6379 (firewall-allowed), only move worker ports
# out of the default metrics_export range.
#SBATCH --job-name=hsg_epN_ab
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --segment=4
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hsg_epN_ab_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-arm.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
mkdir -p $RES
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15RS=$BASE/rl_token_routing/r15_router_swap
RAY_VER="${RAY_VER:-2.48.0}"
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NN=${SLURM_NNODES}
TP=$(( NN * 4 ))
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')
# Firewall-friendly: keep GCS at default 6379
PORT=6379
# Move worker port range OUT of metrics_export=14898 default
WORKER_MIN=40000
WORKER_MAX=49999
RAY_START_COMMON="--min-worker-port=$WORKER_MIN --max-worker-port=$WORKER_MAX --num-gpus=4 --disable-usage-stats"

echo "=== HSG EP=$TP A/B pre vs r15rs decode $(date), nodes=$NN ==="
echo "Ports: GCS=$PORT worker=[$WORKER_MIN,$WORKER_MAX]"
[ -d "$R15RS" ] && [ "$(ls $R15RS/*.safetensors | wc -l)" -ge 100 ] || { echo "ERROR r15rs missing"; exit 1; }

if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "pip install --quiet ray==${RAY_VER} 2>/dev/null || true
      for i in \$(seq 1 60); do ray start --address=$HEAD_IP:$PORT $RAY_START_COMMON && break; sleep 10; done
      sleep 10800" &
fi

srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   pip install --quiet ray==${RAY_VER} 2>/dev/null || true
   export TMPDIR=/tmp/epNab_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
   export RAY_scheduler_pending_placement_group_creation_timeout_s=600
   mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm
   ray start --head --node-ip-address=$HEAD_IP --port=$PORT $RAY_START_COMMON
   export RAY_ADDRESS=$HEAD_IP:$PORT
   for i in \$(seq 1 60); do
     TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf \"%d\",\$1}')
     [ \"\${TOT:-0}\" -ge \"$TP\" ] && break
     sleep 10
   done
   ray status
   echo '=== EP=$TP A/B decode plen=256 bs={512,2048,8192} matching ORD Phase C ==='
   python3 $SCRIPT --models pre_ep${TP}=$PRE,r15rs_ep${TP}=$R15RS --tp-size $TP \\
     --distributed-executor-backend ray \\
     --prompt-lengths 256 --batch-sizes 512,2048,8192 --max-tokens 16 \\
     --num-warmup 2 --num-trials 15 --cuda-graphs \\
     --output $RES/hsg_ep${TP}ab_decode_\$SLURM_JOB_ID.json
   ray stop 2>/dev/null || true
  "
EXIT=$?
echo "=== done EXIT=$EXIT $(date) ==="
exit $EXIT
