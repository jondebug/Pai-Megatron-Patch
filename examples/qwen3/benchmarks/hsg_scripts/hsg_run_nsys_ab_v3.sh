#!/bin/bash
# nsys v3: SPORK-compatible multi-node Ray+vLLM tracing.
#
# Fix vs v2: `ray start` daemonizes, so wrapping it with nsys traced only the
# short-lived CLI (12MB head-only reps, no worker kernels). v3 runs
# `ray start --block` in the foreground under nsys on every worker node, so the
# raylet and all vLLM worker processes it spawns stay inside nsys's process
# tree. The head signals completion via a sentinel file on lustre; workers then
# `ray stop` locally, which makes --block return and nsys finalize its report.
#
# Sanity gate: after finalize, worker 0 exports its rep to sqlite and checks
# NVTX_EVENTS for 'nccl' rows + CUPTI kernel rows, printing SPORK_GATE=PASS/FAIL.
#
# Env vars: CELL_ID, SWAP_DIR, TARGET_EP, optional BENCH_MODE=prefill|decode
#SBATCH --job-name=hsg_nsys_v3
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hsg_nsys_v3_%j.out
set -uo pipefail

: "${CELL_ID:?Set CELL_ID env var}"
: "${SWAP_DIR:?Set SWAP_DIR env var}"
: "${TARGET_EP:?Set TARGET_EP env var}"
BENCH_MODE=${BENCH_MODE:-prefill}

BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-arm.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
NSYS_DIR=$RES/nsys_v3_${CELL_ID}_${BENCH_MODE}_ep${TARGET_EP}_$SLURM_JOB_ID
mkdir -p $NSYS_DIR
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NN=${SLURM_NNODES}
TP=$TARGET_EP
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
NW=$(( NN - 1 ))
WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
HEAD_IP_FILE=$NSYS_DIR/head_ip.txt
SENTINEL=$NSYS_DIR/BENCH_DONE
rm -f $HEAD_IP_FILE $SENTINEL

case $BENCH_MODE in
  prefill) PLENS=8192; BSS="1,16";  MT=4;  NTRIALS=3 ;;
  decode)  PLENS=256;  BSS="64,512"; MT=32; NTRIALS=3 ;;
  *) echo "bad BENCH_MODE=$BENCH_MODE"; exit 1 ;;
esac

echo "=== nsys v3 A/B $CELL_ID mode=$BENCH_MODE EP=$TP nodes=$NN $(date) ==="

# ---------------- HEAD ----------------
srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -s <<HEADEOF &
set +e
HEAD_IP=\$(hostname --ip-address)
echo "\$HEAD_IP" > $HEAD_IP_FILE
pip install --quiet ray==2.48.0 2>/dev/null || true
export TMPDIR=/tmp/nsysv3_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
export RAY_TMPDIR=/tmp/ray_h_\$SLURM_JOB_ID RAY_DISABLE_IMPORT_WARNING=1
mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm \$RAY_TMPDIR
ray start --head --node-ip-address=\$HEAD_IP --port=6379 --dashboard-port=8265 \\
  --min-worker-port=10002 --max-worker-port=19999 \\
  --metrics-export-port=22222 --dashboard-agent-grpc-port=22223 --runtime-env-agent-port=22224 --dashboard-agent-listen-port=52365 \\
  --num-gpus=4 --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS=\$HEAD_IP:6379
for i in \$(seq 1 60); do
  TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf "%d",\$1}')
  [ "\${TOT:-0}" -ge "$TP" ] && break
  sleep 10
done
if [ "\${TOT:-0}" -lt "$TP" ]; then
  echo "ERROR \${TOT:-0}/$TP GPUs"; touch $SENTINEL; ray stop; exit 1
fi
echo "=== BENCH $BENCH_MODE A/B (head also nsys-traced) ==="
nsys profile -t cuda,nvtx -s none --cpuctxsw=none --output=$NSYS_DIR/head_\${SLURM_JOB_ID} --force-overwrite=true \\
  python3 $SCRIPT --models pre_ep${TP}=$PRE,${CELL_ID}_ep${TP}=$SWAP_DIR --tp-size $TP \\
  --distributed-executor-backend ray \\
  --prompt-lengths $PLENS --batch-sizes $BSS --max-tokens $MT \\
  --num-warmup 1 --num-trials $NTRIALS --cuda-graphs \\
  --output $RES/hsg_${CELL_ID}_nsysv3_${BENCH_MODE}_ep${TP}_\$SLURM_JOB_ID.json
echo "=== bench done, signaling workers ==="
touch $SENTINEL
sleep 90   # give worker nsys instances time to finalize before job teardown
ray stop 2>&1 | tail -1
HEADEOF
HPID=$!
sleep 15

# ---------------- WORKERS ----------------
if [ $NW -gt 0 ]; then
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -s <<WORKEREOF &
set +e
MY_HOST=\$(hostname); MY_IP=\$(hostname --ip-address)
for i in \$(seq 1 60); do [ -s $HEAD_IP_FILE ] && break; sleep 3; done
HEAD_IP=\$(cat $HEAD_IP_FILE)
python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(10)
try: s.connect(('\$HEAD_IP', 6379)); s.close()
except Exception as e: print('TCP fail:', e); sys.exit(1)
" || exit 1
pip install --quiet ray==2.48.0 2>/dev/null || true
export RAY_TMPDIR=/tmp/ray_w_\${SLURM_JOB_ID}_\$MY_HOST RAY_DISABLE_IMPORT_WARNING=1
mkdir -p \$RAY_TMPDIR

# v3 core: ray start --block stays in the foreground under nsys, so the raylet
# and every vLLM worker it spawns are in nsys's traced process tree.
nsys profile -t cuda,nvtx -s none --cpuctxsw=none \\
  --output=$NSYS_DIR/worker_\${MY_HOST} --force-overwrite=true \\
  ray start --address=\$HEAD_IP:6379 --node-ip-address=\$MY_IP --block \\
    --min-worker-port=10002 --max-worker-port=19999 \\
    --metrics-export-port=22222 --dashboard-agent-grpc-port=22223 --runtime-env-agent-port=22224 --dashboard-agent-listen-port=52365 \\
    --num-gpus=4 --disable-usage-stats &
NSYS_PID=\$!

# Wait for head to finish the bench.
for i in \$(seq 1 1000); do
  [ -f $SENTINEL ] && break
  kill -0 \$NSYS_PID 2>/dev/null || break
  sleep 5
done
ray stop 2>&1 | tail -1
sleep 5
# 4060919 lesson: ray start --block does NOT exit after ray stop, so nsys never
# finalizes on its own. SIGINT nsys directly — it stops collection, tears down
# the child, and writes the report. 4063844 lesson: report conversion was killed
# at 75% by a 10-min TERM escalation; -s none shrinks the trace and the window
# is now 25 min. TERM only as last resort.
kill -INT \$NSYS_PID 2>/dev/null
for i in \$(seq 1 150); do
  kill -0 \$NSYS_PID 2>/dev/null || break
  sleep 10
done
kill -TERM \$NSYS_PID 2>/dev/null
wait \$NSYS_PID
echo "worker \$MY_HOST nsys finalized: \$(ls -la $NSYS_DIR/worker_\${MY_HOST}.nsys-rep 2>/dev/null)"

# Sanity gate: first worker (alphabetically first host on the node list) exports
# its rep and verifies SPORK-required tables are populated.
FIRST_WORKER=\$(echo "$WORKERS" | cut -d, -f1)
if [ "\$MY_HOST" = "\$FIRST_WORKER" ] && [ -f $NSYS_DIR/worker_\${MY_HOST}.nsys-rep ]; then
  nsys export --type=sqlite --force-overwrite true $NSYS_DIR/worker_\${MY_HOST}.nsys-rep \\
    --output $NSYS_DIR/worker_\${MY_HOST}.sqlite 2>&1 | tail -2
  python3 - <<'PYGATE'
import sqlite3, glob, sys
f = glob.glob("$NSYS_DIR/worker_*.sqlite")[0]
conn = sqlite3.connect(f)
def count(q):
    try: return conn.execute(q).fetchone()[0]
    except Exception: return -1
kern = count("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
nccl = count("SELECT COUNT(*) FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id WHERE s.value LIKE '%nccl%'")
print(f"SPORK_GATE kernels={kern} nccl_nvtx={nccl}")
print("SPORK_GATE=PASS" if (kern > 1000 and nccl > 0) else "SPORK_GATE=FAIL")
PYGATE
fi
WORKEREOF
  WPID=$!
fi

wait $HPID
EXIT=$?
# Workers finalize their nsys reports after the sentinel; wait for the worker
# step or SLURM kills nsys mid-write (root cause of the 4059123 empty run).
if [ $NW -gt 0 ]; then
  echo "=== waiting for worker nsys finalization ==="
  wait $WPID
fi
echo "=== v3 done EXIT=$EXIT $(date) ==="
ls -la $NSYS_DIR/ | head -20
exit $EXIT
