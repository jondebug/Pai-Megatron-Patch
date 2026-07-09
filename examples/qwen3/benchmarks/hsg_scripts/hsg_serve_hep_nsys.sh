#!/bin/bash
# nsys-traced hybrid_ep serve: ONE model per job (clean per-model traces, no
# window splitting). Wraps vllm serve on every node in nsys (-t cuda,nvtx
# -s none --cpuctxsw=none), benches 1 rep x 2 regimes, then sentinel-triggered
# SIGINT so nsys finalizes. Exports head-node sqlite + gate check in-job.
# Env: MODEL_PATH (weights dir), TAG (trace name prefix).
#SBATCH --job-name=hep_nsys
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hep_nsys_%j.out
set -uo pipefail
: "${MODEL_PATH:?}"; : "${TAG:?}"
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm_deci_handoff.sqsh
RES=$BASE/rl_token_routing/cp_latency_results
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
RACKS=$(printf "%s\n" "${NODES[@]}" | cut -d- -f1 | sort -u)
if [ "$(echo "$RACKS" | wc -l)" -ne 1 ]; then
  echo "FATAL: allocation spans multiple NVL72 racks: $RACKS"; exit 1
fi
echo "rack-locality OK: ${#NODES[@]} nodes on rack $RACKS"
HEAD=${NODES[0]}
HEAD_IP=$(getent hosts $HEAD | awk '{print $1}' | head -1)
DP_TOTAL=$((SLURM_JOB_NUM_NODES * 4))
NSYS_DIR=$BASE/nsys_hep
mkdir -p $NSYS_DIR
SENT=$NSYS_DIR/done_$SLURM_JOB_ID
rm -f $SENT
OUT_JSON=$RES/hsg_${TAG}_nsys_hybrid_ep_dp${DP_TOTAL}_$SLURM_JOB_ID.json
echo "=== nsys hybrid_ep serve $TAG DP=$DP_TOTAL $(date) ==="

COMMON_FLAGS="--tensor-parallel-size 1 --data-parallel-size $DP_TOTAL --data-parallel-size-local 4 \
 --data-parallel-address $HEAD_IP --data-parallel-rpc-port 13345 \
 --enable-expert-parallel --all2all-backend hybrid_ep \
 --max-model-len 10240 --gpu-memory-utilization 0.85 --trust-remote-code \
 --enable-chunked-prefill --max-num-batched-tokens 2048"
COMMON_ENV="export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=1800;
export PATH=\$PATH:$BASE/tools/nsys_pkg/bin:$BASE/tools/nsys_pkg/target-linux-sbsa-armv8;
export TMPDIR=/tmp/hep_\$SLURM_JOB_ID; mkdir -p \$TMPDIR;
export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x;
mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm"

PIDS=()
RANK=0
for N in "${NODES[@]}"; do
  if [ "$N" = "$HEAD" ]; then
    SERVE_CMD="vllm serve $MODEL_PATH --port 8000 --host 0.0.0.0 $COMMON_FLAGS"
  else
    SERVE_CMD="vllm serve $MODEL_PATH --headless --data-parallel-start-rank $RANK $COMMON_FLAGS"
  fi
  srun --nodes=1 --ntasks=1 -w $N --overlap --mem=0 --gpu-bind=none \
       --container-image=$CONTAINER --container-mounts=$MNT \
    bash -c "$COMMON_ENV; export CUDA_VISIBLE_DEVICES=0,1,2,3; \
      nsys profile -t cuda,nvtx -s none --cpuctxsw=none --force-overwrite=true \
        -o $NSYS_DIR/${TAG}_dp${DP_TOTAL}_${SLURM_JOB_ID}_\$(hostname) \
        $SERVE_CMD & NP=\$!; \
      while [ ! -f $SENT ]; do sleep 5; kill -0 \$NP 2>/dev/null || exit 1; done; \
      kill -INT \$NP; wait \$NP; echo \"nsys finalized on \$(hostname)\"" &
  PIDS+=($!)
  RANK=$((RANK + 4))
  sleep 5
done

UP=0
for i in $(seq 1 180); do
  if curl -s -m 3 http://$HEAD_IP:8000/health >/dev/null 2>&1; then UP=1; break; fi
  kill -0 ${PIDS[0]} 2>/dev/null || break
  sleep 10
done
if [ $UP -ne 1 ]; then echo "SERVER_FAILED $TAG"; touch $SENT; sleep 60; kill "${PIDS[@]}" 2>/dev/null; exit 1; fi
echo "=== $TAG up; benching under trace ==="
for CFG in "64 256 256 32 std" "64 128 128 256 dec"; do
  set -- $CFG
  srun --nodes=1 --ntasks=1 -w $HEAD --overlap --mem=8G \
       --container-image=$CONTAINER --container-mounts=$MNT \
    python3 $BASE/spork_configs/http_bench.py --url http://$HEAD_IP:8000 \
      --name ${TAG}_${5}_nsys --out $OUT_JSON \
      --concurrency $1 --prompts $2 --plen $3 --gen $4
done
echo "=== bench done; triggering nsys finalize $(date) ==="
touch $SENT
for P in "${PIDS[@]}"; do wait $P; done
echo "=== all nsys finalized $(date) ==="

REP=$NSYS_DIR/${TAG}_dp${DP_TOTAL}_${SLURM_JOB_ID}_${HEAD}.nsys-rep
srun --nodes=1 --ntasks=1 -w $HEAD --overlap --mem=32G \
     --container-image=$CONTAINER --container-mounts=$MNT \
  bash -c "export PATH=\$PATH:$BASE/tools/nsys_pkg/bin:$BASE/tools/nsys_pkg/target-linux-sbsa-armv8; \
    nsys export --type=sqlite --force-overwrite=true -o ${REP%.nsys-rep}.sqlite $REP && \
    python3 -c \"
import sqlite3
c = sqlite3.connect('${REP%.nsys-rep}.sqlite')
k = c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
try:
    n = c.execute(\\\"select count(*) from NVTX_EVENTS where text like '%nccl%'\\\").fetchone()[0]
except Exception as e:
    n = f'ERR {e}'
print(f'GATE: kernels={k} nccl_nvtx={n}')
print('SPORK_GATE=' + ('PASS' if k > 100000 else 'CHECK'))
\""
ls -sh $NSYS_DIR/${TAG}_dp${DP_TOTAL}_${SLURM_JOB_ID}_*.nsys-rep
echo "=== done $(date) ==="
