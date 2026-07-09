#!/bin/bash
# hybrid_ep A/B via `vllm serve` (Nemotron-validated path) in the rebuilt
# deci-handoff container. Serves model A (pretrained), benches over HTTP,
# tears down, serves model B (cell), benches. TP=1, DP = 4 x nodes = EP.
# Submit with --nodes=N --segment=N; DP scales automatically.
#SBATCH --job-name=hsg_serve_hep
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hsg_serve_hep_%j.out
set -uo pipefail
: "${CELL_ID:?}"; : "${SWAP_DIR:?}"
A2A_BACKEND=${A2A_BACKEND:-hybrid_ep}
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm_deci_handoff.sqsh
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
# Rack-locality check: all nodes must be trays of ONE NVL72 rack (hostname
# scheme nvl72<rack>-T<tray>). --segment should guarantee this; abort loudly
# if the allocation ever violates it, since MNNVL/hybrid_ep needs one fabric.
RACKS=$(printf "%s\n" "${NODES[@]}" | cut -d- -f1 | sort -u)
if [ "$(echo "$RACKS" | wc -l)" -ne 1 ]; then
  echo "FATAL: allocation spans multiple NVL72 racks: $RACKS (nodes: ${NODES[*]})"
  exit 1
fi
echo "rack-locality OK: all ${#NODES[@]} nodes on rack $RACKS"
HEAD=${NODES[0]}
HEAD_IP=$(getent hosts $HEAD | awk '{print $1}' | head -1)
DP_TOTAL=$((SLURM_JOB_NUM_NODES * 4))
OUT_JSON=$RES/hsg_${CELL_ID}_serve_${A2A_BACKEND}_dp${DP_TOTAL}_$SLURM_JOB_ID.json
echo "=== serve hybrid_ep A/B $CELL_ID backend=$A2A_BACKEND DP=$DP_TOTAL nodes=$SLURM_JOB_NUM_NODES head=$HEAD_IP $(date) ==="

COMMON_FLAGS="--tensor-parallel-size 1 --data-parallel-size $DP_TOTAL --data-parallel-size-local 4 \
 --data-parallel-address $HEAD_IP --data-parallel-rpc-port 13345 \
 --enable-expert-parallel --all2all-backend $A2A_BACKEND \
 --max-model-len 10240 --gpu-memory-utilization 0.85 --trust-remote-code \
 --enable-chunked-prefill --max-num-batched-tokens 2048 ${EXTRA_FLAGS:-}"
# CG_FORCE=1: force CUDA-graph mode (JSON must survive the inner bash -c re-parse,
# so it is single-quoted here rather than passed through sbatch --export)
if [ "${CG_FORCE:-0}" = "1" ]; then
  COMMON_FLAGS="$COMMON_FLAGS --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'"
fi
COMMON_ENV="export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=1800;
export TMPDIR=/tmp/hep_\$SLURM_JOB_ID; mkdir -p \$TMPDIR;
export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x;
mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm"

bench_model () {
  local NAME=$1 MODEL=$2
  echo "=== serving $NAME ($MODEL) DP=$DP_TOTAL ==="
  local PIDS=()
  srun --nodes=1 --ntasks=1 -w $HEAD --overlap --mem=0 --gpu-bind=none \
       --container-image=$CONTAINER --container-mounts=$MNT \
    bash -c "$COMMON_ENV; export CUDA_VISIBLE_DEVICES=0,1,2,3; \
      vllm serve $MODEL --port 8000 --host 0.0.0.0 $COMMON_FLAGS" &
  PIDS+=($!)
  sleep 10
  local RANK=4
  for W in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w $W --overlap --mem=0 --gpu-bind=none \
         --container-image=$CONTAINER --container-mounts=$MNT \
      bash -c "$COMMON_ENV; export CUDA_VISIBLE_DEVICES=0,1,2,3; \
        vllm serve $MODEL --headless --data-parallel-start-rank $RANK $COMMON_FLAGS" &
    PIDS+=($!)
    RANK=$((RANK + 4))
  done
  # wait for health (up to 25 min)
  local UP=0
  for i in $(seq 1 150); do
    if curl -s -m 3 http://$HEAD_IP:8000/health >/dev/null 2>&1; then UP=1; break; fi
    kill -0 ${PIDS[0]} 2>/dev/null || break
    sleep 10
  done
  if [ $UP -ne 1 ]; then echo "SERVER_FAILED $NAME"; kill "${PIDS[@]}" 2>/dev/null; return 1; fi
  echo "=== $NAME up; benching ==="
  for REP in 1 2 3; do
    for CFG in "64 256 256 32 std" "64 128 128 256 dec"; do
      set -- $CFG
      srun --nodes=1 --ntasks=1 -w $HEAD --overlap --mem=8G \
           --container-image=$CONTAINER --container-mounts=$MNT \
        python3 $BASE/spork_configs/http_bench.py --url http://$HEAD_IP:8000 \
          --name ${NAME}_${5}_rep${REP} --out $OUT_JSON \
          --concurrency $1 --prompts $2 --plen $3 --gen $4
    done
  done
  echo "=== $NAME bench done; tearing down ==="
  kill -INT "${PIDS[@]}" 2>/dev/null; sleep 20
  kill -9 "${PIDS[@]}" 2>/dev/null || true
  sleep 30
}

bench_model pre "$PRE"
bench_model "$CELL_ID" "$SWAP_DIR"
echo "=== all done $(date) ==="
