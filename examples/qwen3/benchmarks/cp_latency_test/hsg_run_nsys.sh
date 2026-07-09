#!/bin/bash
# nsys-wrapped bench for HSG. Container v2 has nsys at /opt/nvidia/nsight-systems-cli/2026.3.0/
# Wraps rank-0 driver in nsys profile. Workers run untraced.
#SBATCH --job-name=hsg_nsys
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/hsg_nsys_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-arm.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
mkdir -p $RES
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
RAY_VER="${RAY_VER:-2.48.0}"
MNT="$BASE:$BASE,/lustre/fs1:/lustre/fs1"
NN=${SLURM_NNODES}
TP=$(( NN * 4 ))
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk "{print \$1}")
PORT=6379
PIP="pip install --quiet ray==${RAY_VER} 2>/dev/null || pip install --quiet ray 2>/dev/null || true"
CFG_NAME="${CFG_NAME:-nsys_ep${TP}}"
BS="${BS:-1}"
PLEN="${PLEN:-8192}"
MAXTOK="${MAXTOK:-8}"
CELL="${CELL:-pre}"   # pre or r15
MODEL=$PRE
[ "$CELL" = "r15" ] && MODEL=$R15

echo "=== nsys trace EP=$TP cell=$CELL bs=$BS plen=$PLEN max_tok=$MAXTOK $(date) ==="

if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "$PIP
      for i in \$(seq 1 40); do ray start --address=$HEAD_IP:$PORT --num-gpus=4 --disable-usage-stats && break; sleep 10; done
      sleep 7200" &
fi

srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   $PIP
   export TMPDIR=/tmp/nsys_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
   mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm
   ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=4 --disable-usage-stats
   export RAY_ADDRESS=$HEAD_IP:$PORT
   for i in \$(seq 1 36); do
     TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf \"%d\",\$1}');
     [ \"\${TOT:-0}\" -ge \"$TP\" ] && break; sleep 10;
   done
   ray status
   NSYS=/opt/nvidia/nsight-systems-cli/2026.3.0/bin/nsys
   [ -x \"\$NSYS\" ] || NSYS=\$(ls /opt/nvidia/nsight-systems-cli/*/bin/nsys 2>/dev/null | head -1)
   echo \"nsys=\$NSYS\"
   TAG=hsg_nsys_${CFG_NAME}_${CELL}_bs${BS}_\$SLURM_JOB_ID
   \$NSYS profile -t cuda,nvtx --output=$RES/\$TAG -f true --trace-fork-before-exec=true \
     python3 $SCRIPT --models ${CELL}_ep${TP}=$MODEL --tp-size $TP \
       --distributed-executor-backend ray \
       --prompt-lengths $PLEN --batch-sizes $BS --max-tokens $MAXTOK \
       --num-warmup 1 --num-trials 5 --cuda-graphs \
       --output $RES/\$TAG.json
   ls -la $RES/\$TAG.nsys-rep $RES/\$TAG.sqlite 2>/dev/null
   ray stop 2>/dev/null || true
  "
EXIT=$?
echo "=== done EXIT=$EXIT $(date) ==="
exit $EXIT
