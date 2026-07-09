#!/bin/bash
#SBATCH --job-name=ep64_ultrabs_decode
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ep64_ultrabs_decode_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
SCRIPT=$DIR/cp_vllm_bench.py
RES=$BASE/rl_token_routing/cp_latency_results
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
R15=$BASE/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp
RAY_VER="${RAY_VER:-2.48.0}"
MNT="$BASE:$BASE"

NN=${SLURM_NNODES}
TP=$(( NN * 8 ))
HEAD=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk "{print \$1}")
PORT=6379
PIP="pip install --quiet ray==${RAY_VER} 2>/dev/null || pip install --quiet ray 2>/dev/null || true"
echo "=== EP=$TP ULTRA-HIGH-BS DECODE — push M past GEMM knee ==="
# bs targets:
#   16384 -> per-local-expert M = 1024 (at knee, partial benefit)
#   32768 -> M = 2048 (above knee, r15 should start winning)
#   65536 -> M = 4096 (deeply saturated, max r15 benefit if regime is FLOPs-bound)
# Memory budget at EP=64: per-rank ≈ model 7GB + KV @ bs=N × 23 GB / 16384 + activations 10GB.
#   bs=32768 -> ~ 7 + 46 + 10 = 63 GB ✓
#   bs=65536 -> ~ 7 + 92 + 10 = 109 GB ✗ likely OOM; we'll try it last

if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "$PIP
      for i in \$(seq 1 40); do ray start --address=$HEAD_IP:$PORT --num-gpus=8 --disable-usage-stats --include-dashboard=false && break; echo 'worker waiting...'; sleep 10; done
      sleep 14400" &
fi

srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   $PIP
   export TMPDIR=/tmp/eubd_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
   export VLLM_USE_RAY_COMPILED_DAG=0 VLLM_USE_RAY_SPMD_WORKER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
   export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/ti VLLM_CACHE_ROOT=\$TMPDIR/v TRITON_CACHE_DIR=\$TMPDIR/tr XDG_CONFIG_HOME=\$TMPDIR/x
   mkdir -p \$TMPDIR/ti \$TMPDIR/v \$TMPDIR/tr \$TMPDIR/x/vllm
   ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=8 --disable-usage-stats --include-dashboard=false
   export RAY_ADDRESS=$HEAD_IP:$PORT
   for i in \$(seq 1 36); do
     TOT=\$(ray status 2>/dev/null | awk -F/ '/GPU/{print \$2}' | awk '{printf \"%d\",\$1}');
     echo \"  ray GPUs=\${TOT:-0}/$TP\"; [ \"\${TOT:-0}\" -ge \"$TP\" ] && break; sleep 10;
   done
   ray status

   # max_tokens=8 keeps per-trial walltime bounded at huge bs.
   for CFG in pretrained_235b=$PRE r15_cp4682=$R15; do
     CNAME=\$(echo \$CFG | cut -d= -f1); CPATH=\$(echo \$CFG | cut -d= -f2)
     for BS in 32768 65536; do
       echo === \$CNAME bs=\$BS ===
       python3 $SCRIPT --models \${CNAME}_ep${TP}=\$CPATH --tp-size $TP \
         --distributed-executor-backend ray \
         --max-num-seqs \$BS --max-num-batched-tokens \$BS --max-model-len 280 \
         --prompt-lengths 256 --batch-sizes \$BS --max-tokens 8 \
         --num-warmup 1 --num-trials 10 \
         --output $RES/ep${TP}_ultrabs_\${CNAME}_bs\${BS}_\$SLURM_JOB_ID.json || echo \"RC=\$?\"
     done
   done
   ray stop 2>/dev/null || true
  "
EXIT=$?
echo "=== done EXIT=$EXIT $(date) ==="
exit $EXIT
