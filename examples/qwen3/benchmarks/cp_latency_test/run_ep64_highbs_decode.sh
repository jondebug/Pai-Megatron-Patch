#!/bin/bash
#SBATCH --job-name=ep64_highbs_decode
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ep64_highbs_decode_%j.out
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
echo "=== EP=$TP HIGH-BS DECODE — test CP->TPOT prediction at M>>knee ==="

if [ "$NN" -gt 1 ]; then
  NW=$(( NN - 1 ))
  WORKERS=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2 | paste -sd,)
  srun --nodes=$NW --ntasks=$NW --ntasks-per-node=1 -w "$WORKERS" --overlap --mem=0 \
       --container-image="$CONTAINER" --container-mounts="$MNT" \
    bash -c "$PIP
      for i in \$(seq 1 40); do ray start --address=$HEAD_IP:$PORT --num-gpus=8 --disable-usage-stats --include-dashboard=false && break; echo 'worker waiting...'; sleep 10; done
      sleep 10800" &
fi

srun --nodes=1 --ntasks=1 -w "$HEAD" --overlap --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" --container-workdir="$DIR" \
  bash -c "
   $PIP
   export TMPDIR=/tmp/ehbd_\$SLURM_JOB_ID VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
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

   # bs=16384 needs max_num_seqs override + KV-cache headroom. Use --enable-chunked-prefill off.
   # plen=256, max_tokens=16 to keep walltime bounded; 20 trials.
   for CFG in pretrained_235b=$PRE r15_cp4682=$R15; do
     CNAME=\$(echo \$CFG | cut -d= -f1); CPATH=\$(echo \$CFG | cut -d= -f2)
     for BS in 2048 8192 16384; do
       echo === \$CNAME bs=\$BS ===
       python3 $SCRIPT --models \${CNAME}_ep${TP}=\$CPATH --tp-size $TP \
         --distributed-executor-backend ray \
         --max-num-seqs \$BS --max-num-batched-tokens \$BS --max-model-len 384 \
         --prompt-lengths 256 --batch-sizes \$BS --max-tokens 16 \
         --num-warmup 1 --num-trials 20 \
         --output $RES/ep${TP}_highbs_\${CNAME}_bs\${BS}_\$SLURM_JOB_ID.json || echo \"RC=\$?\"
     done
   done
   ray stop 2>/dev/null || true
  "
EXIT=$?
echo "=== done EXIT=$EXIT $(date) ==="
exit $EXIT
