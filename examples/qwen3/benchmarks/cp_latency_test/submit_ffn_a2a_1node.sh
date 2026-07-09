#!/bin/bash
#SBATCH --job-name=cp_ffn_a2a_nvl
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ffn_a2a_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ffn_a2a_%x_%j.err

# Phase A: 235B expert-FFN compute curve (rank0) + intra-node NVLink all-to-all (8 GPUs).
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
SCRIPT=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test/cp_ffn_a2a_bench.py
OUT=$BASE/rl_token_routing/cp_latency_results/ffn_a2a_nvlink_ep8_${SLURM_JOB_ID}.json
MNT="$BASE:$BASE"

echo "=== Phase A FFN+NVLink a2a  node=$(hostname)  $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" \
     --container-workdir="$(dirname $SCRIPT)" \
  bash -c "
    export TMPDIR=/tmp/ffna2a_\$SLURM_JOB_ID
    mkdir -p \$TMPDIR
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29577
    torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29577 \
      $SCRIPT --output $OUT --ffn-repeats 80 --ffn-warmup 20 --a2a-repeats 50 --a2a-warmup 15
  "
RC=$?
echo "=== done rc=$RC  $(date) ==="
exit $RC
