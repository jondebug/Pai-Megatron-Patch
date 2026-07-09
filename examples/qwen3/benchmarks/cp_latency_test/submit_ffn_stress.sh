#!/bin/bash
#SBATCH --job-name=cp_ffn_stress
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=230000M
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ffn_stress_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/cp_ffn_stress_%j.err
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
RES=$BASE/rl_token_routing/cp_latency_results
SCRIPT=$DIR/cp_ffn_stress.py
echo "=== FFN STRESS-TEST  node=$(hostname)  $(date) ==="
srun --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=230000M \
     --container-image="$CONTAINER" --container-mounts="$BASE:$BASE" \
     --container-workdir="$DIR" \
  bash -c "
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    python3 $SCRIPT --repeats 100 --warmup 30 --trials 5 \
      --n-experts 16 --light 4 --wscale-b 8 \
      --out $RES/ffn_stress_\${SLURM_JOB_ID}.json
  "
RC=$?
echo "=== done rc=$RC $(date) ==="
exit $RC
