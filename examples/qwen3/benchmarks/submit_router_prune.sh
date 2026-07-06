#!/bin/bash
#SBATCH --job-name=router_prune
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/router_prune_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/router_prune_%j.err
# Router-preserving prune inside the pai-megatron container (py3.12 + torch + megatron deps).
set -uo pipefail
N=/lustre/fsw/portfolios/nvr/users/jonathanp
P=$N/rl_token_routing/Pai-Megatron-Patch
srun --container-image=$N/containers/pai-megatron-patch_25.04.sqsh --container-mounts="$N:$N" bash -c "
  export PYTHONPATH=$P:$P/backends/megatron/Megatron-LM-250624
  GAPMIN=${GAPMIN:-1.5} DELETE=${DELETE:-0} LIMIT=${LIMIT:-10000} \
    python3 $P/examples/qwen3/benchmarks/router_extract_and_prune.py
"
