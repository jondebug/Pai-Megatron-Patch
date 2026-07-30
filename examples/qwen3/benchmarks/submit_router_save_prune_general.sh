#!/bin/bash
#SBATCH --job-name=rsprune_gen
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/rsprune_gen_%j.out
# Self-chaining router-SAVE prune of old general checkpoints (container CPU job).
# Queues a singleton successor FIRST (so it survives the 4h wall), runs one pass, and
# cancels the pending successor once a pass reports 0 remaining targets => the chain stops itself.
set -uo pipefail
N=/lustre/fsw/portfolios/nvr/users/jonathanp
P=$N/rl_token_routing/Pai-Megatron-Patch
BM=$P/examples/qwen3/benchmarks
SELF=$BM/submit_router_save_prune_general.sh
if [ "${DELETE:-0}" = "1" ] && [ "${NOCHAIN:-0}" != "1" ]; then
  DELETE=1 sbatch --parsable "$SELF" >/dev/null 2>&1 || true      # successor waits via singleton
fi
LOG=/tmp/rsp_$$.log
srun --container-image=$N/containers/pai-megatron-patch_25.04.sqsh --container-mounts="$N:$N" bash -c "
  export PYTHONPATH=$P:$P/backends/megatron/Megatron-LM-250624
  DELETE=${DELETE:-0} LIMIT=${LIMIT:-1000000} SHOWKEEP=${SHOWKEEP:-0} \
    python3 $BM/router_save_prune_general.py
" 2>&1 | tee "$LOG"
if grep -q "=== 0 on-disk checkpoints TARGETED" "$LOG"; then
  scancel -u jonathanp -n rsprune_gen -t PENDING 2>/dev/null || true
  echo "PRUNE COMPLETE — 0 targets remain; pending successors cancelled"
fi
rm -f "$LOG" 2>/dev/null || true
