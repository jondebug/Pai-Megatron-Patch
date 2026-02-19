#!/bin/bash
#SBATCH --job-name=test_sweep
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:10:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/test_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/test_%j.err

# ============================================================
# TEST: Verify chain and multi-agent mechanism
# ============================================================
CHAIN_DEPTH="${1:-0}"
AGENTS_PER_JOB="${2:-2}"
MAX_CHAIN=2  # Only chain twice for testing

SCRIPT_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/test_sweep_chain.sh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs"

echo "============================================================"
echo "TEST SWEEP CHAIN"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Chain depth: $CHAIN_DEPTH of $MAX_CHAIN"
echo "Agents per job: $AGENTS_PER_JOB"
echo "Start: $(date)"
echo "============================================================"

mkdir -p "$LOG_DIR"

# Submit continuation job if not at max depth
if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN" ]; then
    NEXT_DEPTH=$((CHAIN_DEPTH + 1))
    NEXT_JOB=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$NEXT_DEPTH" "$AGENTS_PER_JOB" 2>&1 | awk '{print $NF}')
    echo "Submitted continuation: $NEXT_JOB (depth $NEXT_DEPTH)"
else
    echo "Max chain depth reached, no continuation."
fi

# Simulate multiple agents
echo ""
echo "Starting $AGENTS_PER_JOB simulated agents..."
PIDS=()
for i in $(seq 1 $AGENTS_PER_JOB); do
    (
        echo "[Agent $i] Started at $(date)"
        # Simulate work (30 seconds per agent)
        sleep 30
        echo "[Agent $i] Completed at $(date)"
    ) &
    PIDS+=($!)
    echo "Launched agent $i (PID: ${PIDS[-1]})"
done

echo ""
echo "Waiting for ${#PIDS[@]} agents: ${PIDS[*]}"
for pid in ${PIDS[@]}; do
    wait $pid
    echo "Agent $pid finished with code $?"
done

echo ""
echo "============================================================"
echo "TEST COMPLETE"
echo "Chain depth: $CHAIN_DEPTH"
echo "End: $(date)"
echo "============================================================"

