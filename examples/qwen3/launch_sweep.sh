#!/bin/bash
# ============================================================
# Sweep Launcher with Auto-Chaining Support
# ============================================================
# Usage:
#   ./launch_sweep.sh <sweep_id>                           # 3 chains, 2 agents each
#   ./launch_sweep.sh <sweep_id> --agents 4                # 3 chains, 4 agents each
#   ./launch_sweep.sh <sweep_id> --parallel 2 --agents 2   # 2 chains, 2 agents each
#
# Examples:
#   ./launch_sweep.sh h2ax29qa                    # 3 chains × 2 agents = 6 concurrent runs
#   ./launch_sweep.sh h2ax29qa --agents 3         # 3 chains × 3 agents = 9 concurrent runs
#   ./launch_sweep.sh h2ax29qa --single --agents 2  # 1 chain × 2 agents = 2 concurrent runs
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_ENTITY="nvr-israel"
WANDB_PROJECT="qwen3-router-training"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

show_help() {
    echo "Usage: $0 <sweep_id> [options]"
    echo ""
    echo "Options:"
    echo "  --parallel N     Submit N parallel chains/allocations (default: 3)"
    echo "  --agents N       Run N wandb agents per allocation (default: 2)"
    echo "  --single         Submit single chain only (same as --parallel 1)"
    echo "  --help           Show this help"
    echo ""
    echo "Note: Each chain runs N agents per 4-hour allocation, then auto-continues."
    echo ""
    echo "Examples:"
    echo "  $0 h2ax29qa                       # 3 chains x 2 agents = 6 concurrent runs"
    echo "  $0 h2ax29qa --agents 3            # 3 chains × 3 agents = 9 concurrent runs"
    echo "  $0 h2ax29qa --parallel 1 --agents 3  # 1 chain × 3 agents = 3 concurrent runs"
}

# Parse arguments
SWEEP_ID=""
PARALLEL=3              # Default to 3 chains
AGENTS_PER_JOB=2        # Default to 2 agents per job

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            shift
            PARALLEL="$1"
            shift
            ;;
        --agents)
            shift
            AGENTS_PER_JOB="$1"
            shift
            ;;
        --single)
            PARALLEL=1
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            SWEEP_ID="$1"
            shift
            ;;
    esac
done

if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: sweep_id required"
    show_help
    exit 1
fi

TOTAL_CONCURRENT=$((PARALLEL * AGENTS_PER_JOB))

echo "============================================================"
echo "Launching Wandb Sweep Chains"
echo "============================================================"
echo "Sweep ID:           $SWEEP_ID"
echo "Full path:          $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
echo "Allocations:        $PARALLEL chain(s) (max: $MAX_PARALLEL)"
echo "Agents per alloc:   $AGENTS_PER_JOB"
echo "Total concurrent:   $TOTAL_CONCURRENT runs"
echo "Per job:            $AGENTS_PER_JOB run(s) per 4-hour allocation, auto-chains"
echo "Log dir:            $LOG_DIR"
echo "============================================================"

# Find the sweep directory by grep'ing for the sweep ID in sweep_id.txt files.
# Can't rely on 'latest' symlink since it gets overwritten by any new sweep creation.
SWEEP_DIR=""
for d in "$SCRIPT_DIR"/sweep_logs/*/; do
    if [ -f "$d/sweep_id.txt" ] && grep -q "^${SWEEP_ID}$" "$d/sweep_id.txt" 2>/dev/null; then
        SWEEP_DIR="$d"
        break
    fi
done
if [ -z "$SWEEP_DIR" ]; then
    echo "WARNING: Could not find sweep directory for $SWEEP_ID, falling back to latest"
    SWEEP_DIR="$SCRIPT_DIR/sweep_logs/$(readlink "$SCRIPT_DIR/sweep_logs/latest" 2>/dev/null)"
fi
echo "Sweep dir:          $SWEEP_DIR"

# Submit jobs
SUBMITTED_JOBS=()
for i in $(seq 1 $PARALLEL); do
    JOB_NAME="sweep_${SWEEP_ID}_chain${i}"
    
    echo "Submitting chain $i of $PARALLEL ($AGENTS_PER_JOB agents)..."
    
    JOB_OUTPUT=$(sbatch \
        --job-name="$JOB_NAME" \
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        "$SCRIPT_DIR/submit_sweep_agent.sh" "$SWEEP_ID" 0 "$AGENTS_PER_JOB" "$SWEEP_DIR" 2>&1)
    
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    SUBMITTED_JOBS+=("$JOB_ID")
    echo "  -> Job $JOB_ID submitted"
    
    # Small delay between submissions
    sleep 1
done

echo ""
echo "============================================================"
echo "Submitted ${#SUBMITTED_JOBS[@]} chain(s): ${SUBMITTED_JOBS[*]}"
echo "Total concurrent runs: $TOTAL_CONCURRENT"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f $LOG_DIR/sweep_${SWEEP_ID}_chain*"
echo ""
echo "Cancel all chains:"
echo "  scancel ${SUBMITTED_JOBS[*]}"
echo ""
echo "View sweep on W&B:"
echo "  https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/sweeps/$SWEEP_ID"
echo "============================================================"

