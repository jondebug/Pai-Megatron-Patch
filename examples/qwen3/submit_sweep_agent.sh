#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/%x_%j.err

# ============================================================
# Configuration - Edit these values
# ============================================================
SWEEP_ID="${1:-YOUR_SWEEP_ID_HERE}"  # Sweep ID
CHAIN_DEPTH="${2:-0}"                 # Chain depth for auto-continuation
AGENTS_PER_JOB="${3:-1}"              # Number of parallel agents per job

WANDB_ENTITY="nvr-israel"
WANDB_PROJECT="qwen3-router-training"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
WORKDIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3"
SCRIPT_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/submit_sweep_agent.sh"
MAX_CHAIN_DEPTH=20  # Maximum number of chained jobs

# ============================================================
# Environment setup
# ============================================================
echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Sweep ID: $SWEEP_ID"
echo "Chain depth: $CHAIN_DEPTH of $MAX_CHAIN_DEPTH"
echo "Agents per job: $AGENTS_PER_JOB"
echo "============================================================"

# Create log directory
mkdir -p /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs

# ============================================================
# Submit continuation job BEFORE starting work
# This ensures continuity even if current job is killed
# ============================================================
if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN_DEPTH" ]; then
    NEXT_DEPTH=$((CHAIN_DEPTH + 1))
    # Submit next job to start after this one ends (with dependency)
    NEXT_JOB=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$SWEEP_ID" "$NEXT_DEPTH" "$AGENTS_PER_JOB" 2>&1 | awk '{print $NF}')
    echo "Submitted continuation job: $NEXT_JOB (chain depth: $NEXT_DEPTH)"
else
    echo "WARNING: Max chain depth reached ($MAX_CHAIN_DEPTH). No continuation job submitted."
fi

# ============================================================
# Run wandb agents inside container via srun
# ============================================================
echo "Starting $AGENTS_PER_JOB wandb agent(s) for sweep: $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$WORKDIR" \
     bash -c "
         pip install wandb --quiet
         export WANDB_RESUME=allow
         cd $WORKDIR
         
         # Launch multiple agents in parallel
         PIDS=()
         for i in \$(seq 1 $AGENTS_PER_JOB); do
             echo \"Starting agent \$i of $AGENTS_PER_JOB\"
             wandb agent --count 1 $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID &
             PIDS+=(\$!)
             sleep 2  # Stagger agent starts
         done
         
         # Wait for all agents to complete
         echo \"Waiting for \${#PIDS[@]} agents: \${PIDS[*]}\"
         for pid in \${PIDS[@]}; do
             wait \$pid
             echo \"Agent \$pid completed with exit code \$?\"
         done
     "

EXIT_CODE=$?

echo "============================================================"
echo "All agents exited. Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE

