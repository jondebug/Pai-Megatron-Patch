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
SWEEP_DIR_ARG="${4:-}"                # Sweep directory (passed by launch_sweep.sh)

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
echo "Sweep dir arg: ${SWEEP_DIR_ARG:-<not set>}"
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
    NEXT_JOB=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$SWEEP_ID" "$NEXT_DEPTH" "$AGENTS_PER_JOB" "$SWEEP_DIR_ARG" 2>&1 | awk '{print $NF}')
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
         
         # Phase 1: Resume incomplete runs from previous allocations.
         # Use the sweep dir passed as argument, or fall back to latest symlink.
         if [ -n \"$SWEEP_DIR_ARG\" ]; then
             SWEEP_DIR=\"$SWEEP_DIR_ARG\"
         else
             SWEEP_DIR=\$(readlink -f sweep_logs/latest 2>/dev/null)
         fi
         if [ -n \"\$SWEEP_DIR\" ] && [ -f \"\$SWEEP_DIR/sweep_combinations.json\" ]; then
             COMBOS=\$SWEEP_DIR/sweep_combinations.json
             CONFIG=\$SWEEP_DIR/sweep_config.json
             OUTPUT_BASE=\$(python3 -c \"import json; print(json.load(open('\$CONFIG')).get('output_basepath',''))\" 2>/dev/null)
             TRAIN_ITERS=\$(python3 -c \"import json; c=json.load(open('\$CONFIG')); print(c.get('train_iters', c.get('fixed_params',{}).get('train_iters',0)))\" 2>/dev/null)
             N_COMBOS=\$(python3 -c \"import json; print(len(json.load(open('\$COMBOS')).get('combinations',[])))\" 2>/dev/null)
             
             if [ -n \"\$OUTPUT_BASE\" ] && [ \"\$TRAIN_ITERS\" -gt 0 ] 2>/dev/null; then
                 echo \"Checking for incomplete runs (target: \$TRAIN_ITERS steps)...\"
                 for IDX in \$(seq 0 \$((N_COMBOS - 1))); do
                     # Build the run name to find its checkpoint directory
                     RUN_NAME=\$(python3 -c \"
import sys; sys.path.insert(0,'.')
from wandb_agent_runner import build_run_name, load_combinations
combos, fixed, _, _ = load_combinations('\$COMBOS')
if \$IDX < len(combos):
    print(build_run_name(combos[\$IDX], \$IDX, fixed))
\" 2>/dev/null)
                     [ -z \"\$RUN_NAME\" ] && continue
                     
                     # Checkpoint lives under: output_base/RUN_NAME/checkpoint/MEGATRON_NAME/
                     ITER_FILE=\$(find \"\$OUTPUT_BASE/\$RUN_NAME/checkpoint\" -name \"latest_checkpointed_iteration.txt\" 2>/dev/null | head -1)
                     if [ -n \"\$ITER_FILE\" ]; then
                         SAVED_ITER=\$(cat \"\$ITER_FILE\" | tr -d '[:space:]')
                         if [ \"\$SAVED_ITER\" -lt \"\$TRAIN_ITERS\" ] 2>/dev/null; then
                             echo \"RESUME: \$RUN_NAME at iter \$SAVED_ITER/\$TRAIN_ITERS\"
                             python3 wandb_agent_runner.py --run_index \$IDX --sweep-dir \"\$SWEEP_DIR\" &
                             RESUME_PID=\$!
                             wait \$RESUME_PID
                             echo \"Resume of \$RUN_NAME completed (exit \$?)\"
                         fi
                     fi
                 done
             fi
         fi
         
         # Phase 2: Launch wandb agents for new (unstarted) configs
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

