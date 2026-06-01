#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --partition=interactive_singlenode
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/%x_%j.err

# ============================================================
# 8-GPU sweep agent (for Qwen3-235B-A22B and other large models)
# Same as submit_sweep_agent.sh but with 8 GPUs + memory optimizations.
# ============================================================
SWEEP_ID="${1:-YOUR_SWEEP_ID_HERE}"
CHAIN_DEPTH="${2:-0}"
AGENTS_PER_JOB="${3:-1}"
SWEEP_DIR_ARG="${4:-}"

WANDB_ENTITY="nvr-israel"
WANDB_PROJECT="qwen3-router-training"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
WORKDIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3"
SCRIPT_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/submit_sweep_agent_8gpu.sh"
MAX_CHAIN_DEPTH=30

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8 (large model)"
echo "Start time: $(date)"
echo "Sweep ID: $SWEEP_ID"
echo "Chain depth: $CHAIN_DEPTH of $MAX_CHAIN_DEPTH"
echo "Agents per job: $AGENTS_PER_JOB"
echo "Sweep dir arg: ${SWEEP_DIR_ARG:-<not set>}"
echo "============================================================"

mkdir -p /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs

if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN_DEPTH" ]; then
    NEXT_DEPTH=$((CHAIN_DEPTH + 1))
    NEXT_JOB=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$SWEEP_ID" "$NEXT_DEPTH" "$AGENTS_PER_JOB" "$SWEEP_DIR_ARG" 2>&1 | awk '{print $NF}')
    echo "Submitted continuation job: $NEXT_JOB (chain depth: $NEXT_DEPTH)"
else
    echo "WARNING: Max chain depth reached ($MAX_CHAIN_DEPTH). No continuation job submitted."
fi

echo "Starting $AGENTS_PER_JOB wandb agent(s) for sweep: $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$WORKDIR" \
     bash -c "
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
         pip install wandb datasets --quiet
         export WANDB_RESUME=allow
         cd $WORKDIR
         
         PIDS=()
         for i in \$(seq 1 $AGENTS_PER_JOB); do
             echo \"Starting agent \$i of $AGENTS_PER_JOB\"
             wandb agent --count 1 $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID &
             PIDS+=(\$!)
             sleep 2
         done
         
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
