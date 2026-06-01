#!/bin/bash
#SBATCH --job-name=continue_3000
#SBATCH --partition=polar4
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
# Note: real --output / --error are set per-job by the wrapper sbatch call
# (the chain successor passes --output/--error explicitly). These defaults
# exist so the script is also usable for ad-hoc sbatch invocations.
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/continue_3000_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/continue_3000_%j.err

# ============================================================
# Continue training for a single 235B v5a / v5b sweep cell from
# its latest distcp checkpoint, bypassing W&B sweep agents.
#
# Usage:
#   sbatch \
#     --output=/lustre/.../sweep_logs/continue_v5_<CELL>_d1_%j.out \
#     --error=/lustre/.../sweep_logs/continue_v5_<CELL>_d1_%j.err \
#     submit_continue_v5.sh <CELL_NAME> [CURRENT_DEPTH]
#
# Self-chaining: pre-submits next job with afterany dependency
# BEFORE launching srun, so the chain continues even on crash.
# MAX_CHAIN_DEPTH=8 (4h * 4 = 16h, enough to reach iter 1500).
# ============================================================

set -euo pipefail

CELL_NAME="${1:?Usage: sbatch submit_continue_v5.sh <CELL_NAME> [CURRENT_DEPTH]}"
CURRENT_DEPTH="${2:-1}"
MAX_CHAIN_DEPTH=8

# ----- paths -----
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
SCRIPT_PATH=$WORKDIR/submit_continue_to_3000.sh
LOG_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh

OUTPUT_BASEPATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
CELL_DIR=$OUTPUT_BASEPATH/$CELL_NAME
NAME_SUFFIX="pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5"
CELL_CKPT_ROOT=$CELL_DIR/checkpoint/$NAME_SUFFIX

DATASET_PATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CKPT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "continue_v5  cell=$CELL_NAME  depth=$CURRENT_DEPTH/$MAX_CHAIN_DEPTH"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}  Node: ${SLURM_NODELIST:-N/A}"
echo "Start: $(date)"
echo "============================================================"

# ----- validation -----
# Validate cell + the source-of-truth iter_1500 path (not the new ti-3000 save target which is empty on first chain)
LOAD_PATH="$CELL_DIR/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-1500-wi-5"
if [ ! -d "$CELL_DIR" ]; then
    echo "ERROR: cell dir does not exist: $CELL_DIR" >&2
    exit 2
fi
if [ ! -d "$LOAD_PATH" ]; then
    echo "ERROR: load path does not exist: $LOAD_PATH" >&2
    exit 2
fi
LATEST_ITER_FILE=$LOAD_PATH/latest_checkpointed_iteration.txt
if [ ! -f "$LATEST_ITER_FILE" ]; then
    echo "ERROR: latest_checkpointed_iteration.txt missing in $LOAD_PATH" >&2
    exit 2
fi
# Ensure the (possibly-new) save dir exists for cleanup logic
mkdir -p "$CELL_CKPT_ROOT" 
LATEST_ITER=$(tr -d '[:space:]' < "$LATEST_ITER_FILE")
ITER_DIR=$(printf "%s/iter_%07d" "$LOAD_PATH" "$LATEST_ITER")
if [ ! -d "$ITER_DIR" ]; then
    echo "ERROR: iter dir not found: $ITER_DIR" >&2
    exit 2
fi
NUM_DISTCP=$(find "$ITER_DIR" -maxdepth 1 -name '*.distcp' | wc -l)
if [ "$NUM_DISTCP" -lt 1 ]; then
    echo "ERROR: no .distcp files in $ITER_DIR" >&2
    exit 2
fi
echo "Validated: $CELL_CKPT_ROOT  latest_iter=$LATEST_ITER  distcp_files=$NUM_DISTCP"

# Choose resume source: continue from ti-3000 save target if it already holds a
# checkpoint, otherwise seed from the iter-1500 source-of-truth.
if [ -f "$CELL_CKPT_ROOT/latest_checkpointed_iteration.txt" ] && ls -d "$CELL_CKPT_ROOT"/iter_* >/dev/null 2>&1; then
    RESUME_LOAD="$CELL_CKPT_ROOT"
else
    RESUME_LOAD="$LOAD_PATH"
fi
echo "RESUME_LOAD=$RESUME_LOAD"

# ----- parse hyperparameters from CELL_NAME -----
# Defaults (from v5a/v5b sweep configs):
USE_RL_LOSS=true
RL_LOSS_COEFF=0.5
AUX_COEFF=0.01
RL_BASELINE_TYPE=mean
RL_DISCOUNT_FACTOR=0
RL_LM_REWARD_COEFF=0
RL_STOCHASTIC_ROUTING=false
RL_STOCHASTIC_TEMPERATURE=0.3
KL_LOSS_COEFF=0

# norl baseline: aux-only
if [[ "$CELL_NAME" == 235bv5a_norl_aux* ]]; then
    USE_RL_LOSS=false
    if [[ "$CELL_NAME" =~ _aux([0-9.]+)_r ]]; then
        AUX_COEFF="${BASH_REMATCH[1]}"
    fi
fi

# v5a main grid: rlc<C>_ppo_aux<X>_rNN
if [[ "$CELL_NAME" =~ ^235bv5a_rlc([0-9.]+)_ppo_aux([0-9.]+)_r[0-9]+$ ]]; then
    RL_LOSS_COEFF="${BASH_REMATCH[1]}"
    AUX_COEFF="${BASH_REMATCH[2]}"
fi

# v5b cells: all share rlc=0.5, aux=0.01, baseline=mean, gamma=0 by default
if [[ "$CELL_NAME" == 235bv5b_* ]]; then
    if [[ "$CELL_NAME" =~ _rlc([0-9.]+)_ ]]; then
        RL_LOSS_COEFF="${BASH_REMATCH[1]}"
    fi
    if [[ "$CELL_NAME" =~ _g([0-9.]+)_ ]]; then
        RL_DISCOUNT_FACTOR="${BASH_REMATCH[1]}"
    fi
    if [[ "$CELL_NAME" =~ _kl([0-9.]+)_r ]]; then
        KL_LOSS_COEFF="${BASH_REMATCH[1]}"
    fi
    if [[ "$CELL_NAME" =~ _lm([0-9.]+)_r ]]; then
        RL_LM_REWARD_COEFF="${BASH_REMATCH[1]}"
    fi
    if [[ "$CELL_NAME" =~ _gumbel_t([0-9.]+)_r ]]; then
        RL_STOCHASTIC_ROUTING=true
        RL_STOCHASTIC_TEMPERATURE="${BASH_REMATCH[1]}"
    fi
    if [[ "$CELL_NAME" == *_critic_ppo_* ]]; then
        RL_BASELINE_TYPE=critic
    fi
fi

echo "Resolved hyperparameters:"
echo "  use_rl_loss=$USE_RL_LOSS  rl_loss_coeff=$RL_LOSS_COEFF  aux_coeff=$AUX_COEFF"
echo "  baseline=$RL_BASELINE_TYPE  gamma=$RL_DISCOUNT_FACTOR  kl=$KL_LOSS_COEFF  lm=$RL_LM_REWARD_COEFF"
echo "  stoch=$RL_STOCHASTIC_ROUTING (T=$RL_STOCHASTIC_TEMPERATURE)"

# ----- pre-submit chain successor BEFORE training -----
if [ "$CURRENT_DEPTH" -lt "$MAX_CHAIN_DEPTH" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    NEXT_DEPTH=$((CURRENT_DEPTH + 1))
    NEXT_OUT=$LOG_DIR/continue_3000_${CELL_NAME}_d${NEXT_DEPTH}_%j.out
    NEXT_ERR=$LOG_DIR/continue_3000_${CELL_NAME}_d${NEXT_DEPTH}_%j.err
    NEXT_JOB=$(sbatch \
        --dependency=afterany:$SLURM_JOB_ID \
        --output="$NEXT_OUT" \
        --error="$NEXT_ERR" \
        --job-name="c3000_${CELL_NAME}_d${NEXT_DEPTH}" \
        "$SCRIPT_PATH" "$CELL_NAME" "$NEXT_DEPTH" 2>&1 | awk '{print $NF}')
    echo "Chain successor submitted: jobid=$NEXT_JOB depth=$NEXT_DEPTH"
elif [ "$CURRENT_DEPTH" -ge "$MAX_CHAIN_DEPTH" ]; then
    echo "Max chain depth reached ($MAX_CHAIN_DEPTH); not submitting successor."
fi

# ----- launch training -----

# ---- Prune older iter_* dirs to keep only the latest (save_interval=3000 + chain saves)
echo "Pruning older iter dirs in $CELL_CKPT_ROOT to save disk..."
MAX_ITER=$(ls -d "$CELL_CKPT_ROOT"/iter_* 2>/dev/null | sed -E 's@.*/iter_0*@@' | sort -n | tail -1 || true)
if [ -n "$MAX_ITER" ]; then
  shopt -s nullglob
  for IT_DIR in "$CELL_CKPT_ROOT"/iter_*; do
    [ -d "$IT_DIR" ] || continue
    THIS_ITER=$(echo "$IT_DIR" | sed -E 's@.*/iter_0*@@')
    if [ "$THIS_ITER" != "$MAX_ITER" ]; then
      echo "  rm $IT_DIR (iter $THIS_ITER, keeping iter $MAX_ITER)"
      rm -rf "$IT_DIR"
    fi
  done
  shopt -u nullglob
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT

# Build the EXTRA_ARGS array (passed positional-after-27 to run_mcore_qwen3.sh)
EXTRA_ARGS=(
    --router-only-training
    --enable-wandb-logging
    --ckpt-assume-constant-structure
    --ckpt-fully-parallel-save
    --rl-normalize-rewards
    --rl-use-ema-loads
    --rl-ppo-reeval
    --wandb-project-name qwen3-router-training
    --wandb-run-name "$CELL_NAME"
    --wandb-run-tags 235b v5-continue router-only resume "depth-$CURRENT_DEPTH"
    --rl-algorithm ppo
    --rl-ppo-entropy-coeff 0.01
    --rl-ppo-baseline-type "$RL_BASELINE_TYPE"
    --rl-critic-hidden-dims 256
    --rl-reward-type per_token_load_weighted
    --rl-reward-topn 2
    --rl-discount-factor "$RL_DISCOUNT_FACTOR"
    --rl-gae-lambda 1.0
    --rl-ppo-epochs 1
    --rl-ppo-extra-lr 0.0001
    --rl-lm-reward-coeff "$RL_LM_REWARD_COEFF"
    --kl-loss-coeff "$KL_LOSS_COEFF"
    --moe-aux-loss-coeff "$AUX_COEFF"
    --exit-duration-in-mins 230
    --train-iters 3000
    --save-interval 500
    --eval-interval 200
    --eval-iters 50
    --empty-unused-memory-level 2
    --load "$RESUME_LOAD"
)

# Boolean toggles that only emit a flag when true
if [ "$USE_RL_LOSS" = "true" ]; then
    EXTRA_ARGS+=( --use_rl_loss )
fi
if [ "$RL_STOCHASTIC_ROUTING" = "true" ]; then
    EXTRA_ARGS+=( --rl-stochastic-routing --rl-stochastic-temperature "$RL_STOCHASTIC_TEMPERATURE" )
fi

# Render extra-args as a single shell-safe string for the srun bash -c invocation
EXTRA_ARGS_STR=$(printf ' %q' "${EXTRA_ARGS[@]}")

echo "============================================================"
echo "Launching srun  master=$MASTER_ADDR:$MASTER_PORT"
echo "EXTRA_ARGS:$EXTRA_ARGS_STR"
echo "============================================================"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$WORKDIR" \
     --nodes=2 --ntasks-per-node=1 \
     bash -c "
         set -euo pipefail
         pip install wandb datasets --quiet
         export WANDB_RESUME=allow
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
         export WORLD_SIZE=2
         export RANK=\${SLURM_PROCID:-0}
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8
         export MASTER_ADDR=$MASTER_ADDR
         export MASTER_PORT=$MASTER_PORT
         echo \"node \$(hostname) RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE}\"
         cd $WORKDIR
         sh run_mcore_qwen3.sh dlc A22B 1 16 1e-4 1e-6 128 128 bf16 1 1 1 1 16 true true true false sel false 3000 \\
           $DATASET_PATH $DATASET_PATH $PRETRAIN_CKPT 1024000 10240 $CELL_DIR$EXTRA_ARGS_STR
     "

EXIT_CODE=$?
echo "============================================================"
echo "srun exit: $EXIT_CODE   End: $(date)"
echo "============================================================"
exit $EXIT_CODE
