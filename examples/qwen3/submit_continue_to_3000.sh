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
# RELAXED VALIDATION (2026-07-01): the ti-1500 seed may have been legitimately pruned by disk
# cleanup AFTER the cell moved on — if the ti-3000 save root already holds iter_* checkpoints,
# resume from there and do NOT require the ti-1500 seed to still exist.
mkdir -p "$CELL_CKPT_ROOT"
TI3000_OK=0
ls -d "$CELL_CKPT_ROOT"/iter_* >/dev/null 2>&1 && TI3000_OK=1
LATEST_ITER=""
if [ -d "$LOAD_PATH" ] && [ -f "$LOAD_PATH/latest_checkpointed_iteration.txt" ]; then
    LATEST_ITER=$(tr -d '[:space:]' < "$LOAD_PATH/latest_checkpointed_iteration.txt")
    ITER_DIR=$(printf "%s/iter_%07d" "$LOAD_PATH" "$LATEST_ITER")
    NUM_DISTCP=0; [ -d "$ITER_DIR" ] && NUM_DISTCP=$(find "$ITER_DIR" -maxdepth 1 -name '*.distcp' | wc -l)
else
    ITER_DIR=""; NUM_DISTCP=0
fi
if [ "$NUM_DISTCP" -lt 1 ] && [ "$TI3000_OK" -ne 1 ]; then
    echo "ERROR: no usable checkpoint (ti-1500 seed missing/empty AND ti-3000 root empty) for $CELL_NAME" >&2
    exit 2
fi
[ "$NUM_DISTCP" -lt 1 ] && echo "WARN: ti-1500 seed pruned; resuming from ti-3000 root (has checkpoints)"
echo "Validated: $CELL_CKPT_ROOT  ti1500_latest=${LATEST_ITER:-none} distcp=$NUM_DISTCP ti3000_ok=$TI3000_OK"

# Choose resume source: continue from ti-3000 save target if it already holds a
# checkpoint, otherwise seed from the iter-1500 source-of-truth.
if [ -f "$CELL_CKPT_ROOT/latest_checkpointed_iteration.txt" ] && ls -d "$CELL_CKPT_ROOT"/iter_* >/dev/null 2>&1; then
    RESUME_LOAD="$CELL_CKPT_ROOT"
else
    RESUME_LOAD="$LOAD_PATH"
fi
echo "RESUME_LOAD=$RESUME_LOAD"
# POINTER-REPAIR (2026-06-05): if latest_checkpointed_iteration.txt points to a missing
# iter dir (stale exit-save pointer), reset it to the max existing iter dir. Prevents
# FileNotFoundError crashes on resume. Never deletes anything.
if [ -d "$RESUME_LOAD" ]; then
  _PF="$RESUME_LOAD/latest_checkpointed_iteration.txt"
  _CUR=$(tr -d "[:space:]" < "$_PF" 2>/dev/null || echo "")
  _PAD=$(printf "%07d" "${_CUR:-0}" 2>/dev/null || echo "")
  if [ -n "$_CUR" ] && [ ! -d "$RESUME_LOAD/iter_$_PAD" ]; then
    _MAX=$(ls -d "$RESUME_LOAD"/iter_* 2>/dev/null | grep -oE "iter_[0-9]+" | sed "s/iter_0*//" | sort -n | tail -1)
    if [ -n "$_MAX" ]; then echo "$_MAX" > "$_PF"; echo "POINTER-REPAIR: $_CUR -> $_MAX (missing iter dir)"; fi
  fi
fi

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
# reward type default matches the long-standing v5a/v5b/v12corner convention.
REWARD_TYPE=per_token_load_weighted
# SEED empty => do not emit --seed (preserves existing cells' trainer default of 1234).
SEED=""
PLR="1e-4"; PMINLR="1e-6"   # default; overridden by _plr<val> in cell name (v17g LR arm)

# generic fallback (2026-06-06): pull rlc/aux from ANY cell name so non-v5a/v5b cells
# (e.g. v12corner) continue with their REAL hyperparams, not the rlc0.5/aux0.01 defaults.
if [[ "$CELL_NAME" =~ rlc([0-9.]+) ]]; then RL_LOSS_COEFF="${BASH_REMATCH[1]}"; fi
if [[ "$CELL_NAME" =~ _aux([0-9.]+) ]]; then AUX_COEFF="${BASH_REMATCH[1]}"; fi

# generic fallback (2026-06-07): thread reward-type / KL / LM-reward from ANY cell name.
# Without this, REWARD_TYPE/KL/LM do NOT survive name-based continuation (the trainer
# silently reverts to per_token_load_weighted / kl0 / lm0), confounding reward-comparison,
# KL, and LM-reward cells. Cell-name encodings:
#   _rwd<type>   e.g. _rwdcritical_path / _rwdentropy / _rwdtopn_binary / _rwdper_token_load_weighted
#   _kl<x>       e.g. _kl0.001
#   _lm<x>       e.g. _lm0.3
# The reward type itself contains underscores, so we first strip any trailing _r<digits>
# seed suffix (bash ERE has no reliable lazy match), then match _rwd<type> anchored at end.
_RWD_STRIPPED="${CELL_NAME%_r[0-9]}"
_RWD_STRIPPED="${_RWD_STRIPPED%_r[0-9][0-9]}"
_RWD_STRIPPED="${_RWD_STRIPPED%_r[0-9][0-9][0-9]}"
if [[ "$_RWD_STRIPPED" =~ _rwd([a-zA-Z0-9_]+)$ ]]; then REWARD_TYPE="${BASH_REMATCH[1]}"; fi
# topn_binary is the human-friendly name; the trainer arg is per_token_topn_binary.
if [[ "$REWARD_TYPE" == "topn_binary" ]]; then REWARD_TYPE=per_token_topn_binary; fi
if [[ "$CELL_NAME" =~ _kl([0-9.]+) ]]; then KL_LOSS_COEFF="${BASH_REMATCH[1]}"; fi
if [[ "$CELL_NAME" =~ _lm([0-9.]+) ]]; then RL_LM_REWARD_COEFF="${BASH_REMATCH[1]}"; fi
# Multi-seed cells (#5): _seed<N> must keep the SAME seed through continuation as the fresh
# launch used, else the two seeds reconverge. Mapping MUST match submit_fresh_corner_ep16.sh's
# launch env: seed1->1234 (Megatron default), seed2->2025.
if [[ "$CELL_NAME" =~ _plr([0-9.e-]+) ]]; then PLR="${BASH_REMATCH[1]}"; PMINLR="1e-7"; fi
if [[ "$CELL_NAME" =~ _seed([0-9]+) ]]; then
    case "${BASH_REMATCH[1]}" in
        1) SEED=1234 ;;
        2) SEED=2025 ;;
        *) SEED="${BASH_REMATCH[1]}" ;;
    esac
fi

# generic fallback (2026-06-07): thread PPO baseline-type from ANY cell name so the
# critic-vs-mean ablation setting SURVIVES the 1500->3000 continuation. Without this the
# trainer silently reverts to the mean baseline default, confounding the comparison.
# Cell-name encodings: _basemean (mean baseline) / _basecritic (learned critic value head).
# No marker => RL_BASELINE_TYPE stays 'mean' (default) => existing cells byte-identical.
if [[ "$CELL_NAME" =~ _basecritic ]]; then
    RL_BASELINE_TYPE=critic
elif [[ "$CELL_NAME" =~ _basemean ]]; then
    RL_BASELINE_TYPE=mean
fi

# generic fallback (2026-06-08): thread the discount factor gamma from ANY cell name so
# gamma SURVIVES the 1500->3000 continuation. Previously gamma was parsed ONLY inside the
# 235bv5b_* block (regex _g(...)_), so non-v5b cells (v7/v10/v11/crit_n1, and any cell whose
# gamma token is not followed by '_') silently reverted to gamma=0 at the handoff -- the
# factorial sweep's gamma>0 cells would have collapsed to gamma0, confounding the discount
# ablation. Cell-name encoding: _g<x> e.g. _g0.3 / _g0.5 / _g0 . The pattern requires a digit
# immediately after 'g' so _gumbel / _gbs / _gpus etc. never false-match. No _g<digit> token
# => RL_DISCOUNT_FACTOR stays at its current value (default 0) => those cells byte-identical.
# Placed AFTER the v5b block so it is a strict superset; v5b cells resolve identically.
if [[ "$CELL_NAME" =~ _g([0-9.]+) ]]; then RL_DISCOUNT_FACTOR="${BASH_REMATCH[1]}"; fi

# norl baseline: aux-only
if [[ "$CELL_NAME" == *norl* ]]; then   # generalized 2026-07-08 (v18s aux-seed cells)
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
echo "  reward_type=$REWARD_TYPE  seed=${SEED:-<default>}"
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
      # FRONTIER-SAFETY (2026-06-05): never delete checkpoints. A pruned iter may be
      # an unbenchmarked frontier point. Disk is reclaimed via the frontier-exempt HF cleanup.
      echo "  [prune-disabled] KEEPING $IT_DIR (iter $THIS_ITER); distcp checkpoints are never deleted"
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
    --rl-critic-lr 1e-3
    --rl-reward-type "$REWARD_TYPE"
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
# Only emit --seed for cells that encode a seed (multi-seed #5); other cells keep the
# trainer default 1234 -> byte-identical behavior to before this change.
if [ -n "$SEED" ]; then
    EXTRA_ARGS+=( --seed "$SEED" )
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
         sh run_mcore_qwen3.sh dlc A22B 1 16 $PLR $PMINLR 128 128 bf16 1 1 1 1 16 true true true false sel false 3000 \\
           $DATASET_PATH $DATASET_PATH $PRETRAIN_CKPT 1024000 10240 $CELL_DIR$EXTRA_ARGS_STR
     "

EXIT_CODE=$?
echo "============================================================"
echo "srun exit: $EXIT_CODE   End: $(date)"
echo "============================================================"
exit $EXIT_CODE
