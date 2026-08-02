#!/bin/bash
#SBATCH --job-name=cp_rlport_rung4
#SBATCH --partition=polar4
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/rlport_capture_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/rlport_capture_%j.err

# =============================================================================
# RL-PORT RUNG-4 50-ITER ON-POLICY TREND
# =============================================================================
# Copy of submit_rlport_capture.sh. Trains 50 on-policy iters with
# LOO_BETA=0.008 so the graduation gate can read the TRENDS: rl_grad_norm
# stability, cov_ptlp_adv sign/direction, num_tokens_on_critical_path trajectory,
# rl_mean_reward, load_balancing_loss, and churn bounds (is_ratio_p99, flip_rate).
# ~5min load + 50*~17s steady-state => ~19min; fits the 1h wall / 55min exit.
#
# Deliberate deltas vs the corner launcher (causal isolation):
#   * AUX=0  KL=0            -> aux/KL OFF, so any router gradient is RL-only
#   * RLC=0.5               -> RL loss coefficient
#   * SEED=1234
#   * RL_TRAIN_ITERS=50     -> rung-4 on-policy trend
#   * --save= (empty)       -> args.save falsy => NO checkpoint written and the post-run
#                              auto-convert/lm-eval block is skipped. (A huge save-interval
#                              alone is NOT enough: Megatron force-saves a final checkpoint at
#                              train-iters completion whenever args.save is set.)
#   * EVAL_ITERS=0          -> in-training eval OFF
#   * PERLAYER_NORM=1       -> the intended single-normalization reduction
#   * 4 port flags added to RL_FLAGS below:
#       --rl-sampling hard_gumbel_pl --rl-candidate-pool 32
#       --rl-global-loads --rl-reward-type loo_smoothmax
#   * --rl-loo-beta 0.008   -> denser credit (0.03 saturates)
#
# Launch:
#   RUN_NAME=235bv21math_rlport_rung4 sbatch submit_rlport_rung4_trend.sh
# =============================================================================
set -euo pipefail

: "${RUN_NAME:?set RUN_NAME (e.g. 235bv21math_rlport_capture)}"

# --- Capture-mode settings (fixed for this gate; overridable via env if needed) ---
RLC="${RLC:-0.5}"
AUX="${AUX:-0}"            # aux OFF (causal isolation)
KL="${KL:-0}"             # KL OFF (causal isolation)
LM="${LM:-0}"
GAMMA="${GAMMA:-0}"
REWARD_TYPE="${REWARD_TYPE:-loo_smoothmax}"
RL_TRAIN_ITERS="${RL_TRAIN_ITERS:-50}"   # RUNG-4 on-policy trend
SEED="${SEED:-1234}"
BASELINE="${BASELINE:-mean}"
USE_RL="${USE_RL:-1}"
PLR="${PLR:-1e-4}"
PMINLR="${PMINLR:-1e-6}"
GAE_LAMBDA="${GAE_LAMBDA:-1.0}"
EXTRA_LR="${EXTRA_LR:-1e-4}"
PERLAYER_NORM="${PERLAYER_NORM:-1}"       # intended single-normalization reduction
LOO_BETA="${LOO_BETA:-0.008}"             # RUNG-3/4: denser credit (0.03 saturates)
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"  # huge (belt); real disabler is --save= below
EVAL_ITERS="${EVAL_ITERS:-0}"             # eval OFF
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
MATH_BLEND="${MATH_BLEND:-}"

CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
OUTPUT_BASEPATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
TARGET_RUN_DIR=$OUTPUT_BASEPATH/$RUN_NAME
DATASET_PATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CKPT="${RESUME_FROM:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore-dist}"  # torch_dist reshardable base (EP=16-validated)

mkdir -p "$TARGET_RUN_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT

WANDB_TAGS="235b v21math rlport-capture router-only g7"

echo "============================================================"
echo "RL-PORT CAPTURE (EP=16, 2 nodes) fresh-from-pretrained  [CAPTURE, NOT TRAINING]"
echo "RUN_NAME=$RUN_NAME"
echo "RLC=$RLC AUX=$AUX REWARD=$REWARD_TYPE LM=$LM KL=$KL GAMMA=$GAMMA SEED=$SEED BASELINE=$BASELINE TRAIN_ITERS=$RL_TRAIN_ITERS"
echo "PERLAYER_NORM=$PERLAYER_NORM LOO_BETA=$LOO_BETA SAVE_INTERVAL=$SAVE_INTERVAL EVAL_ITERS=$EVAL_ITERS"
echo "PORT FLAGS: --rl-sampling hard_gumbel_pl --rl-candidate-pool 32 --rl-global-loads --rl-reward-type loo_smoothmax --rl-loo-beta $LOO_BETA"
echo "TARGET_RUN_DIR=$TARGET_RUN_DIR"
echo "LOAD (fresh): $PRETRAIN_CKPT"
echo "Master: $MASTER_ADDR:$MASTER_PORT   Start: $(date)"
echo "============================================================"

# RL flag block (base corner set + the 4 port flags + loo beta)
if [ "$USE_RL" = "1" ]; then
  RL_FLAGS="--use_rl_loss --rl-normalize-rewards --rl-use-ema-loads --rl-ppo-reeval \
    --rl-algorithm ppo --rl-loss-coeff $RLC --rl-ppo-entropy-coeff 0.01 \
    --rl-ppo-baseline-type $BASELINE --rl-critic-hidden-dims 256 --rl-critic-lr 1e-3 \
    --rl-reward-type $REWARD_TYPE --rl-reward-topn 2 --rl-discount-factor $GAMMA \
    --rl-gae-lambda $GAE_LAMBDA --rl-ppo-epochs ${PPO_EPOCHS:-1} --rl-ppo-extra-lr $EXTRA_LR \
    --rl-lm-reward-coeff $LM \
    --rl-sampling hard_gumbel_pl --rl-candidate-pool 32 --rl-global-loads --rl-loo-beta $LOO_BETA"
  [ "${PERLAYER_NORM:-0}" = "1" ] && RL_FLAGS="$RL_FLAGS --rl-perlayer-norm"
else
  RL_FLAGS=""
fi

# Data path override (last --data-path wins in argparse)
if [ -n "$MATH_BLEND" ]; then
  DATA_OVERRIDE="--data-path $MATH_BLEND"
else
  DATA_OVERRIDE=""
fi

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
         export RANK=\${SLURM_PROCID}
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8
         export MASTER_ADDR=\"$MASTER_ADDR\"
         export MASTER_PORT=\"$MASTER_PORT\"
         echo \"node \$(hostname) RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE}\"
         cd $WORKDIR
         sh run_mcore_qwen3.sh dlc A22B 1 16 $PLR $PMINLR 128 128 bf16 1 1 1 1 16 true true true false sel false $RL_TRAIN_ITERS \\
           $DATASET_PATH $DATASET_PATH $PRETRAIN_CKPT 1024000 10240 $TARGET_RUN_DIR \\
           --router-only-training --enable-wandb-logging --seed $SEED $RL_FLAGS --kl-loss-coeff $KL \\
           --moe-aux-loss-coeff $AUX \\
           --exit-duration-in-mins 55 --train-iters $RL_TRAIN_ITERS \\
           --save-interval $SAVE_INTERVAL --eval-interval $EVAL_INTERVAL --eval-iters $EVAL_ITERS \\
           --save= \\
           --empty-unused-memory-level 2 \\
           --wandb-run-tags $WANDB_TAGS $DATA_OVERRIDE
     "

echo "============================================================"
echo "End: $(date)"
echo "============================================================"
