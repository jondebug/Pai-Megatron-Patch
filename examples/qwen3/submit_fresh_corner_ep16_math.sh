#!/bin/bash
#SBATCH --job-name=fresh_corner_ep16
#SBATCH --partition=polar4
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/fresh_corner_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/fresh_corner_%j.err

# Parameterized FRESH-from-pretrained EP=16 launcher for the corner attack.
# Derived verbatim from the proven submit_train_235b_16gpu.sh fresh invocation;
# flag set mirrors the v12corner cells exactly (per_token_load_weighted, ppo, mean
# baseline, gamma 0). Trains 0 -> RL_TRAIN_ITERS (default 1500) into a NEW cell dir,
# producing a ti-1500 checkpoint that the existing babysit_3000 supervisor then
# continues to 3000 and inf-evaluates (cell name MUST start 235bv12corner_ so the
# supervisor globs + the generic rlc/aux name-parser pick it up).
#
# MATH-BLEND VARIANT (2026-07-23): Also accepts 235bv21math_ prefix; adds GAE_LAMBDA, EXTRA_LR, SEED env;
# injects MATH_BLEND as data override; tags "v21math" to fence from babysit_3000 auto-continue.
#
# Required env: RUN_NAME (e.g. 235bv12corner_per_token_load_weighted_rlc0.03_aux0.001 or 235bv21math_...)
# Optional env (defaults match v12corner): RLC=0.05 AUX=0.001 REWARD_TYPE=per_token_load_weighted
#                                          LM=0 KL=0 GAMMA=0 RL_TRAIN_ITERS=1500
#                                          GAE_LAMBDA=1.0 EXTRA_LR=1e-4 SEED=1234
#                                          MATH_BLEND="30 <gen> 70 <math>" (override data)
set -euo pipefail

: "${RUN_NAME:?set RUN_NAME (must start with 235bv12corner_ or 235bv21math_)}"
RLC="${RLC:-0.05}"
AUX="${AUX:-0.001}"
REWARD_TYPE="${REWARD_TYPE:-per_token_load_weighted}"
LM="${LM:-0}"
KL="${KL:-0}"
GAMMA="${GAMMA:-0}"
RL_TRAIN_ITERS="${RL_TRAIN_ITERS:-1500}"
SEED="${SEED:-1234}"   # Megatron default is 1234; vary per multi-seed cell.
BASELINE="${BASELINE:-mean}"
USE_RL="${USE_RL:-1}"      # 0 -> aux-only (no RL flags at all; v18s seed baseline)
PLR="${PLR:-1e-4}"       # policy learning rate (v17g LR arm); min-lr scales /100
PMINLR="${PMINLR:-1e-6}"   # mean (REINFORCE-with-mean-baseline) or critic (learned value head).
                               # critic args below are inert under mean, used under critic.
GAE_LAMBDA="${GAE_LAMBDA:-1.0}"   # GAE lambda (1.0 = no GAE, <1.0 = exponential average)
EXTRA_LR="${EXTRA_LR:-1e-4}"      # PPO extra learning rate (for ppo_epochs >= 2)
MATH_BLEND="${MATH_BLEND:-}"      # If set, override data-path with blend (e.g., "30 <gen> 70 <math>")
EVAL_ITERS="${EVAL_ITERS:-50}"      # in-training eval iters; set 0 to DISABLE periodic eval (we eval checkpoints ourselves)
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"  # in-training eval interval

CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
OUTPUT_BASEPATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
TARGET_RUN_DIR=$OUTPUT_BASEPATH/$RUN_NAME
DATASET_PATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CKPT="${RESUME_FROM:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore-dist}"  # torch_dist reshardable base (EP=16-validated 2026-06-07); legacy -to-mcore is EP8-only and OOMs

mkdir -p "$TARGET_RUN_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT

# Determine wandb tags based on RUN_NAME prefix
if [[ "$RUN_NAME" =~ ^235bv21math_ ]]; then
  WANDB_TAGS="235b v21math fresh-corner router-only"
else
  WANDB_TAGS="235b v12corner fresh-corner router-only"
fi

echo "============================================================"
echo "FRESH CORNER (EP=16, 2 nodes) fresh-from-pretrained"
echo "RUN_NAME=$RUN_NAME"
echo "RLC=$RLC AUX=$AUX REWARD=$REWARD_TYPE LM=$LM KL=$KL GAMMA=$GAMMA SEED=$SEED BASELINE=$BASELINE TRAIN_ITERS=$RL_TRAIN_ITERS"
echo "GAE_LAMBDA=$GAE_LAMBDA EXTRA_LR=$EXTRA_LR"
[ -n "$MATH_BLEND" ] && echo "MATH_BLEND=$MATH_BLEND"
echo "TARGET_RUN_DIR=$TARGET_RUN_DIR"
echo "LOAD (fresh): $PRETRAIN_CKPT"
echo "Master: $MASTER_ADDR:$MASTER_PORT   Start: $(date)"
echo "============================================================"

# RL flag block (omitted entirely for aux-only cells)
if [ "$USE_RL" = "1" ]; then
  RL_FLAGS="--use_rl_loss --rl-normalize-rewards --rl-use-ema-loads --rl-ppo-reeval \
    --rl-algorithm ppo --rl-loss-coeff $RLC --rl-ppo-entropy-coeff 0.01 \
    --rl-ppo-baseline-type $BASELINE --rl-critic-hidden-dims 256 --rl-critic-lr 1e-3 \
    --rl-reward-type $REWARD_TYPE --rl-reward-topn 2 --rl-discount-factor $GAMMA \
    --rl-gae-lambda $GAE_LAMBDA --rl-ppo-epochs ${PPO_EPOCHS:-1} --rl-ppo-extra-lr $EXTRA_LR \
    --rl-lm-reward-coeff $LM"
  [ "${NO_ADV_NORM:-0}" = "1" ] && RL_FLAGS="$RL_FLAGS --rl-no-advantage-norm"
[ "${DISCONNECT_REPRO:-0}" = "1" ] && RL_FLAGS="$RL_FLAGS --rl-disconnect-repro"
[ "${GLOBAL_LOAD:-0}" = "1" ] && RL_FLAGS="$RL_FLAGS --rl-global-load"
  [ "${CREDIT_CF:-0}" = "1" ] && RL_FLAGS="$RL_FLAGS --rl-credit-counterfactual"
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
           --exit-duration-in-mins 230 --train-iters $RL_TRAIN_ITERS \\
           --save-interval 500 --eval-interval $EVAL_INTERVAL --eval-iters $EVAL_ITERS \\
           --empty-unused-memory-level 2 \\
           --wandb-run-tags $WANDB_TAGS $DATA_OVERRIDE
     "

echo "============================================================"
echo "End: $(date)"
echo "============================================================"
