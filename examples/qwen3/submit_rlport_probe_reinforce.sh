#!/bin/bash
#SBATCH --job-name=cp_rlport_probe
#SBATCH --partition=polar3
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/rlport_probe_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/rlport_probe_%j.err

# =============================================================================
# RL-PORT PROBED REINFORCE RUN (reviewer-consolidated measurement apparatus)
# =============================================================================
# The decisive "does RL lower the DETERMINISTIC CP" cell. Uses the CORRECT loss
# path: REINFORCE (per-token) + hard_gumbel_pl, so the ordered Plackett-Luce
# log-prob (rl_ordered_logprob) is actually differentiated. Pairing hard_gumbel_pl
# with PPO is now a fail-fast assert (P1) -- PPO bypasses the PL log-prob.
#
# Measurement infrastructure (all default OFF; enabled here):
#   P1  fail-fast hard_gumbel_pl->reinforce assert + [RL CONFIG BANNER]
#   P2  --rl-probe-interval/--rl-probe-batches: fixed deterministic-CP probe  -> [PROBE]
#   P3  --rl-audit-interval: frozen-rollout causal audit                      -> [AUDIT]
#   P4  churn (old-vs-new deterministic top-k) reported in [PROBE]
#
# Config (task spec): EP16/DP16/TP1/PP1, seq128, gbs16, 128 experts top-8,
#   REINFORCE + hard_gumbel_pl + candidate_pool 32 + global loads +
#   loo_smoothmax beta=0.01 + aux0 kl0 + perlayer_norm.
#
# Modes (env-driven; the SAME script serves both):
#   SMOKE :  RL_TRAIN_ITERS=8  PROBE_INTERVAL=2  PROBE_BATCHES=2 AUDIT_INTERVAL=1
#   RUN   :  RL_TRAIN_ITERS=150 PROBE_INTERVAL=10 PROBE_BATCHES=16 AUDIT_INTERVAL=1 (defaults)
#
# Launch (smoke):
#   RUN_NAME=235bv21math_rlport_probe_smoke RL_TRAIN_ITERS=8 PROBE_INTERVAL=2 \
#     PROBE_BATCHES=2 AUDIT_INTERVAL=1 sbatch submit_rlport_probe_reinforce.sh
# Launch (150-iter probed run):
#   RUN_NAME=235bv21math_rlport_probe_run sbatch submit_rlport_probe_reinforce.sh
# =============================================================================
set -euo pipefail

: "${RUN_NAME:?set RUN_NAME (e.g. 235bv21math_rlport_probe_run)}"

# --- Config (fixed for this gate; overridable via env) ---
RLC="${RLC:-0.5}"
AUX="${AUX:-0}"            # aux OFF (causal isolation)
KL="${KL:-0}"             # KL OFF (causal isolation)
GAMMA="${GAMMA:-0}"       # per-token discount over layers
REWARD_TYPE="${REWARD_TYPE:-loo_smoothmax}"
RL_TRAIN_ITERS="${RL_TRAIN_ITERS:-150}"
SEED="${SEED:-1234}"
USE_RL="${USE_RL:-1}"
PLR="${PLR:-1e-4}"
PMINLR="${PMINLR:-1e-6}"
PERLAYER_NORM="${PERLAYER_NORM:-1}"       # intended single-normalization reduction
LOO_BETA="${LOO_BETA:-0.01}"              # task: beta=0.01
# --- measurement cadence (P2/P3) ---
PROBE_INTERVAL="${PROBE_INTERVAL:-10}"
PROBE_BATCHES="${PROBE_BATCHES:-16}"
AUDIT_INTERVAL="${AUDIT_INTERVAL:-1}"
SAVE_CKPT="${SAVE_CKPT:-1}"               # 1 = write checkpoint (run_mcore's real --save path); 0 = no checkpoint
SAVE_INTERVAL="${SAVE_INTERVAL:-150}"     # save once at iter 150 (run_mcore uses --no-save-optim => model-only)
EVAL_ITERS="${EVAL_ITERS:-0}"             # eval OFF
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EXIT_MINS="${EXIT_MINS:-55}"
MATH_BLEND="${MATH_BLEND:-}"

CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
OUTPUT_BASEPATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
TARGET_RUN_DIR=$OUTPUT_BASEPATH/$RUN_NAME
DATASET_PATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CKPT="${RESUME_FROM:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore-dist}"

mkdir -p "$TARGET_RUN_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT

WANDB_TAGS="235b v21math rlport-probe router-only reinforce"

echo "============================================================"
echo "RL-PORT PROBED REINFORCE (EP=16, 2 nodes) fresh-from-pretrained"
echo "RUN_NAME=$RUN_NAME"
echo "ALGO=reinforce RLC=$RLC AUX=$AUX REWARD=$REWARD_TYPE KL=$KL GAMMA=$GAMMA SEED=$SEED TRAIN_ITERS=$RL_TRAIN_ITERS"
echo "PERLAYER_NORM=$PERLAYER_NORM LOO_BETA=$LOO_BETA"
echo "PROBE_INTERVAL=$PROBE_INTERVAL PROBE_BATCHES=$PROBE_BATCHES AUDIT_INTERVAL=$AUDIT_INTERVAL"
echo "PORT FLAGS: --rl-sampling hard_gumbel_pl --rl-candidate-pool 32 --rl-global-loads --rl-reward-type $REWARD_TYPE --rl-loo-beta $LOO_BETA"
echo "TARGET_RUN_DIR=$TARGET_RUN_DIR"
echo "LOAD (fresh): $PRETRAIN_CKPT"
echo "Master: $MASTER_ADDR:$MASTER_PORT   Start: $(date)"
echo "============================================================"

# RL flag block -- REINFORCE (NOT ppo): the per-token REINFORCE loss is the ONLY path that
# differentiates the ordered Plackett-Luce log-prob. + P2/P3 measurement flags.
if [ "$USE_RL" = "1" ]; then
  RL_FLAGS="--use_rl_loss --rl-algorithm reinforce --rl-loss-coeff $RLC \
    --rl-reward-type $REWARD_TYPE --rl-discount-factor $GAMMA \
    --rl-sampling hard_gumbel_pl --rl-candidate-pool 32 --rl-global-loads --rl-loo-beta $LOO_BETA \
    --rl-probe-interval $PROBE_INTERVAL --rl-probe-batches $PROBE_BATCHES --rl-audit-interval $AUDIT_INTERVAL"
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

# Checkpoint save toggle. SAVE_CKPT=1 (default) keeps run_mcore_qwen3.sh's real
# --save ${OUTPUT_BASEPATH}/checkpoint/${NAME} (with --no-save-optim => model-only ~470GB),
# saving at --save-interval (150 => once at the final iter) so the accuracy eval can convert
# it. SAVE_CKPT=0 appends a trailing --save= that blanks args.save (probe-only, no checkpoint).
if [ "$SAVE_CKPT" = "1" ]; then
  SAVE_OVERRIDE=""
else
  SAVE_OVERRIDE="--save="
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
           --exit-duration-in-mins $EXIT_MINS --train-iters $RL_TRAIN_ITERS \\
           --save-interval $SAVE_INTERVAL --eval-interval $EVAL_INTERVAL --eval-iters $EVAL_ITERS \\
           $SAVE_OVERRIDE \\
           --empty-unused-memory-level 2 \\
           --wandb-run-tags $WANDB_TAGS $DATA_OVERRIDE
     "

echo "============================================================"
echo "End: $(date)"
echo "============================================================"
