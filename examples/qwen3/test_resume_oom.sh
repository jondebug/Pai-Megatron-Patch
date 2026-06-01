#!/bin/bash
#SBATCH --job-name=resume_test
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/resume_test_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/resume_test_%j.err

# Validate that resume-from-saved-checkpoint works with patches 1-7 applied.
# Loads 235bv2p1_norl_aux0.005_r12 (iter 500), trains to iter 530, exits.
# PASS if iter 530 reached cleanly.
# FAIL if 76 MiB Inductor OOM or 608 byte NCCL OOM at first forward.

set -uo pipefail
CONTAINER=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
TARGET_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/235bv2p1_norl_aux0.005_r12
DATASET=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore

echo "============================================================"
echo "Resume-OOM validation test"
echo "SLURM job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Target: $TARGET_DIR (resuming from iter $(cat $TARGET_DIR/checkpoint/pretrain-mcore-*/latest_checkpointed_iteration.txt 2>/dev/null))"
echo "Start: $(date)"
echo "Train until iter 530, then exit."
echo "============================================================"

srun --container-image=$CONTAINER \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir=$WORKDIR \
     bash -c "
       set -uo pipefail
       pip install wandb datasets --quiet
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
       cd $WORKDIR
       sh run_mcore_qwen3.sh dsw A22B 1 8 1e-4 1e-6 128 128 bf16 1 1 1 1 8 true true true false sel false 99999 \
         $DATASET $DATASET $PRETRAIN 1024000 10240 $TARGET_DIR \
         --router-only-training --use_rl_loss --rl-normalize-rewards --rl-use-ema-loads --rl-ppo-reeval \
         --ckpt-assume-constant-structure --ckpt-fully-parallel-save \
         --rl-algorithm ppo --rl-loss-coeff 0.5 --rl-ppo-entropy-coeff 0.01 \
         --rl-ppo-baseline-type critic --rl-critic-hidden-dims 256 \
         --rl-reward-type per_token_load_weighted --rl-reward-topn 2 \
         --rl-discount-factor 0 --moe-aux-loss-coeff 0.005 \
         --rl-ppo-epochs 1 --rl-ppo-extra-lr 0.0001 --rl-lm-reward-coeff 0 \
         --kl-loss-coeff 0 --moe-router-critical-path-topn 1 --moe-router-critical-path-alpha 0.01 \
         --train-iters 530 \
         --eval-interval 999 --eval-iters 1 \
         --empty-unused-memory-level 2
     "
EC=$?

# Pull verdict
LOG=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/resume_test_${SLURM_JOB_ID}.out
LAST_ITER=$(grep -oE "iteration[[:space:]]+[0-9]+/" $LOG 2>/dev/null | tail -1)
OOM_OUT=$(grep -cE "OutOfMemoryError|NCCL WARN Cuda failure" $LOG 2>/dev/null)
OOM_ERR=$(grep -cE "OutOfMemoryError|NCCL WARN Cuda failure" ${LOG%.out}.err 2>/dev/null)
OOM=$((OOM_OUT + OOM_ERR))
echo
echo "============================================================"
echo "Exit code: $EC"
echo "Last iter: $LAST_ITER"
echo "OOM count: $OOM"
if [ "$OOM" -gt 0 ]; then
  echo "VERDICT: RESUME FAILS (OOM hit) - chain pattern not viable"
elif echo "$LAST_ITER" | grep -qE "(52[0-9]|53[0-9])/"; then
  echo "VERDICT: RESUME WORKS - reached iter $LAST_ITER, chain pattern viable"
else
  echo "VERDICT: INCONCLUSIVE - last iter=$LAST_ITER"
fi
echo "============================================================"
