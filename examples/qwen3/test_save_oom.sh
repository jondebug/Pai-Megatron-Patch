#!/bin/bash
#SBATCH --job-name=save_oom_test
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/save_oom_test_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/save_oom_test_%j.err

# Hypothesis test: save_checkpoint at iter 20 and 40 leaves residual GPU memory
# that causes 76 MiB OOM in the subsequent forward step. Patch 7 (empty_cache
# after save in training.py:~1940) should prevent this.
#
# PASS: training reaches iter 50 cleanly, 2 checkpoints saved cleanly
# FAIL: OOM at iter 21 or iter 41 in first forward after save

set -uo pipefail

CONTAINER=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
DATASET=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore
TESTDIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/save_oom_test_${SLURM_JOB_ID}

echo "============================================================"
echo "Save-OOM hypothesis test"
echo "SLURM Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "50 iters total, save_interval=20 (saves at iter 20 and 40)"
echo "Test dir (will be deleted): $TESTDIR"
echo "============================================================"
mkdir -p $TESTDIR

srun --container-image=$CONTAINER \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir=$WORKDIR \
     bash -c "
       set -uo pipefail
       pip install wandb datasets --quiet
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
       cd $WORKDIR
       sh run_mcore_qwen3.sh dsw A22B 1 8 1e-4 1e-6 128 128 bf16 1 1 1 1 8 true true true false sel false 20 \
         $DATASET $DATASET $PRETRAIN 1024000 10240 $TESTDIR \
         --router-only-training --use_rl_loss --rl-normalize-rewards --rl-use-ema-loads --rl-ppo-reeval \
         --ckpt-assume-constant-structure --ckpt-fully-parallel-save \
         --rl-algorithm ppo --rl-loss-coeff 0.5 --rl-ppo-entropy-coeff 0.01 \
         --rl-ppo-baseline-type critic --rl-critic-hidden-dims 256 \
         --rl-reward-type per_token_load_weighted --rl-reward-topn 2 \
         --rl-discount-factor 0 --moe-aux-loss-coeff 0.005 \
         --rl-ppo-epochs 1 --rl-ppo-extra-lr 0.0001 --rl-lm-reward-coeff 0 \
         --kl-loss-coeff 0 --moe-router-critical-path-topn 1 --moe-router-critical-path-alpha 0.01 \
         --exit-duration-in-mins 60 --train-iters 50 \
         --eval-interval 999 --eval-iters 1 \
         --empty-unused-memory-level 2
     "
EXIT_CODE=$?

echo
echo "============================================================"
echo "Training exited with code: $EXIT_CODE  at $(date)"
echo "============================================================"

# Pull verdict from the slurm .out itself (we have set -uo not -e, no fancy log search)
THIS_LOG=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/save_oom_test_${SLURM_JOB_ID}.out
LAST_ITER=$(grep -oE "iteration[[:space:]]+[0-9]+/" $THIS_LOG 2>/dev/null | tail -1)
OOM=$(grep -cE "OutOfMemoryError|NCCL WARN Cuda failure" $THIS_LOG 2>/dev/null)
SAVES=$(grep -cE "save-checkpoint" $THIS_LOG 2>/dev/null)
echo "Last iter line   : $LAST_ITER"
echo "OOM count        : $OOM"
echo "save-checkpoint  : $SAVES (expect 2 from saves at iter 20 and 40)"
if [ "$OOM" -gt 0 ]; then
  echo "VERDICT: FAIL (save-OOM still present)"
elif echo "$LAST_ITER" | grep -qE "[4-5][0-9]/"; then
  echo "VERDICT: PASS (training reached iter $LAST_ITER cleanly)"
else
  echo "VERDICT: INCONCLUSIVE (last iter=$LAST_ITER)"
fi

# Cleanup test dir
echo "Deleting test dir: $TESTDIR"
rm -rf $TESTDIR

echo "============================================================"
echo "End: $(date)"
exit $EXIT_CODE
