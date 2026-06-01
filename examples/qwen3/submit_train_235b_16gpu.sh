#!/bin/bash
#SBATCH --job-name=train_16gpu_resume
#SBATCH --partition=polar4
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/train_16gpu_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/train_16gpu_%j.err

# Test: resume an existing non-CPB partial run on 16 GPUs (EP=16, 2 nodes)
# to validate that more expert parallelism solves the resume-time OOM.
# Target run: 235b-rladv_c256_ppo_aux0.015_r06 (last iter 1282).
set -euo pipefail

CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
WORKDIR=$REPO_ROOT/examples/qwen3
TARGET_RUN_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/235b-rladv_c256_ppo_aux0.015_r06
DATASET_PATH=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CKPT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT

echo "============================================================"
echo "16-GPU RESUME TEST (EP=16, 2 nodes)"
echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Target run dir: $TARGET_RUN_DIR"
echo "Latest checkpointed iter: $(cat $TARGET_RUN_DIR/checkpoint/*/latest_checkpointed_iteration.txt 2>/dev/null)"
echo "Start: $(date)"
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
         export RANK=\${SLURM_PROCID}
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8
         export MASTER_ADDR="$MASTER_ADDR"
         export MASTER_PORT="$MASTER_PORT"
         echo \"node \$(hostname) RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE}\"
         cd $WORKDIR
         sh run_mcore_qwen3.sh dlc A22B 1 16 1e-4 1e-6 128 128 bf16 1 1 1 1 16 true true true false sel false 1500 \\
           $DATASET_PATH $DATASET_PATH $PRETRAIN_CKPT 1024000 10240 $TARGET_RUN_DIR \\
           --router-only-training --enable-wandb-logging --use_rl_loss \\
           --rl-normalize-rewards --rl-use-ema-loads --rl-ppo-reeval \\
           --ckpt-assume-constant-structure --ckpt-fully-parallel-save \\
           --wandb-project-name qwen3-router-training \\
           --wandb-run-name 235b-rladv_c256_ppo_aux0.015_r06_16gpu \\
           --rl-algorithm ppo --rl-loss-coeff 0.5 --rl-ppo-entropy-coeff 0.01 \\
           --rl-ppo-baseline-type mean --rl-reward-type per_token_load_weighted \\
           --rl-reward-topn 2 --rl-discount-factor 0 \\
           --moe-aux-loss-coeff 0.015 --rl-ppo-epochs 1 --rl-ppo-extra-lr 0.0001 \\
           --rl-lm-reward-coeff 0 --kl-loss-coeff 0 \\
           --moe-router-critical-path-topn 1 --moe-router-critical-path-alpha 0.01 \\
           --exit-duration-in-mins 230 --train-iters 1500 \\
           --eval-interval 200 --eval-iters 50 \\
           --empty-unused-memory-level 2 \\
           --wandb-run-tags 235b rl-advantage 16gpu-test resume-test
     "

echo "============================================================"
echo "End: $(date)"
echo "============================================================"
