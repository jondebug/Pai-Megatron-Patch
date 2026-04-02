#!/bin/bash
#SBATCH --job-name=qwen3_235b_smoke
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/235b_smoke_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/235b_smoke_%j.err

# ============================================================
# Qwen3-235B-A22B Router-Only Training Smoke Test
# ============================================================
# Prerequisites:
#   1. Download HF checkpoint:
#      huggingface-cli download Qwen/Qwen3-235B-A22B \
#        --local-dir /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
#
#   2. Convert HF → Megatron (run ONCE, takes ~30-60 min on CPU):
#      cd /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
#      bash hf2mcore_qwen3_convertor.sh \
#        A22B \
#        /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B \
#        /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore \
#        1 1 1 8 bf16 false
#        ^TP ^PP ^ETP ^EP
#
#   3. Then submit this script:
#      sbatch test_235b_smoke.sh
# ============================================================

CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
WORKDIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3"
CKPT_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore"
DATASET_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document"
OUTPUT_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/235b_smoke_test"

echo "============================================================"
echo "Qwen3-235B-A22B Router-Only Smoke Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 8"
echo "Config: EP=8, TP=1, PP=1"
echo "Checkpoint: ${CKPT_PATH}"
echo "Start: $(date)"
echo "============================================================"

# Verify checkpoint exists
if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: Megatron checkpoint not found at ${CKPT_PATH}"
    echo "Run the conversion step first (see instructions above)."
    exit 1
fi

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${WORKDIR}" \
     bash -c "
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
         pip install wandb datasets --quiet 2>/dev/null

         cd ${WORKDIR}

         bash run_mcore_qwen3.sh \
             dsw \
             A22B \
             1 \
             8 \
             1e-4 \
             1e-6 \
             128 \
             128 \
             bf16 \
             1 \
             1 \
             1 \
             1 \
             8 \
             true \
             true \
             true \
             false \
             sel \
             false \
             50 \
             ${DATASET_PATH} \
             ${DATASET_PATH} \
             ${CKPT_PATH} \
             1024000 \
             10240 \
             ${OUTPUT_DIR} \
             --router-only-training \
             --use_rl_loss \
             --rl-algorithm ppo \
             --rl-ppo-baseline-type critic \
             --rl-reward-type per_token_load_weighted \
             --rl-loss-coeff 0.5 \
             --rl-normalize-rewards \
             --rl-use-ema-loads \
             --rl-reward-topn 2 \
             --rl-discount-factor 0 \
             --rl-ppo-entropy-coeff 0.01 \
             --rl-ppo-reeval \
             --rl-ppo-epochs 1 \
             --rl-lm-reward-coeff 1.0 \
             --moe-aux-loss-coeff 0.01 \
             --train-iters 20 \
             --eval-interval 10 \
             --eval-iters 2 \
             --exit-duration-in-mins 230 \
             --ckpt-assume-constant-structure \
             --ckpt-fully-parallel-save \
             --empty-unused-memory-level 2
     "

EXIT_CODE=$?

echo "============================================================"
echo "Smoke test finished. Exit code: ${EXIT_CODE}"
echo "End: $(date)"
echo "============================================================"

exit ${EXIT_CODE}
