#!/bin/bash
#SBATCH --job-name=convert_235b
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=939368M
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/convert_235b_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/convert_235b_%j.err

# ============================================================
# Convert Qwen3-235B-A22B from HuggingFace to Megatron format
# Usage: sbatch convert_235b.sh
# ============================================================

REPO="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONVERTOR_DIR="${REPO}/toolkits/model_checkpoints_convertor/qwen"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"

HF_CKPT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B"
MCORE_CKPT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore"

# Target parallelism for the saved checkpoint.
TP=1
PP=1
ETP=1
EP=8

echo "============================================================"
echo "Qwen3-235B-A22B HF → Megatron Conversion"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "HF ckpt:   ${HF_CKPT}"
echo "Mcore out: ${MCORE_CKPT}"
echo "Target:    TP=${TP} PP=${PP} ETP=${ETP} EP=${EP}"
echo "Start:     $(date)"
echo "============================================================"

if [ ! -d "${HF_CKPT}" ]; then
    echo "ERROR: HF checkpoint not found at ${HF_CKPT}"
    echo "Download first:"
    echo "  ~/.local/bin/huggingface-cli download Qwen/Qwen3-235B-A22B --local-dir ${HF_CKPT}"
    exit 1
fi

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${CONVERTOR_DIR}" \
     bash -c "
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${REPO}:${REPO}/backends/megatron/Megatron-LM-250624:\${PYTHONPATH:-}

cd ${CONVERTOR_DIR}

MASTER_PORT=\$(shuf -n 1 -i 10000-65535)

echo '[CONVERT] Starting torchrun at \$(date)'
echo '[CONVERT] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}'
echo '[CONVERT] PYTHONPATH=\${PYTHONPATH}'

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    --master_addr localhost --master_port \${MASTER_PORT} \
    hf2mcore_qwen2_moe.py \
    --load ${HF_CKPT} \
    --save ${MCORE_CKPT} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --num-layers 94 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --max-position-embeddings 10 \
    --max-padding-length 10 \
    --seq-length 10 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen3Tokenizer \
    --extra-vocab-size 293 \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --rotary-base 1000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --transformer-impl transformer_engine \
    --attention-backend fused \
    --dist-ckpt-strictness ignore_all \
    --qk-layernorm \
    --kv-channels 128 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk 8 \
    --num-experts 128 \
    --target-expert-tensor-parallel-size ${ETP} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq '([1]*94)' \
    --moe-router-pre-softmax \
    --bf16 \
    --use-cpu-initialization \
    --group-query-attention \
    --num-query-groups 4 \
    --untie-embeddings-and-output-weights

echo '[CONVERT] Done at \$(date)'
"

EXIT_CODE=$?

echo "============================================================"
echo "Conversion finished. Exit code: ${EXIT_CODE}"
echo "End: $(date)"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Checkpoint saved to: ${MCORE_CKPT}"
    echo "Next: sbatch test_235b_smoke.sh"
else
    echo "FAILED — check logs above"
fi
echo "============================================================"

exit ${EXIT_CODE}
