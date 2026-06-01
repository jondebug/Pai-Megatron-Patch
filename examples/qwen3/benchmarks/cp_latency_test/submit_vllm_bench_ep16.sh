#!/bin/bash
#SBATCH --job-name=cp_vllm_bench_ep16
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_bench_ep16_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_bench_ep16_%j.err

# =============================================================================
# CP→Latency End-to-End vLLM Benchmark
#
# Boots vLLM with TP=8 + expert parallelism (16 experts/GPU) and times real
# generation on a fixed prompt set for both models.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks/cp_latency_test"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
RESULTS_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

BASELINE_MODEL="${BASELINE_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete}"
TRAINED_MODEL="${TRAINED_MODEL:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/pareto_g0_c256_ppo_aux0.01_cpb_n1_a0.01_r80/checkpoint/pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-5000-wi-10/hf_converted_iter2000}"
BASELINE_NAME="${BASELINE_NAME:-pretrained_cp4780}"
TRAINED_NAME="${TRAINED_NAME:-r80_iter2000_cp2659}"
TP_SIZE="${TP_SIZE:-16}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,1024}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32}"
MAX_TOKENS="${MAX_TOKENS:-256}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
OUTPUT_PATH="${RESULTS_DIR}/vllm_${BASELINE_NAME}_vs_${TRAINED_NAME}_${TAG}.json"

echo "============================================================"
echo "CP→LATENCY vLLM BENCHMARK"
echo "============================================================"
echo "SLURM Job ID:   ${SLURM_JOB_ID:-manual}"
echo "Baseline model: ${BASELINE_MODEL}"
echo "Trained model:  ${TRAINED_MODEL}"
echo "TP/EP size:     ${TP_SIZE}"
echo "Prompt lengths: ${PROMPT_LENGTHS}"
echo "Batch sizes:    ${BATCH_SIZES}"
echo "Max tokens:     ${MAX_TOKENS}"
echo "Trials:         ${NUM_TRIALS}"
echo "Output:         ${OUTPUT_PATH}"
echo "Start: $(date)"
echo "============================================================"

# Multi-node coordination (TP=16 spans 2 nodes)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -n 1 -i 30000-50000)
export MASTER_ADDR MASTER_PORT
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     --nodes=2 --ntasks-per-node=1 \
     bash -c "
         set -euo pipefail
         # Multi-node coords (inner shell needs them)
         export MASTER_ADDR=\$MASTER_ADDR
         export MASTER_PORT=\$MASTER_PORT
         # vLLM Ray multi-node backend
         export VLLM_USE_RAY_SPMD_WORKER=1
         # NCCL / RDMA tuning
         export NCCL_SOCKET_IFNAME=^docker,lo
         export NCCL_DEBUG=WARN

         # Install vLLM if not present. Pin to a Qwen3-MoE-friendly version.
         python3 -c 'import vllm' 2>/dev/null || pip install --quiet 'vllm>=0.7,<0.9' 'transformers>=4.51'

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         export VLLM_WORKER_MULTIPROC_METHOD=spawn
         # Same safeguards as our lm-eval inner script (escape \$ so inner shell expands, not outer)
         export VLLM_NO_USAGE_STATS=1
         export DO_NOT_TRACK=1
         export TMPDIR=/tmp/vllmbench_\${SLURM_JOB_ID:-\$\$}
         export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/torch_inductor
         export VLLM_CACHE_ROOT=\$TMPDIR/vllm
         export XDG_CONFIG_HOME=\$TMPDIR/xdg
         export TRITON_CACHE_DIR=\$TMPDIR/triton
         mkdir -p \$TMPDIR \$TORCHINDUCTOR_CACHE_DIR \$VLLM_CACHE_ROOT \$XDG_CONFIG_HOME/vllm \$TRITON_CACHE_DIR
         touch \$XDG_CONFIG_HOME/vllm/do_not_track 2>/dev/null
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         python3 ${SCRIPT_DIR}/cp_vllm_bench.py \
             --baseline-model  '${BASELINE_MODEL}' \
             --trained-model   '${TRAINED_MODEL}' \
             --baseline-name   '${BASELINE_NAME}' \
             --trained-name    '${TRAINED_NAME}' \
             --output          '${OUTPUT_PATH}' \
             --tp-size         ${TP_SIZE} \
             --prompt-lengths  '${PROMPT_LENGTHS}' \
             --batch-sizes     '${BATCH_SIZES}' \
             --max-tokens      ${MAX_TOKENS} \
             --num-trials      ${NUM_TRIALS}
     "

EXIT=$?
echo "============================================================"
echo "End: $(date)   Exit: $EXIT"
echo "Results JSON: ${OUTPUT_PATH}"
echo "============================================================"
exit $EXIT
