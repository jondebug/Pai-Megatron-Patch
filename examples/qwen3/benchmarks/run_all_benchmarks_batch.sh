#!/bin/bash
#SBATCH --job-name=benchmarks_all
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_scne
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmarks_all_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/benchmarks_all_%j.err

# =============================================================================
# Run lm-eval benchmarks on multiple checkpoints sequentially in one SLURM job.
# Each checkpoint uses all GPUs via accelerate for faster evaluation.
# =============================================================================
set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
TASKS="hellaswag,arc_challenge,winogrande"
BATCH_SIZE=8
BASE="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CKPT_SUB="pretrain-mcore-qwen3-moe-megatron-A3B-lr-1e-4-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-10"
PRETRAIN_CKPT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore"

# List of runs to benchmark (all ptload from ic7dzzax sweep + norl baseline + pretrained)
RUNS=(
    "ptload_n2_rlc0.1_norm_PPO_ent0.01_c256_ema_aux0.01_r02"
    "ptload_n2_rlc0.1_norm_PPO_ent0.01_c256_ema_lm1.0_aux0.01_r03"
    "ptload_n2_rlc0.5_norm_PPO_ent0.01_c256_ema_aux0.01_r06"
    "ptload_n2_rlc0.5_norm_PPO_ent0.01_c256_ema_lm1.0_aux0.01_r07"
    "ptload_n2_rlc1.0_norm_PPO_ent0.01_c256_ema_aux0.01_r10"
    "ptload_n2_rlc1.0_norm_PPO_ent0.01_c256_ema_lm1.0_aux0.01_r11"
    "norl_aux0.01_r12"
)

# Comments for each run (matched by index)
COMMENTS=(
    "RL+aux rlc=0.1, load-weighted reward, EMA loads"
    "RL+aux+LM_reward rlc=0.1, tests LM reward protection"
    "RL+aux rlc=0.5, moderate RL strength"
    "RL+aux+LM_reward rlc=0.5, best Pareto balance"
    "RL+aux rlc=1.0, strongest RL, best critical path"
    "RL+aux+LM_reward rlc=1.0, strong RL with LM protection"
    "Aux-only baseline, no RL"
)

echo "============================================================"
echo "BATCH BENCHMARK - ${#RUNS[@]} checkpoints"
echo "SLURM Job ID: ${SLURM_JOB_ID:-manual}"
echo "Tasks: ${TASKS}"
echo "Start: $(date)"
echo "============================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${REPO_ROOT}/examples/qwen3/benchmarks" \
     bash -c "
         pip install wandb 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache/datasets
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}

         RUNS_STR='${RUNS[@]}'

         for run in \${RUNS_STR}; do
             HF_MODEL='${BASE}/'\${run}'/checkpoint/${CKPT_SUB}/hf_converted'
             RESULTS_DIR='${BASE}/'\${run}'/checkpoint/${CKPT_SUB}/benchmark_results'
             mkdir -p \${RESULTS_DIR}

             echo ''
             echo '============================================================'
             echo \"BENCHMARKING: \${run}\"
             echo \"  Model: \${HF_MODEL}\"
             echo \"  Tasks: ${TASKS}\"
             echo '============================================================'

             if [ ! -d \"\${HF_MODEL}\" ] || ! ls \${HF_MODEL}/*.safetensors 1>/dev/null 2>&1; then
                 echo \"ERROR: No HF checkpoint found at \${HF_MODEL}, skipping\"
                 continue
             fi

             # Run each task separately to get per-task timing
             TOTAL_START=\$(date +%s)
             TIMING_JSON='{\"per_task\": {'
             FIRST_TASK=true

             for task in hellaswag arc_challenge winogrande; do
                 echo \"  Running task: \${task}\"
                 TASK_START=\$(date +%s)

                 python3 -m lm_eval \
                     --model hf \
                     --model_args \"pretrained=\${HF_MODEL},trust_remote_code=True,dtype=bfloat16,device_map=auto\" \
                     --tasks \${task} \
                     --batch_size ${BATCH_SIZE} \
                     --output_path \${RESULTS_DIR} \
                     --limit 1000

                 TASK_END=\$(date +%s)
                 TASK_ELAPSED=\$((\${TASK_END} - \${TASK_START}))
                 echo \"  Task \${task}: \${TASK_ELAPSED} seconds\"

                 if [ \"\${FIRST_TASK}\" = true ]; then
                     TIMING_JSON=\"\${TIMING_JSON}\\\"\${task}\\\": \${TASK_ELAPSED}\"
                     FIRST_TASK=false
                 else
                     TIMING_JSON=\"\${TIMING_JSON}, \\\"\${task}\\\": \${TASK_ELAPSED}\"
                 fi
             done

             TOTAL_END=\$(date +%s)
             TOTAL_ELAPSED=\$((\${TOTAL_END} - \${TOTAL_START}))
             TIMING_JSON=\"\${TIMING_JSON}}, \\\"total_seconds\\\": \${TOTAL_ELAPSED}}\"

             echo \"\${TIMING_JSON}\" > \${RESULTS_DIR}/timing.json
             echo \"BENCHMARK_TIME: \${run} \${TOTAL_ELAPSED} seconds\"
             echo \"lm-eval complete for: \${run}\"
             echo ''

             # Parse results and update CSV after each run
             python3 '${REPO_ROOT}/examples/qwen3/benchmarks/_parse_lm_eval_results.py' \${RESULTS_DIR} || true
             python3 '${REPO_ROOT}/examples/qwen3/benchmarks/collect_benchmark_results.py' || true
         done

         # --- Pretrained baseline (original Qwen3 weights, no training) ---
         echo ''
         echo '============================================================'
         echo 'BENCHMARKING: pretrained_baseline (original Qwen3-30B-A3B)'
         echo '============================================================'
         PRETRAIN_HF='/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete'
         PRETRAIN_RESULTS='${BASE}/pretrained_baseline/benchmark_results'
         mkdir -p \${PRETRAIN_RESULTS}

         TOTAL_START=\$(date +%s)
         TIMING_JSON='{\"per_task\": {'
         FIRST_TASK=true

         for task in hellaswag arc_challenge winogrande; do
             echo \"  Running task: \${task}\"
             TASK_START=\$(date +%s)
             python3 -m lm_eval \
                 --model hf \
                 --model_args \"pretrained=\${PRETRAIN_HF},trust_remote_code=True,dtype=bfloat16,device_map=auto\" \
                 --tasks \${task} \
                 --batch_size ${BATCH_SIZE} \
                 --output_path \${PRETRAIN_RESULTS} \
                 --limit 1000
             TASK_END=\$(date +%s)
             TASK_ELAPSED=\$((\${TASK_END} - \${TASK_START}))
             echo \"  Task \${task}: \${TASK_ELAPSED} seconds\"
             if [ \"\${FIRST_TASK}\" = true ]; then
                 TIMING_JSON=\"\${TIMING_JSON}\\\"\${task}\\\": \${TASK_ELAPSED}\"
                 FIRST_TASK=false
             else
                 TIMING_JSON=\"\${TIMING_JSON}, \\\"\${task}\\\": \${TASK_ELAPSED}\"
             fi
         done
         TOTAL_END=\$(date +%s)
         TOTAL_ELAPSED=\$((\${TOTAL_END} - \${TOTAL_START}))
         TIMING_JSON=\"\${TIMING_JSON}}, \\\"total_seconds\\\": \${TOTAL_ELAPSED}}\"
         echo \"\${TIMING_JSON}\" > \${PRETRAIN_RESULTS}/timing.json
         echo \"BENCHMARK_TIME: pretrained_baseline \${TOTAL_ELAPSED} seconds\"
         python3 '${REPO_ROOT}/examples/qwen3/benchmarks/_parse_lm_eval_results.py' \${PRETRAIN_RESULTS} || true
         python3 '${REPO_ROOT}/examples/qwen3/benchmarks/collect_benchmark_results.py' || true

         echo ''
         echo '============================================================'
         echo 'ALL LM-EVAL BENCHMARKS COMPLETE'
         echo '============================================================'
     "

# =============================================================================
# Phase 2: Megatron EP latency measurement (uses EP=4, measures per-iteration time)
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: EP Latency Measurement (EP=4, 4 GPUs)"
echo "============================================================"

DATASET_PATH="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document"
PRETRAIN_BASE="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore"
EP_EVAL_ITERS=50

for run in "${RUNS[@]}"; do
    CKPT_DIR="${BASE}/${run}/checkpoint/${CKPT_SUB}"

    if [ ! -d "${CKPT_DIR}" ]; then
        echo "SKIP: No checkpoint at ${CKPT_DIR}"
        continue
    fi

    echo ""
    echo "EP LATENCY: ${run}"
    EP_START=$(date +%s)

    # Run Megatron eval-only: train-iters=1 triggers immediate eval at step 0
    srun --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
         --container-workdir="${REPO_ROOT}/examples/qwen3" \
         --ntasks=1 --gpus-per-task=4 \
         bash -c "
             cd '${REPO_ROOT}/examples/qwen3'
             torchrun --nproc_per_node 4 --nnodes 1 --master_port 29501 pretrain_qwen.py \
                 --num-layers 48 --hidden-size 2048 --num-attention-heads 32 --ffn-hidden-size 6144 \
                 --seq-length 128 --max-position-embeddings 40960 --max-padding-length 128 \
                 --micro-batch-size 1 --global-batch-size 4 \
                 --train-iters 1 --eval-interval 1 --eval-iters ${EP_EVAL_ITERS} \
                 --lr 1e-6 --min-lr 1e-6 --lr-decay-style cosine --weight-decay 0.01 \
                 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 \
                 --bf16 --attention-backend flash \
                 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
                 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 \
                 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 \
                 --moe-grouped-gemm --moe-token-dispatcher-type alltoall \
                 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0 \
                 --moe-layer-freq '([1]*48)' \
                 --extra-vocab-size 293 --patch-tokenizer-type Qwen3Tokenizer \
                 --swiglu --normalization RMSNorm --norm-epsilon 1e-6 \
                 --use-rotary-position-embeddings --position-embedding-type rope --disable-bias-linear --rotary-base 1000000 \
                 --qk-layernorm --kv-channels 128 \
                 --group-query-attention --num-query-groups 4 \
                 --untie-embeddings-and-output-weights \
                 --use-distributed-optimizer --no-save-optim --no-load-optim --no-load-rng \
                 --ckpt-format torch_dist --transformer-impl transformer_engine \
                 --recompute-activations \
                 --data-path '${DATASET_PATH}' --split 99,1,0 --dataset MMAP \
                 --load '${CKPT_DIR}' \
                 --save '/tmp/ep_latency_dummy_${SLURM_JOB_ID}' --save-interval 999999 \
                 --log-interval 1 --log-throughput \
                 2>&1 | tee '${BASE}/${run}/checkpoint/${CKPT_SUB}/benchmark_results/ep_latency.log'
         " || echo "EP eval failed for ${run}"

    EP_END=$(date +%s)
    EP_ELAPSED=$((EP_END - EP_START))
    echo "EP LATENCY TIME: ${run} = ${EP_ELAPSED} seconds for ${EP_EVAL_ITERS} eval iters"
done

echo ""
echo "============================================================"
echo "ALL BENCHMARKS (accuracy + EP latency) COMPLETE"
echo "Batch benchmark finished: $(date)"
echo "============================================================"

echo ""
echo "============================================================"
echo "Batch benchmark finished: $(date)"
echo "============================================================"
