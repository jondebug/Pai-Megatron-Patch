#!/bin/bash
set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

if [ $ENV = dsw ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### BASE CONFIG ###
MODEL_SIZE=$2
BATCH_SIZE=$3
GLOBAL_BATCH_SIZE=$4
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
PR=$9
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=${10}
PP=${11}
CP=${12}
ETP=${13}
EP=${14}
SP=${15}
DO=${16}
FL=${17}
SFT=${18}
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${19}
OPTIMIZER_OFFLOAD=${20}
SAVE_INTERVAL=${21}
DATASET_PATH=${22}
VALID_DATASET_PATH=${23}
PRETRAIN_CHECKPOINT_PATH=${24}

# the following two values will not be used when SFT is true
TRAIN_TOKENS=${25}
WARMUP_TOKENS=${26}
###############################

OUTPUT_BASEPATH=${27}

# Capture any additional arguments (like --router-only-training)
shift 27  # Remove the first 27 positional arguments
EXTRA_ARGS="$@"  # Capture all remaining arguments

# Check if --moe-router-load-balancing-type is passed in EXTRA_ARGS to override default (aux_loss)
# Must be parsed early since moe_options uses ${MOE_LB_TYPE} during model size config
MOE_LB_TYPE="aux_loss"
prev_arg=""
for arg in ${EXTRA_ARGS}; do
    if [ "$prev_arg" = "--moe-router-load-balancing-type" ]; then
        MOE_LB_TYPE=$arg
        echo "Using --moe-router-load-balancing-type override: ${MOE_LB_TYPE}"
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--moe-router-load-balancing-type [a-z_]*//g')
        break
    fi
    prev_arg=$arg
done

### OTHERS ###


if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
        --attention-backend flash
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=" \
        --attention-backend fused
    "
fi

if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 4B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=9728
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
    moe_options=""
elif [ $MODEL_SIZE = 14B ]; then 
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=17408
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=25600
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""
elif [ $MODEL_SIZE = A3B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=32
    NUM_LAYERS=48
    INTERMEDIATE_SIZE=6144
    MOE_INTERMEDIATE_SIZE=768
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type ${MOE_LB_TYPE} \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*48)' \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
elif [ $MODEL_SIZE = A22B ]; then
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=64
    NUM_LAYERS=94
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6


    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type ${MOE_LB_TYPE} \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*94)' \
        --moe-router-pre-softmax
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
fi


# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ -z ${MP_VP} ]; then
    vp_option=""
else
    vp_option=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"


if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} "
            # --recompute-granularity full
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-recipe blockwise"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_option=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_option=" \
                    "
fi


if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
                    "
fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_option=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

# Auto-resume is handled after NAME is defined (see below, after megatron_options)

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=${25}
    LR_WARMUP_ITERS=${26}
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-qwen3-moe-megatron-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_options=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    # Check if --train-iters is passed in EXTRA_ARGS to override calculation
    TRAIN_ITERS_OVERRIDE=""
    for arg in ${EXTRA_ARGS}; do
        if [ "$prev_arg" = "--train-iters" ]; then
            TRAIN_ITERS_OVERRIDE=$arg
            break
        fi
        prev_arg=$arg
    done
    
    if [ -n "$TRAIN_ITERS_OVERRIDE" ]; then
        TRAIN_ITERS=$TRAIN_ITERS_OVERRIDE
        echo "Using --train-iters override: ${TRAIN_ITERS}"
        # Remove --train-iters from EXTRA_ARGS since we handle it explicitly
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--train-iters [0-9]*//g')
    else
        TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    fi
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} ))
    PREFIX="pretrain-mcore-qwen3-moe-megatron-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_options=" \
        --train-mode pretrain"
fi

# Check if --eval-interval is passed in EXTRA_ARGS to override default
EVAL_INTERVAL=10000
prev_arg=""
for arg in ${EXTRA_ARGS}; do
    if [ "$prev_arg" = "--eval-interval" ]; then
        EVAL_INTERVAL=$arg
        echo "Using --eval-interval override: ${EVAL_INTERVAL}"
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--eval-interval [0-9]*//g')
        break
    fi
    prev_arg=$arg
done

# Check if --eval-iters is passed in EXTRA_ARGS to override default
EVAL_ITERS=10
prev_arg=""
for arg in ${EXTRA_ARGS}; do
    if [ "$prev_arg" = "--eval-iters" ]; then
        EVAL_ITERS=$arg
        echo "Using --eval-iters override: ${EVAL_ITERS}"
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--eval-iters [0-9]*//g')
        break
    fi
    prev_arg=$arg
done

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
else 
    dataset_options=" \
        --data-path ${DATASET_PATH} \
        --split 99,1,0 \
        --dataset MMAP"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# Auto-resume: if a saved checkpoint exists from a previous allocation, load from there.
if [ -f "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]; then
    SAVED_ITER=$(cat "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    echo "AUTO-RESUME: Found saved checkpoint at iteration ${SAVED_ITER} in ${SAVED_PRETRAIN_CHECKPOINT_PATH}"
    load_option=" --load ${SAVED_PRETRAIN_CHECKPOINT_PATH}"
fi

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# Copy config files from pretrained checkpoint (skip if already resuming from the same dir)
if [ "${PRETRAIN_CHECKPOINT_PATH}" != "${SAVED_PRETRAIN_CHECKPOINT_PATH}" ]; then
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH} 2>/dev/null || true
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH} 2>/dev/null || true
fi

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 32 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen3Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --no-save-optim \
        --ckpt-format torch_dist \
        --transformer-impl transformer_engine \
        --cross-entropy-loss-fusion \
        --qk-layernorm \
        --kv-channels 128 \
        --te-rng-tracker 


        "
        #TODO: Jonathanp: removed these from args to get rl loss to work:
            # --recompute-granularity selective \
            # --recompute-modules moe
            # --external-cuda-graph \
            # --cuda-graph-scope attn \


#        --add-qkv-bias \ # no qkv bias
#        --no-rope-fusion \
#        --no-bias-swiglu-fusion \
#       --decoder-first-pipeline-num-layers 10
#         --te-rng-tracker \         --external-cuda-graph \        --cuda-graph-scope attn


run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_qwen.py \
 ${megatron_options} ${dataset_options} ${pr_options} ${load_option} ${activation_checkpoint_options} \
 ${do_option} ${sp_option} ${moe_options} ${offload_option} ${sft_options} ${vp_option} ${packing_options} ${uneven_split_option} ${attn_backend_option} ${tie_option} ${gqa_options} ${EXTRA_ARGS}"

echo ${run_cmd}
eval ${run_cmd}
TRAIN_EXIT_CODE=$?
set +x

# Post-training inline HellaSwag benchmark (if checkpoint exists)
if [ -f "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]; then
    BENCH_ITER=$(cat "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    HF_OUTPUT="${SAVED_PRETRAIN_CHECKPOINT_PATH}/hf_converted"
    BENCH_RESULTS="${SAVED_PRETRAIN_CHECKPOINT_PATH}/benchmark_results"
    ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-complete"
    CONVERTOR_DIR="$(cd "$(dirname "$0")/../../toolkits/distributed_checkpoints_convertor" && pwd)"
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

    # Skip if already benchmarked at this iteration
    if [ -f "${BENCH_RESULTS}/accuracy_summary.json" ]; then
        echo "BENCHMARK: Already benchmarked at iter ${BENCH_ITER}, skipping"
    else
        echo "============================================================"
        echo "INLINE BENCHMARK: HellaSwag at iter ${BENCH_ITER}"
        echo "============================================================"

        # Step 1: Convert checkpoint to HF format
        if ! ls "${HF_OUTPUT}"/*.safetensors 1>/dev/null 2>&1; then
            echo "  Converting Megatron -> HF..."
            export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:${PYTHONPATH:-}"
            export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4'
            export KUBERNETES_CONTAINER_RESOURCE_GPU=4
            cd "${CONVERTOR_DIR}"
            bash scripts/qwen3/run_8xH20.sh A3B \
                "${SAVED_PRETRAIN_CHECKPOINT_PATH}" "${HF_OUTPUT}" \
                true true bf16 "${ORIGINAL_HF}" 2>&1 || echo "  WARNING: Conversion failed"
            cd "${SCRIPT_DIR}"
        fi

        # Step 2: Run HellaSwag only
        if ls "${HF_OUTPUT}"/*.safetensors 1>/dev/null 2>&1; then
            mkdir -p "${BENCH_RESULTS}"
            export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
            export HF_DATASETS_CACHE="${HF_HOME}/datasets"
            mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
            pip install 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null

            echo "  Running HellaSwag (1000 samples)..."
            BENCH_START=$(date +%s)
            python3 -m lm_eval \
                --model hf \
                --model_args "pretrained=${HF_OUTPUT},trust_remote_code=True,dtype=bfloat16" \
                --tasks hellaswag \
                --batch_size 8 \
                --output_path "${BENCH_RESULTS}" \
                --device cuda:0 \
                --limit 1000 2>&1 || echo "  WARNING: lm_eval failed"
            BENCH_END=$(date +%s)
            echo "{\"total_seconds\": $((BENCH_END - BENCH_START))}" > "${BENCH_RESULTS}/timing.json"

            # Parse results
            python3 "${SCRIPT_DIR}/benchmarks/_parse_lm_eval_results.py" "${BENCH_RESULTS}" 2>&1 || true

            # Upload to WandB
            python3 -c "
import json, os, glob
results_files = glob.glob('${BENCH_RESULTS}/**/results.json', recursive=True)
if results_files:
    with open(max(results_files, key=os.path.getmtime)) as f:
        data = json.load(f)
    hs = data.get('results',{}).get('hellaswag',{})
    acc = hs.get('acc_norm,none', hs.get('acc,none'))
    if acc is not None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({'benchmark/hellaswag_accuracy': acc * 100, 'benchmark/hellaswag_iter': ${BENCH_ITER}})
                print(f'  Logged to WandB: hellaswag={acc*100:.1f}% at iter ${BENCH_ITER}')
        except Exception as e:
            print(f'  WandB log failed: {e}')
        print(f'  HellaSwag accuracy: {acc*100:.1f}%')
" 2>&1 || true

            echo "BENCHMARK COMPLETE: iter ${BENCH_ITER}"
        else
            echo "  WARNING: No HF checkpoint found, skipping benchmark"
        fi
    fi
fi

exit ${TRAIN_EXIT_CODE}
