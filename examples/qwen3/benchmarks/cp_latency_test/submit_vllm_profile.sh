#!/bin/bash
#SBATCH --job-name=vllm_profile
#SBATCH --partition=polar4
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_profile_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_profile_%j.err

# vLLM MoE profiler — captures per-rank kernel trace + token-count log
# Required env (or use defaults below):
#   MODEL_PATH      HF model dir (default: pretrained 235B)
#   MODEL_NAME      short label for output filenames
#   TP_SIZE         tensor_parallel_size (default 8)
#   PROMPT_LEN      (default 256)
#   BATCH_SIZE      (default 8)
#   MAX_TOKENS      decode tokens to record (default 64; keep small)
#   NUM_RECORD     generate() calls to record (default 2)

set -uo pipefail

REPO_ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
SCRIPT_DIR=$REPO_ROOT/examples/qwen3/benchmarks/cp_latency_test
CONTAINER=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh
RESULTS_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B}"
MODEL_NAME="${MODEL_NAME:-pretrained_235b}"
TP_SIZE="${TP_SIZE:-8}"
PROMPT_LEN="${PROMPT_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
NUM_RECORD="${NUM_RECORD:-2}"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "vLLM MoE PROFILE  job=${SLURM_JOB_ID:-manual}"
echo "Model:      $MODEL_PATH"
echo "Name:       $MODEL_NAME"
echo "TP/EP:      $TP_SIZE"
echo "Prompt len: $PROMPT_LEN  bs=$BATCH_SIZE  max_tokens=$MAX_TOKENS  num_record=$NUM_RECORD"
echo "Output dir: $RESULTS_DIR/profile_${MODEL_NAME}_*"
echo "Start:      $(date)"
echo "============================================================"

srun --container-image="$CONTAINER" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="$SCRIPT_DIR" \
     bash -c "
        set -uo pipefail
        export TMPDIR=/tmp/profile_${SLURM_JOB_ID:-\$\$}
        export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/torch_inductor
        export VLLM_CACHE_ROOT=\$TMPDIR/vllm
        export TRITON_CACHE_DIR=\$TMPDIR/triton
        export XDG_CONFIG_HOME=\$TMPDIR/xdg
        export VLLM_NO_USAGE_STATS=1
        export VLLM_ALLOW_INSECURE_SERIALIZATION=1
        export DO_NOT_TRACK=1
        mkdir -p \$TMPDIR \$TORCHINDUCTOR_CACHE_DIR \$VLLM_CACHE_ROOT \$TRITON_CACHE_DIR \$XDG_CONFIG_HOME/vllm
        touch \$XDG_CONFIG_HOME/vllm/do_not_track 2>/dev/null

        python3 $SCRIPT_DIR/cp_vllm_profile.py \
            --model '$MODEL_PATH' \
            --name '$MODEL_NAME' \
            --tp-size $TP_SIZE \
            --prompt-len $PROMPT_LEN \
            --batch-size $BATCH_SIZE \
            --max-tokens $MAX_TOKENS \
            --num-record-steps $NUM_RECORD \
            --output-dir '$RESULTS_DIR'
     "

RC=$?
echo "============================================================"
echo "End: $(date)   Exit: $RC"
echo "Outputs:"
ls -la $RESULTS_DIR/profile_${MODEL_NAME}_* 2>/dev/null | head -10
echo "============================================================"
exit $RC
