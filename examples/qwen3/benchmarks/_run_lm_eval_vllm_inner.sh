#!/bin/bash
set -uo pipefail
: "${HF_DIR:?HF_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${LIMIT:=1000}"

pip install 'lm_eval[vllm]' --quiet 2>/dev/null || pip install 'lm_eval' --quiet 2>/dev/null

# All caches → lustre or per-job TMPDIR (NOT overlay /root, which is tiny on compute nodes)
export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR=/tmp/lmeval_${SLURM_JOB_ID:-$$}
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torch_inductor
export VLLM_CACHE_ROOT=$TMPDIR/vllm
# Disable vLLM's anonymous usage stats (writes to ~/.config/vllm/usage_stats.json in tiny /root overlay)
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Just in case, point XDG_CONFIG_HOME to lustre too
export XDG_CONFIG_HOME=$TMPDIR/xdg
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$RESULTS_DIR" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CONFIG_HOME/vllm"
# Belt-and-suspenders: touch do_not_track in the place vllm checks
touch $XDG_CONFIG_HOME/vllm/do_not_track 2>/dev/null

echo "Running lm_eval --model vllm on $HF_DIR (TP=8, enforce_eager=True, no_usage_stats)"
echo "TMPDIR=$TMPDIR  XDG_CONFIG_HOME=$XDG_CONFIG_HOME"

ARGS=(
  --model vllm
  --model_args "pretrained=$HF_DIR,tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True,enforce_eager=True"
  --tasks hellaswag,arc_challenge,winogrande
  --batch_size auto
  --output_path "$RESULTS_DIR"
)
if [ -n "$LIMIT" ] && [ "$LIMIT" != "inf" ] && [ "$LIMIT" != "0" ]; then
  ARGS+=(--limit "$LIMIT")
fi
python3 -m lm_eval "${ARGS[@]}"
RC=$?
rm -rf "$TMPDIR" 2>/dev/null
exit $RC
