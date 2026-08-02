#!/bin/bash
#SBATCH --job-name=lmeval_hf_235b
#SBATCH --partition=interactive,polar,polar3,polar4,grizzly
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/lmeval_hf_%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/lmeval_hf_%x_%j.err
# =============================================================================
# Standalone 1-node vLLM lm-eval on an ALREADY-CONVERTED HF dir (no reconvert).
# submit_convert_and_eval_235b.sh is convert-only (exit 0 before STEP 2); this fires
# STEP 2 on the produced HF. Reuses that script's exact vLLM invocation.
#
# Required env:
#   HF_DIR        - HF model dir (118 safetensors + model.safetensors.index.json + config/tokenizer)
#   BENCHMARK_DIR - where to write results + accuracy_summary.json
# Optional:
#   RUN_NAME (logging), ITER_NUM (logging), LIMIT (empty=full; e.g. 1000),
#   TASKS (default hellaswag,arc_challenge,winogrande)
# =============================================================================
set -uo pipefail
HF_DIR="${HF_DIR:?Set HF_DIR}"
BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR}"
RUN_NAME="${RUN_NAME:-unknown}"
ITER_NUM="${ITER_NUM:-0}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-hellaswag,arc_challenge,winogrande}"
LIMIT_TAG="${LIMIT:-inf}"
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh
mkdir -p "${BENCHMARK_DIR}"
if [ -n "${LIMIT}" ]; then LIMIT_ARG="--limit ${LIMIT}"; else LIMIT_ARG=""; fi

echo "LM-EVAL (1-node vLLM TP=8, minimal args): run=${RUN_NAME} iter=${ITER_NUM} limit=${LIMIT_TAG} tasks=${TASKS}"
echo "HF=${HF_DIR}"
echo "OUT=${BENCHMARK_DIR}"
echo "Start: $(date)"

srun --ntasks=1 --nodes=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="/lustre/fsw/portfolios/nvr/users/jonathanp:/lustre/fsw/portfolios/nvr/users/jonathanp" \
     --container-workdir="${BENCHMARK_DIR}" \
     bash -c "
         set -uo pipefail
         # accelerate>=1.2.0 first: lm_eval->peft imports clear_device_cache from accelerate.utils.memory
         # (added in 1.2.0); the container ships an older accelerate, so the import fails without this.
         pip install 'accelerate>=1.2.0' --quiet 2>&1 | tail -1
         pip install 'lm_eval' --quiet 2>&1 | tail -1
         python3 -c 'from lm_eval import simple_evaluate; print(\"lm_eval import OK\")' || { echo 'lm_eval import FAILED'; exit 1; }
         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=\${HF_HOME}/datasets
         export TMPDIR=/tmp/lmeval_\${SLURM_JOB_ID:-$$}
         export TORCHINDUCTOR_CACHE_DIR=\${TMPDIR}/torch_inductor
         export VLLM_CACHE_ROOT=\${TMPDIR}/vllm
         export VLLM_NO_USAGE_STATS=1
         export DO_NOT_TRACK=1
         export XDG_CONFIG_HOME=\${TMPDIR}/xdg
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE} \${TMPDIR} \${TORCHINDUCTOR_CACHE_DIR} \${VLLM_CACHE_ROOT} \${XDG_CONFIG_HOME}/vllm
         touch \${XDG_CONFIG_HOME}/vllm/do_not_track 2>/dev/null
         # 235B (A22B): vLLM shards the MoE experts across TP=8 natively (HF parallelize=True
         # OOMs on GPU0 for 235B). Use MINIMAL vLLM model_args -- this container's lm_eval VLLM
         # wrapper rejects enforce_eager/max_model_len as kwargs, so pass only the core ones.
         python3 -m lm_eval --model vllm \
             --model_args pretrained='${HF_DIR}',tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.90,trust_remote_code=True \
             --tasks ${TASKS} ${LIMIT_ARG} \
             --batch_size auto \
             --output_path '${BENCHMARK_DIR}'
         echo \"lm-eval exit: \$?\"
         python3 -c \"
import json
from pathlib import Path
bd = Path('${BENCHMARK_DIR}')
files = sorted(bd.rglob('results*.json'))
if not files:
    print('No results file'); raise SystemExit(1)
data = json.loads(files[-1].read_text())
r = data.get('results', {})
acc = {
    'hellaswag': r.get('hellaswag', {}).get('acc_norm,none', 0),
    'arc_challenge': r.get('arc_challenge', {}).get('acc_norm,none', 0),
    'winogrande': r.get('winogrande', {}).get('acc,none', 0),
}
avg = sum(acc.values()) / 3 * 100
s = {'run_name': '${RUN_NAME}', 'iteration': ${ITER_NUM}, 'limit': '${LIMIT_TAG}',
     'benchmark_avg': round(avg, 2), **{k: round(v*100, 2) for k,v in acc.items()}}
Path('${BENCHMARK_DIR}/accuracy_summary.json').write_text(json.dumps(s, indent=2))
print('ACCURACY_SUMMARY ' + json.dumps(s))
\"
     "
echo "End: $(date)"
