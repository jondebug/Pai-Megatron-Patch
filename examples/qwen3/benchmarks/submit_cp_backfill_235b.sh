#!/bin/bash
#SBATCH --job-name=cp_backfill_235b
#SBATCH --partition=polar4,polar3,polar,interactive,grizzly
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/cp_backfill_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/cp_backfill_%j.err
# CP backfill: for each manifest entry lacking a CP json, convert distcp->HF (if needed),
# measure critical path, write <cell>_iter<it>_critpath.json (ingest picks it up), reap HF.
# Sequential, reap-as-you-go (disk-bounded), auto-resubmits until all done. 2 nodes (convert needs PP2/EP8).
set -uo pipefail
N=/lustre/fsw/portfolios/nvr/users/jonathanp
PROOT=$N/rl_token_routing/Pai-Megatron-Patch
REPO_ROOT=$PROOT
CONVERTOR_DIR=$PROOT/toolkits/distributed_checkpoints_convertor
QBENCH=$PROOT/examples/qwen3/benchmarks
PMP_CONTAINER=$N/containers/pai-megatron-patch_25.04.sqsh
VLLM_CONTAINER=$N/containers/vllm-openai-latest.sqsh
ORIGINAL_HF=$N/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
RESULTS=$N/rl_token_routing/lm_eval_results_ord
MANIFEST=$QBENCH/cp_backfill_manifest.json
SELF="$0"
mkdir -p "$RESULTS"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_ADDR

remaining=0
n=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))")
for i in $(seq 0 $((n-1))); do
  read CK CELL IT < <(python3 -c "import json;e=json.load(open('$MANIFEST'))[$i];print(e['checkpoint_dir'],e['run_name'],e['iteration'])")
  CPJSON="$RESULTS/${CELL}_iter${IT}_critpath.json"
  [ -f "$CPJSON" ] && { echo "[$((i+1))/$n] SKIP $CELL@$IT (cp json exists)"; continue; }
  HF="${CK}/hf_converted_iter${IT}_cp"
  echo "============ [$((i+1))/$n] $CELL @ $IT ============"
  # --- convert if HF missing ---
  if ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
    echo "converting -> $HF"
    LF="${CK}/latest_checkpointed_iteration.txt"; ORIG=""; [ -f "$LF" ] && ORIG=$(cat "$LF"); echo "$IT" > "$LF"
    mkdir -p "$HF"; MP=$(shuf -n1 -i 30000-50000); export MASTER_PORT=$MP
    srun --container-image="$PMP_CONTAINER" --container-mounts="$N:$N" --container-workdir="$CONVERTOR_DIR" \
         --nodes=2 --ntasks-per-node=1 bash -c "
           set -euo pipefail
           export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/backends/megatron/Megatron-LM-250624:$CONVERTOR_DIR/impl:\${PYTHONPATH:-}
           export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --expert-model-parallel-size 8'
           export KUBERNETES_CONTAINER_RESOURCE_GPU=8 WORLD_SIZE=2 RANK=\${SLURM_PROCID}
           export MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MP'
           bash scripts/qwen3/run_A22B_16xH20.sh A22B '$CK' '$HF' true true bf16 '$ORIGINAL_HF'
         "
    [ -n "$ORIG" ] && echo "$ORIG" > "$LF"
    ns=$(ls "$HF"/*.safetensors 2>/dev/null | wc -l)
    [ "$ns" -lt 100 ] && { echo "convert FAILED ($ns shards) — skip $CELL@$IT"; rm -rf "$HF"; continue; }
  fi
  # --- measure critical path (1 node, device_map shards 235B over 8 GPU) ---
  srun --nodes=1 --ntasks=1 --container-image="$VLLM_CONTAINER" --container-mounts="$N:$N" bash -c "
      set -uo pipefail
      pip install datasets transformers --quiet 2>/dev/null || true
      export TMPDIR=/tmp/cp_\$SLURM_JOB_ID HF_HOME=$N/rl_token_routing/.hf_cache HF_DATASETS_CACHE=$N/rl_token_routing/.hf_cache/datasets
      mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE
      python3 $QBENCH/measure_critical_path.py --model-path '$HF' --output-path '$CPJSON' --num-batches 50 --batch-size 2 --seq-length 2048 2>&1
    "
  [ -f "$CPJSON" ] && echo "CP measured: $CELL@$IT" || { echo "CP FAILED $CELL@$IT"; remaining=$((remaining+1)); }
  # --- reap HF (keep disk bounded) ---
  rm -rf "$HF" && echo "reaped HF $CELL@$IT"
done
# auto-resubmit if any entry still lacks a cp json
left=$(python3 -c "
import json,os
m=json.load(open('$MANIFEST')); R='$RESULTS'
print(sum(1 for e in m if not os.path.exists(os.path.join(R, e['run_name']+'_iter'+str(e['iteration'])+'_critpath.json'))))
")
echo "=== CP backfill pass done; $left entries remaining ==="
[ "$left" -gt 0 ] && { echo "resubmitting continuation..."; sbatch "$SELF" 2>/dev/null || true; }
