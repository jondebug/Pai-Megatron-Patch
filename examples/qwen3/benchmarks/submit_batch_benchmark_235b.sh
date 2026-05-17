#!/bin/bash
#SBATCH --job-name=bench_batch_235b
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/bench_batch_235b_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs/bench_batch_235b_%j.err

# Batch benchmark for Qwen3-235B-A22B: runs multiple checkpoints sequentially.
# Adapted from submit_batch_benchmark.sh for the 235B model (EP=8, 8 GPUs, A22B).
#
# Usage: sbatch submit_batch_benchmark_235b.sh --manifest /path/to/manifest.json [--limit 1000]

set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
SCRIPT_DIR="${REPO_ROOT}/examples/qwen3/benchmarks"
CONVERTOR_DIR="${REPO_ROOT}/toolkits/distributed_checkpoints_convertor"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
ORIGINAL_HF="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B"

MANIFEST=""
LIMIT=""
TASKS="hellaswag,arc_challenge,winogrande"
WANDB_PROJECT="qwen3-router-training"
BATCH_SIZE=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)       MANIFEST="$2"; shift 2 ;;
        --limit)          LIMIT="$2"; shift 2 ;;
        --tasks)          TASKS="$2"; shift 2 ;;
        --wandb-project)  WANDB_PROJECT="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        *)                shift ;;
    esac
done

if [ -z "${MANIFEST}" ] || [ ! -f "${MANIFEST}" ]; then
    echo "Error: --manifest is required and must point to a valid JSON file"
    exit 1
fi

mkdir -p /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs

TOTAL=$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))))")
echo "============================================================"
echo "BATCH BENCHMARK 235B: ${TOTAL} checkpoints"
echo "SLURM Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Manifest:        ${MANIFEST}"
echo "Tasks:           ${TASKS}"
echo "Limit:           ${LIMIT:-full}"
echo "Original HF:     ${ORIGINAL_HF}"
echo "Start Time:      $(date)"
echo "============================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     --container-workdir="${SCRIPT_DIR}" \
     bash -c "
         pip install wandb 'lm_eval' 'accelerate>=1.2.0' --quiet 2>/dev/null

         export HF_HOME=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/.hf_cache
         export HF_DATASETS_CACHE=\${HF_HOME}/datasets
         mkdir -p \${HF_HOME} \${HF_DATASETS_CACHE}
         export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:\${PYTHONPATH:-}
         # 235B uses EP=8 (matches training config and checkpoint sharding)
         export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 8'
         export KUBERNETES_CONTAINER_RESOURCE_GPU=8

         python3 -c \"
import json, subprocess, os, sys, glob

manifest = json.load(open('${MANIFEST}'))
total = len(manifest)
limit = '${LIMIT}' or None
tasks = '${TASKS}'
batch_size = '${BATCH_SIZE}'
wandb_project = '${WANDB_PROJECT}'

for idx, entry in enumerate(manifest):
    ckpt_dir = entry['checkpoint_dir']
    run_name = entry.get('run_name', 'unknown')
    wandb_run_id = entry.get('wandb_run_id', '')
    bench_iter = entry.get('iteration', None)
    
    limit_tag = f'_limit{limit}' if limit else '_full'
    if bench_iter and str(bench_iter) != 'final':
        hf_output = os.path.join(ckpt_dir, f'hf_converted_iter{bench_iter}')
        results_dir = os.path.join(ckpt_dir, f'benchmark_iter{bench_iter}{limit_tag}')
    else:
        hf_output = os.path.join(ckpt_dir, 'hf_converted')
        results_dir = os.path.join(ckpt_dir, f'benchmark_latest{limit_tag}')
    os.makedirs(results_dir, exist_ok=True)
    
    summary_file = os.path.join(results_dir, 'accuracy_summary.json')
    if os.path.exists(summary_file):
        iter_display = bench_iter if bench_iter else 'latest'
        print(f'[{idx+1}/{total}] SKIP {run_name} (iter={iter_display}, {limit_tag}): already benchmarked')
        continue
    
    print(f'')
    print(f'============================================================')
    iter_display = bench_iter if bench_iter else 'latest'
    print(f'[{idx+1}/{total}] {run_name} (iter={iter_display})')
    print(f'============================================================')
    
    # Point converter at the right iteration
    latest_iter_file = os.path.join(ckpt_dir, 'latest_checkpointed_iteration.txt')
    original_latest = None
    if bench_iter and str(bench_iter) != 'final':
        iter_dir = os.path.join(ckpt_dir, f'iter_{int(bench_iter):07d}')
        if not os.path.isdir(iter_dir):
            print(f'  WARNING: iter dir {iter_dir} not found, skipping')
            continue
        if os.path.exists(latest_iter_file):
            original_latest = open(latest_iter_file).read().strip()
        with open(latest_iter_file, 'w') as f:
            f.write(str(bench_iter))
    
    # Step 1: Convert Megatron -> HF (235B uses A22B, EP=8)
    safetensors = glob.glob(os.path.join(hf_output, '*.safetensors'))
    if not safetensors:
        print(f'  Converting Megatron -> HF (A22B, EP=8)...')
        os.chdir('${CONVERTOR_DIR}')
        rc = os.system(
            f'bash scripts/qwen3/run_A22B_16xH20.sh A22B '
            f'\\\"{ckpt_dir}\\\" \\\"{hf_output}\\\" '
            f'true true bf16 \\\"${ORIGINAL_HF}\\\"'
        )
        os.chdir('${SCRIPT_DIR}')
        if rc != 0:
            print(f'  WARNING: Conversion failed, skipping')
            if original_latest is not None:
                with open(latest_iter_file, 'w') as f:
                    f.write(original_latest)
            continue
        safetensors = glob.glob(os.path.join(hf_output, '*.safetensors'))
    
    if original_latest is not None:
        with open(latest_iter_file, 'w') as f:
            f.write(original_latest)
    
    if not safetensors:
        print(f'  WARNING: No HF checkpoint, skipping')
        continue
    
    # Step 2: Run lm_eval with parallelize=True (multi-GPU model loading for 235B)
    limit_display = limit if limit else 'full'
    print(f'  Running lm_eval (tasks={tasks}, limit={limit_display}, parallelize=True)...')
    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', 'hf',
        '--model_args', f'pretrained={hf_output},trust_remote_code=True,dtype=bfloat16,parallelize=True',
        '--tasks', tasks,
        '--batch_size', str(batch_size),
        '--output_path', results_dir,
    ]
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'  WARNING: lm_eval failed')
        continue
    
    # Step 3: Parse results
    subprocess.run([sys.executable, '${SCRIPT_DIR}/_parse_lm_eval_results.py', results_dir])
    
    # Step 4: Log to WandB
    if wandb_run_id:
        try:
            result_files = glob.glob(os.path.join(results_dir, '**', 'results.json'), recursive=True)
            if result_files:
                data = json.load(open(max(result_files, key=os.path.getmtime)))
                results = data.get('results', {})
                benchmarks = {
                    'hellaswag': 'acc_norm,none',
                    'arc_challenge': 'acc_norm,none',
                    'winogrande': 'acc,none',
                }
                scores = {}
                for task_key, metric_key in benchmarks.items():
                    task_result = results.get(task_key, {})
                    score = task_result.get(metric_key)
                    if score is not None:
                        scores[f'benchmark/{task_key}'] = score * 100
                if scores:
                    avg = sum(scores.values()) / len(scores)
                    scores['benchmark/average'] = avg
                    import wandb
                    run = wandb.init(project=wandb_project, id=wandb_run_id, resume='must')
                    wandb.log(scores)
                    for k, v in scores.items():
                        wandb.run.summary[k] = v
                    wandb.finish()
                    print(f'  Logged to WandB: avg={avg:.1f}%')
        except Exception as e:
            print(f'  WandB logging failed: {e}')
    
    print(f'  DONE: {run_name}')

print(f'')
print(f'============================================================')
print(f'Batch benchmark 235B complete.')
print(f'============================================================')
\"
     "

echo ""
echo "============================================================"
echo "Batch benchmark 235B finished."
echo "End Time: $(date)"
echo "============================================================"
