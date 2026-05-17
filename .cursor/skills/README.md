# Project Skills — RL Token Routing for MoE Load Balancing

This folder teaches a remote / autorun Cursor agent how to do everything a
human researcher has done on this project: launch training sweeps, submit
single jobs, convert checkpoints, run accuracy benchmarks, find Pareto
points, run CP→latency microbenchmarks, and run vLLM end-to-end latency
benchmarks.

The agent runs with **full autonomy** — it submits SLURM jobs, polls
results, and acts on them without asking. The skills below encode the
exact commands and the failure modes we have hit before so the agent does
not re-discover them.

## Project context (must-read before any action)

Always start from these (kept in repo root):

- `CLAUDE.md` — high-level project context, dataset/checkpoint paths, sweep IDs, findings.
- `CLAUDE_CODE_CONTEXT.md` — additional working notes.

Hard infrastructure facts the agent must respect:

- **SLURM account**: `nvr_israel_rlop` (the old `nvr_israel_scne` is deprecated — never use it).
- **Partition**: `interactive` (4 h max wall-time).
- **QOS limits**: 3 concurrent jobs, 24 total GPUs across them.
- **Container (training/convert/eval)**: `/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh`.
- **Container (vLLM bench)**: `/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh`.
- **Always mount**: `$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing`.
- **Always set** `#SBATCH --cpus-per-gpu=2` on new submit scripts (default is ~31 CPUs/GPU which hogs the cluster).
- **W&B**: entity `nvr-israel`, project `qwen3-router-training`.

## Skill index

| Skill | When to use |
|---|---|
| [`launch-sweep`](launch-sweep/SKILL.md) | Create a new W&B sweep from a config JSON and start auto-chained SLURM agents. |
| [`slurm-wandb-sweep`](slurm-wandb-sweep/SKILL.md) | Sweep design + launch invariants (235B EP=8 → `--agents 1`, `save_interval < train_iters`, `fresh_start: false`, etc.). |
| [`submit-training-job`](submit-training-job/SKILL.md) | Submit one single SLURM training job (single config, no sweep). |
| [`resume-training-run`](resume-training-run/SKILL.md) | Continue a partially-trained run to its target `train_iters` (auto-resume mechanism, `wandb_run_id.txt`, cross-sweep matching). Run before launching new sweeps. |
| [`convert-mcore-to-hf`](convert-mcore-to-hf/SKILL.md) | Convert a Megatron checkpoint to HuggingFace format (30B 1-node, 235B 2-node). |
| [`run-lm-eval-benchmark`](run-lm-eval-benchmark/SKILL.md) | Run lm-evaluation-harness (HellaSwag / ARC / WinoGrande) on a checkpoint. |
| [`iter-selection-for-pareto`](iter-selection-for-pareto/SKILL.md) | Pick `i_early / i_mid / i_late` per Pareto-candidate run before benchmarking; accounts for accuracy regression with over-training. |
| [`pareto-benchmark`](pareto-benchmark/SKILL.md) | Find Pareto-optimal checkpoints from a sweep and submit benchmarks for them. |
| [`cp-microbench`](cp-microbench/SKILL.md) | Run the CP→latency microbenchmark (simulated EP step time from real routing traces). |
| [`vllm-latency-bench`](vllm-latency-bench/SKILL.md) | Run end-to-end vLLM latency benchmark (N models, multiple prompt × batch cells). |
| [`collect-benchmark-results`](collect-benchmark-results/SKILL.md) | Aggregate all lm-eval results into the master `benchmark_results.csv`. |
| [`generate-pareto-chart`](generate-pareto-chart/SKILL.md) | Generate the interactive HTML Pareto-frontier chart from the master CSV. |
| [`checkpoint-disk-management`](checkpoint-disk-management/SKILL.md) | Manage Lustre disk usage during long sweeps; identify and delete safe-to-remove checkpoints (non-frontier, far from frontier) when usage > 80%. |
| [`training-log-forensics`](training-log-forensics/SKILL.md) | Parse Megatron training logs into per-run comparison tables. |
| [`training-bug-investigation`](training-bug-investigation/SKILL.md) | Diagnose suspected silent training bugs (5-layer mechanism-vs-outcome checklist). |
| [`hypothesis-reassessment`](hypothesis-reassessment/SKILL.md) | Re-examine an earlier conclusion when challenged or invalidated by a bug fix. |

## Standard SLURM monitoring pattern

After every `sbatch`, the agent should:

```bash
JOB_ID=$(... | awk '{print $NF}')
# Poll until allocated, then babysit
for i in $(seq 1 60); do
    sleep 60
    STATE=$(squeue -j $JOB_ID -h -o %T 2>/dev/null)
    [ -z "$STATE" ] && break               # finished (or gone)
    echo "[$i min] $JOB_ID state=$STATE"
done
sacct -j $JOB_ID -o JobID,JobName,State,Elapsed,ExitCode --noheader
tail -n 200 /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/<log_dir>/<job>_${JOB_ID}.out
```

If `squeue` / `sacct` hangs (cluster congestion), fall back to tailing the
`.out` log file directly.
