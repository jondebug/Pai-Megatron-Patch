---
name: accuracy-benchmark
description: Benchmark a Qwen3 MoE router-training checkpoint for accuracy (HellaSwag/ARC-C/WinoGrande acc_norm) by converting mcore/distcp → HF then running lm-eval. Use when you need accuracy numbers for Pareto points, or to evaluate new training checkpoints. Covers the 235B manifest-driven batch pipeline, the NRT vLLM lm-eval path, and the convert gotchas.
---

# Accuracy benchmarking (mcore/distcp checkpoint → accuracy)

A Pareto point needs **two** numbers: **accuracy** (this skill) and **critical path / CP**
(`eval_crit_path`, which comes from *training* wandb `num_tokens_on_critical_path`, NOT from
lm-eval). lm-eval gives accuracy only.

## Pipeline
`distcp/mcore checkpoint → HF convert → lm_eval (acc_norm) → benchmark_avg → CSV → Pareto plot`

- Tasks: `hellaswag,arc_challenge,winogrande`, metric `acc_norm` (winogrande uses `acc`).
- Run **both** `--limit 1000` (fast, for quick frontier checks) and `--limit inf` / full
  (canonical; full datasets read ~3–4 pp higher than limit=1000 — don't mix them on one plot).
- **lm_loss proxy:** `acc ≈ 95.17 − 7.59·lm_loss` (Pearson r = −0.995). Use to triage which
  checkpoints are worth a full eval.

## 235B (A22B) — manifest-driven batch (ORD canonical pipeline)
`examples/qwen3/benchmarks/submit_batch_benchmark_235b.sh --manifest m.json [--limit 1000]`
- Single-node, **EP=8** (must match the checkpoint's 8 expert shards), 8×GPU.
- Converts distcp→HF via `toolkits/distributed_checkpoints_convertor` `run_A22B_*.sh`.
- Manifest = JSON list of `{checkpoint_dir, run_name, wandb_run_id, iteration}` where
  `checkpoint_dir` is the `...ti-<N>-wi-<w>` dir that *contains* `iter_<NNNNNNN>`. The script
  sets `latest_checkpointed_iteration.txt` to `iteration`, writes `hf_converted_iter<N>`, and
  **skips if `accuracy_summary.json` already exists**.
- One cell ≈ 1.5 h (convert + eval). For N cells, submit **per-cell manifests as parallel
  jobs** (don't put all in one job — they run sequentially and blow the 4 h walltime). See
  [[gpu-job-dispatch]] for spreading across partitions.

## NRT — lm-eval via vLLM (when the HF dir is already on NRT)
`lm_eval --model vllm --model_args pretrained=<HF_DIR>,tensor_parallel_size=8,dtype=bfloat16,enforce_eager=True,trust_remote_code=True`
- Runtime `pip install 'lm_eval[vllm]'` works (NRT compute has internet).
- ⚠️ **NRT lm-eval results don't auto-land in the ORD CSV** — the collector runs on ORD. Copy
  results to ORD or extend the collector, else the new point never reaches the frontier.

## Convert gotchas (hard-won)
- HF dir must have **all shards + `model.safetensors.index.json`** (235B = 118 shards). Missing
  index / 0-byte `config.json` → reconvert.
- **EP at convert/load must match the checkpoint's shard count.** A legacy mcore base saved
  EP=8 (8 `mp_rank` dirs) cannot load at EP=16; distcp (`torch_dist`) format *reshards* freely.
- Transferring a distcp checkpoint between clusters: include the hidden **`.metadata`** index
  and the top-level tokenizer files (`config.json`, `tokenizer.json`, `vocab.json`,
  `merges.txt`, `tokenizer_config.json`) — rsync of just `iter_*/` misses them and load fails
  (misleading `NatConfig KeyError` for tokenizer, `FileNotFoundError .metadata` for weights).

## Aggregate + plot
`collect_results_unified.py` → `benchmark_results.csv` (one row per run×iter×limit, joins
walltime + scrapes CP from logs) → `generate_pareto.py --limit-filter inf` → `pareto_*.html`.
