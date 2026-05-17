---
name: submit-training-job
description: Submit a single SLURM Megatron training job (one config, no sweep) for Qwen3 MoE router training. Use when the user wants a one-off training run, a debug/smoke training of N iterations, a manual rerun of a specific config, or to test changes to pretrain_qwen.py or run_mcore_qwen3.sh without going through W&B.
---

# Submit a single training job

## When

User says: "run a quick training", "submit one job with these args",
"smoke-test 20 iters", "rerun config X without sweep", or wants to
validate changes to `pretrain_qwen.py` / `run_mcore_qwen3.sh` /
`megatron_patch/template/helper.py` before launching a real sweep.

## The two entry points

| Use case | Script |
|---|---|
| Inside a SLURM allocation (e.g. via `srun ... bash -c`) | `examples/qwen3/run_mcore_qwen3.sh` |
| Submit as its own SLURM job | wrap the above in an `sbatch` (see "From scratch" below). |

`run_mcore_qwen3.sh` is the model launcher. It takes ~25 positional args
plus an `--extra-args` tail forwarded straight to `pretrain_qwen.py`.

## Interactive smoke-test (single allocation, 20 iters)

```bash
srun --account=nvr_israel_rlop --partition=interactive \
     --nodes=1 --gpus-per-node=4 --cpus-per-gpu=2 --time=00:15:00 \
     --container-image="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     bash -c "cd examples/qwen3 && sh run_mcore_qwen3.sh \
        dsw A3B 1 8 1e-4 1e-6 128 128 bf16 1 1 1 1 4 true true true false sel false 100 \
        <DATASET_PATH> <DATASET_PATH> <CKPT_PATH> 1024000 10240 <OUTPUT_PATH> \
        --router-only-training --use_rl_loss --train-iters 20 --eval-interval 20 --eval-iters 2"
```

Default paths (use these unless the user specifies otherwise):

- Dataset prefix: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document`
- 30B mcore ckpt: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore`
- 235B mcore ckpt: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B-to-mcore`
- Output base: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/<run_name>/`

## RL-relevant flags (in `--extra-args`)

| Flag | Notes |
|---|---|
| `--router-only-training` | Freeze everything except router weights (48 layers × [2048,128]). |
| `--use_rl_loss` | Enable RL loss path. |
| `--rl-loss-coeff <f>` | RL loss weight. |
| `--rl-algorithm ppo` | `ppo` or `reinforce`. |
| `--rl-ppo-baseline-type {mean,critic}` | Mean is the cheaper and (empirically) stronger CP-reducer. |
| `--rl-reward-type per_token_load_weighted` | Standard reward. |
| `--moe-router-critical-path-bias` | CPB — strongest CP lever, but **does not generalize when combined with RL** (see CLAUDE.md). |
| `--kl-loss-coeff <f>` | Any non-zero KL is a binary switch: protects accuracy, +500 CP. |
| `--rl-stochastic-routing --rl-gumbel-temperature 0.3` | Only viable temperature; higher destroys LM. |
| `--rl-discount-factor 0.8 --rl-gae-lambda 0.95` | GAE only works with `gamma>0`; `gamma=0.8` makes critic loss explode (~10000+). |

See `megatron_patch/arguments.py` for the full list.

## Submit as an sbatch (from scratch)

Use `examples/qwen3/submit_sweep_agent.sh` as the template — keep its
`--account`, `--partition`, `--cpus-per-gpu`, container, and mount lines
verbatim and replace the `wandb agent` body with the `run_mcore_qwen3.sh`
invocation above.

## Monitoring

```bash
JOB_ID=<from sbatch output>
tail -f /lustre/.../sweep_logs/<job>_${JOB_ID}.out
# Look for: "iteration ... | lm loss: ... | grad norm: ..."
# RL-specific: "rl/critic_loss", "rl/policy_loss", "critical_path"
```

## Gotchas

- **Never** set `#SBATCH --gpus-per-node=4` without `--cpus-per-gpu=2` — defaults will eat ~120 CPUs.
- `--train-iters` overrides the sweep config value; the eval cadence must divide it.
- If you see `RuntimeError: ...world_size not divisible by ...`, the TP/PP/EP product mismatches `--gpus-per-node`. For 30B A3B use `TP=1 PP=1 EP=4`; for 235B A22B 1-node use `TP=8 PP=1 EP=8`; for 2-node 235B conversion use `TP=1 PP=2 EP=8`.
