---
name: launch-sweep
description: Create a new W&B sweep from a config JSON in examples/qwen3/run_config_jsons/ and launch auto-chaining SLURM agents on the nvr_israel_rlop account. Use when the user wants to start a new hyperparameter sweep, kick off training runs from a sweep config, run a grid search, or resume/extend an existing sweep with more agents.
---

# Launch a W&B sweep

## When

User says any of: "launch a sweep", "kick off <name> sweep", "run the <X>
configs", "submit agents for sweep <id>", "extend sweep <id>".

## Inputs needed

- **Sweep config JSON** — must live under `examples/qwen3/run_config_jsons/`. List existing ones first; ask the user to pick or to provide a new JSON.
- **Parallelism** — `--parallel N` chains × `--agents M` agents per chain. Total concurrent runs = `N×M`. Hard QOS ceiling is 3 SLURM jobs × ~6 GPUs/job. Default safe: `--parallel 2 --agents 2`.
- **Model size flag** — for 235B use `--8gpu` (selects `submit_sweep_agent_8gpu.sh`); 30B is default 4-GPU.

## Workflow

```bash
cd examples/qwen3

# Step 1: create the sweep on W&B (prints SWEEP_ID at the end)
python3 wandb_sweep_config.py --config run_config_jsons/<name>.json
# Capture the SWEEP_ID it prints (8-char alphanumeric).

# Step 2: launch SLURM agent chains
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 2          # 30B
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 2 --8gpu   # 235B
```

`launch_sweep.sh` submits N independent SLURM chains. Each chain runs M
parallel `wandb agent` processes inside one 4-hour allocation, then
auto-submits a continuation job (`--dependency=afterany`) so the sweep
keeps progressing across allocations. Max chain depth is 20.

## After submission (mandatory monitoring)

```bash
squeue -u $USER -o "%.10i %.20j %.2t %.10M %R"
# Logs are under:
ls -lt /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/sweep_<SWEEP_ID>_chain*
tail -f /lustre/.../sweep_logs/sweep_<SWEEP_ID>_chain1_<JOB_ID>.out
```

Verify within ~5 minutes that the job allocated and `wandb agent` started.
URL to share with user: `https://wandb.ai/nvr-israel/qwen3-router-training/sweeps/<SWEEP_ID>`.

## Known gotchas

- **Wrong account**: SBATCH scripts must use `--account=nvr_israel_rlop`. If a script still references `nvr_israel_scne`, fix it before submitting.
- **Sweep dir lookup**: `launch_sweep.sh` greps `sweep_logs/*/sweep_id.txt` for the sweep ID. If the sweep dir is missing, agents will fail at config-loading time — re-run step 1.
- **Resume**: Re-running `launch_sweep.sh <SWEEP_ID>` extends an existing sweep with more chains. Resume of a specific run is handled inside `wandb_agent_runner.py` (cross-sweep directory matching).
- **Container `pip install`**: Each agent install of `wandb`+`datasets` adds ~30 s startup. Don't be alarmed if first log lines look slow.

## Cancellation

```bash
scancel <JOB_ID> [<JOB_ID> ...]    # cancel one chain; continuation jobs still exist, cancel those too
scancel -u $USER -n "sweep_<SWEEP_ID>_chain*"   # nuke whole sweep
```
