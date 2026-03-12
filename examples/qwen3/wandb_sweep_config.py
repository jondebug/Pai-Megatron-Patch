"""
Generate and initialize a wandb sweep from sweep_config.json.

Usage:
    python wandb_sweep_config.py --config sweep_config.json
"""
import argparse, json, itertools, shutil, yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


def create_sweep_directory(sweep_name: str, base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = base_dir / "sweep_logs" / f"{sweep_name}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "logs").mkdir(exist_ok=True)
    return sweep_dir


def generate_filtered_sweep_config(config: Dict[str, Any]):
    """Pre-compute all valid grid combinations after applying filters."""
    filters = config.get("filters", [])
    sweep_keys = []
    sweep_values = []
    fixed_params = {}

    for key, value in config.items():
        if key.startswith("_") or key == "filters":
            continue
        if isinstance(value, list) and key != "wandb_run_tags":
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed_params[key] = value

    all_combos = [dict(zip(sweep_keys, combo)) for combo in itertools.product(*sweep_values)]

    def make_hashable(v):
        if isinstance(v, list):
            return tuple(make_hashable(x) for x in v)
        return v

    seen_signatures = set()
    valid_combos = []
    for cfg in all_combos:
        skip = False
        collapse_keys = []
        for flt in filters:
            condition = flt.get("if", {})
            if all(cfg.get(k) == v for k, v in condition.items()):
                if flt.get("skip", False):
                    skip = True
                    break
                if "collapse" in flt:
                    collapse_keys.extend(flt["collapse"])
        if skip:
            continue
        sig_cfg = {k: make_hashable(v) for k, v in cfg.items() if k not in collapse_keys}
        sig = tuple(sorted(sig_cfg.items()))
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            valid_combos.append(cfg)

    print(f"Generated {len(valid_combos)} valid configurations after filtering")

    wandb_config = {
        'program': 'wandb_agent_runner.py',
        'method': 'grid',
        'name': config.get('wandb_run_name_base', 'rl-router-sweep'),
        'project': config.get('wandb_project_name', 'qwen3-router-training'),
        'parameters': {'run_index': {'values': list(range(len(valid_combos)))}},
        'command': ['${env}', '${interpreter}', '${program}', '--config', 'sweep_config.json', '${args}'],
    }
    return wandb_config, valid_combos, fixed_params


def build_sweep_summary(valid_combos: List[Dict], fixed_params: Dict) -> Dict[str, Any]:
    if not valid_combos:
        return {'swept_params': {}, 'fixed_params_subset': {}, 'grid_description': 'empty', 'total_combinations': 0}

    all_keys = set()
    for combo in valid_combos:
        all_keys.update(combo.keys())

    swept_params = {}
    for key in sorted(all_keys):
        values = set()
        for combo in valid_combos:
            v = combo.get(key)
            if isinstance(v, list):
                v = tuple(v)
            values.add(v)
        if len(values) > 1:
            sorted_vals = sorted(values, key=lambda x: (str(type(x)), str(x)))
            swept_params[key] = [list(v) if isinstance(v, tuple) else v for v in sorted_vals]

    grid_parts = [f"{len(values)} {key}" for key, values in swept_params.items()]
    grid_description = " x ".join(grid_parts) if grid_parts else "no variation"

    interesting_fixed_keys = ['rl_algorithm', 'rl_reward_type', 'rl_per_token_rewards',
                              'rl_ppo_baseline_type', 'use_rl_loss', 'moe_aux_loss_coeff',
                              'train_iters', 'seq_len', 'global_batch_size']
    fixed_subset = {k: fixed_params[k] for k in interesting_fixed_keys if k in fixed_params}
    return {
        'swept_params': swept_params,
        'fixed_params_subset': fixed_subset,
        'grid_description': grid_description,
        'total_combinations': len(valid_combos),
    }


def save_sweep_artifacts(wandb_config, valid_combos, fixed_params, original_config, output_dir, config_path):
    config_copy_path = output_dir / "sweep_config.json"
    with open(config_copy_path, 'w') as f:
        json.dump(original_config, f, indent=2)
    print(f"Saved config copy: {config_copy_path}")

    sweep_yaml_path = output_dir / "wandb_sweep.yaml"
    with open(sweep_yaml_path, 'w') as f:
        yaml.dump(wandb_config, f, default_flow_style=False)
    print(f"Saved wandb sweep config: {sweep_yaml_path}")

    combos_path = output_dir / "sweep_combinations.json"
    sweep_summary = build_sweep_summary(valid_combos, fixed_params)
    with open(combos_path, 'w') as f:
        json.dump({
            'combinations': valid_combos,
            'fixed_params': fixed_params,
            'sweep_dir': str(output_dir),
            'sweep_summary': sweep_summary,
        }, f, indent=2)
    print(f"Saved sweep combinations: {combos_path}")

    print(f"\n  Sweep grid: {sweep_summary['grid_description']}")
    print(f"  Total combinations: {sweep_summary['total_combinations']}")
    if sweep_summary['swept_params']:
        print("  Swept parameters:")
        for k, v in sweep_summary['swept_params'].items():
            print(f"    {k}: {v}")

    latest_link = output_dir.parent / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)
    print(f"Created symlink: {latest_link} -> {output_dir.name}")

    main_combos_path = Path(__file__).parent / "sweep_combinations.json"
    with open(main_combos_path, 'w') as f:
        json.dump({
            'combinations': valid_combos,
            'fixed_params': fixed_params,
            'sweep_dir': str(output_dir),
            'sweep_summary': sweep_summary,
        }, f, indent=2)

    return sweep_yaml_path, combos_path


def main():
    parser = argparse.ArgumentParser(description="Generate and initialize wandb sweep")
    parser.add_argument("--config", type=str, default="sweep_config.json")
    parser.add_argument("--create", action="store_true", default=True)
    parser.add_argument("--entity", type=str, default="nvr-israel")
    args = parser.parse_args()

    config = load_config(args.config)
    original_config = config.copy()
    base_dir = Path(__file__).parent
    sweep_name = config.get("sweep_name", "unnamed_sweep")
    sweep_dir = create_sweep_directory(sweep_name, base_dir)
    print(f"\nCreated sweep directory: {sweep_dir}")

    wandb_config, valid_combos, fixed_params = generate_filtered_sweep_config(config)
    wandb_config["command"] = ["${env}", "${interpreter}", "${program}",
                               "--sweep-dir", str(sweep_dir), "${args}"]

    sweep_yaml_path, combos_path = save_sweep_artifacts(
        wandb_config, valid_combos, fixed_params, original_config, sweep_dir, args.config)

    with open(sweep_dir / "sweep_info.txt", 'w') as f:
        f.write(f"Sweep Name: {sweep_name}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Configurations: {len(valid_combos)}\n")
        f.write(f"Directory: {sweep_dir}\n")
        f.write("\nSweep Parameters:\n")
        for i, combo in enumerate(valid_combos):
            f.write(f"  [{i}] {combo}\n")

    print("\nValid sweep configurations:")
    for i, combo in enumerate(valid_combos):
        parts = [f'{k}={"T" if v else "F"}' if isinstance(v, bool) else f"{k}={v}" for k, v in combo.items()]
        print(f'  [{i}] {", ".join(parts)}')

    if args.create:
        import wandb
        project = config.get("wandb_project_name", "qwen3-router-training")
        sweep_id = wandb.sweep(sweep=wandb_config, project=project, entity=args.entity)
        with open(sweep_dir / "sweep_id.txt", 'w') as f:
            f.write(sweep_id)
        print(f"\n{'='*60}")
        print(f"SWEEP CREATED: {sweep_id}")
        print(f"{'='*60}")
        print(f"Sweep directory: {sweep_dir}")
        print(f"\nTo run agents on any machine:")
        print(f"  wandb agent {args.entity}/{project}/{sweep_id}")
        print(f"\nOr run multiple agents:")
        print(f"  for i in {{1..4}}; do wandb agent {project}/{sweep_id} & done")
    else:
        print(f"\n{'='*60}")
        print("SWEEP CONFIG GENERATED")
        print(f"{'='*60}")
        print(f"Sweep directory: {sweep_dir}")
        print(f"\nTo create the sweep on wandb, run:")
        print(f"  python wandb_sweep_config.py --config {args.config} --create")
        print(f"\nOr manually with:")
        print(f"  wandb sweep {sweep_yaml_path}")


if __name__ == "__main__":
    main()
