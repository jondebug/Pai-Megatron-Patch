#!/usr/bin/env python3
"""
Generate and initialize a wandb sweep from sweep_config.json.

Usage:
    # Create sweep and print sweep ID
    python wandb_sweep_config.py --config sweep_config.json
    
    # Then run agents on any machine:
    wandb agent <entity>/<project>/<sweep_id>
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_sweep_directory(sweep_name: str, base_dir: Path) -> Path:
    """Create a directory for this sweep with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir_name = f"{sweep_name}_{timestamp}"
    sweep_dir = base_dir / "sweep_logs" / sweep_dir_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (sweep_dir / "logs").mkdir(exist_ok=True)
    
    return sweep_dir


def build_wandb_sweep_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert our sweep config to wandb sweep format."""
    
    # Extract sweep parameters (lists in our config)
    parameters = {}
    fixed_params = {}
    filters = config.get('filters', [])
    
    # Parameters that are sweep variables
    sweep_param_keys = ['rl_algorithm', 'rl_per_token_rewards', 'rl_ppo_entropy_coeff', 'rl_loss_coeff']
    
    for key, value in config.items():
        if key.startswith('_') or key == 'filters':
            continue
        
        if isinstance(value, list) and key != 'wandb_run_tags':
            # This is a sweep parameter
            parameters[key] = {'values': value}
        else:
            # Fixed parameter
            fixed_params[key] = value
    
    # Build wandb sweep config
    wandb_config = {
        'program': 'wandb_agent_runner.py',
        'method': 'grid',  # Run all combinations
        'name': config.get('wandb_run_name_base', 'rl-router-sweep'),
        'project': config.get('wandb_project_name', 'qwen3-router-training'),
        'parameters': parameters,
        
        # Pass fixed params as command args
        'command': [
            '${env}',
            '${interpreter}',
            '${program}',
            '--config', str(Path(__file__).parent / 'sweep_config.json'),
            '${args}'
        ]
    }
    
    return wandb_config, fixed_params, filters


def generate_filtered_sweep_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate wandb sweep config with filters applied.
    
    Since wandb doesn't support conditional parameters natively,
    we pre-compute all valid combinations and use 'grid' method.
    """
    import itertools
    
    filters = config.get('filters', [])
    
    # Get sweep parameters
    sweep_keys = []
    sweep_values = []
    fixed_params = {}
    
    for key, value in config.items():
        if key.startswith('_') or key == 'filters':
            continue
        if isinstance(value, list) and key != 'wandb_run_tags':
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed_params[key] = value
    
    # Generate all combinations
    all_combos = []
    for combo in itertools.product(*sweep_values):
        cfg = dict(zip(sweep_keys, combo))
        all_combos.append(cfg)
    
    # Apply filters
    seen_signatures = set()
    valid_combos = []
    
    for cfg in all_combos:
        skip = False
        collapse_keys = []
        
        for flt in filters:
            condition = flt.get('if', {})
            matches = all(cfg.get(k) == v for k, v in condition.items())
            
            if matches:
                if flt.get('skip', False):
                    skip = True
                    break
                if 'collapse' in flt:
                    collapse_keys.extend(flt['collapse'])
        
        if skip:
            continue
        
        # Create signature for deduplication
        
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(make_hashable(x) for x in v)
            return v
        sig_cfg = {k: make_hashable(v) for k, v in cfg.items() if k not in collapse_keys}
        sig = tuple(sorted(sig_cfg.items()))
        
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            valid_combos.append(cfg)
    
    print(f"Generated {len(valid_combos)} valid configurations after filtering")
    
    # Create wandb sweep with explicit run configs
    # We use a single parameter that indexes into our valid configs
    wandb_config = {
        'program': 'wandb_agent_runner.py',
        'method': 'grid',
        'name': config.get('wandb_run_name_base', 'rl-router-sweep'),
        'project': config.get('wandb_project_name', 'qwen3-router-training'),
        'parameters': {
            'run_index': {
                'values': list(range(len(valid_combos)))
            }
        },
        'command': [
            '${env}',
            '${interpreter}',
            '${program}',
            '--config', 'sweep_config.json',
            '${args}'
        ]
    }
    
    return wandb_config, valid_combos, fixed_params


def build_sweep_summary(valid_combos: List[Dict], fixed_params: Dict) -> Dict[str, Any]:
    """Build a human-readable summary of what the sweep is actually varying.
    
    Returns a dict with:
      - 'swept_params': {param_name: sorted unique values} for params that vary
      - 'fixed_params_subset': key fixed params relevant to the experiment
      - 'grid_description': human-readable string like "3 gammas x 4 topn x ..."
    """
    if not valid_combos:
        return {'swept_params': {}, 'fixed_params_subset': {}, 'grid_description': 'empty'}
    
    # Find which keys vary across combinations
    all_keys = set()
    for combo in valid_combos:
        all_keys.update(combo.keys())
    
    swept_params = {}
    for key in sorted(all_keys):
        values = set()
        for combo in valid_combos:
            v = combo.get(key)
            # Make hashable
            if isinstance(v, list):
                v = tuple(v)
            values.add(v)
        if len(values) > 1:
            # Convert back for JSON serialization
            sorted_vals = sorted(values, key=lambda x: (str(type(x)), str(x)))
            swept_params[key] = [list(v) if isinstance(v, tuple) else v for v in sorted_vals]
    
    # Build grid description
    grid_parts = []
    for key, values in swept_params.items():
        grid_parts.append(f"{len(values)} {key}")
    grid_description = " x ".join(grid_parts) if grid_parts else "no variation"
    
    # Key fixed params worth showing
    interesting_fixed_keys = [
        'rl_algorithm', 'rl_reward_type', 'rl_per_token_rewards',
        'rl_ppo_baseline_type', 'use_rl_loss', 'moe_aux_loss_coeff',
        'train_iters', 'seq_len', 'global_batch_size',
    ]
    fixed_subset = {k: fixed_params[k] for k in interesting_fixed_keys if k in fixed_params}
    
    return {
        'swept_params': swept_params,
        'fixed_params_subset': fixed_subset,
        'grid_description': grid_description,
        'total_combinations': len(valid_combos),
    }


def save_sweep_artifacts(wandb_config: Dict, valid_combos: List[Dict], fixed_params: Dict, 
                         original_config: Dict, output_dir: Path, config_path: str):
    """Save sweep configuration files."""
    
    # Save original config as reference
    config_copy_path = output_dir / 'sweep_config.json'
    with open(config_copy_path, 'w') as f:
        json.dump(original_config, f, indent=2)
    print(f"Saved config copy: {config_copy_path}")
    
    # Save wandb sweep YAML
    sweep_yaml_path = output_dir / 'wandb_sweep.yaml'
    with open(sweep_yaml_path, 'w') as f:
        yaml.dump(wandb_config, f, default_flow_style=False)
    print(f"Saved wandb sweep config: {sweep_yaml_path}")
    
    # Save valid combinations as JSON (for agent runner to use)
    combos_path = output_dir / 'sweep_combinations.json'
    
    # Build and include sweep summary
    sweep_summary = build_sweep_summary(valid_combos, fixed_params)
    
    with open(combos_path, 'w') as f:
        json.dump({
            'combinations': valid_combos,
            'fixed_params': fixed_params,
            'sweep_dir': str(output_dir),
            'sweep_summary': sweep_summary,
        }, f, indent=2)
    print(f"Saved sweep combinations: {combos_path}")
    
    # Print sweep summary
    print(f"\n  Sweep grid: {sweep_summary['grid_description']}")
    print(f"  Total combinations: {sweep_summary['total_combinations']}")
    if sweep_summary['swept_params']:
        print(f"  Swept parameters:")
        for k, v in sweep_summary['swept_params'].items():
            print(f"    {k}: {v}")
    
    # Create a symlink to the latest sweep for easy access
    base_dir = output_dir.parent
    latest_link = base_dir / 'latest'
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)
    print(f"Created symlink: {latest_link} -> {output_dir.name}")
    
    # Also update the main sweep_combinations.json for agent runner
    main_combos_path = Path(__file__).parent / 'sweep_combinations.json'
    with open(main_combos_path, 'w') as f:
        json.dump({
            'combinations': valid_combos,
            'fixed_params': fixed_params,
            'sweep_dir': str(output_dir),
            'sweep_summary': sweep_summary,
        }, f, indent=2)
    
    return sweep_yaml_path, combos_path


def main():
    parser = argparse.ArgumentParser(description='Generate and initialize wandb sweep')
    parser.add_argument('--config', type=str, default='sweep_config.json', 
                        help='Path to sweep config JSON')
    parser.add_argument('--create', action='store_true', default=True,
                        help='Create the sweep on wandb (requires wandb login)')
    parser.add_argument('--entity', type=str, default="nvr-israel",
                        help='Wandb entity (team/username)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    original_config = config.copy()  # Keep original for saving
    base_dir = Path(__file__).parent
    
    # Get sweep name
    sweep_name = config.get('sweep_name', 'unnamed_sweep')
    
    # Create sweep directory
    sweep_dir = create_sweep_directory(sweep_name, base_dir)
    print(f"\nCreated sweep directory: {sweep_dir}")
    
    # Generate filtered sweep config
    wandb_config, valid_combos, fixed_params = generate_filtered_sweep_config(config)
    
    # Update wandb config to use correct paths
    # Use ${args} to get --key value format instead of ${args_no_hyphens} which gives key=value
    wandb_config['command'] = [
        '${env}',
        '${interpreter}',
        '${program}',
        '--sweep-dir', str(sweep_dir),
        '${args}'
    ]
    
    # Save artifacts to sweep directory
    sweep_yaml_path, combos_path = save_sweep_artifacts(
        wandb_config, valid_combos, fixed_params, original_config, sweep_dir, args.config
    )
    
    # Save sweep info
    sweep_info_path = sweep_dir / 'sweep_info.txt'
    with open(sweep_info_path, 'w') as f:
        f.write(f"Sweep Name: {sweep_name}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Configurations: {len(valid_combos)}\n")
        f.write(f"Directory: {sweep_dir}\n")
        f.write(f"\nSweep Parameters:\n")
        for i, combo in enumerate(valid_combos):
            f.write(f"  [{i}] {combo}\n")
    
    # Print combinations
    print("\nValid sweep configurations:")
    for i, combo in enumerate(valid_combos):
        parts = []
        for k, v in combo.items():
            if isinstance(v, bool):
                parts.append(f"{k}={'T' if v else 'F'}")
            else:
                parts.append(f"{k}={v}")
        print(f"  [{i}] {', '.join(parts)}")
    
    if args.create:
        import wandb
        
        project = config.get('wandb_project_name', 'qwen3-router-training')
        sweep_id = wandb.sweep(
            sweep=wandb_config,
            project=project,
            entity=args.entity
        )
        
        # Save sweep ID
        sweep_id_path = sweep_dir / 'sweep_id.txt'
        with open(sweep_id_path, 'w') as f:
            f.write(sweep_id)
        
        print("\n" + "=" * 60)
        print(f"SWEEP CREATED: {sweep_id}")
        print("=" * 60)
        print(f"Sweep directory: {sweep_dir}")
        print(f"\nTo run agents on any machine:")
        if args.entity:
            print(f"  wandb agent {args.entity}/{project}/{sweep_id}")
        else:
            print(f"  wandb agent {project}/{sweep_id}")
        print(f"\nOr run multiple agents:")
        print(f"  for i in {{1..4}}; do wandb agent {project}/{sweep_id} & done")
    else:
        print("\n" + "=" * 60)
        print("SWEEP CONFIG GENERATED")
        print("=" * 60)
        print(f"Sweep directory: {sweep_dir}")
        print(f"\nTo create the sweep on wandb, run:")
        print(f"  python wandb_sweep_config.py --config {args.config} --create")
        print(f"\nOr manually with:")
        print(f"  wandb sweep {sweep_yaml_path}")


if __name__ == '__main__':
    main()

