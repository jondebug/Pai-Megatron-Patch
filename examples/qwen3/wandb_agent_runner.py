#!/usr/bin/env python3
"""
Wandb sweep agent runner.

This script is called by wandb agents. It:
1. Receives the run_index from wandb sweep
2. Loads the corresponding configuration from sweep_combinations.json
3. Runs the training script with the correct parameters
4. Saves logs to the sweep directory
"""

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime


def load_combinations(combos_path: str):
    """Load sweep combinations from JSON."""
    with open(combos_path, 'r') as f:
        data = json.load(f)
    return data['combinations'], data['fixed_params'], data.get('sweep_dir', None)


def build_run_name(sweep_params: dict, run_index: int, fixed_params: dict = None) -> str:
    """Build a descriptive run name."""
    # Merge fixed and sweep params (sweep overrides fixed)
    all_params = {}
    if fixed_params:
        all_params.update(fixed_params)
    all_params.update(sweep_params)
    
    parts = []
    
    alg = all_params.get('rl_algorithm', 'unknown').lower()
    parts.append(alg.upper())
    
    if all_params.get('rl_per_token_rewards', False):
        parts.append('token')
    else:
        parts.append('batch')
    
    if alg == 'ppo':
        ent = all_params.get('rl_ppo_entropy_coeff', 0)
        parts.append(f'ent{ent}')
        
        # Add baseline type info
        baseline = all_params.get('rl_ppo_baseline_type', 'mean')
        if baseline == 'critic':
            dims = all_params.get('rl_critic_hidden_dims', [256])
            if isinstance(dims, list):
                dims_str = 'x'.join(str(d) for d in dims)
            else:
                dims_str = str(dims)
            parts.append(f'critic{dims_str}')
        else:
            parts.append('mean')
    
    rlc = all_params.get('rl_loss_coeff', 0)
    parts.append(f'rlc{rlc}')
    
    parts.append(f'r{run_index:02d}')
    
    return '_'.join(parts)


def build_command(fixed_params: dict, sweep_params: dict, run_name: str) -> list:
    """Build the training command."""
    
    # Merge params (sweep overrides fixed)
    config = {**fixed_params, **sweep_params}
    config['wandb_run_name'] = run_name
    
    # Build positional arguments
    positional_args = [
        config.get('env', 'dsw'),
        config.get('model_size', 'A3B'),
        str(config.get('batch_size', 1)),
        str(config.get('global_batch_size', 8)),
        str(config.get('lr', '1e-4')),
        str(config.get('min_lr', '1e-6')),
        str(config.get('seq_len', 128)),
        str(config.get('pad_len', 128)),
        config.get('precision', 'bf16'),
        str(config.get('tp', 1)),
        str(config.get('pp', 1)),
        str(config.get('cp', 1)),
        str(config.get('etp', 1)),
        str(config.get('ep', 4)),
        str(config.get('sp', 'true')).lower() if isinstance(config.get('sp'), bool) else str(config.get('sp', 'true')).lower(),
        str(config.get('do', 'true')).lower() if isinstance(config.get('do'), bool) else str(config.get('do', 'true')).lower(),
        str(config.get('fl', 'true')).lower() if isinstance(config.get('fl'), bool) else str(config.get('fl', 'true')).lower(),
        str(config.get('sft', 'false')).lower() if isinstance(config.get('sft'), bool) else str(config.get('sft', 'false')).lower(),
        config.get('ac', 'sel'),
        str(config.get('optimizer_offload', 'false')).lower() if isinstance(config.get('optimizer_offload'), bool) else str(config.get('optimizer_offload', 'false')).lower(),
        str(config.get('save_interval', 900)),
        config.get('dataset_path'),
        config.get('valid_dataset_path'),
        config.get('pretrain_checkpoint_path'),
        str(config.get('train_tokens', 1024000)),
        str(config.get('warmup_tokens', 10240)),
        config.get('output_basepath'),
    ]
    
    # Build extra args
    extra_args = []
    
    # Boolean flags
    bool_flags = [
        ('router_only_training', '--router-only-training'),
        ('enable_wandb_logging', '--enable-wandb-logging'),
        ('use_rl_loss', '--use_rl_loss'),
        ('rl_per_token_rewards', '--rl-per-token-rewards'),
        ('rl_use_entropy_reward', '--rl-use-entropy-reward'),
    ]
    
    for config_key, flag in bool_flags:
        if config.get(config_key, False):
            extra_args.append(flag)
    
    # Value arguments
    value_args = [
        ('wandb_project_name', '--wandb-project-name'),
        ('wandb_run_name', '--wandb-run-name'),
        ('rl_algorithm', '--rl-algorithm'),
        ('rl_loss_coeff', '--rl-loss-coeff'),
        ('rl_ppo_entropy_coeff', '--rl-ppo-entropy-coeff'),
        ('rl_ppo_baseline_type', '--rl-ppo-baseline-type'),
        ('moe_aux_loss_coeff', '--moe-aux-loss-coeff'),
        ('train_iters', '--train-iters'),
    ]
    
    for config_key, flag in value_args:
        if config_key in config and config[config_key] is not None:
            extra_args.extend([flag, str(config[config_key])])
    
    # List arguments (like rl_critic_hidden_dims)
    if 'rl_critic_hidden_dims' in config and config['rl_critic_hidden_dims'] is not None:
        dims = config['rl_critic_hidden_dims']
        if isinstance(dims, list):
            extra_args.append('--rl-critic-hidden-dims')
            extra_args.extend([str(d) for d in dims])
    
    # Wandb tags
    if 'wandb_run_tags' in config and config['wandb_run_tags']:
        extra_args.append('--wandb-run-tags')
        if isinstance(config['wandb_run_tags'], list):
            extra_args.extend(config['wandb_run_tags'])
        else:
            extra_args.append(config['wandb_run_tags'])
    
    script_dir = Path(__file__).parent
    script_path = script_dir / 'run_mcore_qwen3.sh'
    
    cmd = ['sh', str(script_path)] + positional_args + extra_args
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Wandb sweep agent runner')
    parser.add_argument('--config', type=str, default='sweep_config.json',
                        help='Path to original sweep config (for reference)')
    parser.add_argument('--run_index', type=int, required=True,
                        help='Index into sweep_combinations.json')
    parser.add_argument('--sweep-dir', type=str, default=None,
                        help='Sweep directory containing sweep_combinations.json')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Determine where to load combinations from
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        combos_path = sweep_dir / 'sweep_combinations.json'
    else:
        combos_path = script_dir / 'sweep_combinations.json'
        sweep_dir = None
    
    if not combos_path.exists():
        print(f"ERROR: {combos_path} not found!")
        print("Run 'python wandb_sweep_config.py --config sweep_config.json' first.")
        return 1
    
    combinations, fixed_params, stored_sweep_dir = load_combinations(combos_path)
    
    # Use stored sweep_dir if not provided
    if sweep_dir is None and stored_sweep_dir:
        sweep_dir = Path(stored_sweep_dir)
    
    if args.run_index < 0 or args.run_index >= len(combinations):
        print(f"ERROR: run_index {args.run_index} out of range [0, {len(combinations)})")
        return 1
    
    sweep_params = combinations[args.run_index]
    run_name = build_run_name(sweep_params, args.run_index, fixed_params)
    
    # Log sweep parameters to wandb (wandb agent already initialized a run)
    try:
        import wandb
        if wandb.run is not None:
            # Update the wandb run config with our sweep parameters
            # Merge fixed and sweep params for logging
            all_params = {**fixed_params, **sweep_params}
            wandb.config.update({
                'rl_algorithm': all_params.get('rl_algorithm'),
                'rl_per_token_rewards': all_params.get('rl_per_token_rewards'),
                'rl_ppo_entropy_coeff': all_params.get('rl_ppo_entropy_coeff'),
                'rl_ppo_baseline_type': all_params.get('rl_ppo_baseline_type'),
                'rl_critic_hidden_dims': all_params.get('rl_critic_hidden_dims'),
                'rl_loss_coeff': all_params.get('rl_loss_coeff'),
                'run_index': args.run_index,
                'run_name': run_name,
            })
            print(f"Logged sweep params to wandb run: {wandb.run.name}")
    except Exception as e:
        print(f"Note: Could not log to wandb: {e}")
    
    # Setup log file
    log_file = None
    if sweep_dir:
        logs_dir = sweep_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{run_name}.log"
    
    cmd = build_command(fixed_params, sweep_params, run_name)
    full_command = ' '.join(cmd)
    
    print("=" * 60)
    print(f"WANDB SWEEP AGENT - Run {args.run_index}")
    print("=" * 60)
    print(f"Run name: {run_name}")
    print(f"Sweep params: {sweep_params}")
    print(f"Fixed params: {fixed_params}")
    if sweep_dir:
        print(f"Sweep dir: {sweep_dir}")
        print(f"Log file: {log_file}")
    print("-" * 60)
    print(f"FULL COMMAND:")
    print(full_command)
    print("=" * 60)
    
    # Log full command to wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                'full_command': full_command,
            }, allow_val_change=True)
            # Also log as a summary for easy access
            wandb.run.summary['command'] = full_command
    except Exception as e:
        print(f"Note: Could not log command to wandb: {e}")
    
    # Run the training with logging
    if log_file:
        with open(log_file, 'w') as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Run index: {args.run_index}\n")
            f.write(f"\nSweep params:\n{json.dumps(sweep_params, indent=2)}\n")
            f.write(f"\nFixed params:\n{json.dumps(fixed_params, indent=2)}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write(f"FULL COMMAND (copy-paste to rerun):\n")
            f.write(f"{full_command}\n")
            f.write("=" * 60 + "\n\n")
        
        # Run with output to both console and log file (tee-like behavior)
        with open(log_file, 'a') as f:
            result = subprocess.run(cmd, cwd=script_dir, stdout=f, stderr=subprocess.STDOUT)
        
        # Append completion status
        with open(log_file, 'a') as f:
            f.write(f"\n" + "=" * 60 + "\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
        
        print(f"\nLog saved to: {log_file}")
    else:
        result = subprocess.run(cmd, cwd=script_dir)
    
    return result.returncode


if __name__ == '__main__':
    exit(main())

