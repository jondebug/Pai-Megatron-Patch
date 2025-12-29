#!/usr/bin/env python3
"""
Sweep runner for qwen3 MoE training experiments.

Usage:
    python sweep_runner.py --config sweep_config.json
    python sweep_runner.py --config sweep_config.json --dry-run
    python sweep_runner.py --config sweep_config.json --parallel 4
"""

import argparse
import json
import itertools
import subprocess
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_filters(configurations: List[Dict[str, Any]], filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply filters to remove invalid configurations.
    
    Each filter is a dict with:
        - "if": condition dict (all must match)
        - "then": constraint dict (values that must match, or "skip": true to skip entirely)
        - "collapse": list of params to collapse (use first value only when condition matches)
    
    Examples:
        {"if": {"rl_algorithm": "reinforce"}, "collapse": ["rl_ppo_entropy_coeff"]}
        {"if": {"rl_loss_coeff": 0}, "skip": true}  # Skip when rl_loss is 0
    """
    if not filters:
        return configurations
    
    filtered = []
    seen_signatures = set()
    
    for cfg in configurations:
        skip = False
        collapse_keys = []
        
        for flt in filters:
            # Check if condition matches
            condition = flt.get('if', {})
            matches = all(cfg.get(k) == v for k, v in condition.items())
            
            if matches:
                # Check for skip
                if flt.get('skip', False):
                    skip = True
                    break
                
                # Collect keys to collapse
                if 'collapse' in flt:
                    collapse_keys.extend(flt['collapse'])
        
        if skip:
            continue
        
        # Create signature for deduplication (excluding collapsed keys)
        sig_cfg = {}
        for k, v in cfg.items():
            if k not in collapse_keys:
                # Convert lists to tuples for hashing
                sig_cfg[k] = tuple(v) if isinstance(v, list) else v
        sig = tuple(sorted(sig_cfg.items()))
        
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            filtered.append(cfg)
    
    return filtered


def expand_sweeps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand sweep parameters into individual configurations.
    
    If a parameter value is a list, it will be swept over.
    Returns a list of all parameter combinations.
    """
    # Extract filters before processing
    filters = config.pop('filters', [])
    
    # Separate sweep params (lists) from fixed params
    # Skip keys starting with "_" (comments)
    sweep_keys = []
    sweep_values = []
    fixed_params = {}
    
    for key, value in config.items():
        if key.startswith('_'):
            continue  # Skip comment keys
        if isinstance(value, list) and key != 'wandb_run_tags':  # Don't expand tags
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed_params[key] = value
    
    # Generate all combinations
    if not sweep_keys:
        return apply_filters([config], filters)
    
    configurations = []
    for combo in itertools.product(*sweep_values):
        cfg = fixed_params.copy()
        for key, val in zip(sweep_keys, combo):
            cfg[key] = val
        configurations.append(cfg)
    
    # Apply filters
    configurations = apply_filters(configurations, filters)
    
    return configurations


def build_run_name(config: Dict[str, Any], base_name: str, run_index: int = None) -> str:
    """Build a descriptive, informative run name from sweep parameters."""
    parts = []
    
    # Algorithm name (human readable)
    alg = config.get('rl_algorithm', 'unknown')
    parts.append(alg.upper())
    
    # Reward type
    if config.get('rl_per_token_rewards', False):
        parts.append('token-reward')
    else:
        parts.append('batch-reward')
    
    # PPO entropy (only meaningful for PPO)
    if alg == 'ppo':
        ent = config.get('rl_ppo_entropy_coeff', 0)
        parts.append(f'ent{ent}')
    
    # RL loss coefficient
    rlc = config.get('rl_loss_coeff', 0)
    parts.append(f'rlcoef{rlc}')
    
    # Add run index if provided
    if run_index is not None:
        parts.append(f'run{run_index:02d}')
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime('%m%d-%H%M')
    parts.append(timestamp)
    
    return '_'.join(parts)


def build_command(config: Dict[str, Any]) -> List[str]:
    """Build the shell command from configuration."""
    
    # Extract positional arguments in order
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
        str(config.get('sp', 'true')).lower(),
        str(config.get('do', 'true')).lower(),
        str(config.get('fl', 'true')).lower(),
        str(config.get('sft', 'false')).lower(),
        config.get('ac', 'sel'),
        str(config.get('optimizer_offload', 'false')).lower(),
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
        ('moe_aux_loss_coeff', '--moe-aux-loss-coeff'),
    ]
    
    for config_key, flag in value_args:
        if config_key in config and config[config_key] is not None:
            extra_args.extend([flag, str(config[config_key])])
    
    # Wandb tags (list)
    if 'wandb_run_tags' in config and config['wandb_run_tags']:
        extra_args.append('--wandb-run-tags')
        extra_args.extend(config['wandb_run_tags'])
    
    # Build full command
    script_dir = Path(__file__).parent
    script_path = script_dir / 'run_mcore_qwen3.sh'
    
    cmd = ['sh', str(script_path)] + positional_args + extra_args
    return cmd


# Lock for thread-safe printing
_print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def run_experiment(config: Dict[str, Any], run_index: int = 0, dry_run: bool = False, 
                   verbose: bool = True, log_dir: str = None) -> Tuple[int, str, str]:
    """Run a single experiment configuration.
    
    Returns:
        Tuple of (return_code, run_name, log_file_path)
    """
    # Generate run name if using wandb
    base_name = config.get('wandb_run_name_base', 'sweep')
    run_name = build_run_name(config, base_name, run_index)
    if config.get('enable_wandb_logging', False):
        config['wandb_run_name'] = run_name
    
    cmd = build_command(config)
    
    if verbose:
        thread_safe_print("\n" + "=" * 80)
        thread_safe_print(f"[Run {run_index}] Starting: {run_name}")
        thread_safe_print("=" * 80)
        if dry_run:
            thread_safe_print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        thread_safe_print(f"[DRY RUN] Would execute: {run_name}")
        return 0, run_name, None
    
    # Create log file if log_dir specified
    log_file = None
    if log_dir:
        log_path = Path(log_dir) / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path)
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n")
        
        # Run with output redirected to log file
        with open(log_file, 'a') as f:
            result = subprocess.run(cmd, cwd=Path(__file__).parent, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode, run_name, log_file


def main():
    parser = argparse.ArgumentParser(description='Run parameter sweeps for qwen3 MoE training')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--list-configs', action='store_true', help='List all configurations and exit')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='Number of experiments to run in parallel (default: 1 = sequential)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save logs for parallel runs (required if --parallel > 1)')
    args = parser.parse_args()
    
    # Validate parallel args
    if args.parallel > 1 and not args.log_dir and not args.dry_run:
        print("ERROR: --log-dir is required when running in parallel (--parallel > 1)")
        print("       This prevents output from multiple jobs getting interleaved.")
        return
    
    config = load_config(args.config)
    configurations = expand_sweeps(config)
    
    print(f"Loaded {len(configurations)} experiment configuration(s)")
    if args.parallel > 1:
        print(f"Running with {args.parallel} parallel workers")
        if args.log_dir:
            print(f"Logs will be saved to: {args.log_dir}")
    
    if args.list_configs:
        for i, cfg in enumerate(configurations):
            # Generate the run name for display
            base_name = cfg.get('wandb_run_name_base', 'sweep')
            run_name = build_run_name(cfg, base_name, i + 1)
            print(f"\n--- Configuration {i+1}: {run_name} ---")
            # Show only sweep-relevant params
            for key in ['rl_algorithm', 'rl_per_token_rewards', 'rl_ppo_entropy_coeff', 'rl_loss_coeff']:
                if key in cfg:
                    print(f"  {key}: {cfg[key]}")
        return
    
    # Prepare jobs as (index, config) tuples
    jobs = [(i + 1, cfg) for i, cfg in enumerate(configurations)]
    results = []
    
    if args.parallel <= 1:
        # Sequential execution
        for run_index, cfg in jobs:
            print(f"\n[{run_index}/{len(configurations)}] Starting experiment...")
            ret, run_name, log_file = run_experiment(
                cfg, run_index=run_index, dry_run=args.dry_run, 
                verbose=args.verbose, log_dir=args.log_dir
            )
            results.append((ret, run_name, log_file))
            
            if ret != 0 and not args.dry_run:
                print(f"Experiment {run_index} ({run_name}) failed with return code {ret}")
    else:
        # Parallel execution
        def worker(job):
            run_index, cfg = job
            return run_experiment(
                cfg, run_index=run_index, dry_run=args.dry_run,
                verbose=args.verbose, log_dir=args.log_dir
            )
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all jobs
            futures = {executor.submit(worker, job): job for job in jobs}
            
            # Process completed jobs
            for future in as_completed(futures):
                run_index, cfg = futures[future]
                try:
                    ret, run_name, log_file = future.result()
                    results.append((ret, run_name, log_file))
                    
                    status = "✓" if ret == 0 else "✗"
                    thread_safe_print(f"[{status}] Run {run_index}/{len(configurations)}: {run_name} (exit: {ret})")
                    if log_file:
                        thread_safe_print(f"    Log: {log_file}")
                except Exception as e:
                    thread_safe_print(f"[✗] Run {run_index} failed with exception: {e}")
                    results.append((-1, f"run_{run_index}", None))
    
    # Summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {len(configurations)}")
    successful = sum(1 for r in results if r[0] == 0)
    failed = sum(1 for r in results if r[0] != 0)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed runs:")
        for ret, run_name, log_file in results:
            if ret != 0:
                print(f"  - {run_name} (exit code: {ret})")
                if log_file:
                    print(f"    Log: {log_file}")


if __name__ == '__main__':
    main()

