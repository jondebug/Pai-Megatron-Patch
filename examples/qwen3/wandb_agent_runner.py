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
    return data['combinations'], data['fixed_params'], data.get('sweep_dir', None), data.get('sweep_summary', None)


def build_run_name(sweep_params: dict, run_index: int, fixed_params: dict = None,
                   sweep_name_prefix: str = None) -> str:
    """Build a descriptive run name from ONLY the sweep-varying parameters.
    
    Fixed params shared across all runs are omitted — they belong in the sweep name.
    If sweep_name_prefix is provided, it is prepended to avoid output directory collisions
    across sweeps (used when fresh_start=true).
    """
    all_params = {}
    if fixed_params:
        all_params.update(fixed_params)
    all_params.update(sweep_params)
    
    parts = []
    
    # Determine if RL is active for this specific run
    rl_active = all_params.get('use_rl_loss', True)
    
    # Only include parameters that are in sweep_params (i.e., they vary across runs).
    # RL-specific params are skipped entirely when use_rl_loss=False.
    
    if rl_active:
        # Reward type (short names)
        if 'rl_reward_type' in sweep_params:
            _REWARD_SHORT = {
                'critical_path': 'crit',
                'topn_load': 'topn',
                'per_token_topn_binary': 'ptbin',
                'per_token_load_weighted': 'ptload',
                'entropy': 'entr',
                'expert0': 'exp0',
            }
            rt = sweep_params['rl_reward_type']
            parts.append(_REWARD_SHORT.get(rt, rt))
        
        # Top-N (only meaningful for topn-based rewards)
        if 'rl_reward_topn' in sweep_params:
            parts.append(f'n{sweep_params["rl_reward_topn"]}')
        
        # RL loss coefficient
        if 'rl_loss_coeff' in sweep_params:
            parts.append(f'rlc{sweep_params["rl_loss_coeff"]}')
        
        # Discount factor
        if 'rl_discount_factor' in sweep_params:
            parts.append(f'g{sweep_params["rl_discount_factor"]}')
        
        # Normalize rewards
        if 'rl_normalize_rewards' in sweep_params:
            parts.append('norm' if sweep_params['rl_normalize_rewards'] else 'nonorm')
        
        # Algorithm (only if it varies)
        if 'rl_algorithm' in sweep_params:
            parts.append(sweep_params['rl_algorithm'].upper())
        
        # Baseline type (only if it varies)
        if 'rl_ppo_baseline_type' in sweep_params:
            parts.append(sweep_params['rl_ppo_baseline_type'])
        
        # Entropy coeff (only if it varies)
        if 'rl_ppo_entropy_coeff' in sweep_params:
            parts.append(f'ent{sweep_params["rl_ppo_entropy_coeff"]}')
        
        # Critic dims (only if it varies)
        if 'rl_critic_hidden_dims' in sweep_params:
            dims = sweep_params['rl_critic_hidden_dims']
            dims_str = 'x'.join(str(d) for d in dims) if isinstance(dims, list) else str(dims)
            parts.append(f'c{dims_str}')
        
        # PPO clip ratio (only if it varies)
        if 'rl_ppo_clip_ratio' in sweep_params:
            parts.append(f'clip{sweep_params["rl_ppo_clip_ratio"]}')
        
        # EMA loads (only if it varies)
        if sweep_params.get('rl_use_ema_loads', False):
            parts.append('ema')
        
        # Layer-aware critic (only if it varies)
        if 'rl_critic_layer_aware' in sweep_params:
            parts.append('lcrit' if sweep_params['rl_critic_layer_aware'] else 'nocrit')
        
        # PPO re-evaluation and multi-epoch
        if sweep_params.get('rl_ppo_reeval', False) or all_params.get('rl_ppo_reeval', False):
            epochs = sweep_params.get('rl_ppo_epochs', all_params.get('rl_ppo_epochs', 1))
            parts.append(f'ppo_k{epochs}' if int(epochs) > 1 else 'ppo')
        
        # LM reward coefficient
        if 'rl_lm_reward_coeff' in sweep_params:
            v = sweep_params['rl_lm_reward_coeff']
            if v and float(v) > 0:
                parts.append(f'lm{v}')
    else:
        # RL is off — just mark it
        parts.append('norl')
    
    # Non-RL params (always included when they vary)
    
    # Aux loss coefficient
    if 'moe_aux_loss_coeff' in sweep_params:
        v = sweep_params['moe_aux_loss_coeff']
        parts.append(f'aux{v}' if v and float(v) > 0 else 'noaux')
    
    # ALF-LB (aux-loss-free load balancing via dynamic expert bias)
    if sweep_params.get('moe_router_enable_expert_bias', False):
        rate = sweep_params.get('moe_router_bias_update_rate', all_params.get('moe_router_bias_update_rate', 0.001))
        parts.append(f'alflb_a{rate}')
    
    # Load balancing type (only if it varies and isn't already covered by ALF-LB)
    if 'moe_router_load_balancing_type' in sweep_params and not sweep_params.get('moe_router_enable_expert_bias', False):
        parts.append(f'lb_{sweep_params["moe_router_load_balancing_type"]}')
    
    # Topology-aware routing
    if all_params.get('moe_router_topology_aware', False):
        lam = sweep_params.get('moe_router_topology_lambda', all_params.get('moe_router_topology_lambda', 0.01))
        parts.append(f'topo{lam}')
    
    # Critical-path dynamic bias
    if all_params.get('moe_router_critical_path_bias', False):
        topn = sweep_params.get('moe_router_critical_path_topn', all_params.get('moe_router_critical_path_topn', 1))
        alpha = sweep_params.get('moe_router_critical_path_alpha', all_params.get('moe_router_critical_path_alpha', 0.001))
        parts.append(f'cpb_n{topn}_a{alpha}')
    
    # KL loss coefficient
    if 'kl_loss_coeff' in sweep_params:
        v = sweep_params['kl_loss_coeff']
        if v and float(v) > 0:
            parts.append(f'kl{v}')
    
    # Fallback: if no parts generated, use a generic label
    if not parts:
        parts.append('baseline')
    
    parts.append(f'r{run_index:02d}')
    
    name = '_'.join(parts)
    if sweep_name_prefix:
        name = f'{sweep_name_prefix}_{name}'
    return name


def build_command(fixed_params: dict, sweep_params: dict, run_name: str) -> list:
    """Build the training command."""
    
    # Merge params (sweep overrides fixed)
    config = {**fixed_params, **sweep_params}
    config['wandb_run_name'] = run_name
    
    # Make output path unique per sweep run to avoid checkpoint collisions
    base_output = config.get('output_basepath', '/tmp/output')
    unique_output = f"{base_output}/{run_name}"
    
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
        unique_output,
    ]
    
    # Build extra args
    extra_args = []
    
    # Boolean flags
    bool_flags = [
        ('router_only_training', '--router-only-training'),
        ('enable_wandb_logging', '--enable-wandb-logging'),
        ('use_rl_loss', '--use_rl_loss'),
        ('rl_per_token_rewards', '--rl-per-token-rewards'),
        ('rl_normalize_rewards', '--rl-normalize-rewards'),
        ('rl_use_ema_loads', '--rl-use-ema-loads'),
        ('rl_critic_layer_aware', '--rl-critic-layer-aware'),
        ('moe_router_enable_expert_bias', '--moe-router-enable-expert-bias'),
        ('rl_ppo_reeval', '--rl-ppo-reeval'),
        ('moe_router_topology_aware', '--moe-router-topology-aware'),
        ('moe_router_critical_path_bias', '--moe-router-critical-path-bias'),
        ('eval_kl_tracking', '--eval-kl-tracking'),
        ('log_expert_heatmap', '--log-expert-heatmap'),
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
        ('rl_reward_type', '--rl-reward-type'),
        ('rl_reward_topn', '--rl-reward-topn'),
        ('rl_discount_factor', '--rl-discount-factor'),
        ('rl_ppo_clip_ratio', '--rl-ppo-clip-ratio'),
        ('moe_aux_loss_coeff', '--moe-aux-loss-coeff'),
        ('moe_router_score_function', '--moe-router-score-function'),
        ('moe_router_bias_update_rate', '--moe-router-bias-update-rate'),
        ('moe_router_load_balancing_type', '--moe-router-load-balancing-type'),
        ('rl_ppo_epochs', '--rl-ppo-epochs'),
        ('rl_lm_reward_coeff', '--rl-lm-reward-coeff'),
        ('kl_loss_coeff', '--kl-loss-coeff'),
        ('moe_router_topology_lambda', '--moe-router-topology-lambda'),
        ('moe_router_critical_path_topn', '--moe-router-critical-path-topn'),
        ('moe_router_critical_path_alpha', '--moe-router-critical-path-alpha'),
        ('exit_duration_in_mins', '--exit-duration-in-mins'),
        ('train_iters', '--train-iters'),
        ('eval_interval', '--eval-interval'),
        ('eval_iters', '--eval-iters'),
        ('hellaswag_eval_interval', '--hellaswag-eval-interval'),
        ('hellaswag_eval_limit', '--hellaswag-eval-limit'),
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
    
    combinations, fixed_params, stored_sweep_dir, sweep_summary = load_combinations(combos_path)
    
    # Use stored sweep_dir if not provided
    if sweep_dir is None and stored_sweep_dir:
        sweep_dir = Path(stored_sweep_dir)
    
    if args.run_index < 0 or args.run_index >= len(combinations):
        print(f"ERROR: run_index {args.run_index} out of range [0, {len(combinations)})")
        return 1
    
    sweep_params = combinations[args.run_index]
    
    # Always apply prefix from wandb_run_name_base to ensure consistent naming across sweeps
    sweep_name_prefix = fixed_params.get('wandb_run_name_base') or fixed_params.get('sweep_name')
    
    run_name = build_run_name(sweep_params, args.run_index, fixed_params,
                              sweep_name_prefix=sweep_name_prefix)
    
    # WandB resume: if this run was previously started, resume the same wandb run.
    # Only attempt resume when fresh_start is NOT set.
    all_params_for_resume = {**fixed_params, **sweep_params}
    base_output = all_params_for_resume.get('output_basepath', '/tmp/output')
    run_output_dir = os.path.join(base_output, run_name)
    wandb_id_file = os.path.join(run_output_dir, 'wandb_run_id.txt')
    
    fresh_start = fixed_params.get('fresh_start', False)
    
    # Cross-sweep resume: if exact run_name dir doesn't exist, search for a dir
    # with the same config content but a different r## index or prefix (from a previous sweep).
    if not fresh_start and not os.path.exists(wandb_id_file):
        import re
        def _strip_for_match(name):
            """Strip r## index and known prefixes for config-based matching."""
            s = re.sub(r'_r\d+$', '', name)
            prefix = sweep_name_prefix + '_' if sweep_name_prefix else ''
            if prefix and s.startswith(prefix):
                s = s[len(prefix):]
            return s
        config_part = _strip_for_match(run_name)
        if os.path.isdir(base_output):
            for d in os.listdir(base_output):
                if d == run_name:
                    continue
                d_config = _strip_for_match(d)
                if d_config == config_part:
                    candidate_id_file = os.path.join(base_output, d, 'wandb_run_id.txt')
                    candidate_ckpt = os.path.join(base_output, d, 'checkpoint')
                    if os.path.exists(candidate_id_file) or os.path.isdir(candidate_ckpt):
                        print(f"CROSS-SWEEP RESUME: matched {d} -> {run_name} (same config, different index)")
                        run_output_dir = os.path.join(base_output, d)
                        wandb_id_file = os.path.join(run_output_dir, 'wandb_run_id.txt')
                        run_name = d
                        break

    try:
        import wandb
        if not fresh_start and os.path.exists(wandb_id_file):
            saved_id = open(wandb_id_file).read().strip()
            print(f"RESUME: Found previous wandb run ID: {saved_id}")
            if wandb.run is not None:
                wandb.finish(quiet=True)
            wandb_project = all_params_for_resume.get('wandb_project_name', 'qwen3-router-training')
            wandb.init(id=saved_id, resume="must", project=wandb_project,
                       entity="nvr-israel", name=run_name)
            print(f"RESUMED wandb run: {saved_id}")
        elif wandb.run is not None:
            os.makedirs(run_output_dir, exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                f.write(wandb.run.id)
            print(f"SAVED wandb run ID: {wandb.run.id} to {wandb_id_file}")
    except Exception as e:
        print(f"Note: WandB resume handling: {e}")

    # Log sweep parameters to wandb
    try:
        import wandb
        if wandb.run is not None:
            # Merge fixed and sweep params -- log ALL actual hyperparameters
            all_params = {**fixed_params, **sweep_params}
            
            # Build a comprehensive config dict with all experiment hyperparameters
            # Include every param that could matter for analysis
            wandb_config = {
                'run_index': args.run_index,
                'run_name': run_name,
            }
            
            # Log all sweep-varying params explicitly (these are the ones that matter most)
            for key, value in sweep_params.items():
                wandb_config[f'sweep/{key}'] = value
            
            # Log all fixed params that are experiment-relevant
            experiment_keys = [
                'rl_algorithm', 'rl_per_token_rewards', 'rl_ppo_entropy_coeff',
                'rl_ppo_baseline_type', 'rl_critic_hidden_dims', 'rl_loss_coeff',
                'rl_reward_type', 'rl_reward_topn', 'rl_discount_factor',
                'use_rl_loss', 'moe_aux_loss_coeff', 'train_iters',
                'global_batch_size', 'seq_len', 'lr', 'min_lr',
            ]
            for key in experiment_keys:
                if key in all_params:
                    wandb_config[key] = all_params[key]
            
            # Also log any remaining params from all_params that aren't infrastructure
            infra_keys = {
                'env', 'model_size', 'batch_size', 'precision', 'tp', 'pp', 'cp',
                'etp', 'ep', 'sp', 'do', 'fl', 'sft', 'ac', 'optimizer_offload',
                'save_interval', 'dataset_path', 'valid_dataset_path',
                'pretrain_checkpoint_path', 'output_basepath', 'train_tokens',
                'warmup_tokens', 'wandb_project_name', 'wandb_run_name',
                'wandb_run_name_base', 'sweep_name', 'wandb_run_tags',
                'router_only_training', 'enable_wandb_logging', 'pad_len',
            }
            for key, value in all_params.items():
                if key not in infra_keys and key not in wandb_config:
                    wandb_config[key] = value
            
            wandb.config.update(wandb_config, allow_val_change=True)
            
            # Log sweep summary as notes if available
            if sweep_summary:
                wandb.run.notes = (
                    f"Grid: {sweep_summary.get('grid_description', 'N/A')}\n"
                    f"Total combos: {sweep_summary.get('total_combinations', 'N/A')}\n"
                    f"Swept: {list(sweep_summary.get('swept_params', {}).keys())}"
                )
            
            print(f"Logged sweep params to wandb run: {wandb.run.name}")
            print(f"  Sweep params: {sweep_params}")
    except Exception as e:
        print(f"Note: Could not log to wandb: {e}")
    
    # Capture WandB run ID for post-training benchmark logging
    wandb_run_id = None
    try:
        import wandb
        if wandb.run is not None:
            wandb_run_id = wandb.run.id
    except Exception:
        pass
    
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
    
    # Save wandb run ID after training so the next allocation can resume the same wandb run.
    # The training script (Megatron) initializes wandb internally, so wandb.run may not be
    # available here. Instead, find the wandb run ID from the wandb output directory.
    try:
        wandb_data_dir = os.path.join(script_dir, '..', '..', '..', 'wandb_data', 'wandb')
        if os.path.isdir(wandb_data_dir):
            run_dirs = sorted(
                [d for d in os.listdir(wandb_data_dir) if d.startswith('run-') and run_name in os.listdir(os.path.join(wandb_data_dir, d)) == False],
                reverse=True
            )
        # Simpler approach: scan for the run ID file that Megatron's wandb created
        # by looking for the most recent wandb run directory
        import glob
        wandb_run_files = sorted(glob.glob(os.path.join(wandb_data_dir, 'run-*', 'run-*.wandb')), reverse=True)
        for wrf in wandb_run_files[:5]:
            run_dir_name = os.path.basename(os.path.dirname(wrf))
            # run dir format: run-YYYYMMDD_HHMMSS-RUNID
            parts = run_dir_name.split('-')
            if len(parts) >= 3:
                found_id = parts[-1]
                # Verify this wandb run matches our run_name by checking the wandb config
                config_file = os.path.join(os.path.dirname(wrf), 'files', 'config.yaml')
                if os.path.exists(config_file):
                    import yaml
                    with open(config_file) as cf:
                        wconfig = yaml.safe_load(cf)
                    if wconfig.get('run_name', {}).get('value', '') == run_name:
                        os.makedirs(run_output_dir, exist_ok=True)
                        with open(wandb_id_file, 'w') as f:
                            f.write(found_id)
                        print(f"POST-TRAINING: Saved wandb run ID {found_id} to {wandb_id_file}")
                        wandb_run_id = found_id
                        break
    except Exception as e:
        print(f"Note: Could not save wandb run ID post-training: {e}")

    # Submit post-training benchmark if enabled and training succeeded
    all_params = {**fixed_params, **sweep_params}
    if result.returncode == 0 and all_params.get('run_benchmarks', False):
        try:
            benchmark_script = os.path.join(script_dir, 'benchmarks', 'submit_benchmark.sh')
            base_output = all_params.get('output_basepath', '/tmp/output')
            checkpoint_dir = os.path.join(base_output, run_name, 'checkpoint')
            # Find the actual checkpoint subdirectory (contains iter_XXXXXX)
            ckpt_subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if ckpt_subdirs:
                checkpoint_dir = os.path.join(checkpoint_dir, ckpt_subdirs[0])
            
            benchmark_cmd = [
                'sbatch',
                benchmark_script,
                '--checkpoint-dir', checkpoint_dir,
                '--run-name', run_name,
            ]
            if wandb_run_id:
                benchmark_cmd.extend(['--wandb-run-id', wandb_run_id])
            wandb_project = all_params.get('wandb_project_name', 'qwen3-router-training')
            benchmark_cmd.extend(['--wandb-project', wandb_project])
            
            print(f"\nSubmitting benchmark job for: {run_name}")
            print(f"  Checkpoint: {checkpoint_dir}")
            print(f"  Command: {' '.join(benchmark_cmd)}")
            bench_result = subprocess.run(benchmark_cmd, capture_output=True, text=True)
            if bench_result.returncode == 0:
                print(f"  Benchmark job submitted: {bench_result.stdout.strip()}")
            else:
                print(f"  WARNING: Benchmark submission failed: {bench_result.stderr.strip()}")
        except Exception as e:
            print(f"  WARNING: Could not submit benchmark job: {e}")
    
    return result.returncode


if __name__ == '__main__':
    exit(main())

