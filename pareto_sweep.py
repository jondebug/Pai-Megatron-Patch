#!/usr/bin/env python3
"""
Pareto curve analysis for WandB sweep results.

Usage:
    python pareto_sweep.py --group-by rl_ppo_entropy_coeff use_rl_loss
    python pareto_sweep.py --group-by moe_aux_loss_coeff rl_loss_coeff
"""

import argparse
import wandb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
ENTITY = "nvr-israel"
PROJECT = "qwen3-router-training"
SWEEP_ID = "84wy7dfg"
LAST_PERCENT = 0.05

# Key metrics for Pareto analysis (prefer eval, fallback to train)
X_METRIC = "eval/num_tokens_on_critical_path"
X_METRIC_FALLBACK = "train/num_tokens_on_critical_path"
Y_METRIC = "eval/lm loss"
Y_METRIC_FALLBACK = "train/lm loss"

# All config keys to show in tooltip (from sweep config + parsed from name)
TOOLTIP_KEYS = [
    "run_state",
    "rl_loss_active",
    "aux_loss_active",
    "use_rl_loss",
    "moe_aux_loss_coeff", 
    "rl_loss_coeff",
    "entropy_coeff",
    "reward_topn",
    "critic_dims",
    "algorithm",
    "reward_type",
    "rl_algorithm",
    "rl_ppo_entropy_coeff",
    "rl_critic_hidden_dims",
    "rl_reward_topn",
    "rl_reward_type",
    "rl_normalize_rewards",
    "rl_discount_factor",
    "gamma",
]


def get_last_percent_average(history_df, metric_name, last_percent=0.05):
    """Compute average of last X% of a metric's values."""
    if metric_name not in history_df.columns:
        return None
    values = history_df[metric_name].dropna()
    if len(values) == 0:
        return None
    n_last = max(1, int(len(values) * last_percent))
    return values.iloc[-n_last:].mean()


def parse_run_name(name):
    """Extract config from run name like 'PPO_batch_ent0_critic256_rlc2_n1_r68'."""
    import re
    parsed = {}
    
    # Extract entropy coeff: ent0, ent0.01
    ent_match = re.search(r'ent([\d.]+)', name)
    if ent_match:
        parsed['entropy_coeff'] = float(ent_match.group(1))
    
    # Extract rl_loss_coeff: rlc2, rlc0.3
    rlc_match = re.search(r'rlc([\d.]+)', name)
    if rlc_match:
        parsed['rl_loss_coeff'] = float(rlc_match.group(1))
    
    # Extract reward topn: topn1, topn12 (new format) or _n1_ (old format)
    topn_match = re.search(r'topn(\d+)', name)
    if topn_match:
        parsed['reward_topn'] = int(topn_match.group(1))
    else:
        n_match = re.search(r'_n(\d+)_', name)
        if n_match:
            parsed['reward_topn'] = int(n_match.group(1))
    
    # Extract critic dims: critic256, critic256x64x32
    critic_match = re.search(r'critic([\dx]+)', name)
    if critic_match:
        parsed['critic_dims'] = critic_match.group(1)
    
    # Extract algorithm
    if name.startswith('PPO'):
        parsed['algorithm'] = 'ppo'
    elif name.startswith('REINFORCE'):
        parsed['algorithm'] = 'reinforce'
    
    # Batch vs per-token
    if '_batch_' in name:
        parsed['reward_type'] = 'batch'
    elif '_token_' in name:
        parsed['reward_type'] = 'per_token'
    
    # Extract discount factor (gamma): g0.4, g0.9
    g_match = re.search(r'_g([\d.]+)_', name)
    if g_match:
        parsed['gamma'] = float(g_match.group(1))
    
    # Extract RL reward function type
    if '_crit_' in name:
        parsed['rl_reward_type'] = 'critical_path'
    elif '_entr_' in name:
        parsed['rl_reward_type'] = 'entropy'
    elif 'topn' in name:
        parsed['rl_reward_type'] = 'topn_load'
    
    # Reward normalization
    if '_norm_' in name:
        parsed['rl_normalize_rewards'] = True
    
    # Aux loss coeff from name: aux0.001, aux0.01
    aux_match = re.search(r'aux([\d.]+)', name)
    if aux_match:
        parsed['moe_aux_loss_coeff'] = float(aux_match.group(1))
    
    return parsed


def fetch_sweep_data(entity, project, sweep_id, last_percent=0.05):
    """Fetch all runs from sweep and compute last-X% averages."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    print(f"Sweep: {sweep.name}")
    print(f"Total runs: {len(sweep.runs)}")
    
    results = []
    
    for run in sweep.runs:
        # Include all runs that have data (finished, crashed, failed)
        print(f"  Processing: {run.name} ({run.state})")
        
        # Get config - flatten nested dicts
        config = {}
        for k, v in run.config.items():
            if isinstance(v, dict):
                # Skip the big 'advanced' dict, just note it exists
                if k == 'advanced':
                    continue
                for k2, v2 in v.items():
                    config[f"{k}.{k2}"] = v2
            elif isinstance(v, list):
                config[k] = str(v)  # Convert lists to string for grouping
            else:
                config[k] = v
        
        # Also parse config from run name
        name_config = parse_run_name(run.name)
        config.update(name_config)
        
        # Get history
        history = run.history(samples=10000)
        if history.empty:
            continue
        
        # Compute last-X% averages and variance for key metrics
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run.id}"
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "run_url": run_url,
            **config,
        }
        
        # Get all numeric metrics - compute mean and variance
        for col in history.columns:
            if col.startswith('_'):
                continue
            if col not in history.columns:
                continue
            values = history[col].dropna()
            if len(values) == 0:
                continue
            n_last = max(1, int(len(values) * last_percent))
            last_values = values.iloc[-n_last:]
            
            # Clean metric name for column (strip train/ or eval/ prefix)
            clean_name = col.replace("train/", "").replace("eval/", "eval_").replace(" ", "_")
            row[clean_name] = last_values.mean()
            row[f"{clean_name}_var"] = last_values.var() if len(last_values) > 1 else 0.0
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print(f"\nCompleted runs: {len(df)}")
    return df


def find_pareto_front(df, x_col, y_col, minimize_x=True, minimize_y=True):
    """Find Pareto-optimal points. Returns boolean mask."""
    points = df[[x_col, y_col]].values
    is_pareto = np.ones(len(points), dtype=bool)
    
    for i, (x, y) in enumerate(points):
        if not is_pareto[i]:
            continue
        for j, (x2, y2) in enumerate(points):
            if i == j or not is_pareto[j]:
                continue
            # Check if point j dominates point i
            if minimize_x and minimize_y:
                dominates = (x2 <= x and y2 <= y) and (x2 < x or y2 < y)
            elif minimize_x and not minimize_y:
                dominates = (x2 <= x and y2 >= y) and (x2 < x or y2 > y)
            elif not minimize_x and minimize_y:
                dominates = (x2 >= x and y2 <= y) and (x2 > x or y2 < y)
            else:
                dominates = (x2 >= x and y2 >= y) and (x2 > x or y2 > y)
            
            if dominates:
                is_pareto[i] = False
                break
    
    return is_pareto


def add_binary_flags(df):
    """Add binary filter columns for rl_loss_active and aux_loss_active."""
    # RL loss is active if use_rl_loss=True AND rl_loss_coeff > 0
    df['rl_loss_active'] = False
    if 'use_rl_loss' in df.columns and 'rl_loss_coeff' in df.columns:
        df['rl_loss_active'] = (df['use_rl_loss'] == True) & (df['rl_loss_coeff'] > 0)
    elif 'use_rl_loss' in df.columns:
        df['rl_loss_active'] = df['use_rl_loss'] == True
    
    # Aux loss is active if moe_aux_loss_coeff > 0
    df['aux_loss_active'] = False
    if 'moe_aux_loss_coeff' in df.columns:
        df['aux_loss_active'] = df['moe_aux_loss_coeff'] > 0
    
    # Override rl_reward_type to "none" for no-RL runs so they form a distinct group
    if 'rl_reward_type' in df.columns:
        df.loc[~df['rl_loss_active'], 'rl_reward_type'] = 'none'
    
    return df


def _clean_metric_name(metric):
    """Convert WandB metric name to dataframe column name."""
    return metric.replace("train/", "").replace("eval/", "eval_").replace(" ", "_")


def create_pareto_plot(df, group_by_cols, x_col, y_col, x_min=None, y_max=None):
    """Create interactive Pareto plot with grouping."""
    
    x_clean = _clean_metric_name(x_col)
    y_clean = _clean_metric_name(y_col)
    
    # Fallback to train metrics if eval columns are missing or all-zero
    x_fb = _clean_metric_name(X_METRIC_FALLBACK)
    y_fb = _clean_metric_name(Y_METRIC_FALLBACK)
    
    if x_clean not in df.columns and x_fb in df.columns:
        print(f"  Falling back from {x_clean} to {x_fb}")
        x_clean = x_fb
    if y_clean not in df.columns and y_fb in df.columns:
        print(f"  Falling back from {y_clean} to {y_fb}")
        y_clean = y_fb
    
    # For runs missing eval metrics, fill from train metrics
    if x_clean in df.columns and x_fb in df.columns and x_clean != x_fb:
        mask = df[x_clean].isna() | (df[x_clean] == 0)
        if mask.any():
            df.loc[mask, x_clean] = df.loc[mask, x_fb]
            print(f"  Filled {mask.sum()} missing {x_clean} values from {x_fb}")
    if y_clean in df.columns and y_fb in df.columns and y_clean != y_fb:
        mask = df[y_clean].isna() | (df[y_clean] == 0)
        if mask.any():
            df.loc[mask, y_clean] = df.loc[mask, y_fb]
            print(f"  Filled {mask.sum()} missing {y_clean} values from {y_fb}")
    
    if x_clean not in df.columns or y_clean not in df.columns:
        print(f"Error: Required columns not found.")
        print(f"  Looking for: {x_clean}, {y_clean}")
        print(f"  Available: {list(df.columns)}")
        return None
    
    # Create group key with descriptive names
    group_cols_present = [c for c in group_by_cols if c in df.columns]
    if not group_cols_present:
        print(f"Warning: None of {group_by_cols} found in data. Using 'all' as group.")
        df['group'] = 'all'
    else:
        # Create descriptive group labels
        def make_group_label(row):
            parts = []
            for col in group_cols_present:
                val = row[col]
                # Make boolean values more descriptive
                if col == 'rl_loss_active':
                    parts.append("RL: ON" if val else "RL: OFF")
                elif col == 'aux_loss_active':
                    parts.append("Aux: ON" if val else "Aux: OFF")
                elif col == 'use_rl_loss':
                    parts.append("RL: ON" if val else "RL: OFF")
                elif isinstance(val, bool):
                    parts.append(f"{col}={'ON' if val else 'OFF'}")
                else:
                    parts.append(f"{col}={val}")
            return " | ".join(parts)
        
        df['group'] = df.apply(make_group_label, axis=1)
    
    # Build hover text with all tooltip keys
    hover_texts = []
    for _, row in df.iterrows():
        lines = [f"<b>{row['run_name']}</b>"]
        
        # Add run URL if available
        if 'run_url' in row and pd.notna(row['run_url']):
            lines.append(f"<a href='{row['run_url']}' target='_blank'>Open in WandB</a>")
        lines.append("")
        
        # Main metrics with variance
        x_var_col = f"{x_clean}_var"
        y_var_col = f"{y_clean}_var"
        x_var = row.get(x_var_col, 0) if x_var_col in row else 0
        y_var = row.get(y_var_col, 0) if y_var_col in row else 0
        
        lines.append(f"<b>{x_clean}:</b> {row[x_clean]:.2f} (±{np.sqrt(x_var):.2f})")
        lines.append(f"<b>{y_clean}:</b> {row[y_clean]:.4f} (±{np.sqrt(y_var):.4f})")
        lines.append("")
        
        lines.append("<b>Config:</b>")
        
        # RL-related keys to skip if use_rl_loss is False
        rl_keys = {'rl_loss_active', 'rl_loss_coeff', 'entropy_coeff', 'reward_topn', 
                   'critic_dims', 'algorithm', 'reward_type', 'rl_algorithm', 
                   'rl_ppo_entropy_coeff', 'rl_critic_hidden_dims', 'rl_reward_topn'}
        
        use_rl = row.get('use_rl_loss', False)
        
        for key in TOOLTIP_KEYS:
            # Skip RL keys if RL is not active
            if not use_rl and key in rl_keys:
                continue
            if key in row:
                val = row[key]
                if pd.notna(val):
                    lines.append(f"  {key}: {val}")
        hover_texts.append("<br>".join(lines))
    
    df['hover_text'] = hover_texts
    
    # Create figure
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    groups = df['group'].unique()
    
    for i, group in enumerate(sorted(groups)):
        group_df = df[df['group'] == group].copy()
        color = colors[i % len(colors)]
        
        # Find Pareto front for this group
        pareto_mask = find_pareto_front(group_df, x_clean, y_clean, 
                                        minimize_x=True, minimize_y=True)
        
        # Plot all points
        fig.add_trace(go.Scatter(
            x=group_df[x_clean],
            y=group_df[y_clean],
            mode='markers',
            name=group,
            marker=dict(size=10, color=color, opacity=0.6),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=group_df['hover_text'],
        ))
        
        # Plot Pareto front line
        pareto_df = group_df[pareto_mask].sort_values(x_clean)
        if len(pareto_df) > 1:
            fig.add_trace(go.Scatter(
                x=pareto_df[x_clean],
                y=pareto_df[y_clean],
                mode='lines',
                name=f'{group} (Pareto)',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))
        
        # Highlight Pareto points
        fig.add_trace(go.Scatter(
            x=pareto_df[x_clean],
            y=pareto_df[y_clean],
            mode='markers',
            name=f'{group} (Pareto)',
            marker=dict(size=14, color=color, symbol='star', 
                       line=dict(width=2, color='white')),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=pareto_df['hover_text'],
            showlegend=False,
        ))
    
    # Set axis ranges
    xaxis_range = None
    yaxis_range = None
    if x_min is not None:
        x_max_val = df[x_clean].max() * 1.05  # Add 5% padding
        xaxis_range = [x_min, x_max_val]
    if y_max is not None:
        y_min_val = df[y_clean].min() * 0.95  # Add 5% padding
        yaxis_range = [y_min_val, y_max]
    
    fig.update_layout(
        title=f"Pareto Curves: {y_clean} vs {x_clean}<br><sup>Grouped by: {', '.join(group_by_cols)}</sup>",
        xaxis_title=x_clean,
        yaxis_title=y_clean,
        xaxis_range=xaxis_range,
        yaxis_range=yaxis_range,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        width=1200,
        height=800,
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Pareto curve analysis for WandB sweep')
    parser.add_argument('--group-by', nargs='+', default=['rl_ppo_entropy_coeff', 'use_rl_loss'],
                       help='Metrics to group curves by')
    parser.add_argument('--sweep-id', default=SWEEP_ID, help='WandB sweep ID')
    parser.add_argument('--refresh', action='store_true', help='Re-fetch data from WandB')
    parser.add_argument('--output', default='pareto_plot.html', help='Output HTML file')
    
    # Binary filters
    parser.add_argument('--rl-loss-active', type=str, choices=['true', 'false', 'any'], default='any',
                       help='Filter by RL loss active (true/false/any)')
    parser.add_argument('--aux-loss-active', type=str, choices=['true', 'false', 'any'], default='any',
                       help='Filter by aux loss active (true/false/any)')
    
    # Axis limits
    parser.add_argument('--y-max', type=float, default=3.5,
                       help='Maximum value for Y axis (lm_loss). Default: 3.5')
    parser.add_argument('--x-min', type=float, default=2500,
                       help='Minimum value for X axis (num_tokens_on_critical_path). Default: 2500')
    
    args = parser.parse_args()
    
    cache_file = f"sweep_{args.sweep_id}_cache.csv"
    
    # Try to load cached data
    try:
        if args.refresh:
            raise FileNotFoundError("Refresh requested")
        df = pd.read_csv(cache_file)
        print(f"Loaded cached data from {cache_file}")
    except FileNotFoundError:
        print("Fetching data from WandB...")
        df = fetch_sweep_data(ENTITY, PROJECT, args.sweep_id, LAST_PERCENT)
        df.to_csv(cache_file, index=False)
        print(f"Cached data to {cache_file}")
    
    if df.empty:
        print("No completed runs found!")
        return
    
    # Add binary filter columns
    df = add_binary_flags(df)
    
    # Apply filters
    original_count = len(df)
    if args.rl_loss_active != 'any':
        filter_val = args.rl_loss_active == 'true'
        df = df[df['rl_loss_active'] == filter_val]
        print(f"Filtered by rl_loss_active={filter_val}: {original_count} -> {len(df)} runs")
    
    if args.aux_loss_active != 'any':
        filter_val = args.aux_loss_active == 'true'
        df = df[df['aux_loss_active'] == filter_val]
        print(f"Filtered by aux_loss_active={filter_val}: {len(df)} runs remaining")
    
    if df.empty:
        print("No runs match the filter criteria!")
        return
    
    print(f"\nCreating Pareto plot grouped by: {args.group_by}")
    print(f"Axis limits: x_min={args.x_min}, y_max={args.y_max}")
    
    fig = create_pareto_plot(df, args.group_by, X_METRIC, Y_METRIC, 
                             x_min=args.x_min, y_max=args.y_max)
    
    if fig:
        fig.write_html(args.output)
        print(f"\nSaved interactive plot to: {args.output}")
        
        # Also print text summary
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        
        x_clean = _clean_metric_name(X_METRIC)
        y_clean = _clean_metric_name(Y_METRIC)
        # Use fallback names if primary not available
        if x_clean not in df.columns:
            x_clean = _clean_metric_name(X_METRIC_FALLBACK)
        if y_clean not in df.columns:
            y_clean = _clean_metric_name(Y_METRIC_FALLBACK)
        
        print(f"\n{x_clean} range: {df[x_clean].min():.2f} - {df[x_clean].max():.2f}")
        print(f"{y_clean} range: {df[y_clean].min():.4f} - {df[y_clean].max():.4f}")
        
        print(f"\nGrouping columns: {args.group_by}")
        for col in args.group_by:
            if col in df.columns:
                print(f"  {col}: {sorted(df[col].unique())}")


if __name__ == "__main__":
    main()

