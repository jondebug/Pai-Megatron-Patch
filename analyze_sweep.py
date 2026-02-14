#!/usr/bin/env python3
"""
Analyze WandB sweep results: compute last-5% average for all metrics.
"""

import wandb
import pandas as pd
import numpy as np
from collections import defaultdict

# Configuration
ENTITY = "nvr-israel"
PROJECT = "qwen3-router-training"
SWEEP_ID = "84wy7dfg"
LAST_PERCENT = 0.05  # Last 5%

def get_last_percent_average(history_df, metric_name, last_percent=0.05):
    """Compute average of last X% of a metric's values."""
    if metric_name not in history_df.columns:
        return None
    
    values = history_df[metric_name].dropna()
    if len(values) == 0:
        return None
    
    n_last = max(1, int(len(values) * last_percent))
    last_values = values.iloc[-n_last:]
    return last_values.mean()

def analyze_sweep(entity, project, sweep_id, last_percent=0.05):
    """Analyze all runs in a sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    print(f"Sweep: {sweep.name}")
    print(f"Total runs: {len(sweep.runs)}")
    print(f"Computing last {last_percent*100:.0f}% averages...\n")
    
    all_results = []
    
    for run in sweep.runs:
        if run.state != "finished":
            print(f"  Skipping {run.name} (state: {run.state})")
            continue
        
        print(f"  Processing: {run.name} ({run.id})")
        
        # Get run config
        config = dict(run.config)
        
        # Get full history (all logged metrics over time)
        history = run.history(samples=10000)  # Increase if needed
        
        if history.empty:
            print(f"    No history data")
            continue
        
        # Compute last-5% average for each metric
        run_result = {
            "run_id": run.id,
            "run_name": run.name,
            **config  # Include hyperparameters
        }
        
        # Get all numeric columns (metrics)
        metric_cols = [c for c in history.columns 
                      if c not in ['_step', '_runtime', '_timestamp'] 
                      and pd.api.types.is_numeric_dtype(history[c])]
        
        for metric in metric_cols:
            avg = get_last_percent_average(history, metric, last_percent)
            if avg is not None:
                run_result[f"{metric}_last5pct"] = avg
        
        all_results.append(run_result)
    
    return pd.DataFrame(all_results)

def main():
    print("=" * 60)
    print("WandB Sweep Analysis")
    print("=" * 60)
    
    df = analyze_sweep(ENTITY, PROJECT, SWEEP_ID, LAST_PERCENT)
    
    if df.empty:
        print("No completed runs found!")
        return
    
    # Save to CSV
    output_file = f"sweep_{SWEEP_ID}_last5pct.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Find metric columns (those ending in _last5pct)
    metric_cols = [c for c in df.columns if c.endswith('_last5pct')]
    
    # Print stats for key metrics
    key_metrics = [
        'lm_loss_last5pct',
        'rl_loss_last5pct', 
        'tokens_routed_to_expert_0_layer_0_last5pct',
        'grad_norm_last5pct'
    ]
    
    print(f"\nNumber of completed runs: {len(df)}")
    print("\nKey Metrics (last 5% averages):")
    print("-" * 40)
    
    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"{metric}:")
                print(f"  mean: {values.mean():.4f}")
                print(f"  std:  {values.std():.4f}")
                print(f"  min:  {values.min():.4f}")
                print(f"  max:  {values.max():.4f}")
    
    # Show all hyperparameters and their values
    config_cols = [c for c in df.columns 
                   if c not in ['run_id', 'run_name'] 
                   and not c.endswith('_last5pct')]
    
    if config_cols:
        print("\nHyperparameters varied in sweep:")
        print("-" * 40)
        for col in config_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) > 1:
                print(f"  {col}: {sorted(unique_vals)}")
    
    # Print full table
    print("\n" + "=" * 60)
    print("FULL RESULTS TABLE")
    print("=" * 60)
    
    # Select most important columns for display
    display_cols = ['run_name'] + config_cols[:5] + key_metrics[:4]
    display_cols = [c for c in display_cols if c in df.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df[display_cols].to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = main()






