#!/usr/bin/env python3
"""Quick summary of sweep results."""

import pandas as pd

df = pd.read_csv("sweep_84wy7dfg_last5pct.csv")

# Key metrics to show
key_cols = [
    'run_name',
    'moe_aux_loss_coeff',
    'train/lm loss_last5pct',
    'train/grad_norm_last5pct',
    'train/tokens_routed_to_expert_0_layer_0_last5pct',
    'train/token_assignment_entropy_last5pct',
    'train/load_balancing_loss_layer_0_last5pct',
]

# Filter to existing columns
key_cols = [c for c in key_cols if c in df.columns]

# Rename for readability
rename = {
    'train/lm loss_last5pct': 'lm_loss',
    'train/grad_norm_last5pct': 'grad_norm',
    'train/tokens_routed_to_expert_0_layer_0_last5pct': 'expert0_tokens',
    'train/token_assignment_entropy_last5pct': 'routing_entropy',
    'train/load_balancing_loss_layer_0_last5pct': 'lb_loss',
}

summary = df[key_cols].rename(columns=rename)

print("\n" + "=" * 80)
print("SWEEP SUMMARY: Last 5% Averages (5 completed runs)")
print("=" * 80)
print()

# Sort by aux loss coeff for clarity
if 'moe_aux_loss_coeff' in summary.columns:
    summary = summary.sort_values('moe_aux_loss_coeff', ascending=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
- moe_aux_loss_coeff varied: 1.0, 0.1, 0.01, 0.001, 0.0
- expert0_tokens: number of tokens routed to expert 0 in layer 0 (always 0!)
- routing_entropy: lower = more concentrated routing
- lb_loss: load balancing loss

Key finding: expert0_tokens is 0 across all runs - RL is not routing to expert 0.
""")






