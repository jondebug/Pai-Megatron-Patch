#!/bin/bash
# Simple example: Add --router-only-training to any existing training command

echo "Router-Only Training Example"
echo "Just add --router-only-training to your existing command:"
echo ""

# Your original command with --router-only-training added:
sh run_mcore_qwen3.sh \
dsw \
A3B \
1 \
8 \
1e-4 \
1e-6 \
128 \
128 \
bf16 \
1 \
1 \
1 \
1 \
8 \
true \
true \
true \
false \
sel \
false \
100000 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
5000 \
500 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning \
--router-only-training

echo ""
echo "==============================================="
echo "ROUTER + RL TRAINING EXAMPLE"
echo "==============================================="
echo "Add RL loss for router optimization:"
echo ""

# Example with RL loss using REINFORCE
sh run_mcore_qwen3.sh \
dsw \
A3B \
1 \
8 \
1e-4 \
1e-6 \
128 \
128 \
bf16 \
1 \
1 \
1 \
1 \
8 \
true \
true \
true \
false \
sel \
false \
100000 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
5000 \
500 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning \
--router-only-training \
--use-rl-loss \
--rl-algorithm reinforce \
--rl-loss-coeff 0.1

echo ""
echo "RL Training Options:"
echo "--use-rl-loss                    # Enable RL loss"
echo "--rl-algorithm [reinforce|ppo]   # Choose algorithm"  
echo "--rl-loss-coeff 0.1              # RL loss coefficient"
echo ""
echo "==============================================="
echo "WANDB LOGGING EXAMPLE"
echo "==============================================="
echo "Add wandb logging for metrics tracking:"
echo ""

# Example with wandb logging
sh run_mcore_qwen3.sh \
dsw \
A3B \
1 \
8 \
1e-4 \
1e-6 \
128 \
128 \
bf16 \
1 \
1 \
1 \
1 \
8 \
true \
true \
true \
false \
sel \
false \
100000 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
5000 \
500 \
/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning \
--router-only-training \
--use-wandb \
--wandb-project my-router-training \
--wandb-name router-experiment-1 \
--wandb-tags router-only moe-training

echo ""
echo "Wandb Options:"
echo "--use-wandb                      # Enable wandb logging"
echo "--wandb-project my-project       # Wandb project name"
echo "--wandb-name my-experiment       # Experiment name"
echo "--wandb-tags tag1 tag2           # Tags for organization"
echo ""
echo "This will log:"
echo "• train/lm_loss                  # Language modeling loss"
echo "• train/load_balancing_loss      # MoE load balancing loss" 
echo "• train/learning_rate            # Learning rate"
echo "• train/grad_norm                # Gradient norm"
echo "• eval/lm_loss                   # Validation loss"
echo "• eval/lm_loss_ppl               # Validation perplexity"
echo "Done! All metrics will be logged to wandb."
