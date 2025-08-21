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
echo "Done! Only router parameters will be trained with RL optimization."
