# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from typing import Dict, Optional, Callable
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
# from megatron.training.utils import print_rank_0
 
debug_mode = False

def wrap_print_rank_0(str):
    if debug_mode:
        print(f"{str}")


class RouterRLLossScaler(torch.autograd.Function):
    """An AutoScaler that adds RL loss gradients to router weights after trajectory completion."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, rl_loss: torch.Tensor, loss_scale: float = 1.0):
        """Save the RL loss for backward pass.
        
        Args:
            output (torch.Tensor): The output tensor (router probs/scores)
            rl_loss (torch.Tensor): The RL loss tensor
            loss_scale (float): Scaling fac@tor for the loss
        """
        ctx.save_for_backward(rl_loss)
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Add scaled RL loss gradient during backward pass."""
        (rl_loss,) = ctx.saved_tensors
        loss_scale = ctx.loss_scale
        
        # Compute gradient for the RL loss
        rl_loss_grad = torch.autograd.grad(
            outputs=rl_loss * loss_scale,
            inputs=rl_loss,
            create_graph=True,
            retain_graph=True
        )[0] if rl_loss.requires_grad else torch.zeros_like(rl_loss)
        
        # Return gradients: (output_grad, rl_loss_grad, loss_scale_grad)
        return grad_output, rl_loss_grad, None


class RouterTrajectoryTracker:
    """Track routing decisions across multiple MoE layers for RL rollouts.
    
    The trajectory consists of routing decisions from each MoE layer in the model.
    Trajectory length = number of MoE layers.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the trajectory for a new forward pass."""
        # Store old trajectory for PPO importance sampling
        self.old_layer_decisions = getattr(self, 'layer_decisions', {}).copy()
        self.layer_decisions = {}  # layer_number -> (latent_token_representations, routing_map, probs, entropy_reward)
        
    def add_layer_decision(self, layer_num: int, latent_token_representations: torch.Tensor, routing_map: torch.Tensor, routing_logits: torch.Tensor):
        """Add routing decision from a MoE layer.
        
        Args:
            layer_num (int): Layer number (1-indexed)
            latent_token_representations (torch.Tensor) - state space
            routing_map (torch.Tensor): Token routing assignments - action space
        """
        # Store detached copies to avoid keeping gradients
        latent_token_representations = latent_token_representations.detach()

        use_entropy_reward = False
        if use_entropy_reward:  
            reward = self.compute_expert_load_entropy(routing_map)
        else:
            reward = self.focus_tokens_on_expert_0_reward(routing_map)
        
        if layer_num == 1:
            wrap_print_rank_0(f"[RL DEBUG] layer {layer_num} add_layer_decision - latent_token_representations req_grad: {latent_token_representations.requires_grad}, use_entropy_reward: {use_entropy_reward} (alternatince is focus_tokens_on_expert_0_reward), reward: {reward.item():.4f}")
        self.layer_decisions[layer_num] = (
            latent_token_representations,
            routing_map,
            routing_logits,
            reward
        )

    def focus_tokens_on_expert_0_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Reward to focus token routing on expert 0.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
                        containing routing probabilities or binary assignments
        
        Returns:
            reward: Scalar tensor representing the reward to focus token routing on expert 0
        """
        return routing_map[:, :, 0].sum()

    
    def compute_expert_load_entropy(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Compute entropy of expert load distribution.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
                        containing routing probabilities or binary assignments
        
        Returns:
            entropy: Scalar tensor representing the entropy of expert load distribution
        """
        # Reshape to [total_tokens, num_experts]
        seq_length, batch_size, num_experts = routing_map.shape
        routing_map_flat = routing_map.view(-1, num_experts)
        
        # Compute expert loads (how many tokens each expert gets)
        # Sum across all tokens to get load per expert
        expert_loads = routing_map_flat.sum(dim=0)  # [num_experts]
        
        # Normalize to get probability distribution
        total_load = expert_loads.sum()
        if total_load > 0:
            expert_probs = expert_loads / total_load
        else:
            raise ValueError(f"Total load is 0 for routing map: {routing_map}")
        
        epsilon = 1e-10
        expert_probs = expert_probs + epsilon
        entropy = -(expert_probs * torch.log(expert_probs)).sum()
        
        # Normalize by log(num_experts) to get value in [0, 1]
        normalized_entropy = entropy / torch.log(torch.tensor(num_experts, dtype=entropy.dtype, device=entropy.device))
        
        return normalized_entropy
    
    # def apply_rl_loss_to_scores(self, layer_num: int, scores: torch.Tensor) -> torch.Tensor:
    #     """Apply RL loss to scores using MoEAuxLossAutoScaler - MINIMAL POC."""
    #     if layer_num not in self.layer_decisions:
    #         return scores
        
    #     # Get stored data
    #     logits, routing_map, entropy_reward = self.layer_decisions[layer_num]
        
    #     # Debug prints for layer 1
    #     if layer_num == 1:
    #         from megatron.training.utils import wrap_print_rank_0
    #         wrap_print_rank_0(f"[RL DEBUG] apply_rl_loss - logits req_grad: {logits.requires_grad}, entropy: {entropy_reward.item():.4f}")
        
    #     # Compute simple RL loss
    #     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    #     chosen_log_probs = log_probs * routing_map.float()
    #     # Simple average reward - detach to treat as constant
    #     rl_loss = -chosen_log_probs.mean() * 4.0  # Use constant reward for now
        
    #     if layer_num == 1:
    #         wrap_print_rank_0(f"[RL DEBUG] rl_loss: {rl_loss.item():.6f}, req_grad: {rl_loss.requires_grad}")
        
    #     # Apply using MoEAuxLossAutoScaler
    #     return MoEAuxLossAutoScaler.apply(scores, rl_loss)
    
    def compute_reinforce_loss(self, trajectory_data: Dict, discount_factor: float = 0.9) -> torch.Tensor:
        """Compute trajectory loss using entropy rewards with layer-wise discounting.
      
        Args:
            trajectory_data: Dictionary of layer decisions  
            discount_factor: Discount factor γ for future rewards
            
        Returns:
            torch.Tensor: trajectory loss with discounted values
        """
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device  # latent_token_representations device
        
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward
        
        # Debug: Show reward information
        reward_tensors = list(layer_rewards.values())
        wrap_print_rank_0(f"REINFORCE DEBUG: num_layers={len(layer_rewards)}, reward_shape={reward_tensors[0].shape if hasattr(reward_tensors[0], 'shape') else 'scalar'}")
        wrap_print_rank_0(f"REINFORCE DEBUG: reward_values={[r.item() if hasattr(r, 'item') else r for r in reward_tensors]}")
        
        #Calculate discounted state values using future rewards in reverse
        layer_values = {}
        accumulated_value = torch.tensor(0.0, device=device)
        
 
        for layer_num in reversed(sorted_layers):
            accumulated_value = layer_rewards[layer_num] + discount_factor * accumulated_value
            layer_values[layer_num] = accumulated_value
        
        # Debug: Show computed state values
        value_tensors = list(layer_values.values())
        wrap_print_rank_0(f"REINFORCE DEBUG: state_values={[v.item() if hasattr(v, 'item') else v for v in value_tensors]}")
        
        total_loss = torch.tensor(0.0, device=device)
        
        for layer_num in sorted_layers:
            latent_token_representations, routing_map, routing_logits, reward = trajectory_data[layer_num]
            state_value = layer_values[layer_num].detach()

            # Get log probabilities of chosen actions from router logits
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            # Average over routed assignments to keep scale comparable to LM loss
            num_tokens_routed = routing_map.sum().clamp_min(1).float()
            layer_log_prob = chosen_log_probs.sum() / num_tokens_routed
            
            # Debug: Show routing information for first layer
            wrap_print_rank_0(f"REINFORCE DEBUG Layer {layer_num}: routing_logits.shape={routing_logits.shape}, routing_map.shape={routing_map.shape}")
            wrap_print_rank_0(f"REINFORCE DEBUG Layer {layer_num}: num_experts_per_token={routing_map.sum(dim=-1).float().mean().item():.2f}")
            wrap_print_rank_0(f"REINFORCE DEBUG Layer {layer_num}: layer_log_prob={layer_log_prob.item():.4f}, state_value={state_value.item():.4f}")

            # REINFORCE loss: -log_prob(action) * state_value
            layer_loss = -layer_log_prob * state_value
            total_loss += layer_loss
        
        # Debug: Show loss information
        
        # Average across layers
        total_loss = total_loss / max(1, len(sorted_layers))
        wrap_print_rank_0(f"REINFORCE DEBUG: total_loss={total_loss.item() if hasattr(total_loss, 'item') else total_loss}, shape={total_loss.shape if hasattr(total_loss, 'shape') else 'scalar'}")
        return total_loss



    def compute_ppo_loss(self, trajectory_data: Dict, old_trajectory_data: Dict = None, 
                        discount_factor: float = 0.99, clip_ratio: float = 0.2, 
                        value_coeff: float = 0.5) -> torch.Tensor:
        """Compute PPO loss with clipped policy gradient and value function loss.
        
        Args:
            trajectory_data: Current trajectory decisions  
            old_trajectory_data: Previous trajectory decisions for importance sampling
            discount_factor: Discount factor γ for future rewards (0.99 = value future highly)
            clip_ratio: PPO clipping parameter (0.2 is standard)
            value_coeff: Coefficient for value function loss
            entropy_coeff: Coefficient for entropy regularization
            
        Returns:
            torch.Tensor: PPO loss combining policy gradient, value loss, and entropy
        """
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        # Extract immediate rewards for each layer
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward
        
        # Calculate returns and values first
        returns = {}
        values = {}
        
        # Initialize final value (no future rewards beyond last layer)
        reward_to_go = torch.tensor(0.0, device=device)
        
        # Calculate reward-to-go (returns) following reference pattern
        for layer_num in reversed(sorted_layers):
            immediate_reward = layer_rewards[layer_num]
            reward_to_go = immediate_reward + discount_factor * reward_to_go
            returns[layer_num] = reward_to_go
        
        # Simple baseline (value function approximation)
        # NOTE: This is a static baseline (mean of returns). For better PPO performance,
        # you would typically train a separate value network that takes latent_token_representations
        # as input to provide state-dependent value estimates. The latent_token_representations
        # are stored in the trajectory for this purpose but currently unused.
        all_returns = torch.stack(list(returns.values()))
        baseline = all_returns.mean()
        
        # Initialize values - using baseline as simple value function
        for layer_num in sorted_layers:
            values[layer_num] = baseline
        
        # Add final value for GAE calculation (value after last layer = 0)
        final_layer = max(sorted_layers)
        values[final_layer + 1] = torch.tensor(0.0, device=device)
        
        # Calculate advantages using GAE following reference implementation
        advantages = {}
        gae_tau = 0.95  # GAE parameter (lambda in the paper)
        adv = torch.tensor(0.0, device=device)
        
        # Calculate GAE advantages in reverse order
        for layer_num in reversed(sorted_layers):
            immediate_reward = layer_rewards[layer_num]
            current_value = values[layer_num]
            next_value = values.get(layer_num + 1, torch.tensor(0.0, device=device))
            
            # TD error calculation following reference:
            # td_error = reward[i] + discount * mask[i] * value[i + 1] - value[i]
            td_error = immediate_reward + discount_factor * next_value - current_value
            
            # GAE calculation following reference:
            # adv = td_error + adv * gae_tau * discount * mask[i]
            adv = td_error + adv * gae_tau * discount_factor
            advantages[layer_num] = adv
        
        # # Normalize advantages for stability (following PPO best practices)
        # advantage_values = torch.stack(list(advantages.values()))
        # if advantage_values.std() > 1e-8:
        #     advantage_mean = advantage_values.mean()
        #     advantage_std = advantage_values.std()
        #     normalized_advantages = (advantage_values - advantage_mean) / (advantage_std + 1e-8)
        #     advantages = {layer_num: normalized_advantages[i] for i, layer_num in enumerate(sorted_layers)}
        
        total_loss = torch.tensor(0.0, device=device)
        
        for layer_num in sorted_layers:
            latent_token_representations, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantage = advantages[layer_num].detach()
            return_value = returns[layer_num].detach()
            baseline_value = values[layer_num]
            
            # Current policy log probabilities
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            current_log_prob = chosen_log_probs.sum()
            
            # Old policy log probabilities (for importance sampling)
            if old_trajectory_data is not None and layer_num in old_trajectory_data:
                old_latent_token_representations, old_routing_map, old_routing_logits, _ = old_trajectory_data[layer_num]
                old_log_probs = torch.nn.functional.log_softmax(old_routing_logits, dim=-1)
                old_chosen_log_probs = old_log_probs * old_routing_map.float()
                old_log_prob = old_chosen_log_probs.sum()
                
                # Importance sampling ratio
                ratio = torch.exp(current_log_prob - old_log_prob)
            else:
                # No old trajectory available, use ratio = 1 (equivalent to REINFORCE)
                ratio = torch.tensor(1.0, device=device)
            
            # PPO clipped objective following reference implementation
            # pg_obj1 = ratio * sampled_advantages
            # pg_obj2 = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * sampled_advantages
            # pg_loss = torch.min(pg_obj1, pg_obj2).mean()
            pg_obj1 = ratio * advantage
            pg_obj2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            policy_loss = -torch.min(pg_obj1, pg_obj2)  # Negative because we want to maximize
            
            # Value function loss following reference: v_loss = 0.5 * torch.square(returns - v).mean()
            value_loss = 0.5 * torch.square(return_value - baseline_value)
            
            # Entropy bonus for exploration
            probs = torch.nn.functional.softmax(routing_logits, dim=-1)
            entropy = -(probs * log_probs).sum()
            
            # Combined loss following reference: loss = -pg_loss - entropy_coeff * entropy + baseline_coeff * v_loss
            # Note: entropy_coeff can be tuned (e.g., 0.01) to encourage exploration
            entropy_coeff = 0.01
            layer_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            total_loss += layer_loss
            
        return total_loss


# Global trajectory tracker instance
_global_trajectory_tracker = None

# singleton behavior
def get_trajectory_tracker() -> RouterTrajectoryTracker:
    """Get the global trajectory tracker instance."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is None:
        _global_trajectory_tracker = RouterTrajectoryTracker()
    return _global_trajectory_tracker


def reset_trajectory_tracker():
    """Reset the global trajectory tracker for a new forward pass."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is not None:
        _global_trajectory_tracker.reset()
