# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
# from megatron.training.utils import print_rank_0
 
debug_mode = False

def wrap_print_rank_0(str):
    if debug_mode:
        print(f"{str}")


class CriticNetwork(nn.Module):
    """MLP critic network for value function estimation.
    
    Takes latent token representations and outputs a scalar value estimate.
    Supports configurable depth via hidden_dims list.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None):
        """Initialize critic network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions. 
                         E.g., [256] for 1 layer, [256, 64, 32] for 3 layers.
                         Default: [256]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims
        
        # Initialize with small weights for stable training
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Latent representations of shape [seq_length, batch_size, hidden_dim]
               or [num_tokens, hidden_dim]
        
        Returns:
            Value estimates of shape [seq_length, batch_size] or [num_tokens]
        """
        original_shape = x.shape[:-1]
        # Flatten to [num_tokens, hidden_dim]
        x_flat = x.view(-1, x.shape[-1])
        values = self.network(x_flat).squeeze(-1)
        # Reshape back
        return values.view(*original_shape)
    
    def __repr__(self):
        return f"CriticNetwork(hidden_dims={self.hidden_dims})"


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
        self.per_token_rewards = True
        self.ppo_entropy_coeff = 0.01
        self.use_entropy_reward = False
        self.baseline_type = "mean"  # "mean" or "critic"
        self.critic_hidden_dims = [256]  # List of hidden layer dimensions
        self.critic_lr = 1e-3  # Critic learning rate
        self._critic = None  # Lazy initialization
        self._critic_optimizer = None
        self._pending_critic_update = False  # Flag to trigger critic update
        # Store last computed loss components for logging
        self.last_loss_components = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_advantage': 0.0,
            'mean_reward': 0.0,
        }

    
    def reset(self):
        """Reset the trajectory for a new forward pass."""
        # Store old trajectory for PPO importance sampling
        self.old_layer_decisions = getattr(self, 'layer_decisions', {}).copy()
        self.layer_decisions = {}  # layer_number -> (latent_token_representations, routing_map, probs, entropy_reward)
    
    def get_critic(self, input_dim: int, device: torch.device) -> CriticNetwork:
        """Get or create the critic network.
        
        Args:
            input_dim: Input dimension (hidden size of latent representations)
            device: Device to place the critic on
            
        Returns:
            CriticNetwork instance
        """
        if self._critic is None:
            self._critic = CriticNetwork(input_dim, self.critic_hidden_dims).to(device)
            self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self.critic_lr)
            wrap_print_rank_0(f"[RL DEBUG] Created critic network: {self._critic}, lr={self.critic_lr}")
        return self._critic
    
    def update_critic(self, value_loss: torch.Tensor):
        """Update critic network weights.
        
        Should be called after backward pass to update critic separately from main model.
        
        Args:
            value_loss: The value function loss tensor (must have gradients)
        """
        if self._critic is None or self._critic_optimizer is None:
            return
        
        # Backward pass for critic (value loss should already be part of total loss,
        # but we need to ensure critic gradients are computed)
        if value_loss.requires_grad:
            self._critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self._critic_optimizer.step()
            wrap_print_rank_0(f"[RL DEBUG] Critic updated, value_loss={value_loss.item():.6f}")
    
    def get_critic_state_dict(self) -> dict:
        """Get critic state for checkpointing."""
        if self._critic is None:
            return {}
        return {
            'critic_state_dict': self._critic.state_dict(),
            'critic_optimizer_state_dict': self._critic_optimizer.state_dict() if self._critic_optimizer else None,
        }
    
    def load_critic_state_dict(self, state_dict: dict, device: torch.device):
        """Load critic state from checkpoint."""
        if 'critic_state_dict' not in state_dict:
            return
        # Ensure critic exists (need input_dim, so we defer if not created yet)
        if self._critic is not None:
            self._critic.load_state_dict(state_dict['critic_state_dict'])
            if self._critic_optimizer and state_dict.get('critic_optimizer_state_dict'):
                self._critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])
            wrap_print_rank_0(f"[RL DEBUG] Loaded critic state from checkpoint")
    
    def compute_critic_baseline(self, latent_representations: torch.Tensor) -> torch.Tensor:
        """Compute baseline values using the critic network.
        
        Args:
            latent_representations: Tensor of shape [seq_length, batch_size, hidden_dim]
            
        Returns:
            Value estimates of shape [seq_length, batch_size]
        """
        device = latent_representations.device
        input_dim = latent_representations.shape[-1]
        critic = self.get_critic(input_dim, device)
        return critic(latent_representations)
        
    def add_layer_decision(self, layer_num: int, latent_token_representations: torch.Tensor, routing_map: torch.Tensor, routing_logits: torch.Tensor):
        """Add routing decision from a MoE layer.
        
        Args:
            layer_num (int): Layer number (1-indexed)
            latent_token_representations (torch.Tensor) - state space
            routing_map (torch.Tensor): Token routing assignments - action space
        """
        # Store detached copies to avoid keeping gradients
        latent_token_representations = latent_token_representations.detach()

        if self.use_entropy_reward:  
            reward = self.compute_expert_load_entropy(routing_map)
        else:
            reward = self.focus_tokens_on_expert_0_reward(routing_map)
        
        if layer_num == 1:
            reward_summary = reward.mean().item() if reward.dim() > 0 else reward.item()
            wrap_print_rank_0(f"[RL DEBUG] layer {layer_num} add_layer_decision - latent_token_representations req_grad: {latent_token_representations.requires_grad}, use_entropy_reward: {self.use_entropy_reward}, reward_shape: {reward.shape}, reward_mean: {reward_summary:.4f}")
        self.layer_decisions[layer_num] = (
            latent_token_representations,
            routing_map,
            routing_logits,
            reward
        )

    def focus_tokens_on_expert_0_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Reward to focus token routing on expert 0.
        
        Uses self.per_token_rewards flag to decide between per-token or scalar reward.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
                        containing routing probabilities or binary assignments
        
        Returns:
            reward: Either per-token [seq_length, batch_size] or scalar tensor
        """
        if self.per_token_rewards:
            return self._per_token_expert_0_reward(routing_map)
        else:
            return self._scalar_expert_0_reward(routing_map)
    
    def _per_token_expert_0_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Per-token reward: 1.0 if token went to expert 0, else 0.0.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
        
        Returns:
            reward: Tensor of shape [seq_length, batch_size]
        """
        return routing_map[:, :, 0].float()
    
    def _scalar_expert_0_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Scalar reward: fraction of tokens routed to expert 0.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
        
        Returns:
            reward: Scalar tensor in [0, 1]
        """
        total_tokens = routing_map.sum()
        tokens_to_expert_0 = routing_map[:, :, 0].sum()
        return tokens_to_expert_0 / total_tokens.clamp_min(1.0)

    
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
        """Compute REINFORCE loss.
        
        Automatically uses per-token or scalar rewards based on self.per_token_rewards flag.
      
        Args:
            trajectory_data: Dictionary of layer decisions  
            discount_factor: Discount factor γ for future rewards
            
        Returns:
            torch.Tensor: trajectory loss
        """
        if self.per_token_rewards:
            return self._compute_reinforce_loss_per_token(trajectory_data, discount_factor)
        else:
            return self._compute_reinforce_loss_scalar(trajectory_data, discount_factor)
    
    def _compute_reinforce_loss_scalar(self, trajectory_data: Dict, discount_factor: float = 0.9) -> torch.Tensor:
        """Compute REINFORCE loss with scalar rewards (original implementation)."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward  # scalar
        
        # Calculate discounted state values using future rewards in reverse
        layer_values = {}
        accumulated_value = torch.tensor(0.0, device=device)
        
        for layer_num in reversed(sorted_layers):
            accumulated_value = layer_rewards[layer_num] + discount_factor * accumulated_value
            layer_values[layer_num] = accumulated_value
        
        total_loss = torch.tensor(0.0, device=device)
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            state_value = layer_values[layer_num].detach()

            # Get log probabilities of chosen actions from router logits
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            # Average over routed assignments to keep scale comparable to LM loss
            num_tokens_routed = routing_map.sum().clamp_min(1).float()
            layer_log_prob = chosen_log_probs.sum() / num_tokens_routed

            # REINFORCE loss: -log_prob(action) * state_value
            layer_loss = -layer_log_prob * state_value
            total_loss += layer_loss
        
        # Average across layers
        total_loss = total_loss / max(1, len(sorted_layers))
        wrap_print_rank_0(f"REINFORCE (scalar) DEBUG: total_loss={total_loss.item():.6f}")
        return total_loss
    
    def _compute_reinforce_loss_per_token(self, trajectory_data: Dict, discount_factor: float = 0.9) -> torch.Tensor:
        """Compute REINFORCE loss with per-token rewards."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        sorted_layers = sorted(trajectory_data.keys())
        
        # Get per-token rewards for each layer: shape [seq_length, batch_size]
        layer_rewards = {}
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward  # [seq_length, batch_size]
        
        first_reward = layer_rewards[sorted_layers[0]]
        wrap_print_rank_0(f"REINFORCE (per-token) DEBUG: num_layers={len(layer_rewards)}, reward_shape={first_reward.shape}, reward_mean={first_reward.mean().item():.4f}")
        
        # Calculate per-token discounted returns (reward-to-go) in reverse layer order
        layer_returns = {}
        reward_to_go = torch.zeros_like(first_reward)  # [seq_length, batch_size]
        
        for layer_num in reversed(sorted_layers):
            reward_to_go = layer_rewards[layer_num] + discount_factor * reward_to_go
            layer_returns[layer_num] = reward_to_go.clone()
        
        # Compute baseline per layer (mean return across tokens) for variance reduction
        layer_baselines = {}
        for layer_num in sorted_layers:
            layer_baselines[layer_num] = layer_returns[layer_num].mean()
        
        # Compute per-token advantages
        layer_advantages = {}
        for layer_num in sorted_layers:
            layer_advantages[layer_num] = (layer_returns[layer_num] - layer_baselines[layer_num]).detach()
        
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantages = layer_advantages[layer_num]  # [seq_length, batch_size]
            
            # Get log probabilities of chosen actions from router logits
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            per_token_log_prob = chosen_log_probs.sum(dim=-1)  # [seq_length, batch_size]
            
            # REINFORCE loss: -log_prob(action) * advantage, per token
            per_token_loss = -per_token_log_prob * advantages
            
            layer_loss = per_token_loss.sum()
            total_loss += layer_loss
            total_tokens += per_token_loss.numel()
            
            if layer_num == sorted_layers[0]:
                wrap_print_rank_0(f"REINFORCE (per-token) Layer {layer_num}: advantages_mean={advantages.mean().item():.4f}")
        
        # Average over all tokens across all layers
        total_loss = total_loss / max(1, total_tokens)
        wrap_print_rank_0(f"REINFORCE (per-token) DEBUG: total_loss={total_loss.item():.6f}, total_tokens={total_tokens}")
        return total_loss



    def compute_ppo_loss(self, trajectory_data: Dict, old_trajectory_data: Dict = None, 
                        discount_factor: float = 0.99, clip_ratio: float = 0.2, 
                        value_coeff: float = 0.5) -> torch.Tensor:
        """Compute PPO loss with clipped policy gradient.
        
        Automatically uses per-token or scalar rewards based on self.per_token_rewards flag.
        
        Args:
            trajectory_data: Current trajectory decisions  
            old_trajectory_data: Previous trajectory decisions for importance sampling
            discount_factor: Discount factor γ for future rewards (0.99 = value future highly)
            clip_ratio: PPO clipping parameter (0.2 is standard)
            value_coeff: Coefficient for value function loss
            
        Returns:
            torch.Tensor: PPO loss combining policy gradient, value loss, and entropy
        """
        if self.per_token_rewards:
            return self._compute_ppo_loss_per_token(trajectory_data, old_trajectory_data, 
                                                     discount_factor, clip_ratio, value_coeff)
        else:
            return self._compute_ppo_loss_scalar(trajectory_data, old_trajectory_data,
                                                  discount_factor, clip_ratio, value_coeff)
    
    def _compute_ppo_loss_scalar(self, trajectory_data: Dict, old_trajectory_data: Dict = None, 
                                  discount_factor: float = 0.99, clip_ratio: float = 0.2, 
                                  value_coeff: float = 0.5) -> torch.Tensor:
        """Compute PPO loss with scalar rewards (original implementation)."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward  # scalar
        
        # Calculate returns
        returns = {}
        reward_to_go = torch.tensor(0.0, device=device)
        
        for layer_num in reversed(sorted_layers):
            reward_to_go = layer_rewards[layer_num] + discount_factor * reward_to_go
            returns[layer_num] = reward_to_go
        
        # Compute baseline based on baseline_type
        if self.baseline_type == "critic":
            # Use critic network: average value across all layer representations
            critic_values = {}
            for layer_num in sorted_layers:
                latent_repr, _, _, _ = trajectory_data[layer_num]
                # For scalar mode, average the critic output across all tokens
                critic_values[layer_num] = self.compute_critic_baseline(latent_repr).mean()
            baseline_values = critic_values
        else:
            # Use simple mean baseline (original behavior)
            all_returns = torch.stack(list(returns.values()))
            mean_baseline = all_returns.mean()
            baseline_values = {ln: mean_baseline for ln in sorted_layers}
        
        # Calculate advantages
        advantages = {}
        for layer_num in sorted_layers:
            advantages[layer_num] = (returns[layer_num] - baseline_values[layer_num]).detach()
        
        total_loss = torch.tensor(0.0, device=device)
        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        total_advantage = torch.tensor(0.0, device=device)
        total_reward = torch.tensor(0.0, device=device)
        entropy_coeff = self.ppo_entropy_coeff
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantage = advantages[layer_num]
            return_value = returns[layer_num].detach()
            
            # Current policy log probabilities (normalized by routed tokens)
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            num_tokens_routed = routing_map.sum().clamp_min(1).float()
            current_log_prob = chosen_log_probs.sum() / num_tokens_routed
            
            # Old policy log probabilities
            if old_trajectory_data is not None and layer_num in old_trajectory_data:
                _, old_routing_map, old_routing_logits, _ = old_trajectory_data[layer_num]
                old_routing_logits = old_routing_logits.detach()
                old_log_probs = torch.nn.functional.log_softmax(old_routing_logits, dim=-1)
                old_chosen_log_probs = old_log_probs * old_routing_map.float()
                old_num_tokens_routed = old_routing_map.sum().clamp_min(1).float()
                old_log_prob = old_chosen_log_probs.sum() / old_num_tokens_routed
                
                log_ratio = torch.clamp(current_log_prob - old_log_prob, min=-10.0, max=10.0)
                ratio = torch.exp(log_ratio)
            else:
                ratio = torch.tensor(1.0, device=device)
            
            # PPO clipped objective
            pg_obj1 = ratio * advantage
            pg_obj2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            policy_loss = -torch.min(pg_obj1, pg_obj2)
            
            # Value function loss
            if self.baseline_type == "critic":
                # Train critic to predict returns
                value_pred = baseline_values[layer_num]
                value_loss = 0.5 * torch.square(return_value - value_pred)
            else:
                value_loss = 0.5 * torch.square(return_value - baseline_values[layer_num])
            
            # Entropy bonus
            probs = torch.nn.functional.softmax(routing_logits, dim=-1)
            entropy = -(probs * log_probs).sum()
            
            layer_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            total_loss += layer_loss
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            total_advantage += advantage
            total_reward += reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, device=device)
        
        num_layers = max(1, len(sorted_layers))
        total_loss = total_loss / num_layers
        
        # Store component losses for logging
        self.last_loss_components = {
            'policy_loss': (total_policy_loss / num_layers).item(),
            'value_loss': (total_value_loss / num_layers).item(),
            'entropy_bonus': (total_entropy / num_layers).item(),
            'mean_advantage': (total_advantage / num_layers).item(),
            'mean_reward': (total_reward / num_layers).item() if isinstance(total_reward, torch.Tensor) else total_reward / num_layers,
        }
        wrap_print_rank_0(f"PPO (scalar, baseline={self.baseline_type}) DEBUG: total_loss={total_loss.item():.6f}")
        return total_loss
    
    def _compute_ppo_loss_per_token(self, trajectory_data: Dict, old_trajectory_data: Dict = None, 
                                     discount_factor: float = 0.99, clip_ratio: float = 0.2, 
                                     value_coeff: float = 0.5) -> torch.Tensor:
        """Compute PPO loss with per-token rewards."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        sorted_layers = sorted(trajectory_data.keys())
        
        # Get per-token rewards for each layer: shape [seq_length, batch_size]
        layer_rewards = {}
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            layer_rewards[layer_num] = reward  # [seq_length, batch_size]
        
        first_reward = layer_rewards[sorted_layers[0]]
        
        # Calculate per-token discounted returns
        layer_returns = {}
        reward_to_go = torch.zeros_like(first_reward)
        
        for layer_num in reversed(sorted_layers):
            reward_to_go = layer_rewards[layer_num] + discount_factor * reward_to_go
            layer_returns[layer_num] = reward_to_go.clone()
        
        # Compute per-token baseline based on baseline_type
        if self.baseline_type == "critic":
            # Use critic network for per-token value estimates
            layer_baselines = {}
            for layer_num in sorted_layers:
                latent_repr, _, _, _ = trajectory_data[layer_num]
                # Critic outputs per-token values: [seq_length, batch_size]
                layer_baselines[layer_num] = self.compute_critic_baseline(latent_repr)
        else:
            # Use simple mean baseline (original behavior)
            all_returns = torch.cat([layer_returns[ln].flatten() for ln in sorted_layers])
            mean_baseline = all_returns.mean()
            layer_baselines = {ln: mean_baseline for ln in sorted_layers}
        
        # Compute per-token advantages
        layer_advantages = {}
        for layer_num in sorted_layers:
            layer_advantages[layer_num] = (layer_returns[layer_num] - layer_baselines[layer_num]).detach()
        
        total_loss = torch.tensor(0.0, device=device)
        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        total_advantage = torch.tensor(0.0, device=device)
        total_reward = torch.tensor(0.0, device=device)
        total_tokens = 0
        entropy_coeff = self.ppo_entropy_coeff
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantages = layer_advantages[layer_num]  # [seq_length, batch_size]
            returns = layer_returns[layer_num].detach()
            
            # Current policy log probabilities per token
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            current_per_token_log_prob = chosen_log_probs.sum(dim=-1)
            
            # Old policy log probabilities
            if old_trajectory_data is not None and layer_num in old_trajectory_data:
                _, old_routing_map, old_routing_logits, _ = old_trajectory_data[layer_num]
                old_routing_logits = old_routing_logits.detach()
                old_log_probs = torch.nn.functional.log_softmax(old_routing_logits, dim=-1)
                old_chosen_log_probs = old_log_probs * old_routing_map.float()
                old_per_token_log_prob = old_chosen_log_probs.sum(dim=-1)
                
                log_ratio = torch.clamp(current_per_token_log_prob - old_per_token_log_prob, min=-10.0, max=10.0)
                ratio = torch.exp(log_ratio)
            else:
                ratio = torch.ones_like(current_per_token_log_prob)
            
            # PPO clipped objective per token
            pg_obj1 = ratio * advantages
            pg_obj2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            per_token_policy_loss = -torch.min(pg_obj1, pg_obj2)
            
            # Value function loss per token
            baseline_for_layer = layer_baselines[layer_num]
            per_token_value_loss = 0.5 * torch.square(returns - baseline_for_layer)
            
            # Entropy bonus per token
            probs = torch.nn.functional.softmax(routing_logits, dim=-1)
            per_token_entropy = -(probs * log_probs).sum(dim=-1)
            
            # Combined per-token loss
            per_token_loss = per_token_policy_loss + value_coeff * per_token_value_loss - entropy_coeff * per_token_entropy
            
            layer_loss = per_token_loss.sum()
            total_loss += layer_loss
            total_policy_loss += per_token_policy_loss.sum()
            total_value_loss += per_token_value_loss.sum()
            total_entropy += per_token_entropy.sum()
            total_advantage += advantages.sum()
            total_reward += reward.sum()
            total_tokens += per_token_loss.numel()
            
            if layer_num == sorted_layers[0]:
                wrap_print_rank_0(f"PPO (per-token, baseline={self.baseline_type}) Layer {layer_num}: advantages_mean={advantages.mean().item():.4f}, ratio_mean={ratio.mean().item():.4f}")
        
        # Normalize and store component losses for logging
        total_loss = total_loss / max(1, total_tokens)
        self.last_loss_components = {
            'policy_loss': (total_policy_loss / max(1, total_tokens)).item(),
            'value_loss': (total_value_loss / max(1, total_tokens)).item(),
            'entropy_bonus': (total_entropy / max(1, total_tokens)).item(),
            'mean_advantage': (total_advantage / max(1, total_tokens)).item(),
            'mean_reward': (total_reward / max(1, total_tokens)).item(),
        }
        wrap_print_rank_0(f"PPO (per-token, baseline={self.baseline_type}) DEBUG: total_loss={total_loss.item():.6f}, total_tokens={total_tokens}")
        return total_loss


# Global trajectory tracker instance
_global_trajectory_tracker = None

# singleton behavior
def get_trajectory_tracker() -> RouterTrajectoryTracker:
    """Get the global trajectory tracker instance."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is None:
        _global_trajectory_tracker = RouterTrajectoryTracker()
        # Configure from command line args if available
        try:
            from megatron import get_args
            args = get_args()
            _global_trajectory_tracker.per_token_rewards = getattr(args, 'rl_per_token_rewards', True)
            _global_trajectory_tracker.ppo_entropy_coeff = getattr(args, 'rl_ppo_entropy_coeff', 0.01)
            _global_trajectory_tracker.use_entropy_reward = getattr(args, 'rl_use_entropy_reward', False)
            _global_trajectory_tracker.baseline_type = getattr(args, 'rl_ppo_baseline_type', 'mean')
            _global_trajectory_tracker.critic_hidden_dims = getattr(args, 'rl_critic_hidden_dims', [256])
        except (ImportError, AssertionError):
            # Args not available yet, use defaults
            pass
    return _global_trajectory_tracker


def reset_trajectory_tracker():
    """Reset the global trajectory tracker for a new forward pass."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is not None:
        _global_trajectory_tracker.reset()
