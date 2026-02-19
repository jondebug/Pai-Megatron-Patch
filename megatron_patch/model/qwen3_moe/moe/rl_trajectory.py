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
        original_dtype = x.dtype
        # Flatten to [num_tokens, hidden_dim]
        x_flat = x.view(-1, x.shape[-1])
        # Cast to float32 for critic computation (critic weights are float32)
        x_flat = x_flat.float()
        values = self.network(x_flat).squeeze(-1)
        # Cast back to original dtype and reshape
        return values.to(original_dtype).view(*original_shape)
    
    def __repr__(self):
        return f"CriticNetwork(hidden_dims={self.hidden_dims})"


class RewardNormalizer:
    """Running mean/std normalizer for reward signals.
    
    Tracks exponential moving average of reward mean and variance,
    then normalizes rewards to have approximately zero mean and unit variance.
    This expands compressed reward ranges (e.g., [0.15, 0.19]) into a useful [-1, +1] range.
    """
    
    def __init__(self, momentum: float = 0.01, eps: float = 1e-8):
        """
        Args:
            momentum: EMA update rate (higher = faster adaptation, more noise)
            eps: Small constant to prevent division by zero
        """
        self.momentum = momentum
        self.eps = eps
        self.running_mean = None
        self.running_var = None
        self._count = 0
    
    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize a reward value using running statistics.
        
        Args:
            reward: Scalar or per-token reward tensor
            
        Returns:
            Normalized reward with approximately zero mean and unit variance
        """
        reward_val = reward.detach().mean().item() if reward.dim() > 0 else reward.detach().item()
        reward_var = reward.detach().var().item() if reward.dim() > 0 and reward.numel() > 1 else 0.0
        
        if self.running_mean is None:
            # Initialize from first observation
            self.running_mean = reward_val
            self.running_var = max(reward_var, self.eps)
            self._count = 1
            # Don't normalize the first observation (no statistics yet)
            return reward
        
        # Update running statistics
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * reward_val
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (reward_val - self.running_mean) ** 2
        self._count += 1
        
        # Normalize: (reward - mean) / std
        std = max(self.running_var ** 0.5, self.eps)
        normalized = (reward - self.running_mean) / std
        
        return normalized
    
    def state_dict(self) -> dict:
        return {
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'count': self._count,
        }
    
    def load_state_dict(self, state: dict):
        self.running_mean = state.get('running_mean')
        self.running_var = state.get('running_var')
        self._count = state.get('count', 0)


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
        self.per_token_rewards = False  # Default False; topn_load, critical_path, entropy are batch-level
        self.ppo_entropy_coeff = 0.01
        self.reward_type = "topn_load"  # "expert0", "entropy", "topn_load", or "critical_path"
        self.reward_topn = 12  # Number of top experts for topn_load reward
        self.baseline_type = "mean"  # "mean" or "critic"
        self.critic_hidden_dims = [256]  # List of hidden layer dimensions
        self.critic_lr = 1e-3  # Critic learning rate
        self._critic = None  # Lazy initialization
        self._critic_optimizer = None
        self._pending_critic_update = False  # Flag to trigger critic update
        self.normalize_rewards = False  # Enable running-mean/std reward normalization
        self._reward_normalizer = RewardNormalizer(momentum=0.01)
        self._num_layers = 48  # Updated from actual data on first forward pass
        self.use_ema_loads = False  # Use EMA expert loads for reward (more stable)
        self._ema_expert_loads = None  # [num_experts] running average of per-expert load
        self._ema_momentum = 0.1  # EMA update rate for expert loads
        # Store last computed loss components for logging
        self.last_loss_components = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_advantage': 0.0,
            'mean_reward': 0.0,
            'avg_topn_load': 0.0,  # Average top-N load across layers (only for topn_load reward)
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
    
    def compute_critic_baseline(self, latent_representations: torch.Tensor, layer_num: int = None) -> torch.Tensor:
        """Compute baseline values using the critic network.
        
        Args:
            latent_representations: Tensor of shape [seq_length, batch_size, hidden_dim]
            layer_num: Layer number (used to provide layer-conditional predictions)
            
        Returns:
            Value estimates of shape [seq_length, batch_size]
        """
        device = latent_representations.device
        # +1 for normalized layer index feature
        input_dim = latent_representations.shape[-1] + 1
        critic = self.get_critic(input_dim, device)
        
        # Append normalized layer index as an extra feature
        num_layers = max(self._num_layers, 1)
        layer_frac = (layer_num / num_layers) if layer_num is not None else 0.0
        layer_feature = torch.full(
            latent_representations.shape[:-1] + (1,),
            layer_frac, device=device, dtype=latent_representations.dtype
        )
        critic_input = torch.cat([latent_representations, layer_feature], dim=-1)
        return critic(critic_input)
        
    def add_layer_decision(self, layer_num: int, latent_token_representations: torch.Tensor, routing_map: torch.Tensor, routing_logits: torch.Tensor):
        """Add routing decision from a MoE layer.
        
        Args:
            layer_num (int): Layer number (1-indexed)
            latent_token_representations (torch.Tensor) - state space
            routing_map (torch.Tensor): Token routing assignments - action space
        """
        # CRITICAL: Check gradient flow - if routing_logits doesn't require grad, policy gradient will be zero!
        if layer_num == 1 and not routing_logits.requires_grad:
            import warnings
            warnings.warn(
                "[RL WARNING] routing_logits.requires_grad=False! "
                "Policy gradients will be zero. Check activation checkpointing settings.",
                RuntimeWarning
            )
        
        # Store detached copies to avoid keeping gradients
        latent_token_representations = latent_token_representations.detach()

        # Track number of layers for critic layer-index feature
        self._num_layers = max(self._num_layers, layer_num)

        # Update EMA expert loads for stable reward computation
        with torch.no_grad():
            batch_loads = routing_map.sum(dim=(0, 1)).float()
            if self._ema_expert_loads is None:
                self._ema_expert_loads = batch_loads.clone()
            else:
                self._ema_expert_loads = (1 - self._ema_momentum) * self._ema_expert_loads + self._ema_momentum * batch_loads

        # Auto-set per_token_rewards for reward types that are inherently per-token
        _PER_TOKEN_REWARD_TYPES = {"per_token_topn_binary", "per_token_load_weighted", "expert0"}
        if self.reward_type in _PER_TOKEN_REWARD_TYPES:
            self.per_token_rewards = True

        # Select reward function based on reward_type
        if self.reward_type == "entropy":
            # Note: entropy reward is always scalar - incompatible with per_token_rewards=True
            if self.per_token_rewards:
                raise ValueError("reward_type='entropy' is incompatible with per_token_rewards=True. "
                                "Entropy reward is computed over the entire batch, not per-token.")
            reward = self.compute_expert_load_entropy(routing_map)
        elif self.reward_type == "topn_load":
            # Note: topn_load is batch-wise only
            if self.per_token_rewards:
                raise ValueError("reward_type='topn_load' is incompatible with per_token_rewards=True. "
                                "Top-N load reward is computed over the entire batch.")
            reward = self.topn_load_reward(routing_map)
        elif self.reward_type == "critical_path":
            # Directly targets max expert load (the critical-path bottleneck)
            if self.per_token_rewards:
                raise ValueError("reward_type='critical_path' is incompatible with per_token_rewards=True. "
                                "Critical path reward is computed over the entire batch.")
            reward = self.critical_path_reward(routing_map)
        elif self.reward_type == "per_token_topn_binary":
            reward = self.per_token_topn_binary_reward(routing_map)
        elif self.reward_type == "per_token_load_weighted":
            reward = self.per_token_load_weighted_reward(routing_map)
        else:  # "expert0" (default)
            reward = self.focus_tokens_on_expert_0_reward(routing_map)
        
        # Apply reward normalization if enabled (expands compressed reward ranges)
        raw_reward_summary = reward.mean().item() if reward.dim() > 0 else reward.item()
        if self.normalize_rewards:
            reward = self._reward_normalizer.normalize(reward)
        
        if layer_num == 1:
            norm_reward_summary = reward.mean().item() if reward.dim() > 0 else reward.item()
            wrap_print_rank_0(f"[RL DEBUG] layer {layer_num} add_layer_decision - reward_type: {self.reward_type}, "
                            f"reward_shape: {reward.shape}, raw_reward: {raw_reward_summary:.4f}, "
                            f"normalized_reward: {norm_reward_summary:.4f}, normalize={self.normalize_rewards}")
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

    def topn_load_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Reward = avg_load / max_load. In (0, 1], 1 = perfect balance."""
        expert_loads = routing_map.sum(dim=(0, 1)).float()  # [num_experts]
        avg_load = expert_loads.mean()
        max_load = expert_loads.topk(self.reward_topn)[0].mean()
        return avg_load / max_load.clamp(min=avg_load)
    
    def critical_path_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Reward that directly targets the critical-path bottleneck (max expert load).
        
        Unlike topn_load_reward which uses avg/topN_mean (compressed into a narrow band),
        this reward uses -max_load directly, producing a wider dynamic range.
        
        The reward is: -(max_load - ideal_load) / ideal_load
        
        Where ideal_load = total_tokens / num_experts (perfect balance).
        
        Range: 0 (perfect balance) to large negative (severe imbalance).
        A max_load that is 2x the ideal gives reward = -1.
        A max_load that is 5x the ideal gives reward = -4.
        
        This linear formulation gives gradient signal proportional to the
        degree of imbalance, unlike the ratio form which compresses everything.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
            
        Returns:
            reward: Scalar tensor (0 = perfect, more negative = worse)
        """
        expert_loads = routing_map.sum(dim=(0, 1)).float()  # [num_experts]
        max_load = expert_loads.max()
        ideal_load = expert_loads.mean()  # = total_tokens / num_experts (constant for fixed batch)
        
        # Negative deviation from ideal, normalized by ideal load
        # This gives reward = 0 at perfect balance, -1 when max is 2x ideal, etc.
        return -(max_load - ideal_load) / ideal_load.clamp(min=1.0)

    def per_token_topn_binary_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Per-token reward based on how many of the token's chosen experts are overloaded.
        
        Counts how many of each token's top-k chosen experts fall in the top-N
        most loaded experts, then scales linearly:
            reward = 1 - 2 * (num_hot / topk)
        
        When use_ema_loads=True, uses exponential moving average of expert loads
        across recent batches instead of current-batch loads. This provides a more
        stable reward signal that doesn't fluctuate with per-batch data variance.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
            
        Returns:
            reward: Tensor of shape [seq_length, batch_size], values in [-1, +1]
        """
        if self.use_ema_loads and self._ema_expert_loads is not None:
            expert_loads = self._ema_expert_loads
        else:
            expert_loads = routing_map.sum(dim=(0, 1)).float()
        _, topn_idx = expert_loads.topk(self.reward_topn)
        
        # Build mask: is each expert in the top-N most loaded?
        is_hot = torch.zeros(routing_map.shape[-1], device=routing_map.device, dtype=torch.float32)
        is_hot[topn_idx] = 1.0
        
        # Count how many of each token's chosen experts are hot
        # routing_map: [seq, batch, E], is_hot: [E] -> [seq, batch]
        num_hot = (routing_map.float() * is_hot).sum(dim=-1)
        
        # Total experts chosen per token (topk)
        num_chosen = routing_map.float().sum(dim=-1).clamp(min=1.0)
        
        # Linear scale: 1 (no hot experts) to -1 (all hot experts)
        reward = 1.0 - 2.0 * (num_hot / num_chosen)
        
        return reward

    def per_token_load_weighted_reward(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Per-token reward proportional to how overloaded the chosen experts are.
        
        When use_ema_loads=True, uses exponential moving average of expert loads
        for more stable rewards across batches.
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
            
        Returns:
            reward: Tensor of shape [seq_length, batch_size], continuously scaled
        """
        if self.use_ema_loads and self._ema_expert_loads is not None:
            expert_loads = self._ema_expert_loads
        else:
            expert_loads = routing_map.sum(dim=(0, 1)).float()
        ideal_load = expert_loads.mean()
        
        # For each token: sum of loads of chosen experts
        # routing_map [seq, batch, E] * expert_loads [E] -> [seq, batch, E] -> sum over E
        token_load = (routing_map.float() * expert_loads).sum(dim=-1)  # [seq, batch]
        
        # Number of experts each token is routed to (topk)
        num_chosen = routing_map.float().sum(dim=-1).clamp(min=1.0)  # [seq, batch]
        avg_chosen_load = token_load / num_chosen
        
        return -(avg_chosen_load - ideal_load) / ideal_load.clamp(min=1.0)

    def compute_topn_load(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Compute average load of top N experts (for logging).
        
        Args:
            routing_map: Tensor of shape [seq_length, batch_size, num_experts]
        
        Returns:
            avg_load: Scalar tensor (positive value, lower = better balance)
        """
        num_experts = routing_map.shape[2]
        
        # Compute expert loads: sum over all tokens in batch
        expert_loads = routing_map.sum(dim=(0, 1))  # [num_experts]
        total_tokens = expert_loads.sum().clamp_min(1.0)
        normalized_loads = expert_loads / total_tokens  # fraction of tokens per expert
        
        # Find top-N loaded experts
        n = min(self.reward_topn, num_experts)
        top_loads, _ = torch.topk(normalized_loads, n)
        
        return top_loads.mean()

    
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
            # Get device from first available tensor or default to cuda if available
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device
        
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_data[layer_num]
            # Ensure reward is scalar (reduce if per-token due to config timing)
            if reward.dim() > 0:
                reward = reward.mean()
            layer_rewards[layer_num] = reward
        
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
        
        # Compute average top-N load across layers (only meaningful for topn_load reward)
        avg_topn_load = 0.0
        if self.reward_type == "topn_load":
            topn_loads = []
            for layer_num in sorted_layers:
                _, routing_map, _, _ = trajectory_data[layer_num]
                topn_loads.append(self.compute_topn_load(routing_map).item())
            avg_topn_load = sum(topn_loads) / len(topn_loads) if topn_loads else 0.0
        
        # Compute mean reward for logging
        mean_reward = sum(r.item() if hasattr(r, 'item') else r for r in layer_rewards.values()) / max(1, len(layer_rewards))
        
        # Store loss components for logging
        self.last_loss_components = {
            'policy_loss': total_loss.item(),
            'value_loss': 0.0,  # REINFORCE doesn't have a critic
            'entropy_bonus': 0.0,  # REINFORCE doesn't use entropy bonus
            'mean_advantage': 0.0,  # REINFORCE uses returns directly
            'mean_reward': mean_reward,
            'avg_topn_load': avg_topn_load,
        }
        
        wrap_print_rank_0(f"REINFORCE (scalar) DEBUG: total_loss={total_loss.item():.6f}")
        return total_loss
    
    def _compute_reinforce_loss_per_token(self, trajectory_data: Dict, discount_factor: float = 0.9) -> torch.Tensor:
        """Compute REINFORCE loss with per-token rewards."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
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
        
        # Normalize advantages across all tokens and layers to zero mean, unit variance
        all_advs = torch.cat([layer_advantages[ln].flatten() for ln in sorted_layers])
        adv_mean = all_advs.mean()
        adv_std = all_advs.std().clamp(min=1e-8)
        for ln in sorted_layers:
            layer_advantages[ln] = (layer_advantages[ln] - adv_mean) / adv_std
        
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantages = layer_advantages[layer_num]  # [seq_length, batch_size]
            
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            per_token_log_prob = chosen_log_probs.sum(dim=-1)  # [seq_length, batch_size]
            
            per_token_loss = -per_token_log_prob * advantages
            
            layer_loss = per_token_loss.sum()
            total_loss += layer_loss
            total_tokens += per_token_loss.numel()
            
            if layer_num == sorted_layers[0]:
                wrap_print_rank_0(f"REINFORCE (per-token) Layer {layer_num}: advantages_mean={advantages.mean().item():.4f}, adv_std={advantages.std().item():.4f}")
        
        # Average over all tokens across all layers
        total_loss = total_loss / max(1, total_tokens)
        
        # Compute mean reward for logging
        mean_reward = sum(r.mean().item() for r in layer_rewards.values()) / max(1, len(layer_rewards))
        
        # Store loss components for logging (topn_load not applicable for per-token)
        self.last_loss_components = {
            'policy_loss': total_loss.item(),
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_advantage': 0.0,
            'mean_reward': mean_reward,
            'avg_topn_load': 0.0,
        }
        
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
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
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
            # Compute critic values and train critic (decoupled from main loss)
            baseline_values = {}
            critic_loss = torch.tensor(0.0, device=device)
            for layer_num in sorted_layers:
                latent_repr, _, _, _ = trajectory_data[layer_num]
                value = self.compute_critic_baseline(latent_repr, layer_num=layer_num).mean()
                target = returns[layer_num].mean().detach() if returns[layer_num].numel() > 1 else returns[layer_num].detach()
                critic_loss = critic_loss + 0.5 * torch.square(target - value)
                baseline_values[layer_num] = value.detach()  # Detach for use in advantages
            critic_loss = critic_loss / len(sorted_layers)
            
            if self._critic_optimizer is not None:
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                self._critic_optimizer.step()
            self._last_critic_loss = critic_loss.item()
        else:
            # Simple mean baseline
            all_returns = torch.stack(list(returns.values()))
            baseline_values = {ln: all_returns.mean() for ln in sorted_layers}
            self._last_critic_loss = 0.0
        
        # Calculate advantages - ensure scalars for scalar loss mode
        advantages = {}
        for layer_num in sorted_layers:
            ret = returns[layer_num]
            baseline = baseline_values[layer_num]
            if ret.numel() > 1:
                ret = ret.mean()
            if baseline.numel() > 1:
                baseline = baseline.mean()
            advantages[layer_num] = (ret - baseline).detach()
        
        # Normalize advantages across all layers to zero mean, unit variance
        all_advs = torch.stack(list(advantages.values()))
        adv_mean = all_advs.mean()
        adv_std = all_advs.std().clamp(min=1e-8)
        for layer_num in sorted_layers:
            advantages[layer_num] = (advantages[layer_num] - adv_mean) / adv_std
        
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
            
            # Value function loss for logging (critic already trained above)
            if self.baseline_type == "critic":
                value_pred = baseline_values[layer_num]
                if return_value.numel() > 1:
                    return_value_scalar = return_value.mean()
                else:
                    return_value_scalar = return_value
                # Just for logging - critic already trained, so detach everything
                value_loss = 0.5 * torch.square(return_value_scalar.detach() - value_pred.detach())
            else:
                # Mean baseline has no learnable value function
                value_loss = torch.tensor(0.0, device=device)
            
            # Entropy bonus (normalized per token)
            probs = torch.nn.functional.softmax(routing_logits, dim=-1)
            per_token_entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len, batch]
            entropy = per_token_entropy.mean()  # Average entropy per token
            
            layer_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            total_loss += layer_loss
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy  # Now normalized per token
            total_advantage += advantage
            # Reduce reward to scalar if needed (for scalar mode)
            if isinstance(reward, torch.Tensor):
                total_reward += reward.mean() if reward.numel() > 1 else reward
            else:
                total_reward += torch.tensor(reward, device=device)
        
        num_layers = max(1, len(sorted_layers))
        total_loss = total_loss / num_layers
        
        # Compute average top-N load across layers (only meaningful for topn_load reward)
        avg_topn_load = 0.0
        if self.reward_type == "topn_load":
            topn_loads = []
            for layer_num in sorted_layers:
                _, routing_map, _, _ = trajectory_data[layer_num]
                topn_loads.append(self.compute_topn_load(routing_map).item())
            avg_topn_load = sum(topn_loads) / len(topn_loads) if topn_loads else 0.0
        
        # Store component losses for logging
        logged_value_loss = self._last_critic_loss if hasattr(self, '_last_critic_loss') else (total_value_loss / num_layers).item()
        all_advs_for_log = torch.stack(list(advantages.values()))
        self.last_loss_components = {
            'policy_loss': (total_policy_loss / num_layers).item(),
            'value_loss': logged_value_loss,
            'entropy_bonus': (total_entropy / num_layers).item(),
            'mean_advantage': (total_advantage / num_layers).item(),
            'mean_reward': (total_reward / num_layers).item() if isinstance(total_reward, torch.Tensor) else total_reward / num_layers,
            'avg_topn_load': avg_topn_load,
            'advantage_std': all_advs_for_log.std().item(),
            'advantage_min': all_advs_for_log.min().item(),
            'advantage_max': all_advs_for_log.max().item(),
        }
        first_layer = sorted_layers[0]
        _, _, routing_logits_first, _ = trajectory_data[first_layer]
        wrap_print_rank_0(f"PPO (scalar, baseline={self.baseline_type}) DEBUG: "
                          f"total_loss={total_loss.item():.6f}, "
                          f"policy_loss={(total_policy_loss / num_layers).item():.6f}, "
                          f"entropy={self.last_loss_components['entropy_bonus']:.4f}, "
                          f"advantage={self.last_loss_components['mean_advantage']:.4f}, "
                          f"adv_std={self.last_loss_components['advantage_std']:.4f}, "
                          f"mean_reward={self.last_loss_components['mean_reward']:.4f}, "
                          f"routing_logits.requires_grad={routing_logits_first.requires_grad}")
        return total_loss
    
    def _compute_ppo_loss_per_token(self, trajectory_data: Dict, old_trajectory_data: Dict = None, 
                                     discount_factor: float = 0.99, clip_ratio: float = 0.2, 
                                     value_coeff: float = 0.5) -> torch.Tensor:
        """Compute PPO loss with per-token rewards."""
        if len(trajectory_data) == 0:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
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
            # Compute critic values and train critic (decoupled from main loss)
            layer_baselines = {}
            critic_loss = torch.tensor(0.0, device=device)
            total_critic_tokens = 0
            for layer_num in sorted_layers:
                latent_repr, _, _, _ = trajectory_data[layer_num]
                value = self.compute_critic_baseline(latent_repr, layer_num=layer_num)
                target = layer_returns[layer_num].detach()
                critic_loss = critic_loss + 0.5 * torch.square(target - value).sum()
                total_critic_tokens += value.numel()
                layer_baselines[layer_num] = value.detach()
            critic_loss = critic_loss / max(1, total_critic_tokens)
            
            if self._critic_optimizer is not None:
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                self._critic_optimizer.step()
            self._last_critic_loss = critic_loss.item()
        else:
            # Simple mean baseline
            all_returns = torch.cat([layer_returns[ln].flatten() for ln in sorted_layers])
            layer_baselines = {ln: all_returns.mean() for ln in sorted_layers}
            self._last_critic_loss = 0.0
        
        # Compute per-token advantages
        layer_advantages = {}
        for layer_num in sorted_layers:
            layer_advantages[layer_num] = (layer_returns[layer_num] - layer_baselines[layer_num]).detach()
        
        # Normalize advantages across all tokens and layers to zero mean, unit variance
        all_advs = torch.cat([layer_advantages[ln].flatten() for ln in sorted_layers])
        adv_mean = all_advs.mean()
        adv_std = all_advs.std().clamp(min=1e-8)
        for ln in sorted_layers:
            layer_advantages[ln] = (layer_advantages[ln] - adv_mean) / adv_std
        
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
            
            # Value function loss per token for logging (critic already trained above)
            baseline_for_layer = layer_baselines[layer_num]
            if self.baseline_type == "critic":
                # Just for logging - critic already trained, so detach everything
                per_token_value_loss = 0.5 * torch.square(returns.detach() - baseline_for_layer.detach())
            else:
                # Mean baseline has no learnable value function
                per_token_value_loss = torch.zeros_like(returns)
            
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
        
        # Normalize
        total_loss = total_loss / max(1, total_tokens)
        
        logged_value_loss = self._last_critic_loss if hasattr(self, '_last_critic_loss') else (total_value_loss / max(1, total_tokens)).item()
        all_advs_log = torch.cat([layer_advantages[ln].flatten() for ln in sorted_layers])
        self.last_loss_components = {
            'policy_loss': (total_policy_loss / max(1, total_tokens)).item(),
            'value_loss': logged_value_loss,
            'entropy_bonus': (total_entropy / max(1, total_tokens)).item(),
            'mean_advantage': (total_advantage / max(1, total_tokens)).item(),
            'mean_reward': (total_reward / max(1, total_tokens)).item(),
            'avg_topn_load': 0.0,
            'advantage_std': all_advs_log.std().item(),
            'advantage_min': all_advs_log.min().item(),
            'advantage_max': all_advs_log.max().item(),
        }
        wrap_print_rank_0(f"PPO (per-token, baseline={self.baseline_type}) DEBUG: total_loss={total_loss.item():.6f}, critic_loss={logged_value_loss:.6f}, adv_std={self.last_loss_components['advantage_std']:.4f}, total_tokens={total_tokens}")
        return total_loss


# Global trajectory tracker instance
_global_trajectory_tracker = None
_tracker_configured = False  # Track if we've successfully configured from args

# singleton behavior
def get_trajectory_tracker() -> RouterTrajectoryTracker:
    """Get the global trajectory tracker instance."""
    global _global_trajectory_tracker, _tracker_configured
    if _global_trajectory_tracker is None:
        _global_trajectory_tracker = RouterTrajectoryTracker()
    
    # Always try to configure from args if not yet configured
    # This handles the case where tracker is created before args are parsed
    if not _tracker_configured:
        try:
            from megatron import get_args
            args = get_args()
            _global_trajectory_tracker.per_token_rewards = getattr(args, 'rl_per_token_rewards', False)
            _global_trajectory_tracker.ppo_entropy_coeff = getattr(args, 'rl_ppo_entropy_coeff', 0.01)
            _global_trajectory_tracker.reward_type = getattr(args, 'rl_reward_type', 'expert0')
            _global_trajectory_tracker.reward_topn = getattr(args, 'rl_reward_topn', 12)
            _global_trajectory_tracker.baseline_type = getattr(args, 'rl_ppo_baseline_type', 'mean')
            _global_trajectory_tracker.critic_hidden_dims = getattr(args, 'rl_critic_hidden_dims', [256])
            _global_trajectory_tracker.critic_lr = getattr(args, 'rl_critic_lr', 1e-3)
            _global_trajectory_tracker.normalize_rewards = getattr(args, 'rl_normalize_rewards', False)
            _tracker_configured = True
            # Always print configuration for verification (not gated by debug_mode)
            import sys
            print(f"[RL CONFIG] Tracker configured: baseline_type={_global_trajectory_tracker.baseline_type}, "
                  f"reward_type={_global_trajectory_tracker.reward_type}, "
                  f"reward_topn={_global_trajectory_tracker.reward_topn}, "
                  f"per_token_rewards={_global_trajectory_tracker.per_token_rewards}, "
                  f"ppo_entropy_coeff={_global_trajectory_tracker.ppo_entropy_coeff}, "
                  f"critic_hidden_dims={_global_trajectory_tracker.critic_hidden_dims}, "
                  f"critic_lr={_global_trajectory_tracker.critic_lr}, "
                  f"normalize_rewards={_global_trajectory_tracker.normalize_rewards}", flush=True)
            sys.stdout.flush()
        except (ImportError, AssertionError) as e:
            # Args not available yet, will retry on next call
            pass
        except Exception as e:
            # Catch any other exceptions and log them
            import sys
            print(f"[RL CONFIG ERROR] Failed to configure tracker: {type(e).__name__}: {e}", flush=True)
            sys.stdout.flush()
    return _global_trajectory_tracker


def reset_trajectory_tracker():
    """Reset the global trajectory tracker for a new forward pass."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is not None:
        _global_trajectory_tracker.reset()
