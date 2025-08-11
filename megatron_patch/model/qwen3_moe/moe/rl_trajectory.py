# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from typing import Dict, Optional, Callable


class RouterRLLossScaler(torch.autograd.Function):
    """An AutoScaler that adds RL loss gradients to router weights after trajectory completion."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, rl_loss: torch.Tensor, loss_scale: float = 1.0):
        """Save the RL loss for backward pass.
        
        Args:
            output (torch.Tensor): The output tensor (router probs/scores)
            rl_loss (torch.Tensor): The RL loss tensor
            loss_scale (float): Scaling factor for the loss
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
    Trajectory length = number of MoE layers (not sequence length).
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the trajectory for a new forward pass."""
        self.layer_decisions = {}  # layer_number -> (logits, routing_map, probs)
        self.sequence_reward = None
        self.expected_num_layers = None
        self.rl_loss_func = None
        self.rl_loss_scale = 1.0
        
    def add_layer_decision(self, layer_num: int, logits: torch.Tensor, routing_map: torch.Tensor, 
                          probs: Optional[torch.Tensor] = None):
        """Add routing decision from a MoE layer.
        
        Args:
            layer_num (int): Layer number (1-indexed)
            logits (torch.Tensor): Router logits for this layer
            routing_map (torch.Tensor): Token routing assignments
            probs (torch.Tensor): Router probabilities (optional)
        """
        # Store detached copies to avoid keeping gradients
        self.layer_decisions[layer_num] = (
            logits.detach().clone(),
            routing_map.detach().clone(),
            probs.detach().clone() if probs is not None else None
        )
        
    def set_rl_loss_function(self, loss_func: Callable, loss_scale: float = 1.0):
        """Set the RL loss function to be applied when trajectory is complete.
        
        Args:
            loss_func: Function that takes trajectory data and returns loss
            loss_scale: Scaling factor for the loss
        """
        self.rl_loss_func = loss_func
        self.rl_loss_scale = loss_scale
        
    def is_trajectory_complete(self, num_moe_layers: int) -> bool:
        """Check if we have decisions from all MoE layers."""
        if self.expected_num_layers is None:
            self.expected_num_layers = num_moe_layers
        return len(self.layer_decisions) == num_moe_layers
    
    def get_trajectory_data(self) -> Dict[int, tuple]:
        """Get all trajectory data."""
        return self.layer_decisions.copy()
    
    def set_sequence_reward(self, reward: torch.Tensor):
        """Set the final reward for this trajectory."""
        self.sequence_reward = reward.detach().clone()
        
    def apply_trajectory_rl_loss(self) -> torch.Tensor:
        """Apply RL loss to the trajectory and return the loss value.
        
        Returns:
            torch.Tensor: The computed RL loss for the trajectory
        """
        if self.rl_loss_func is None or len(self.layer_decisions) == 0:
            return torch.tensor(0.0)
            
        # Compute RL loss for the entire trajectory
        trajectory_loss = self.rl_loss_func(self.layer_decisions, self.sequence_reward)
        
        # Apply RL loss scaling to each layer's probabilities using the autograd function
        for layer_num, (logits, routing_map, probs) in self.layer_decisions.items():
            if probs is not None and probs.requires_grad:
                # Apply RL loss to this layer's probabilities
                RouterRLLossScaler.apply(probs, trajectory_loss, self.rl_loss_scale)
                
        return trajectory_loss
    
    def compute_entropy_only_loss(self, trajectory_data: Dict, sequence_reward: torch.Tensor = None, 
                                 discount_factor: float = 0.99) -> torch.Tensor:
        """Compute trajectory loss using entropy rewards with layer-wise discounting.
        
        Calculates state values using discounted future rewards:
        V(layer_i) = reward_i + γ * reward_{i+1} + γ² * reward_{i+2} + ...
        
        Args:
            trajectory_data: Dictionary of layer decisions  
            sequence_reward: Ignored - not used in entropy-only approach
            discount_factor: Discount factor γ for future rewards (0.99 = value future highly)
            
        Returns:
            torch.Tensor: Entropy-based trajectory loss with discounted values
        """
        if len(trajectory_data) == 0:
            return torch.tensor(0.0)
            
        # Get device from first layer's logits
        first_layer = next(iter(trajectory_data.values()))
        device = first_layer[0].device  # logits device
        
        # Step 1: Compute immediate entropy rewards for each layer
        layer_rewards = {}
        sorted_layers = sorted(trajectory_data.keys())
        
        for layer_num in sorted_layers:
            logits, routing_map, probs = trajectory_data[layer_num]
            
            # Compute entropy for this layer's routing decisions
            # H(p) = -sum(p * log(p)) where p = softmax(logits)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs_from_logits = torch.nn.functional.softmax(logits, dim=-1)
            
            # Entropy per token
            token_entropies = -(probs_from_logits * log_probs).sum(dim=-1)
            
            # Average entropy across all tokens in this layer = immediate reward
            layer_rewards[layer_num] = token_entropies.mean()
        
        # Step 2: Calculate discounted state values using future rewards
        # V(layer_i) = reward_i + γ * reward_{i+1} + γ² * reward_{i+2} + ...
        layer_values = {}
        num_layers = len(sorted_layers)
        
        for i, layer_num in enumerate(sorted_layers):
            state_value = torch.tensor(0.0, device=device)
            
            # Sum discounted future rewards (including current)
            for j in range(i, num_layers):
                future_layer = sorted_layers[j]
                discount = discount_factor ** (j - i)  # γ^(steps_ahead)
                state_value += discount * layer_rewards[future_layer]
                
            layer_values[layer_num] = state_value
        
        # Step 3: Apply REINFORCE loss using state values
        total_loss = torch.tensor(0.0, device=device)
        
        for layer_num in sorted_layers:
            logits, routing_map, probs = trajectory_data[layer_num]
            state_value = layer_values[layer_num]
            
            # Get log probabilities of chosen actions
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            layer_log_prob = chosen_log_probs.sum()
            
            # REINFORCE loss: -log_prob(action) * state_value
            # Higher state value -> higher expected future return -> encourage this action
            layer_loss = -layer_log_prob * state_value
            total_loss += layer_loss
            
        return total_loss


# Global trajectory tracker instance
_global_trajectory_tracker = None


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


def apply_trajectory_rl_loss_if_complete(num_moe_layers: int, 
                                       rl_loss_func: Optional[Callable] = None,
                                       rl_loss_scale: float = 1.0) -> bool:
    """Apply RL loss if trajectory is complete.
    
    Args:
        num_moe_layers: Total number of MoE layers expected
        rl_loss_func: Optional custom RL loss function
        rl_loss_scale: Scaling factor for RL loss
        
    Returns:
        bool: True if RL loss was applied, False otherwise
    """
    tracker = get_trajectory_tracker()
    
    if not tracker.is_trajectory_complete(num_moe_layers):
        return False
        
    # Set default RL loss function if not provided
    if rl_loss_func is None:
        rl_loss_func = tracker.compute_entropy_only_loss
        
    tracker.set_rl_loss_function(rl_loss_func, rl_loss_scale)
    tracker.apply_trajectory_rl_loss()
    
    return True


def complete_trajectory_and_apply_rl_loss(reward: torch.Tensor = None, 
                                         num_moe_layers: int = 1,
                                         rl_loss_scale: float = 1.0,
                                         discount_factor: float = 0.99) -> bool:
    """Complete trajectory and apply entropy-only RL loss with discounted state values.
    
    Args:
        reward: Ignored - entropy-only approach doesn't use sequence reward
        num_moe_layers: Expected number of MoE layers
        rl_loss_scale: Scaling factor for RL loss (similar to aux loss coeff)
        discount_factor: Discount factor γ for future rewards (0.99 = value future highly)
        
    Returns:
        bool: True if RL loss was applied, False if trajectory incomplete
        
    Example usage:
        # Entropy-only RL with discounting (values early layers more):
        complete_trajectory_and_apply_rl_loss(
            num_moe_layers=config.num_moe_layers, 
            rl_loss_scale=0.1,
            discount_factor=0.95  # Lower = prioritize current layer more
        )
    """
    tracker = get_trajectory_tracker()
    
    if not tracker.is_trajectory_complete(num_moe_layers):
        return False
    
    # Apply entropy-only RL loss with discounting
    loss_func = lambda traj_data, seq_reward: tracker.compute_entropy_only_loss(
        traj_data, seq_reward, discount_factor
    )
    return apply_trajectory_rl_loss_if_complete(
        num_moe_layers, loss_func, rl_loss_scale
    ) 