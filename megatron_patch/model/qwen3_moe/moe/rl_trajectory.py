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


def _rl_print_rank0(msg):
    """Raw stdout print on global rank 0, always on (unlike wrap_print_rank_0, which is
    gated by debug_mode=False). Used for the [RL TELEM] causal-check line so SIGNED metrics
    (e.g. cov_ptlp_adv) reach the SLURM .out even though Megatron's training_log only prints
    loss_dict keys with avg > 0.0. Mirrors the [G7] raw-print pattern."""
    try:
        import torch.distributed as _d
        if (not _d.is_available()) or (not _d.is_initialized()) or _d.get_rank() == 0:
            print(msg, flush=True)
    except Exception:
        try:
            print(msg, flush=True)
        except Exception:
            pass


# =============================================================================
# H1 (Gumbel-top-k + ordered Plackett-Luce) + H2 (global leave-one-out smooth-max
# reward). PORT of the reviewer-verified single-process reference in
# ~/rl_repro_5b9/{pl_verify,unit_gates,g7_test}.py (7/7 gates pass).
#
# These module-level functions are the FP32 "RL island": exact, importable,
# stateless algorithms validated by the acceptance gates (port_gates.py imports
# THESE). They are only reached when --rl-sampling hard_gumbel_pl and/or
# --rl-reward loo_smoothmax are set; defaults preserve current behavior.
# See rl_reviewer_docs/PORT_SPEC.md.
# =============================================================================

RL_PL_DEFAULT_TAU = 1.0        # Plackett-Luce temperature (reference tau = 1.0)
RL_LOO_DEFAULT_BETA = 0.3      # smooth-max sharpness (reference beta = 0.3)


def rl_fp32(t: torch.Tensor) -> torch.Tensor:
    """FP32 RL island: upcast a router tensor to float32 for log-prob / reward /
    reduction math (the rest of the model stays BF16). No-op for float32 inputs."""
    return t if t.dtype == torch.float32 else t.float()


def rl_ordered_logprob(logits: torch.Tensor, pool_idx: torch.Tensor,
                       pos: torch.Tensor, tau: float = RL_PL_DEFAULT_TAU) -> torch.Tensor:
    """Ordered Plackett-Luce log-prob over a FIXED detached candidate pool.

    Faithful port of pl_pool() in unit_gates.py / pl() in pl_verify.py. Unbiased
    for the pool-restricted conditional policy ONLY (Q4).

    Args:
        logits:   [..., E]     grad-tracked router logits (current policy).
        pool_idx: [..., POOL]  FIXED global expert ids of the detached pool (from OLD logits).
        pos:      [..., k]     ordered positions WITHIN the pool of the sampled action.
        tau:      Plackett-Luce temperature.
    Returns:
        lp: [...]  ordered log-prob of the sampled action (grad flows through `logits`).
    """
    logits = rl_fp32(logits)
    s = logits.gather(-1, pool_idx) / tau            # [..., POOL] scores over the FIXED pool
    lp = s.new_zeros(s.shape[:-1])
    masked = s.clone()
    k = pos.shape[-1]
    for j in range(k):
        pj = pos[..., j]                             # [...] position of j-th chosen in pool
        lp = lp + s.gather(-1, pj[..., None]).squeeze(-1) - torch.logsumexp(masked, dim=-1)
        masked = masked.scatter(-1, pj[..., None], float('-inf'))
    return lp


def rl_loo_smoothmax_reward(global_counts: torch.Tensor, S: torch.Tensor,
                            beta: float = RL_LOO_DEFAULT_BETA) -> torch.Tensor:
    """Global leave-one-out smooth-max (congestion) reward, CORRECTED SIGN.

    Faithful port of loo() in unit_gates.py (G3/G4):
        J(n) = logsumexp(beta*n)/beta                        # smooth max = congestion cost
        r_t  = J(n - Delta_{A_t}) - J(n)   <= 0              # remove token t's k assignments
    r_t is MORE NEGATIVE for tokens on hot experts (leaving relieves more congestion).
    This is NOT J(n)-J(n-Delta) (that would REINFORCE hot experts).

    Args:
        global_counts: [E]     GLOBAL per-expert counts (all-reduced, FP32).
        S:             [T, k]  each token's k GLOBAL expert ids (the sampled set; order irrelevant).
        beta:          smooth-max sharpness.
    Returns:
        r: [T]  per-token reward (<= 0).
    """
    n = rl_fp32(global_counts)
    J_full = torch.logsumexp(beta * n, dim=0) / beta            # scalar J(n)
    T = S.shape[0]
    nm = n[None, :].repeat(T, 1)                                # [T, E]
    nm.scatter_add_(1, S, -torch.ones_like(S, dtype=nm.dtype))  # n - Delta_{A_t}
    return torch.logsumexp(beta * nm, dim=1) / beta - J_full    # [T] <= 0


def rl_policy_loss_reduction(advantages: torch.Tensor, ptlp: torch.Tensor,
                             denominator, coeff: float = 1.0):
    """Single-normalization reduction contract (PORT_SPEC 'Reduction contract').

        numerator = -(A.detach() * ptlp).sum()
        rl_loss   = coeff * numerator / denominator   # tune strength via coeff, NOT a hidden /94

    Returns (rl_loss, diag) where diag logs every factor for the reviewer.
    """
    A = advantages.detach()
    numerator = -(A * ptlp).sum()
    if not isinstance(denominator, torch.Tensor):
        denominator = torch.as_tensor(float(denominator), device=ptlp.device, dtype=ptlp.dtype)
    denom = denominator.clamp(min=1.0)
    rl_loss = coeff * numerator / denom
    diag = {
        'numerator': float(numerator.detach().item()),
        'denominator': float(denom.detach().item()),
        'coeff': float(coeff),
        'num_terms': int(ptlp.numel()),
    }
    return rl_loss, diag


def rl_all_reduce_global_loads(local_counts: torch.Tensor, group=None) -> torch.Tensor:
    """All-reduce (SUM) local per-expert counts into GLOBAL counts for the H2 reward.

    ***NEEDS REVIEW / G7***: the correct collective is the TOKEN-COVERING group, which must
    be discovered/asserted at RUNTIME on the real topology -- it is NOT necessarily the
    data-parallel group. The group must sum each expert's count exactly once over the ranks
    holding DISTINCT tokens, with NO TP/SP/PP-replicated double counting (g7_test.py
    invariants: sum(counts)==unique_tokens*k; collective==offline reconstruction; CP match;
    a double-counting group is CAUGHT by the sum gate). This helper deliberately takes the
    process group as an ARGUMENT and does not hardcode DP.
    """
    counts = rl_fp32(local_counts)
    if group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=group)
    return counts


def rl_resolve_token_covering_group():
    """G7: return the process group over which per-expert token counts must be summed
    EXACTLY ONCE (the 'token-covering collective').

    On the validated topology TP=1, PP=1, no sequence/context parallel, the DATA-PARALLEL
    group holds DISTINCT tokens on every rank with NO TP/SP/PP-replicated double counting, so
    the token-covering group == the DP group. The group is returned here and passed as an
    ARGUMENT into rl_all_reduce_global_loads (never hardcoded at the call site); the G7
    no-update capture (sum gate: Sum_e count == unique_tokens*k) validates it on the live
    hardware before use. Returns None on any failure => rl_all_reduce_global_loads() falls
    back to LOCAL counts (a safe no-op that preserves current behavior).
    """
    try:
        from megatron.core import parallel_state
        return parallel_state.get_data_parallel_group()
    except Exception:
        return None


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
        self.replay_buffer_size = 0  # 0 = disabled; >0 stores past trajectories for replay
        self._replay_buffer = []  # List of past trajectory dicts (stored on CPU)
        self.reset()
        self.paused = False  # When True, add_layer_decision() calls are skipped (used during KL reference forward)
        self.per_token_rewards = False  # Default False; topn_load, critical_path, entropy are batch-level
        self.ppo_entropy_coeff = 0.01
        self.gae_lambda = 1.0  # 1.0 = MC returns (current behavior), 0.95 = standard GAE
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
        # --- H1/H2 port (default OFF => current behavior preserved) ---
        self.rl_sampling = 'argmax'          # 'argmax' (default) or 'hard_gumbel_pl'
        self.rl_candidate_pool = 0           # Gumbel-top-k pool size N (0 => all experts)
        self.rl_stochastic_temperature = 1.0 # tau for Gumbel sampling + PL log-prob
        self.global_loads = False            # H2: all-reduce loads over token-covering group (G7)
        self.loo_beta = RL_LOO_DEFAULT_BETA  # smooth-max sharpness for loo_smoothmax reward
        self.pl_decisions = {}               # layer -> {pool_idx, pos, experts, old_ptlp} (safeguard bundle)
        self.use_ema_loads = False  # Use EMA expert loads for reward (more stable)
        self._ema_expert_loads = None  # [num_experts] running average of per-expert load
        self._ema_momentum = 0.1  # EMA update rate for expert loads
        self.ppo_reeval = False  # Proper PPO: re-evaluate old states under current policy
        self.reeval_logits = {}  # Populated by router.forward() when ppo_reeval=True
        self.ppo_epochs = 1  # K: number of PPO epochs per training step
        self.ppo_legacy_mode = False  # A/B switch: keep previous PPO behavior when True
        self._router_modules = {}  # layer_num -> router gating module (set by pretrain_qwen)
        self._ppo_optimizer = None  # Separate optimizer for multi-epoch PPO
        self.ppo_extra_lr = 1e-4  # LR for extra PPO epochs (scaled by 1/(K-1))
        self._pending_post_step_ppo = None  # Deferred extra-epoch update payload (run post optimizer.step)
        self._heatmap_accum = {}  # layer_num -> accumulated expert loads [num_experts]
        self._heatmap_steps = 0  # Steps since last heatmap log
        self._heatmap_log_interval = 50  # Log heatmap every N steps
        # Store last computed loss components for logging
        self.last_loss_components = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_advantage': 0.0,
            'mean_reward': 0.0,
            'avg_topn_load': 0.0,  # Average top-N load across layers (only for topn_load reward)
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }

    
    def reset(self):
        """Reset the trajectory for a new forward pass."""
        # Store old trajectory for PPO importance sampling
        self.old_layer_decisions = getattr(self, 'layer_decisions', {}).copy()
        
        # Populate replay buffer with the completed trajectory (moved to CPU)
        if self.replay_buffer_size > 0 and self.old_layer_decisions:
            cpu_trajectory = {}
            for ln, (latent, rmap, logits, reward) in self.old_layer_decisions.items():
                cpu_trajectory[ln] = (
                    latent.detach().cpu(),
                    rmap.detach().cpu(),
                    logits.detach().cpu(),
                    reward.detach().cpu() if isinstance(reward, torch.Tensor) else reward,
                )
            self._replay_buffer.append(cpu_trajectory)
            while len(self._replay_buffer) > self.replay_buffer_size:
                self._replay_buffer.pop(0)
        
        self.layer_decisions = {}
        self.pl_decisions = {}  # H1: fixed pool + ordered action + old PL log-prob, per layer
        self.reeval_logits = {}  # layer_num -> re-evaluated logits (current weights, old states)

    def schedule_extra_ppo_epochs(self, rl_loss_coeff: float, discount_factor: float, clip_ratio: float):
        """Schedule deferred extra PPO epochs to run after optimizer.step().

        We overwrite any previous pending payload because only the latest completed
        rollout (the one associated with the just-finished backward/step) should run.
        """
        self._pending_post_step_ppo = {
            'rl_loss_coeff': float(rl_loss_coeff),
            'discount_factor': float(discount_factor),
            'clip_ratio': float(clip_ratio),
        }

    def run_scheduled_extra_ppo_epochs(self):
        """Run deferred extra PPO epochs once, if a payload is pending."""
        if not self._pending_post_step_ppo:
            return False
        payload = self._pending_post_step_ppo
        self._pending_post_step_ppo = None
        self.run_extra_ppo_epochs(
            rl_loss_coeff=payload['rl_loss_coeff'],
            discount_factor=payload['discount_factor'],
            clip_ratio=payload['clip_ratio'],
        )
        return True
    
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
        if self._critic is not None:
            try:
                self._critic.load_state_dict(state_dict['critic_state_dict'])
                if self._critic_optimizer and state_dict.get('critic_optimizer_state_dict'):
                    self._critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])
                wrap_print_rank_0(f"[RL DEBUG] Loaded critic state from checkpoint")
            except RuntimeError as e:
                wrap_print_rank_0(f"[RL WARNING] Could not load critic checkpoint (dimension mismatch?): {e}. Starting critic fresh.")
    
    def compute_critic_baseline(self, latent_representations: torch.Tensor, layer_num: int = None) -> torch.Tensor:
        """Compute baseline values using the critic network.
        
        Args:
            latent_representations: Tensor of shape [seq_length, batch_size, hidden_dim]
            layer_num: Layer number (used for layer-conditional predictions when critic_layer_aware=True)
            
        Returns:
            Value estimates of shape [seq_length, batch_size]
        """
        device = latent_representations.device
        layer_aware = getattr(self, 'critic_layer_aware', False)
        
        if layer_aware:
            input_dim = latent_representations.shape[-1] + 1
            critic = self.get_critic(input_dim, device)
            num_layers = max(self._num_layers, 1)
            layer_frac = (layer_num / num_layers) if layer_num is not None else 0.0
            layer_feature = torch.full(
                latent_representations.shape[:-1] + (1,),
                layer_frac, device=device, dtype=latent_representations.dtype
            )
            critic_input = torch.cat([latent_representations, layer_feature], dim=-1)
        else:
            input_dim = latent_representations.shape[-1]
            critic = self.get_critic(input_dim, device)
            critic_input = latent_representations
        
        return critic(critic_input)
        
    def add_layer_decision(self, layer_num: int, latent_token_representations: torch.Tensor, routing_map: torch.Tensor, routing_logits: torch.Tensor, rl_action: dict = None):
        """Add routing decision from a MoE layer.
        
        Args:   
            layer_num (int): Layer number (1-indexed)
            latent_token_representations (torch.Tensor) - state space
            routing_map (torch.Tensor): Token routing assignments - action space
        """
        # [FIX 2026-07-30] Honor the paused flag: during the KL reference forward (torch.no_grad) the
        # router still calls this method and would overwrite the trajectory's in-graph logits with
        # detached ones -> rl_loss.requires_grad=False -> RL policy gradient severed (no-op). Mirrors
        # the _kl_state['capture_disabled'] guard already used for KL logit capture.
        if getattr(self, 'paused', False) and not getattr(self, 'rl_disconnect_repro', False):
            return
        # CRITICAL: Check gradient flow - if routing_logits doesn't require grad, policy gradient will be zero!
        if layer_num == 1 and not routing_logits.requires_grad and torch.is_grad_enabled():
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
            if getattr(self, 'global_load', False):
                try:
                    from megatron.core import parallel_state as _ps
                    _grp = _ps.get_data_parallel_group()
                    if torch.distributed.is_initialized() and torch.distributed.get_world_size(group=_grp) > 1:
                        torch.distributed.all_reduce(batch_loads, group=_grp)
                except Exception:
                    pass
            if self._ema_expert_loads is None:
                self._ema_expert_loads = batch_loads.clone()
            else:
                self._ema_expert_loads = (1 - self._ema_momentum) * self._ema_expert_loads + self._ema_momentum * batch_loads

        # Auto-set per_token_rewards for reward types that are inherently per-token
        _PER_TOKEN_REWARD_TYPES = {"per_token_topn_binary", "per_token_load_weighted", "expert0", "diff_lse_load", "loo_smoothmax"}
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
        elif self.reward_type == "diff_lse_load":
            reward = self.diff_lse_load_reward(routing_map, routing_logits)
        elif self.reward_type == "loo_smoothmax":
            # H2: global leave-one-out smooth-max reward (corrected sign). Uses the ordered
            # sampled action when threaded (rl_action), else reconstructs the set from routing_map.
            reward = self.loo_smoothmax_reward(routing_map, rl_action)
        else:  # "expert0" (default)
            reward = self.focus_tokens_on_expert_0_reward(routing_map)
        
        # Apply reward normalization if enabled (expands compressed reward ranges)
        raw_reward_summary = reward.mean().item() if reward.dim() > 0 else reward.item()
        self._last_raw_reward_std = float(reward.std().item()) if (reward.dim() > 0 and reward.numel() > 1) else 0.0
        if layer_num == 1:
            try:
                _r0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
                if _r0 and reward.dim() > 0 and reward.numel() > 1:
                    print(f'[RL REWARD DEBUG] raw_reward std={reward.std().item():.4e} mean={reward.mean().item():.4e} min={reward.min().item():.4e} max={reward.max().item():.4e} numel={reward.numel()} global_load={getattr(self,"global_load",False)}', flush=True)
            except Exception:
                pass
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

        # H1 safeguard: store (FIXED detached pool, ordered sampled GLOBAL expert ids, old
        # ordered PL log-prob) TOGETHER, and assert the stored pool+positions reconstruct the
        # exact sampled action (identical pool + action guaranteed at recompute time).
        if rl_action is not None:
            _tau = float(getattr(self, 'rl_stochastic_temperature', 1.0)) or 1.0
            with torch.no_grad():
                _old_ptlp = rl_ordered_logprob(
                    rl_fp32(routing_logits).detach(), rl_action['pool_idx'], rl_action['pos'], _tau)
            assert torch.equal(rl_action['pool_idx'].gather(-1, rl_action['pos']), rl_action['experts']), \
                "[RL] stored pool+pos does not reconstruct the sampled action"
            self.pl_decisions[layer_num] = {
                'pool_idx': rl_action['pool_idx'],   # FIXED detached pool   [seq, batch, POOL]
                'pos': rl_action['pos'],             # ordered positions     [seq, batch, k]
                'experts': rl_action['experts'],     # ordered GLOBAL ids    [seq, batch, k]
                'old_ptlp': _old_ptlp,               # old-policy PL logprob [seq, batch]
            }

        # Accumulate expert loads for heatmap visualization
        with torch.no_grad():
            loads = routing_map.sum(dim=(0, 1)).float().detach().cpu()
            if layer_num not in self._heatmap_accum:
                self._heatmap_accum[layer_num] = loads
            else:
                self._heatmap_accum[layer_num] += loads


    def inject_lm_reward(self, per_token_losses: torch.Tensor, lm_reward_coeff: float):
        """Add per-token LM cross-entropy as an additional reward component to all layers.

        Called from loss_func_with_rl after the forward pass produces per-token losses.
        The LM reward is: -cross_entropy (lower loss = higher reward), centered per-batch
        to remove data-driven variance and isolate routing-driven quality effects.

        Args:
            per_token_losses: Per-token cross-entropy, shape [seq_len * batch_size] or [seq_len, batch_size]
            lm_reward_coeff: Scaling factor (beta) for the LM reward component
        """
        if not self.layer_decisions or lm_reward_coeff == 0:
            return

        example_layer = next(iter(self.layer_decisions.values()))
        target_shape = example_layer[3].shape  # reward shape: [seq_len, batch_size] or scalar

        lm_reward = -per_token_losses.float().detach()

        if lm_reward.dim() == 1 and len(target_shape) == 2:
            seq_len, batch_size = target_shape
            lm_reward = lm_reward.view(seq_len, batch_size)

        # Center per-batch to remove data-driven variance
        lm_reward = lm_reward - lm_reward.mean()

        for layer_num, (latent, routing_map, logits, load_reward) in self.layer_decisions.items():
            if load_reward.dim() == 0 and lm_reward.dim() > 0:
                combined = load_reward + lm_reward_coeff * lm_reward.mean()
            else:
                combined = load_reward + lm_reward_coeff * lm_reward
            self.layer_decisions[layer_num] = (latent, routing_map, logits, combined)

    def log_heatmap(self, iteration):
        """Log expert load heatmap to wandb.
        
        Creates a 2D heatmap (layers x experts) showing accumulated token routing
        since the last log. Logged every _heatmap_log_interval steps.
        """
        self._heatmap_steps += 1
        if self._heatmap_steps < self._heatmap_log_interval:
            return
        if not self._heatmap_accum:
            return

        self._heatmap_steps = 0

        try:
            import wandb
            if wandb.run is None:
                return

            import numpy as np
            sorted_layers = sorted(self._heatmap_accum.keys())
            num_experts = len(self._heatmap_accum[sorted_layers[0]])
            
            heatmap_data = np.zeros((len(sorted_layers), num_experts))
            for i, ln in enumerate(sorted_layers):
                heatmap_data[i] = self._heatmap_accum[ln].numpy()

            # Normalize per-layer to show relative distribution
            row_sums = heatmap_data.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1.0)
            normalized = heatmap_data / row_sums

            # Create wandb Table for heatmap
            columns = ["layer"] + [f"expert_{j}" for j in range(num_experts)]
            table_data = []
            for i, ln in enumerate(sorted_layers):
                row = [ln] + normalized[i].tolist()
                table_data.append(row)
            
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({
                "expert_heatmap": wandb.plot.HeatMap(
                    columns[1:],
                    [str(ln) for ln in sorted_layers],
                    normalized.tolist(),
                    show_text=False,
                ),
            }, commit=False)

            # Reset accumulator
            self._heatmap_accum = {}

        except Exception as e:
            wrap_print_rank_0(f"[HEATMAP] Failed to log: {e}")
            self._heatmap_accum = {}

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

    def diff_lse_load_reward(self, routing_map: torch.Tensor, routing_logits: torch.Tensor) -> torch.Tensor:
        """P1 pilot (2026-07-12): exact per-token DIFFERENCE reward on a smoothed max-load objective.

        Objective: R(loads) = -tau * logsumexp(loads / tau) (smooth proxy of -max_e load; LogSumExp
        gives non-bottleneck tokens nonzero credit — the Dr.Reinforce max-objective caveat).
        Counterfactual per token: move its PRIMARY assignment (highest-logit chosen expert) to its
        best UNCHOSEN expert (by its own logits — where the router would actually send it).
        Difference reward D_t = R(loads) - R(loads_cf), exact and critic-free: negative when the
        token sits on a hot expert with a cooler alternative, so the policy gradient lowers the
        probability of the hot assignment. Scale-stable via log-ratio * mean load.
        """
        with torch.no_grad():
            rm = routing_map.float()
            loads = rm.sum(dim=(0, 1))                                  # [E]
            mean_load = loads.mean().clamp(min=1.0)
            tau = mean_load * float(getattr(self, "diff_lse_tau", 0.25))
            z = loads / tau
            zmax = z.max()
            S = torch.exp(z - zmax).sum()                               # stable partition

            logits = routing_logits.float()
            neg_inf = float("-inf")
            # primary chosen expert (source) and best unchosen expert (destination) per token
            chosen_logits = logits.masked_fill(~routing_map.bool(), neg_inf)
            e_src = chosen_logits.argmax(dim=-1)                        # [seq, batch]
            unchosen_logits = logits.masked_fill(routing_map.bool(), neg_inf)
            e_cf = unchosen_logits.argmax(dim=-1)                       # [seq, batch]
            l_src = loads[e_src]                                        # [seq, batch]
            l_cf = loads[e_cf]

            def ex(l):
                return torch.exp(l / tau - zmax)
            dS = ex((l_src - 1).clamp(min=0.0)) - ex(l_src) + ex(l_cf + 1.0) - ex(l_cf)
            S_cf = (S + dS).clamp(min=1e-30)
            # D = R - R_cf = tau * (log S_cf - log S); rescale to O(1) per-mean-load units
            reward = (torch.log(S_cf) - torch.log(S)) * mean_load
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

    def loo_smoothmax_reward(self, routing_map: torch.Tensor, rl_action: dict = None) -> torch.Tensor:
        """H2: global leave-one-out smooth-max (congestion) reward, corrected sign.

        r_t = J(n - Delta_{A_t}) - J(n) <= 0 with J = logsumexp(beta*n)/beta and n the
        GLOBAL per-expert counts (all-reduced over the token-covering collective, G7).
        Per-token (state = single token); layer-local mean baseline is applied later in the
        per-token REINFORCE loss.

        Args:
            routing_map: [seq, batch, E] boolean assignment mask.
            rl_action:   optional {'experts': [seq,batch,k], ...} -- the exact ordered sampled
                         action; when present its k-subset is used (order irrelevant for the set
                         reward), else the k-subset is reconstructed from routing_map.
        Returns:
            reward: [seq, batch] (<= 0).
        """
        rm = routing_map.float()
        seq, bsz, E = rm.shape
        T = seq * bsz
        if rl_action is not None and rl_action.get('experts', None) is not None:
            S = rl_action['experts'].reshape(T, -1).long()                      # [T, k]
        else:
            k = int(rm.reshape(T, E).sum(dim=-1).max().item())
            S = rm.reshape(T, E).topk(max(1, k), dim=-1).indices.long()         # [T, k]
        local_counts = rm.reshape(T, E).sum(dim=0)                              # [E] LOCAL counts
        group = rl_resolve_token_covering_group() if getattr(self, 'global_loads', False) else None
        global_counts = rl_all_reduce_global_loads(local_counts, group=group)   # [E] GLOBAL (G7)
        beta = float(getattr(self, 'loo_beta', RL_LOO_DEFAULT_BETA))
        r = rl_loo_smoothmax_reward(global_counts, S, beta=beta)                # [T] <= 0
        # --- G7 sum-gate diagnostic (first <=5 calls; rank-0 prints; NEVER crashes training) ---
        # Verifies the token-covering collective on the live topology:
        #   Sum_e global_count  should == global_tokens*k  == config ground truth
        #   (mb*seq*dp*k). ratio ~1.0 => each token routed to exactly k experts, no double count;
        #   a double-counting group shows ratio/counts ~2.0 vs the config ground truth.
        # The all_reduce below is run in LOCKSTEP on ALL ranks (counter is identical across
        # ranks), only rank 0 prints -- so it can never deadlock.
        try:
            _g7_n = getattr(self, '_g7_log_calls', 0)
            if _g7_n < 5:
                _k_g7 = int(S.shape[1])
                _dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
                _gt = torch.tensor([float(T)], device=global_counts.device, dtype=torch.float32)
                if group is not None and _dist_ok and torch.distributed.get_world_size(group=group) > 1:
                    torch.distributed.all_reduce(_gt, op=torch.distributed.ReduceOp.SUM, group=group)
                    _ws = torch.distributed.get_world_size(group=group)
                else:
                    _ws = 1
                self._g7_log_calls = _g7_n + 1  # advance in lockstep on every rank
                _is_rank0 = (not _dist_ok) or torch.distributed.get_rank() == 0
                if _is_rank0:
                    _global_tokens = float(_gt.item())
                    _sum_count = float(global_counts.sum().item())
                    _ratio = _sum_count / max(_global_tokens * _k_g7, 1.0)
                    _cfg_gt = -1
                    try:
                        from megatron.training import get_args
                        from megatron.core import parallel_state as _ps
                        _a = get_args()
                        _dp = _ps.get_data_parallel_world_size()
                        _cfg_gt = int(_a.micro_batch_size) * int(_a.seq_length) * int(_dp) * _k_g7
                    except Exception:
                        _cfg_gt = -1
                    print(f"[G7] call={_g7_n} global_loads={getattr(self, 'global_loads', False)} "
                          f"sum_global_count={_sum_count:.1f} group_world_size={_ws} "
                          f"local_tokens={T} global_tokens={_global_tokens:.1f} k={_k_g7} "
                          f"ratio_sumcount_over_globaltokens_k={_ratio:.4f} "
                          f"config_ground_truth_mb_seq_dp_k={_cfg_gt}", flush=True)
        except Exception:
            pass
        return r.reshape(seq, bsz)

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
        self._dg_ptlp = []; self._dg_adv = []
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
        # [experiment flag] --rl-no-advantage-norm skips this to preserve the reward calibrated scale.
        if not getattr(self, 'no_advantage_norm', False):
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
            _pl_ok = (getattr(self, 'rl_sampling', 'argmax') == 'hard_gumbel_pl'
                      and layer_num in getattr(self, 'pl_decisions', {}))
            if _pl_ok:
                # H1: ordered Plackett-Luce log-prob over the FIXED detached pool (grad-tracked
                # through routing_logits). Replaces the summed-independent-softmax log-prob.
                _pld = self.pl_decisions[layer_num]
                assert torch.equal(_pld['pool_idx'].gather(-1, _pld['pos']), _pld['experts']), \
                    "[RL] PL recompute pool/action mismatch (safeguard)"
                _tau = float(getattr(self, 'rl_stochastic_temperature', 1.0)) or 1.0
                per_token_log_prob = rl_ordered_logprob(
                    rl_fp32(routing_logits), _pld['pool_idx'], _pld['pos'], _tau)  # [seq, batch]
                with torch.no_grad():
                    _isr = (per_token_log_prob - _pld['old_ptlp']).exp().flatten().float()
                    self._rl_is_ratio_p99 = float(torch.quantile(_isr, 0.99).item())
                    self._rl_is_ratio_mean = float(_isr.mean().item())
            elif getattr(self, 'credit_counterfactual', False):
                # [experiment flag] directed credit toward the counterfactual destination:
                # per_token_log_prob = logP(primary-chosen e_src) - logP(best-unchosen e_cf).
                # loss = -per_token_log_prob*A ; diff_lse_load sign (A<0 => beneficial move) =>
                # this lowers P(e_src) and raises P(e_cf) for tokens that should move.
                _neg_inf = float('-inf')
                _chosen = routing_logits.masked_fill(~routing_map.bool(), _neg_inf)
                _e_src = _chosen.argmax(dim=-1, keepdim=True)
                _unchosen = routing_logits.masked_fill(routing_map.bool(), _neg_inf)
                _e_cf = _unchosen.argmax(dim=-1, keepdim=True)
                _lp_src = torch.gather(log_probs, -1, _e_src).squeeze(-1)
                _lp_cf = torch.gather(log_probs, -1, _e_cf).squeeze(-1)
                per_token_log_prob = _lp_src - _lp_cf  # [seq_length, batch_size]
            else:
                chosen_log_probs = log_probs * routing_map.float()
                per_token_log_prob = chosen_log_probs.sum(dim=-1)  # [seq_length, batch_size]
            
            per_token_loss = -per_token_log_prob * advantages
            self._dg_ptlp.append(per_token_log_prob.detach().flatten()); self._dg_adv.append(advantages.detach().flatten())
            
            layer_loss = per_token_loss.mean() if getattr(self, 'perlayer_norm', False) else per_token_loss.sum()
            total_loss += layer_loss
            total_tokens += per_token_loss.numel()
            
            if layer_num == sorted_layers[0]:
                wrap_print_rank_0(f"REINFORCE (per-token) Layer {layer_num}: advantages_mean={advantages.mean().item():.4f}, adv_std={advantages.std().item():.4f}")
        
        # Average over all tokens across all layers
        total_loss = total_loss if getattr(self, 'perlayer_norm', False) else total_loss / max(1, total_tokens)
        
        # Compute mean reward for logging
        mean_reward = sum(r.mean().item() for r in layer_rewards.values()) / max(1, len(layer_rewards))
        
        # Store loss components for logging (topn_load not applicable for per-token)
        try:
            _pp = torch.cat(self._dg_ptlp); _aa = torch.cat(self._dg_adv)
            _ptlp_std = float(_pp.std().item()); _adv_std_used = float(_aa.std().item())
            _cov = float(((_pp - _pp.mean()) * (_aa - _aa.mean())).mean().item())
            _rrstd = float(getattr(self, '_last_raw_reward_std', 0.0))
        except Exception:
            _ptlp_std = _adv_std_used = _cov = _rrstd = 0.0
        self.last_loss_components = {
            'policy_loss': total_loss.item(),
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_advantage': 0.0,
            'mean_reward': mean_reward,
            'avg_topn_load': 0.0,
            'ptlp_std': _ptlp_std,
            'adv_std_used': _adv_std_used,
            'cov_ptlp_adv': _cov,
            'raw_reward_std': _rrstd,
        }
        # H1/H2 telemetry (labeled for the reviewer): sampling mode, IS ratio, flip rate,
        # deterministic-top-k-in-pool rate, advantage/log-prob direction agreement (corr proxy),
        # and the single-normalization reduction factors.
        try:
            _corr = (_cov / (max(_ptlp_std, 1e-8) * max(_adv_std_used, 1e-8))) if (_ptlp_std and _adv_std_used) else 0.0
            _num = float((-torch.cat(self._dg_adv) * torch.cat(self._dg_ptlp)).sum().item())
            _den = 1.0 if getattr(self, 'perlayer_norm', False) else float(max(1, total_tokens))
        except Exception:
            _corr, _num, _den = 0.0, 0.0, 1.0
        self.last_loss_components.update({
            'rl_sampling': getattr(self, 'rl_sampling', 'argmax'),
            'is_ratio_p99': float(getattr(self, '_rl_is_ratio_p99', 0.0)),
            'is_ratio_mean': float(getattr(self, '_rl_is_ratio_mean', 1.0)),
            'flip_rate': float(getattr(self, '_rl_flip_rate', 0.0)),
            'det_topk_in_pool_rate': float(getattr(self, '_rl_det_topk_in_pool_rate', 1.0)),
            'adv_dir_agreement': float(_corr),
            'reduction_numerator': _num,
            'reduction_denominator': _den,
            'reduction_num_layers': int(len(sorted_layers)),
        })
        
        # Surface the H1/H2 reviewer telemetry to the .out (values live in last_loss_components).
        try:
            _lc = self.last_loss_components
            _rl_print_rank0(
                f"[RL TELEM] path=reinforce_per_token sampling={_lc.get('rl_sampling','?')} "
                f"cov_ptlp_adv={_lc.get('cov_ptlp_adv',0.0):.6e} "
                f"is_ratio_mean={_lc.get('is_ratio_mean',0.0):.4f} "
                f"is_ratio_p99={_lc.get('is_ratio_p99',0.0):.4f} "
                f"flip_rate={_lc.get('flip_rate',0.0):.4f} "
                f"det_topk_in_pool_rate={_lc.get('det_topk_in_pool_rate',0.0):.4f} "
                f"adv_dir_agreement={_lc.get('adv_dir_agreement',0.0):.4f} "
                f"mean_reward={_lc.get('mean_reward',0.0):.6f} "
                f"raw_reward_std={_lc.get('raw_reward_std',0.0):.6f} "
                f"num_layers={_lc.get('reduction_num_layers',0)}")
        except Exception:
            pass
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
        
        # Calculate advantages using GAE(lambda) or simple return-baseline
        advantages = {}
        gae_lambda = getattr(self, 'gae_lambda', 1.0)
        if gae_lambda < 1.0 and len(sorted_layers) > 1:
            # GAE: compute TD residuals then exponentially-weighted sum
            td_residuals = {}
            for i, layer_num in enumerate(sorted_layers):
                r = layer_rewards[layer_num]
                if r.dim() > 0:
                    r = r.mean()
                v = baseline_values[layer_num]
                if v.numel() > 1:
                    v = v.mean()
                if i < len(sorted_layers) - 1:
                    next_v = baseline_values[sorted_layers[i + 1]]
                    if next_v.numel() > 1:
                        next_v = next_v.mean()
                    td_residuals[layer_num] = (r + discount_factor * next_v - v).detach()
                else:
                    td_residuals[layer_num] = (r - v).detach()
            gae = torch.tensor(0.0, device=device)
            for layer_num in reversed(sorted_layers):
                gae = td_residuals[layer_num] + discount_factor * gae_lambda * gae
                advantages[layer_num] = gae.clone().detach()
        else:
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
        total_approx_kl = torch.tensor(0.0, device=device)
        total_clip_fraction = torch.tensor(0.0, device=device)
        entropy_coeff = self.ppo_entropy_coeff
        legacy_mode = getattr(self, 'ppo_legacy_mode', False)
        
        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantage = advantages[layer_num]
            return_value = returns[layer_num].detach()
            
            # Current policy log probabilities (normalized by routed tokens)
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            num_tokens_routed = routing_map.sum().clamp_min(1).float()
            current_log_prob = chosen_log_probs.sum() / num_tokens_routed
            
            if legacy_mode:
                # Legacy behavior (REINFORCE-style main PPO update).
                ratio = torch.tensor(1.0, device=device)
                policy_loss = -current_log_prob * advantage
                approx_kl = torch.tensor(0.0, device=device)
                clip_fraction = torch.tensor(0.0, device=device)
            else:
                # True clipped PPO objective using behavior log-probs captured
                # from the same rollout (detached from graph).
                old_log_probs = torch.nn.functional.log_softmax(routing_logits.detach(), dim=-1)
                old_chosen_log_probs = old_log_probs * routing_map.float()
                old_log_prob = old_chosen_log_probs.sum() / num_tokens_routed

                log_ratio = torch.clamp(current_log_prob - old_log_prob, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = ratio * advantage
                pg2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                policy_loss = -torch.min(pg1, pg2)

                approx_kl = (old_log_prob - current_log_prob).detach()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).float().detach()
            
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
            total_approx_kl += approx_kl
            total_clip_fraction += clip_fraction
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
            'approx_kl': (total_approx_kl / num_layers).item(),
            'clip_fraction': (total_clip_fraction / num_layers).item(),
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

        # #endregion

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
        
        # Compute per-token advantages using GAE(lambda) or simple return-baseline
        layer_advantages = {}
        gae_lambda = getattr(self, 'gae_lambda', 1.0)
        if gae_lambda < 1.0 and len(sorted_layers) > 1:
            td_residuals = {}
            for i, layer_num in enumerate(sorted_layers):
                r = layer_rewards[layer_num]
                v = layer_baselines[layer_num]
                if i < len(sorted_layers) - 1:
                    next_v = layer_baselines[sorted_layers[i + 1]]
                    td_residuals[layer_num] = (r + discount_factor * next_v - v).detach()
                else:
                    td_residuals[layer_num] = (r - v).detach()
            gae = torch.zeros_like(first_reward)
            for layer_num in reversed(sorted_layers):
                gae = td_residuals[layer_num] + discount_factor * gae_lambda * gae
                layer_advantages[layer_num] = gae.clone().detach()
        else:
            for layer_num in sorted_layers:
                layer_advantages[layer_num] = (layer_returns[layer_num] - layer_baselines[layer_num]).detach()
        
        # Normalize advantages across all tokens and layers to zero mean, unit variance
        # [experiment flag] --rl-no-advantage-norm skips this to preserve the reward calibrated scale.
        if not getattr(self, 'no_advantage_norm', False):
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
        total_approx_kl = torch.tensor(0.0, device=device)
        total_clip_fraction = torch.tensor(0.0, device=device)
        total_tokens = 0
        entropy_coeff = self.ppo_entropy_coeff
        legacy_mode = getattr(self, 'ppo_legacy_mode', False)
        # H1/H2 causal-check telemetry accumulators (observational; detached).
        self._dg_ptlp = []; self._dg_adv = []; _is_ratios = []

        for layer_num in sorted_layers:
            _, routing_map, routing_logits, reward = trajectory_data[layer_num]
            advantages = layer_advantages[layer_num]  # [seq_length, batch_size]
            returns = layer_returns[layer_num].detach()
            
            # Current policy log probabilities per token
            log_probs = torch.nn.functional.log_softmax(routing_logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            current_per_token_log_prob = chosen_log_probs.sum(dim=-1)
            
            if legacy_mode:
                # Legacy behavior (REINFORCE-style main PPO update).
                ratio = torch.ones_like(current_per_token_log_prob)
                per_token_policy_loss = -current_per_token_log_prob * advantages
                approx_kl = torch.zeros_like(current_per_token_log_prob)
                clip_fraction = torch.zeros_like(current_per_token_log_prob)
            else:
                old_log_probs = torch.nn.functional.log_softmax(routing_logits.detach(), dim=-1)
                old_chosen_log_probs = old_log_probs * routing_map.float()
                old_per_token_log_prob = old_chosen_log_probs.sum(dim=-1)

                log_ratio = torch.clamp(current_per_token_log_prob - old_per_token_log_prob, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = ratio * advantages
                pg2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                per_token_policy_loss = -torch.min(pg1, pg2)

                approx_kl = (old_per_token_log_prob - current_per_token_log_prob).detach()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).float().detach()
            
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
            
            layer_loss = per_token_loss.mean() if getattr(self, 'perlayer_norm', False) else per_token_loss.sum()
            total_loss += layer_loss
            total_policy_loss += per_token_policy_loss.sum()
            total_value_loss += per_token_value_loss.sum()
            total_entropy += per_token_entropy.sum()
            total_advantage += advantages.sum()
            total_reward += reward.sum()
            total_approx_kl += approx_kl.sum()
            total_clip_fraction += clip_fraction.sum()
            total_tokens += per_token_loss.numel()
            # causal-check telemetry (detached): per-token logprob, advantage, PPO IS ratio
            self._dg_ptlp.append(current_per_token_log_prob.detach().flatten())
            self._dg_adv.append(advantages.detach().flatten())
            _is_ratios.append(ratio.detach().flatten().float())

            if layer_num == sorted_layers[0]:
                wrap_print_rank_0(f"PPO (per-token, baseline={self.baseline_type}) Layer {layer_num}: advantages_mean={advantages.mean().item():.4f}, ratio_mean={ratio.mean().item():.4f}")
        
        # Normalize
        total_loss = total_loss if getattr(self, 'perlayer_norm', False) else total_loss / max(1, total_tokens)
        
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
            'approx_kl': (total_approx_kl / max(1, total_tokens)).item(),
            'clip_fraction': (total_clip_fraction / max(1, total_tokens)).item(),
        }
        # H1/H2 causal-check telemetry (observational; no training effect). cov_ptlp_adv > 0
        # means the advantage points the loss the right way (REINFORCE view: loss = -Cov).
        # is_ratio_* is the PPO importance ratio (=1 on-policy, single epoch); flip_rate /
        # det_topk_in_pool_rate are set by the Gumbel-PL router forward and read off the tracker.
        try:
            _pp = torch.cat(self._dg_ptlp); _aa = torch.cat(self._dg_adv)
            _ptlp_std = float(_pp.std().item()); _adv_std_used = float(_aa.std().item())
            _cov = float(((_pp - _pp.mean()) * (_aa - _aa.mean())).mean().item())
            _isr = torch.cat(_is_ratios) if _is_ratios else torch.ones(1, device=device)
            _is_p99 = float(torch.quantile(_isr, 0.99).item()); _is_mean = float(_isr.mean().item())
            _rrstd = float(getattr(self, '_last_raw_reward_std', 0.0))
        except Exception:
            _ptlp_std = _adv_std_used = _cov = _rrstd = 0.0; _is_p99 = _is_mean = 1.0
        self.last_loss_components.update({
            'ptlp_std': _ptlp_std,
            'adv_std_used': _adv_std_used,
            'cov_ptlp_adv': _cov,
            'raw_reward_std': _rrstd,
            'is_ratio_p99': _is_p99,
            'is_ratio_mean': _is_mean,
            'flip_rate': float(getattr(self, '_rl_flip_rate', 0.0)),
            'det_topk_in_pool_rate': float(getattr(self, '_rl_det_topk_in_pool_rate', 1.0)),
            'rl_sampling': getattr(self, 'rl_sampling', 'argmax'),
        })
        try:
            _lc = self.last_loss_components
            _rl_print_rank0(
                f"[RL TELEM] path=ppo_per_token sampling={_lc.get('rl_sampling','?')} "
                f"cov_ptlp_adv={_lc.get('cov_ptlp_adv',0.0):.6e} "
                f"is_ratio_mean={_lc.get('is_ratio_mean',1.0):.4f} "
                f"is_ratio_p99={_lc.get('is_ratio_p99',1.0):.4f} "
                f"flip_rate={_lc.get('flip_rate',0.0):.4f} "
                f"det_topk_in_pool_rate={_lc.get('det_topk_in_pool_rate',1.0):.4f} "
                f"ptlp_std={_lc.get('ptlp_std',0.0):.4f} adv_std={_lc.get('adv_std_used',0.0):.4f} "
                f"mean_reward={_lc.get('mean_reward',0.0):.6f} raw_reward_std={_lc.get('raw_reward_std',0.0):.6f} "
                f"approx_kl={_lc.get('approx_kl',0.0):.4e} num_layers={len(sorted_layers)}")
        except Exception:
            pass
        wrap_print_rank_0(f"PPO (per-token, baseline={self.baseline_type}) DEBUG: total_loss={total_loss.item():.6f}, critic_loss={logged_value_loss:.6f}, adv_std={self.last_loss_components['advantage_std']:.4f}, total_tokens={total_tokens}")


        return total_loss


    def _recompute_rollout_advantages(self, discount_factor):
        """Recompute rewards and advantages based on current router weights and stored routing maps.

        This prevents advantages from becoming stale across extra PPO epochs by
        re-evaluating the reward function with the original routing decisions but
        using the current expert load statistics.
        """
        sorted_layers = sorted(self.old_layer_decisions.keys())
        layer_rewards = {}
        for layer_num in sorted_layers:
            _, routing_map, _, _ = self.old_layer_decisions[layer_num]
            if self.reward_type == 'per_token_load_weighted':
                reward = self.per_token_load_weighted_reward(routing_map)
            elif self.reward_type == 'diff_lse_load':
                _, _, _rl_logits, _ = self.old_layer_decisions[layer_num]
                reward = self.diff_lse_load_reward(routing_map, _rl_logits)
            elif self.reward_type == 'per_token_topn_binary':
                reward = self.per_token_topn_binary_reward(routing_map)
            elif self.reward_type == 'topn_load':
                reward = self.topn_load_reward(routing_map)
            elif self.reward_type == 'critical_path':
                reward = self.critical_path_reward(routing_map)
            else:
                _, _, _, reward = self.old_layer_decisions[layer_num]
            layer_rewards[layer_num] = reward.detach() if isinstance(reward, torch.Tensor) else reward

        layer_returns = {}
        if self.per_token_rewards:
            reward_to_go = torch.zeros_like(layer_rewards[sorted_layers[0]])
        else:
            device = layer_rewards[sorted_layers[0]].device if isinstance(layer_rewards[sorted_layers[0]], torch.Tensor) else 'cuda'
            reward_to_go = torch.tensor(0.0, device=device)

        for layer_num in reversed(sorted_layers):
            r = layer_rewards[layer_num]
            if not self.per_token_rewards and isinstance(r, torch.Tensor) and r.dim() > 0:
                r = r.mean()
            reward_to_go = r + discount_factor * reward_to_go
            layer_returns[layer_num] = reward_to_go.clone() if isinstance(reward_to_go, torch.Tensor) else reward_to_go

        if self.per_token_rewards:
            all_returns = torch.cat([layer_returns[ln].flatten() for ln in sorted_layers])
        else:
            all_returns = torch.stack([layer_returns[ln] if isinstance(layer_returns[ln], torch.Tensor) else torch.tensor(layer_returns[ln]) for ln in sorted_layers])
        ret_mean = all_returns.mean()
        ret_std = all_returns.std().clamp(min=1e-8)
        return {ln: ((layer_returns[ln] - ret_mean) / ret_std).detach() for ln in sorted_layers}

    def _move_rollout_to_device(self, trajectory, device):
        """Move a CPU-stored trajectory back to GPU."""
        gpu_trajectory = {}
        for ln, (latent, rmap, logits, reward) in trajectory.items():
            gpu_trajectory[ln] = (
                latent.to(device, non_blocking=True),
                rmap.to(device, non_blocking=True),
                logits.to(device, non_blocking=True),
                reward.to(device, non_blocking=True) if isinstance(reward, torch.Tensor) else reward,
            )
        return gpu_trajectory

    def _compute_ppo_epoch_loss(self, trajectory, original_logits, layer_advantages,
                                 clip_ratio, rl_loss_coeff, device):
        """Compute PPO clipped loss for a single trajectory (shared between current + replay)."""
        sorted_layers = sorted(trajectory.keys())
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0

        for layer_num in sorted_layers:
            if layer_num not in self._router_modules:
                continue
            old_latent, old_routing_map, _, _ = trajectory[layer_num]
            router_module = self._router_modules[layer_num]
            advantages = layer_advantages.get(layer_num)
            if advantages is None:
                continue

            new_logits = router_module.gating(old_latent)
            new_lp = torch.nn.functional.log_softmax(new_logits, dim=-1)
            orig_lp = torch.nn.functional.log_softmax(original_logits[layer_num], dim=-1)
            old_rm = old_routing_map.float()

            if self.per_token_rewards:
                new_per_token = (new_lp * old_rm).sum(dim=-1)
                orig_per_token = (orig_lp * old_rm).sum(dim=-1)
                log_ratio = torch.clamp(new_per_token - orig_per_token, -10.0, 10.0)
                ratio = torch.exp(log_ratio)

                pg1 = ratio * advantages
                pg2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                per_token_loss = -torch.min(pg1, pg2)
                total_loss += per_token_loss.sum()
                total_tokens += per_token_loss.numel()
            else:
                adv = advantages if isinstance(advantages, torch.Tensor) else torch.tensor(advantages, device=device)
                if adv.dim() > 0:
                    adv = adv.mean()
                per_token_log_ratio = torch.clamp(
                    (new_lp * old_rm).sum(dim=-1) - (orig_lp * old_rm).sum(dim=-1), -10.0, 10.0)
                per_token_ratio = torch.exp(per_token_log_ratio)
                pg1 = per_token_ratio * adv
                pg2 = torch.clamp(per_token_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                per_token_loss = -torch.min(pg1, pg2)
                num_routed = old_rm.sum(dim=-1).clamp_min(1).bool().float().sum().clamp_min(1)
                total_loss += per_token_loss.sum() / num_routed
                total_tokens += 1

        return total_loss / max(1, total_tokens) * rl_loss_coeff, total_tokens

    def run_extra_ppo_epochs(self, rl_loss_coeff: float, discount_factor: float = 0.0,
                              clip_ratio: float = 0.2):
        """Run K-1 additional PPO epochs on the stored trajectory (+ replay buffer).

        Called after the main training step (forward + backward + optimizer.step()).
        Each epoch: recomputes advantages under current weights, re-evaluates
        stored states, computes PPO loss, and does a gradient step on router weights only.

        Key design decisions:
        1. extra_lr scaled by 1/(K-1) to prevent compounding updates
        2. Global gradient clipping across all router weights
        3. Advantages are recomputed each epoch to prevent staleness
        4. Replay buffer trajectories are included when available
        """
        if self.ppo_epochs <= 1 or not self._router_modules:
            return
        if not self.old_layer_decisions:
            return

        sorted_layers = sorted(self.old_layer_decisions.keys())
        if not sorted_layers:
            return

        first_router = next(iter(self._router_modules.values()))
        device = first_router.weight.device

        # Get the original (pre-update) logits from the stored trajectory
        original_logits = {}
        for layer_num in sorted_layers:
            _, _, logits, _ = self.old_layer_decisions[layer_num]
            original_logits[layer_num] = logits.detach()

        legacy_mode = getattr(self, 'ppo_legacy_mode', False)

        # Scale learning rate by 1/(K-1) to prevent compounding updates
        extra_lr = getattr(self, 'ppo_extra_lr', 1e-4) / max(1, self.ppo_epochs - 1)

        # Lazy-init separate optimizer for router weights
        router_params = [p for router in self._router_modules.values() for p in router.parameters()]
        if not router_params:
            return
        if self._ppo_optimizer is None:
            if legacy_mode:
                self._ppo_optimizer = torch.optim.SGD(router_params, lr=extra_lr)
            else:
                self._ppo_optimizer = torch.optim.Adam(router_params, lr=extra_lr, eps=1e-5)
        else:
            for pg in self._ppo_optimizer.param_groups:
                pg['lr'] = extra_lr

        # Prepare replay buffer rollouts on GPU
        replay_rollouts = []
        replay_orig_logits = []
        if self.replay_buffer_size > 0 and self._replay_buffer:
            for cpu_traj in self._replay_buffer:
                gpu_traj = self._move_rollout_to_device(cpu_traj, device)
                replay_rollouts.append(gpu_traj)
                rp_orig = {ln: data[2].detach() for ln, data in gpu_traj.items()}
                replay_orig_logits.append(rp_orig)

        # Legacy mode keeps a single cached advantage tensor across all extra epochs.
        # New mode recomputes each epoch to reduce stale-policy drift.
        cached_advantages = self._recompute_rollout_advantages(discount_factor) if legacy_mode else None

        # Run K-1 extra epochs
        for epoch in range(self.ppo_epochs - 1):
            layer_advantages = cached_advantages if legacy_mode else self._recompute_rollout_advantages(discount_factor)

            # Loss from current trajectory
            total_loss, total_tokens = self._compute_ppo_epoch_loss(
                self.old_layer_decisions, original_logits, layer_advantages,
                clip_ratio, rl_loss_coeff, device)

            # Add losses from replay buffer trajectories
            for rp_traj, rp_orig in zip(replay_rollouts, replay_orig_logits):
                rp_layers = sorted(rp_traj.keys())
                rp_rewards = {}
                for ln in rp_layers:
                    _, rm, _, _ = rp_traj[ln]
                    if self.reward_type == 'per_token_load_weighted':
                        rp_rewards[ln] = self.per_token_load_weighted_reward(rm)
                    elif self.reward_type == 'per_token_topn_binary':
                        rp_rewards[ln] = self.per_token_topn_binary_reward(rm)
                    else:
                        rp_rewards[ln] = rp_traj[ln][3]

                # Simple advantage computation for replay data
                if self.per_token_rewards:
                    all_rp = torch.cat([r.flatten() for r in rp_rewards.values()])
                else:
                    all_rp = torch.stack([r.mean() if isinstance(r, torch.Tensor) and r.dim() > 0 else r for r in rp_rewards.values()])
                rp_mean = all_rp.mean()
                rp_std = all_rp.std().clamp(min=1e-8)
                rp_advs = {ln: ((rp_rewards[ln] - rp_mean) / rp_std).detach() for ln in rp_layers}

                rp_loss, rp_tokens = self._compute_ppo_epoch_loss(
                    rp_traj, rp_orig, rp_advs, clip_ratio, rl_loss_coeff, device)
                total_loss = total_loss + rp_loss
                total_tokens += rp_tokens

            # Average across all trajectories
            n_trajectories = 1 + len(replay_rollouts)
            total_loss = total_loss / n_trajectories

            self._ppo_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, 1.0)
            self._ppo_optimizer.step()

        opt_name = self._ppo_optimizer.__class__.__name__ if self._ppo_optimizer is not None else "None"
        print(f"[PPO MULTI-EPOCH] Ran {self.ppo_epochs - 1} extra epochs "
                         f"(lr={extra_lr:.2e}, opt={opt_name}, legacy={legacy_mode}, buf={len(replay_rollouts)}), "
                         f"last_loss={total_loss.item():.6f}")


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
            _global_trajectory_tracker.ppo_legacy_mode = getattr(args, 'rl_ppo_legacy_mode', False)
            _tracker_configured = True
            try:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[RL CONFIG] baseline={_global_trajectory_tracker.baseline_type}, "
                          f"reward={_global_trajectory_tracker.reward_type}, "
                          f"topn={_global_trajectory_tracker.reward_topn}, "
                          f"per_token={_global_trajectory_tracker.per_token_rewards}, "
                          f"normalize={_global_trajectory_tracker.normalize_rewards}, "
                          f"legacy={_global_trajectory_tracker.ppo_legacy_mode}", flush=True)
            except Exception:
                pass
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


def run_post_step_ppo_if_pending():
    """Run deferred extra PPO epochs after optimizer.step(), if scheduled."""
    global _global_trajectory_tracker
    if _global_trajectory_tracker is None:
        return False
    return _global_trajectory_tracker.run_scheduled_extra_ppo_epochs()
