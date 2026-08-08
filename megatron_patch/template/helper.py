# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain GPT."""

import os
import torch
import inspect

from functools import partial
from megatron.core import mpu

from megatron.training import get_args, get_timers
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)

from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron_patch.data.utils import (
    get_batch_on_this_tp_rank_original, 
    get_batch_on_this_tp_rank_idxmap_sft,
    get_position_id_on_this_tp_rank_idxmap_sft_packing
)
from megatron.training.utils import print_rank_0

# --- KL divergence constraint state ---
# Stores frozen reference router weights and captured logits for KL loss computation.
# Initialized lazily on first forward_step call when kl_loss_coeff > 0.
_kl_state = {
    'ref_router_weights': None,   # dict: param_name -> frozen tensor
    'current_logits': None,       # captured by output_layer hook (in grad graph)
    'ref_logits': None,           # from reference forward (detached)
    'initialized': False,
    # When True, the output_layer hook does NOT overwrite current_logits.
    # Used to protect the training-forward logits during the reference
    # forward pass (which also calls output_layer and would otherwise
    # clobber current_logits with the frozen-router outputs, making the
    # subsequent KL computation collapse to ~0).
    'capture_disabled': False,
    # Router-KL: when True, the core router stashes per-layer routing logits into
    # ref_routing_logits during the frozen-router reference forward (analogous to
    # capture_disabled). Only set True when router_kl_coeff>0.
    'capture_ref_routing': False,
}

def _kl_capture_logits_hook(module, input, output):
    """Forward hook on output_layer to capture logits from normal forward pass."""
    if _kl_state.get('capture_disabled', False):
        return
    # output is (logits, bias) tuple from ColumnParallelLinear
    logits = output[0] if isinstance(output, tuple) else output
    _kl_state['current_logits'] = logits

def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        packed_seq_params = None
        if args.dataset == 'MMAP' and args.train_mode == "finetune" and args.reset_position_ids:
            position_ids = get_position_id_on_this_tp_rank_idxmap_sft_packing(data_iterator)
            position_ids = position_ids[0] # shape: [seq_length]
            start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
            seqlens = start_indices[1:] - start_indices[:-1]
            # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
            cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
            cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
            cu_seqlens[-1] = position_ids.shape[0]
            max_seqlen = torch.max(seqlens.max(), position_ids.max() + 1)
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format='thd',
                max_seqlen_q = max_seqlen,
                max_seqlen_kv = max_seqlen,
            )

        return None, None, None, None, None, None, packed_seq_params

    if args.dataset == 'JSON-SFT':
        if args.train_mode == "pretrain":
            raise ValueError('The JSON-SFT dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=True)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif args.dataset == 'MMAP':
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            batch = get_batch_on_this_tp_rank(data_iterator)
        else:
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=True)
        
        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0] # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                max_seqlen = torch.max(seqlens.max(), position_ids.max() + 1)
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd',
                    max_seqlen_q = max_seqlen,
                    max_seqlen_kv = max_seqlen,
                )
        
        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


def loss_func(loss_mask: torch.Tensor, num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    # NOTE: for each seq, sum(loss_mask) == 1 if num_seqs is not None, 
    # otherwise sum(loss_mask) == n_tokens
    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    # Create loss dictionary starting with main loss
    loss_dict = {"lm loss": averaged_loss}
    
    # Collect auxiliary losses from MoE layers (like load_balancing_loss)
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker,
        reduce_aux_losses_tracker_across_ranks,
        clear_aux_losses_tracker,
    )
    
    # Get the auxiliary losses tracker
    tracker = get_moe_layer_wise_logging_tracker()
    
    if tracker:  # Only process if there are auxiliary losses
        # Reduce auxiliary losses across ranks
        reduce_aux_losses_tracker_across_ranks()
        
        # Add auxiliary losses to the loss dictionary
        for name, loss_data in tracker.items():
            if 'values' in loss_data:
                loss_values = loss_data['values'].float()
                
                # Average across all MoE layers
                loss_avg = loss_values.sum() / max(1, len(loss_values.nonzero()))
                loss_dict[name] = loss_avg
                max_loss_value = torch.max(loss_values)
                min_loss_value = torch.min(loss_values)
                layer_0_loss_value = loss_values[0]
                loss_dict[f"{name}_max"] = max_loss_value
                loss_dict[f"{name}_min"] = min_loss_value
                loss_dict[f"{name}_layer_0"] = layer_0_loss_value
        
        # Compute num_tokens_on_critical_path: sum of max tokens per expert over all layers
        # This represents the compute critical path (slowest expert per layer, summed)
        if 'max_tokens_per_expert' in tracker and 'values' in tracker['max_tokens_per_expert']:
            max_tokens_values = tracker['max_tokens_per_expert']['values'].float()
            loss_dict['num_tokens_on_critical_path'] = max_tokens_values.sum()
        
        # Clear the tracker for next iteration
        clear_aux_losses_tracker()
    
    if num_seqs is None:
        # average on token-level
        return loss[0] / loss[1] * args.context_parallel_size, loss_dict
    return loss[0] * args.context_parallel_size, num_seqs.sum(), loss_dict


def loss_func_with_rl(loss_mask: torch.Tensor, num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function that includes RL auxiliary loss.

    Assumes trajectory tracking is available and supported.
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    # NOTE: for each seq, sum(loss_mask) == 1 if num_seqs is not None, 
    # otherwise sum(loss_mask) == n_tokens
    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    # Create loss dictionary starting with main loss
    loss_dict = {"lm loss": averaged_loss}
    
    # Critical metrics dict (logged directly to wandb without train/ prefix)
    critical_metrics = {"critical/lm_loss": averaged_loss.item() if hasattr(averaged_loss, 'item') else float(averaged_loss)}
    
    # Get reward_type for conditional metric logging
    reward_type = getattr(args, 'rl_reward_type', 'expert0')
    
    # Collect auxiliary losses from MoE layers (like load_balancing_loss)
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker,
        reduce_aux_losses_tracker_across_ranks,
        clear_aux_losses_tracker,
    )
    
    # Get the auxiliary losses tracker
    tracker = get_moe_layer_wise_logging_tracker()
    
    if tracker:  # Only process if there are auxiliary losses
        # Reduce auxiliary losses across ranks
        reduce_aux_losses_tracker_across_ranks()
        
        # Add auxiliary losses to the loss dictionary
        for name, loss_data in tracker.items():
            # Skip expert_0 metrics unless using expert0 reward
            if 'expert_0' in name and reward_type != 'expert0':
                continue
                
            if 'values' in loss_data:
                loss_values = loss_data['values'].float()
                
                # Average across all MoE layers
                loss_avg = loss_values.sum() / max(1, len(loss_values.nonzero()))
                loss_dict[name] = loss_avg
                max_loss_value = torch.max(loss_values)
                min_loss_value = torch.min(loss_values)
                layer_0_loss_value = loss_values[0]
                loss_dict[f"{name}_max"] = max_loss_value
                loss_dict[f"{name}_min"] = min_loss_value
                loss_dict[f"{name}_layer_0"] = layer_0_loss_value
        
        # Compute num_tokens_on_critical_path: sum of max tokens per expert over all layers
        # This represents the compute critical path (slowest expert per layer, summed)
        if 'max_tokens_per_expert' in tracker and 'values' in tracker['max_tokens_per_expert']:
            max_tokens_values = tracker['max_tokens_per_expert']['values'].float()
            num_critical_path = max_tokens_values.sum()
            loss_dict['num_tokens_on_critical_path'] = num_critical_path
            critical_metrics['critical/num_tokens_on_critical_path'] = num_critical_path.item()
        
        if 'locality_ratio' in tracker and 'values' in tracker['locality_ratio']:
            lr_values = tracker['locality_ratio']['values'].float()
            lr_avg = lr_values.sum() / max(1, len(lr_values.nonzero()))
            critical_metrics['critical/locality_ratio'] = lr_avg.item()
            critical_metrics['critical/locality_ratio_max'] = lr_values.max().item()
            critical_metrics['critical/locality_ratio_min'] = lr_values.min().item()
        
        # Clear the tracker for next iteration
        clear_aux_losses_tracker()

    # RL loss computation (assumes tracker exists)
    from megatron_patch.model.qwen3_moe.moe.rl_trajectory import (
        get_trajectory_tracker,
        reset_trajectory_tracker,
    )

    trajectory_tracker = get_trajectory_tracker()
    
    # Always force configuration from args -- get_trajectory_tracker() may have
    # silently failed to read args during early model construction.
    trajectory_tracker.baseline_type = getattr(args, 'rl_ppo_baseline_type', 'mean')
    trajectory_tracker.reward_type = getattr(args, 'rl_reward_type', 'expert0')
    trajectory_tracker.reward_topn = getattr(args, 'rl_reward_topn', 12)
    trajectory_tracker.reward_topm = getattr(args, 'rl_reward_topm', 0)
    trajectory_tracker.reward_c = getattr(args, 'rl_reward_c', 0.1)
    trajectory_tracker.reward_c_end = getattr(args, 'rl_reward_c_end', 0.0)
    trajectory_tracker.ep_size = getattr(args, 'expert_model_parallel_size', 1)
    trajectory_tracker.per_token_rewards = getattr(args, 'rl_per_token_rewards', False)
    # Auto-detect per_token_rewards for reward types that are inherently per-token
    _PER_TOKEN_REWARD_TYPES = {"per_token_topn_binary", "per_token_load_weighted", "per_token_topm", "per_token_smoothmax", "loo_smoothmax", "loo_maxrelative"}
    if trajectory_tracker.reward_type in _PER_TOKEN_REWARD_TYPES:
        trajectory_tracker.per_token_rewards = True
    trajectory_tracker.ppo_entropy_coeff = getattr(args, 'rl_ppo_entropy_coeff', 0.01)
    trajectory_tracker.critic_hidden_dims = getattr(args, 'rl_critic_hidden_dims', [256])
    trajectory_tracker.critic_lr = getattr(args, 'rl_critic_lr', 1e-3)
    trajectory_tracker.normalize_rewards = getattr(args, 'rl_normalize_rewards', False)
    trajectory_tracker.ppo_clip_ratio = getattr(args, 'rl_ppo_clip_ratio', 0.2)
    trajectory_tracker.use_ema_loads = getattr(args, 'rl_use_ema_loads', False)
    trajectory_tracker.critic_layer_aware = getattr(args, 'rl_critic_layer_aware', False)
    trajectory_tracker.ppo_reeval = getattr(args, 'rl_ppo_reeval', False)
    trajectory_tracker.ppo_epochs = getattr(args, 'rl_ppo_epochs', 1)
    trajectory_tracker.replay_buffer_size = getattr(args, 'rl_replay_buffer_size', 0)
    trajectory_tracker.ppo_extra_lr = getattr(args, 'rl_ppo_extra_lr', 1e-4)
    trajectory_tracker.ppo_legacy_mode = getattr(args, 'rl_ppo_legacy_mode', False)
    trajectory_tracker.gae_lambda = getattr(args, 'rl_gae_lambda', 1.0)
    trajectory_tracker.no_advantage_norm = getattr(args, 'rl_no_advantage_norm', False)
    trajectory_tracker.credit_counterfactual = getattr(args, 'rl_credit_counterfactual', False)
    trajectory_tracker.rl_disconnect_repro = getattr(args, 'rl_disconnect_repro', False)
    trajectory_tracker.global_load = getattr(args, 'rl_global_load', False)
    trajectory_tracker.perlayer_norm = getattr(args, 'rl_perlayer_norm', False)
    # --- H1/H2 port config (defaults OFF => current behavior) ---
    trajectory_tracker.rl_sampling = getattr(args, 'rl_sampling', 'argmax')
    trajectory_tracker.rl_candidate_pool = getattr(args, 'rl_candidate_pool', 0)
    trajectory_tracker.rl_stochastic_temperature = getattr(args, 'rl_stochastic_temperature', 1.0)
    trajectory_tracker.global_loads = getattr(args, 'rl_global_loads', False)
    trajectory_tracker.loo_beta = getattr(args, 'rl_loo_beta', 0.3)
    print(f"[RL CONFIG] reward_type={trajectory_tracker.reward_type}, "
          f"baseline_type={trajectory_tracker.baseline_type}, "
          f"per_token_rewards={trajectory_tracker.per_token_rewards}, "
          f"reward_topn={trajectory_tracker.reward_topn}, "
          f"normalize_rewards={trajectory_tracker.normalize_rewards}, "
          f"critic_hidden_dims={trajectory_tracker.critic_hidden_dims}, "
          f"clip_ratio={trajectory_tracker.ppo_clip_ratio}, "
          f"use_ema_loads={trajectory_tracker.use_ema_loads}, "
          f"legacy_mode={trajectory_tracker.ppo_legacy_mode}, "
          f"gae_lambda={trajectory_tracker.gae_lambda}", flush=True)
    
    rl_loss_coeff = getattr(args, 'rl_loss_coeff', 0.1)
    if getattr(args, 'rl_cosine_schedule', False):
        import math
        iteration = getattr(args, 'iteration', 0) or 0
        total_iters = max(1, getattr(args, 'train_iters', 5000))
        warmup_frac = 0.1
        min_coeff = rl_loss_coeff * 0.1
        if iteration < total_iters * warmup_frac:
            rl_loss_coeff = rl_loss_coeff * (iteration / max(1, total_iters * warmup_frac))
        else:
            progress = (iteration - total_iters * warmup_frac) / max(1, total_iters * (1 - warmup_frac))
            rl_loss_coeff = min_coeff + 0.5 * (rl_loss_coeff - min_coeff) * (1 + math.cos(math.pi * progress))
        critical_metrics['critical/rl_loss_coeff_scheduled'] = rl_loss_coeff
    rl_algorithm = getattr(args, 'rl_algorithm', 'reinforce').lower()
    rl_discount_factor = getattr(args, 'rl_discount_factor', 0.9)
    
    # Inject per-token LM cross-entropy as additional reward if enabled
    lm_reward_coeff = getattr(args, 'rl_lm_reward_coeff', 0.0)
    if lm_reward_coeff > 0:
        trajectory_tracker.inject_lm_reward(output_tensor, lm_reward_coeff)
    
    # Compute RL loss based on selected algorithm
    if getattr(args, 'rl_soft_smoothmax', False):
        rl_loss = trajectory_tracker.compute_soft_smoothmax_loss(trajectory_tracker.layer_decisions)
    elif rl_algorithm == 'ppo':
        rl_loss = trajectory_tracker.compute_ppo_loss(
            trajectory_tracker.layer_decisions,
            trajectory_tracker.old_layer_decisions,
            discount_factor=rl_discount_factor,
            clip_ratio=getattr(trajectory_tracker, 'ppo_clip_ratio', 0.2),
            value_coeff=0.5
        )
    else:  # reinforce
        rl_loss = trajectory_tracker.compute_reinforce_loss(
            trajectory_tracker.layer_decisions,
            discount_factor=rl_discount_factor
        )
    
    # Scale and add RL loss
    rl_loss = rl_loss * rl_loss_coeff
    loss_dict["rl_loss"] = rl_loss.detach()
    
    # Add component losses for detailed wandb logging (must be tensors for Megatron)
    if hasattr(trajectory_tracker, 'last_loss_components'):
        components = trajectory_tracker.last_loss_components
        policy_loss = torch.tensor(components.get('policy_loss', 0.0))
        value_loss = torch.tensor(components.get('value_loss', 0.0))
        
        loss_dict["rl_policy_loss"] = policy_loss
        loss_dict["rl_value_loss"] = value_loss
        loss_dict["rl_entropy_bonus"] = torch.tensor(components.get('entropy_bonus', 0.0))
        loss_dict["rl_mean_advantage"] = torch.tensor(components.get('mean_advantage', 0.0))
        loss_dict["rl_mean_reward"] = torch.tensor(components.get('mean_reward', 0.0))
        loss_dict["rl_advantage_std"] = torch.tensor(components.get('advantage_std', 0.0))
        loss_dict["rl_advantage_min"] = torch.tensor(components.get('advantage_min', 0.0))
        loss_dict["rl_advantage_max"] = torch.tensor(components.get('advantage_max', 0.0))
        loss_dict["rl_approx_kl"] = torch.tensor(components.get('approx_kl', 0.0))
        loss_dict["rl_clip_fraction"] = torch.tensor(components.get('clip_fraction', 0.0))
        loss_dict["rl_ptlp_std"] = torch.tensor(components.get('ptlp_std', 0.0))
        loss_dict["rl_cov_ptlp_adv"] = torch.tensor(components.get('cov_ptlp_adv', 0.0))
        loss_dict["rl_raw_reward_std"] = torch.tensor(components.get('raw_reward_std', 0.0))
        # H1/H2 causal-check telemetry -> iteration line (positive-valued; survive the
        # training_log `avg > 0.0` print filter). Signed cov_ptlp_adv is ALSO emitted as a
        # raw [RL TELEM] stdout line by the loss method so its sign is never hidden.
        loss_dict["rl_is_ratio_p99"] = torch.tensor(components.get('is_ratio_p99', 0.0))
        loss_dict["rl_is_ratio_mean"] = torch.tensor(components.get('is_ratio_mean', 0.0))
        loss_dict["rl_flip_rate"] = torch.tensor(components.get('flip_rate', 0.0))
        loss_dict["rl_det_topk_in_pool_rate"] = torch.tensor(components.get('det_topk_in_pool_rate', 0.0))

        # Critical metrics for policy and value loss
        critical_metrics['critical/policy_loss'] = policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss)
        critical_metrics['critical/value_loss'] = value_loss.item() if hasattr(value_loss, 'item') else float(value_loss)
        critical_metrics['critical/advantage_std'] = float(components.get('advantage_std', 0.0))
        critical_metrics['critical/approx_kl'] = float(components.get('approx_kl', 0.0))
        critical_metrics['critical/clip_fraction'] = float(components.get('clip_fraction', 0.0))
        
        # Only log avg_topn_load when using topn_load reward
        avg_topn = components.get('avg_topn_load', 0.0)
        if avg_topn > 0:
            loss_dict["rl_avg_topn_load"] = torch.tensor(avg_topn)
            critical_metrics['critical/avg_topn_load'] = float(avg_topn)

    # RL health telemetry — computed on ALL ranks so the keys survive the cross-rank loss_dict
    # reduction (rank-0-only keys get dropped). rl_loss_requires_grad=1 => RL plumbed into the graph;
    # rl_grad_diag: 1 ok / 0 no-layers / -1 loss-detached / -2 logits-no-grad_fn / -3 grad-None /
    # -4 grad-error. rl_grad_norm_on_logits = how hard RL pushes the router (0.0 if not computable).
    try:
        _gn = 0.0; _gm = 0.0; _diag = 0.0
        if len(trajectory_tracker.layer_decisions) == 0:
            _diag = 0.0
        elif not rl_loss.requires_grad:
            _diag = -1.0
        else:
            _sl = min(trajectory_tracker.layer_decisions.keys())
            _rlogits = trajectory_tracker.layer_decisions[_sl][2]
            if not (getattr(_rlogits, "requires_grad", False) and getattr(_rlogits, "grad_fn", None) is not None):
                _diag = -2.0
            else:
                try:
                    _g = torch.autograd.grad(rl_loss, _rlogits, retain_graph=True, allow_unused=True)[0]
                    if _g is None:
                        _diag = -3.0
                    else:
                        _gn = _g.norm().item(); _gm = _g.abs().mean().item(); _diag = 1.0
                except Exception:
                    _diag = -4.0
        loss_dict["rl_loss_requires_grad"] = torch.tensor(1.0 if rl_loss.requires_grad else 0.0)
        loss_dict["rl_grad_diag"] = torch.tensor(float(_diag))
        loss_dict["rl_grad_norm_on_logits"] = torch.tensor(float(_gn))
        loss_dict["rl_grad_mean_on_logits"] = torch.tensor(float(_gm))
    except Exception:
        pass

    # Expert heatmap logging (if enabled)
    log_heatmap = getattr(args, 'log_expert_heatmap', False)
    if log_heatmap and hasattr(trajectory_tracker, 'log_heatmap'):
        try:
            from megatron.training import get_num_microbatches
            iteration = getattr(args, 'iteration', 0)
            trajectory_tracker.log_heatmap(iteration)
        except Exception:
            pass

    # Output distribution metrics (entropy, top-1 prob, router weight drift)
    if _kl_state.get('current_logits') is not None:
        try:
            cur_logits = _kl_state['current_logits'].detach().float()
            probs = torch.softmax(cur_logits, dim=-1)
            log_probs = torch.log_softmax(cur_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            top1_prob = probs.max(dim=-1).values.mean()
            loss_dict["output_entropy"] = entropy
            loss_dict["output_top1_prob"] = top1_prob
            critical_metrics['critical/output_entropy'] = entropy.item()
            critical_metrics['critical/output_top1_prob'] = top1_prob.item()
        except Exception:
            pass

    # Router weight drift from initial checkpoint
    if _kl_state.get('ref_router_weights'):
        try:
            total_drift = 0.0
            n_params = 0
            for name, param in model.named_parameters() if hasattr(model, 'named_parameters') else []:
                if name in _kl_state['ref_router_weights']:
                    drift = (param.data - _kl_state['ref_router_weights'][name]).norm().item()
                    total_drift += drift
                    n_params += 1
            if n_params > 0:
                avg_drift = total_drift / n_params
                loss_dict["router_weight_drift"] = torch.tensor(avg_drift)
                critical_metrics['critical/router_weight_drift'] = avg_drift
        except Exception:
            pass

    # Log critical metrics directly to wandb (without train/ prefix)
    try:
        from megatron.core import parallel_state as mpu
        if mpu.get_data_parallel_rank() == 0:
            import wandb
            if wandb.run is not None:
                wandb.log(critical_metrics, commit=False)
    except Exception:
        pass

    # --- KL divergence constraint ---
    kl_loss_coeff = getattr(args, 'kl_loss_coeff', 0.0)
    if kl_loss_coeff > 0:
        _kl_cur = _kl_state['current_logits']
        _kl_ref = _kl_state['ref_logits']
        if _kl_cur is None or _kl_ref is None:
            if not hasattr(loss_func_with_rl, '_kl_debug_printed'):
                print(f"[KL DEBUG] KL skipped: coeff={kl_loss_coeff}, current_logits={'None' if _kl_cur is None else _kl_cur.shape}, ref_logits={'None' if _kl_ref is None else _kl_ref.shape}, initialized={_kl_state['initialized']}", flush=True)
                loss_func_with_rl._kl_debug_printed = True
    if kl_loss_coeff > 0 and _kl_state['current_logits'] is not None and _kl_state['ref_logits'] is not None:
        try:
            import torch.nn.functional as F
            cur_logits = _kl_state['current_logits']  # [seq, batch, vocab] — in grad graph
            ref_logits = _kl_state['ref_logits']       # [batch, seq, vocab] — detached

            # Align shapes: ref_logits is [batch, seq, vocab] (transposed by _postprocess when labels=None)
            # cur_logits is [seq, batch, vocab] (raw from output_layer)
            ref_logits = ref_logits.transpose(0, 1)  # [seq, batch, vocab]

            # Compute KL(current || reference) per token, then average
            cur_log_probs = F.log_softmax(cur_logits.float(), dim=-1)
            ref_probs = F.softmax(ref_logits.float(), dim=-1)
            # F.kl_div expects log_probs as input, probs as target
            kl_per_token = F.kl_div(cur_log_probs, ref_probs, reduction='none').sum(dim=-1)  # [seq, batch]
            kl_loss = kl_per_token.mean()

            # Scale and add to RL loss (will be combined with LM loss below)
            scaled_kl = kl_loss_coeff * kl_loss
            rl_loss = rl_loss + scaled_kl

            loss_dict['kl_loss'] = kl_loss.detach()
            critical_metrics['critical/kl_loss'] = kl_loss.item()
            # Print KL value for the first several calls + sparsely afterwards so we
            # can confirm KL grows as router weights drift (cur vs ref divergence).
            _kl_n = getattr(loss_func_with_rl, '_kl_print_count', 0)
            if _kl_n < 10 or _kl_n % 100 == 0:
                cur_mean = cur_logits.float().mean().item()
                ref_mean = ref_logits.float().mean().item()
                cur_ptr = cur_logits.data_ptr()
                ref_ptr = ref_logits.data_ptr()
                print(
                    f"[KL DEBUG] call={_kl_n} kl_loss={kl_loss.item():.4e} "
                    f"scaled={scaled_kl.item():.4e} coeff={kl_loss_coeff} "
                    f"cur_mean={cur_mean:.4f} ref_mean={ref_mean:.4f} "
                    f"same_ptr={cur_ptr == ref_ptr}",
                    flush=True,
                )
            loss_func_with_rl._kl_print_count = _kl_n + 1
        except Exception as e:
            print_rank_0(f"[KL] WARNING: KL computation failed: {e}")

    # --- Router-KL anchor (sibling of head-KL; independently gated) ---
    router_kl_coeff = getattr(args, 'router_kl_coeff', 0.0)
    if router_kl_coeff > 0 and _kl_state.get('ref_routing_logits'):
        try:
            import torch.nn.functional as F
            ref_routing = _kl_state['ref_routing_logits']
            per_layer_kl = []
            for layer_num, decision in trajectory_tracker.layer_decisions.items():
                if layer_num not in ref_routing:
                    continue
                cur_l = decision[2]                # [seq, batch, E] — in grad graph
                ref_l = ref_routing[layer_num]     # [seq, batch, E] — frozen, detached
                if cur_l.shape != ref_l.shape:
                    continue
                # Match head-KL EXACTLY: KL(ref || cur), reverse-KL, T=1.
                cur_lp = F.log_softmax(cur_l.float(), dim=-1)
                ref_p = F.softmax(ref_l.float(), dim=-1)
                kl_l = F.kl_div(cur_lp, ref_p, reduction='none').sum(dim=-1).mean()
                per_layer_kl.append(kl_l)
            if per_layer_kl:
                router_kl = torch.stack(per_layer_kl).mean()   # average over layers
                rl_loss = rl_loss + router_kl_coeff * router_kl
                loss_dict['router_kl_loss'] = router_kl.detach()
                critical_metrics['critical/router_kl_loss'] = router_kl.item()
                _rk_n = getattr(loss_func_with_rl, '_router_kl_print_count', 0)
                if _rk_n < 10 or _rk_n % 100 == 0:
                    print(f"[ROUTER-KL DEBUG] call={_rk_n} router_kl={router_kl.item():.4e} "
                          f"scaled={(router_kl_coeff * router_kl).item():.4e} "
                          f"coeff={router_kl_coeff} n_layers={len(per_layer_kl)}", flush=True)
                loss_func_with_rl._router_kl_print_count = _rk_n + 1
        except Exception as e:
            print_rank_0(f"[ROUTER-KL] WARNING: router-KL computation failed: {e}")

    # P3: snapshot the just-completed rollout (sampled actions/pools/advantages/old log-prob)
    # BEFORE the reset clears pl_decisions. The post-step recompute happens at the top of the
    # next forward_step (after this rollout's optimizer.step()). Best-effort; never fatal.
    try:
        from megatron_patch.model.qwen3_moe.moe import rl_probe as _rl_probe
        if _rl_probe.audit_enabled():
            _audit_iter = getattr(args, 'curr_iteration', getattr(args, 'iteration', 0)) or 0
            _rl_probe.snapshot_audit(trajectory_tracker, _audit_iter)
    except Exception as _e_audit_snap:
        print_rank_0(f"[AUDIT] WARNING: snapshot hook failed: {_e_audit_snap}")

    # Reset trajectory for next iteration (moves current layer_decisions → old_layer_decisions)
    reset_trajectory_tracker()

    # Run extra PPO epochs.
    # New path: defer to a strict post-optimizer-step hook in training.py.
    # Legacy path: keep inline execution inside loss construction.
    if trajectory_tracker.ppo_epochs > 1 and trajectory_tracker.ppo_reeval:
        clip_ratio = getattr(trajectory_tracker, 'ppo_clip_ratio', 0.2)
        if getattr(trajectory_tracker, 'ppo_legacy_mode', False):
            trajectory_tracker.run_extra_ppo_epochs(
                rl_loss_coeff=rl_loss_coeff,
                discount_factor=rl_discount_factor,
                clip_ratio=clip_ratio,
            )
        else:
            trajectory_tracker.schedule_extra_ppo_epochs(
                rl_loss_coeff=rl_loss_coeff,
                discount_factor=rl_discount_factor,
                clip_ratio=clip_ratio,
            )

    # #region agent log
    import json, time as _t
    _dbg = {"sessionId":"63a0ae","hypothesisId":"H5_loss_combine","location":"helper.py:loss_combine","timestamp":int(_t.time()*1000),
            "message":"rl_loss_combination",
            "data":{"rl_loss_raw":float(rl_loss.item()),"rl_loss_requires_grad":bool(rl_loss.requires_grad),
                    "lm_loss_sum":float(loss[0].item()),"loss_count":float(loss[1].item()),
                    "rl_loss_coeff":float(rl_loss_coeff),
                    "rl_loss_scaled":float((rl_loss * loss[1]).item()),
                    "lm_loss_avg":float((loss[0]/loss[1]).item()),
                    "combined_loss_avg":float(((loss[0] + rl_loss * loss[1])/loss[1]).item())}}
    pass  # disabled 2026-07-12: per-step all-rank writes to one lustre file grew to 14.5GB (IO hazard)
    # #endregion

    # Scale RL loss to match the LM loss scale (RL loss is averaged, LM loss is summed)
    use_only_rl_loss = False
    if use_only_rl_loss:
        loss = torch.stack([rl_loss * loss[1], loss[1]])
    else:        
        loss = torch.stack([loss[0] + rl_loss * loss[1], loss[1]])
    if num_seqs is None:
        # average on token-level
        return loss[0] / loss[1] * args.context_parallel_size, loss_dict
    return loss[0] * args.context_parallel_size, num_seqs.sum(), loss_dict   

def _init_kl_state(model, snapshot_weights=True):
    """One-time initialization: register logit capture hook and optionally snapshot router weights."""
    if _kl_state['initialized']:
        return
    
    print(f"[KL DEBUG] _init_kl_state called, snapshot_weights={snapshot_weights}, model type={type(model).__name__}", flush=True)
    
    # Snapshot all router weights (frozen reference) — only when KL loss is enabled
    if snapshot_weights:
        ref_weights = {}
        for name, param in model.named_parameters():
            if 'router' in name and 'weight' in name:
                ref_weights[name] = param.data.clone().detach()
        _kl_state['ref_router_weights'] = ref_weights
        print(f"[KL DEBUG] Snapshotted {len(ref_weights)} router weight tensors", flush=True)
    
    # Walk model hierarchy to find output_layer.
    # Megatron wraps: DistributedDataParallel -> Float16Module -> GPTModel
    found = False
    obj = model
    path = 'model'
    for depth in range(5):
        if hasattr(obj, 'output_layer'):
            obj.output_layer.register_forward_hook(_kl_capture_logits_hook)
            print(f"[KL DEBUG] Registered logit capture hook on {path}.output_layer", flush=True)
            found = True
            break
        if hasattr(obj, 'module'):
            obj = obj.module
            path += '.module'
        else:
            break
    
    if not found:
        print(f"[KL DEBUG] WARNING: Could not find output_layer after {path}. Final obj type={type(obj).__name__}, attrs={[a for a in dir(obj) if not a.startswith('_')][:15]}", flush=True)
    
    _kl_state['initialized'] = True


def _run_reference_forward(model, tokens, position_ids, attention_mask, packed_seq_params):
    """Run a no-grad forward with frozen router weights to get reference logits."""
    ref_weights = _kl_state['ref_router_weights']
    if not ref_weights:
        return
    
    # Save current router weights and swap in frozen reference
    saved_weights = {}
    for name, param in model.named_parameters():
        if name in ref_weights:
            saved_weights[name] = param.data.clone()
            param.data.copy_(ref_weights[name])
    
    # Disable trajectory tracking during reference forward to avoid
    # polluting routing statistics with decisions from frozen router weights
    try:
        from megatron_patch.model.qwen3_moe.moe.rl_trajectory import get_trajectory_tracker
        tracker = get_trajectory_tracker()
        tracker.paused = True
    except Exception:
        tracker = None

    # Save and clear the aux losses tracker so the reference forward's
    # routing statistics (max_tokens_per_expert, tokens_routed_to_expert_0,
    # etc.) don't contaminate training metrics via the += accumulator.
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker,
        clear_aux_losses_tracker,
    )
    aux_tracker = get_moe_layer_wise_logging_tracker()
    saved_aux = {name: {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in entry.items()}
                 for name, entry in aux_tracker.items()}

    clear_aux_losses_tracker()

    # Disable the output_layer logit-capture hook so the reference forward
    # does NOT overwrite current_logits (the training forward's logits, in
    # the grad graph). Without this guard, current_logits and ref_logits
    # both end up holding the frozen-router output and KL collapses to ~0.
    _kl_state['capture_disabled'] = True

    # Router-KL: capture the frozen-router per-layer routing logits during this
    # reference forward. Only when router-KL is enabled (keeps head-KL-only runs unchanged).
    _router_kl_on = getattr(get_args(), 'router_kl_coeff', 0.0) > 0
    if _router_kl_on:
        _kl_state['ref_routing_logits'] = {}
        _kl_state['capture_ref_routing'] = True

    # Reference forward (no grad, labels=None to get logits)
    try:
        with torch.no_grad():
            ref_logits = model(tokens, position_ids, attention_mask,
                               labels=None, packed_seq_params=packed_seq_params)
            # ref_logits shape: [batch, seq, vocab] (transposed in _postprocess when labels=None)
            _kl_state['ref_logits'] = ref_logits.detach()
            if not hasattr(_run_reference_forward, '_debug_printed'):
                cur = _kl_state['current_logits']
                cur_desc = 'None' if cur is None else f"shape={tuple(cur.shape)}, mean={cur.float().mean().item():.4f}"
                ref_desc = f"shape={tuple(ref_logits.shape)}, mean={ref_logits.float().mean().item():.4f}"
                print(f"[KL DEBUG] Reference forward OK: ref_logits {ref_desc}; current_logits {cur_desc}", flush=True)
                _run_reference_forward._debug_printed = True
    except Exception as e:
        print(f"[KL DEBUG] Reference forward FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        _kl_state['ref_logits'] = None
    finally:
        # Re-enable logit capture before any subsequent forwards
        _kl_state['capture_disabled'] = False
        _kl_state['capture_ref_routing'] = False
        # Restore current router weights
        for name, param in model.named_parameters():
            if name in saved_weights:
                param.data.copy_(saved_weights[name])
        # Re-enable trajectory tracking
        if tracker is not None:
            tracker.paused = False
        # Restore the aux losses tracker so training metrics are uncontaminated
        clear_aux_losses_tracker()
        for name, entry in saved_aux.items():
            aux_tracker[name] = entry


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    kl_loss_coeff = getattr(args, 'kl_loss_coeff', 0.0)
    router_kl_coeff = getattr(args, 'router_kl_coeff', 0.0)

    # One-time initialization: always register logit capture hook (for output metrics),
    # only snapshot router weights when KL loss is enabled
    if not _kl_state['initialized']:
        _init_kl_state(model, snapshot_weights=(kl_loss_coeff > 0 or router_kl_coeff > 0))

    # --- P2/P3: measurement infrastructure. Cheap/idempotent; all no-ops unless the
    # probe/audit intervals are set. snapshot_theta0 runs on the very first forward_step,
    # BEFORE the first optimizer step, so it captures the INITIAL router theta0. ---
    try:
        from megatron_patch.model.qwen3_moe.moe import rl_probe
        rl_probe.configure(args)
        rl_probe.banner_once(args)  # P1 banner on the reliably-flushed training path
        if getattr(args, 'use_rl_loss', False) and (rl_probe.probe_enabled() or rl_probe.audit_enabled()):
            rl_probe.snapshot_theta0(model)
    except Exception as _e_probe_cfg:
        rl_probe = None

    # P3: run the frozen-rollout causal audit for the PREVIOUS step's rollout. The
    # optimizer step for that rollout has completed by the top of this forward_step, so
    # recomputing the frozen action's log-prob here reflects the post-step router.
    _probe_iter = getattr(args, 'curr_iteration', getattr(args, 'iteration', 0)) or 0
    if rl_probe is not None and rl_probe.audit_enabled() and torch.is_grad_enabled():
        try:
            from megatron_patch.model.qwen3_moe.moe.rl_trajectory import get_trajectory_tracker
            rl_probe.run_audit(get_trajectory_tracker(), _probe_iter)
        except Exception as _e_audit:
            print_rank_0(f"[AUDIT] WARNING: audit failed: {_e_audit}")

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()

    # P2: freeze the first N training microbatches as the immutable probe set.
    if rl_probe is not None and getattr(args, 'use_rl_loss', False) and torch.is_grad_enabled():
        rl_probe.maybe_capture(
            (tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params))

    # Clear previous logits
    _kl_state['current_logits'] = None
    if router_kl_coeff > 0:
        _kl_state['ref_routing_logits'] = {}

    if 'loss_mask' in inspect.signature(GPTModel.forward).parameters:
        # NOTE: MTP-head (since 0328) requires loss_mask to compute correct loss scale.
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params, loss_mask=loss_mask)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    # After normal forward, _kl_state['current_logits'] is populated by the hook (if KL enabled)

    # Run reference forward for KL constraint (only during training with grad)
    if (kl_loss_coeff > 0 or router_kl_coeff > 0) and torch.is_grad_enabled():
        _run_reference_forward(model, tokens, position_ids, attention_mask, packed_seq_params)

    # Periodic HellaSwag benchmark via subprocess after checkpoint saves.
    # Uses latest_checkpointed_iteration.txt so triggering is aligned with real saves.
    save_interval = getattr(args, 'save_interval', 0)
    hellaswag_interval = getattr(args, 'hellaswag_eval_interval', 0)
    hellaswag_limit = max(1, int(getattr(args, 'hellaswag_eval_limit', 100)))
    if hellaswag_interval > 0 and save_interval > 0 and torch.is_grad_enabled():
        if not hasattr(forward_step, '_bench_process'):
            forward_step._bench_process = None
            forward_step._bench_log_fh = None
            forward_step._bench_last_checkpoint_iter = -1

        # Check if a previous async benchmark finished and collect status.
        if forward_step._bench_process is not None and forward_step._bench_process.poll() is not None:
            try:
                from megatron.core import parallel_state as mpu
                if mpu.get_data_parallel_rank() == 0:
                    rc = forward_step._bench_process.returncode
                    print(f"[BENCHMARK] Background benchmark finished (exit={rc})", flush=True)
            except Exception:
                pass
            if getattr(forward_step, '_bench_log_fh', None) is not None:
                try:
                    forward_step._bench_log_fh.close()
                except Exception:
                    pass
                forward_step._bench_log_fh = None
            forward_step._bench_process = None

        try:
            from megatron.core import parallel_state as mpu
            if mpu.get_data_parallel_rank() == 0:
                save_dir = getattr(args, 'save', None)
                if save_dir and os.path.isdir(save_dir):
                    latest_iter_file = os.path.join(save_dir, 'latest_checkpointed_iteration.txt')
                    if os.path.isfile(latest_iter_file):
                        with open(latest_iter_file, 'r') as f:
                            latest_iter = int(f.read().strip() or '0')

                        # Run benchmark once for each new saved checkpoint.
                        should_launch = (
                            latest_iter > 0
                            and latest_iter != forward_step._bench_last_checkpoint_iter
                            and latest_iter % save_interval == 0
                            and forward_step._bench_process is None
                        )
                        if should_launch:
                            import subprocess
                            script_dir = os.path.join(
                                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                'examples',
                                'qwen3',
                            )
                            bench_script = os.path.join(script_dir, 'run_inline_benchmark.sh')
                            if os.path.exists(bench_script):
                                bench_iter = latest_iter
                                bench_log = os.path.join(save_dir, f'inline_benchmark_iter{bench_iter}.log')
                                print(
                                    f"[BENCHMARK] Launching HellaSwag at iter {bench_iter} "
                                    f"(limit={hellaswag_limit}, async, log={bench_log})",
                                    flush=True,
                                )
                                bench_log_fh = open(bench_log, 'a')
                                forward_step._bench_log_fh = bench_log_fh
                                forward_step._bench_process = subprocess.Popen(
                                    ['bash', bench_script, save_dir, str(bench_iter), str(hellaswag_limit)],
                                    stdout=bench_log_fh,
                                    stderr=subprocess.STDOUT,
                                )
                                forward_step._bench_last_checkpoint_iter = bench_iter
        except Exception as e:
            print(f"[BENCHMARK] WARNING: failed to launch: {e}", flush=True)

    # P2: fixed deterministic-CP probe (runs its own eval/no_grad forwards on the frozen
    # probe set; temporarily swaps in theta0 for the baseline). Best-effort; never fatal.
    if rl_probe is not None and torch.is_grad_enabled() and rl_probe.should_probe(_probe_iter):
        try:
            from megatron_patch.model.qwen3_moe.moe.rl_trajectory import get_trajectory_tracker
            rl_probe.run_probe(model, get_trajectory_tracker(), _probe_iter)
        except Exception as _e_probe:
            print_rank_0(f"[PROBE] WARNING: probe failed: {_e_probe}")

    # Choose loss function based on CLI arg parsed by Megatron
    use_rl_loss = getattr(args, 'use_rl_loss', False) and torch.is_grad_enabled()
    selected_loss_func = loss_func_with_rl if use_rl_loss else loss_func
    return output_tensor, partial(selected_loss_func, loss_mask, num_seqs)
