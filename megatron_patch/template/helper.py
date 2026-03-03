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
}

def _kl_capture_logits_hook(module, input, output):
    """Forward hook on output_layer to capture logits from normal forward pass."""
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
    trajectory_tracker.per_token_rewards = getattr(args, 'rl_per_token_rewards', False)
    # Auto-detect per_token_rewards for reward types that are inherently per-token
    _PER_TOKEN_REWARD_TYPES = {"per_token_topn_binary", "per_token_load_weighted"}
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
    print(f"[RL CONFIG] reward_type={trajectory_tracker.reward_type}, "
          f"baseline_type={trajectory_tracker.baseline_type}, "
          f"per_token_rewards={trajectory_tracker.per_token_rewards}, "
          f"reward_topn={trajectory_tracker.reward_topn}, "
          f"normalize_rewards={trajectory_tracker.normalize_rewards}, "
          f"critic_hidden_dims={trajectory_tracker.critic_hidden_dims}, "
          f"clip_ratio={trajectory_tracker.ppo_clip_ratio}, "
          f"use_ema_loads={trajectory_tracker.use_ema_loads}", flush=True)
    
    rl_loss_coeff = getattr(args, 'rl_loss_coeff', 0.1)
    rl_algorithm = getattr(args, 'rl_algorithm', 'reinforce').lower()
    rl_discount_factor = getattr(args, 'rl_discount_factor', 0.9)
    
    # Inject per-token LM cross-entropy as additional reward if enabled
    lm_reward_coeff = getattr(args, 'rl_lm_reward_coeff', 0.0)
    if lm_reward_coeff > 0:
        trajectory_tracker.inject_lm_reward(output_tensor, lm_reward_coeff)
    
    # Compute RL loss based on selected algorithm
    if rl_algorithm == 'ppo':
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
        
        # Critical metrics for policy and value loss
        critical_metrics['critical/policy_loss'] = policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss)
        critical_metrics['critical/value_loss'] = value_loss.item() if hasattr(value_loss, 'item') else float(value_loss)
        critical_metrics['critical/advantage_std'] = float(components.get('advantage_std', 0.0))
        
        # Only log avg_topn_load when using topn_load reward
        avg_topn = components.get('avg_topn_load', 0.0)
        if avg_topn > 0:
            loss_dict["rl_avg_topn_load"] = torch.tensor(avg_topn)
            critical_metrics['critical/avg_topn_load'] = float(avg_topn)

    # Gradient magnitude diagnostic: compute RL gradient norms on routing logits
    # Uses layer_decisions (current iteration) since rl_loss was computed from them.
    # Previously this used old_layer_decisions which are from the previous iteration
    # and have no computational connection to rl_loss, so gradients were always None.
    try:
        from megatron.core import parallel_state as mpu
        if mpu.get_data_parallel_rank() == 0:
            print_rank_0(f"[RL DEBUG] rl_loss={rl_loss.item():.4e}, lm_loss={averaged_loss.item():.4e}", override_debug_mode=False)
            
            # Compute gradient norms of RL loss w.r.t. routing logits (sample first layer)
            if rl_loss.requires_grad and len(trajectory_tracker.layer_decisions) > 0:
                sample_layer = min(trajectory_tracker.layer_decisions.keys())
                _, _, routing_logits_sample, _ = trajectory_tracker.layer_decisions[sample_layer]
                if routing_logits_sample.requires_grad and routing_logits_sample.grad_fn is not None:
                    try:
                        rl_grads = torch.autograd.grad(
                            rl_loss, routing_logits_sample,
                            retain_graph=True, allow_unused=True
                        )
                        if rl_grads[0] is not None:
                            rl_grad_norm = rl_grads[0].norm().item()
                            rl_grad_mean = rl_grads[0].abs().mean().item()
                            loss_dict["rl_grad_norm_on_logits"] = torch.tensor(rl_grad_norm)
                            loss_dict["rl_grad_mean_on_logits"] = torch.tensor(rl_grad_mean)
                    except Exception:
                        pass  # Don't crash training for diagnostic logging
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
            print_rank_0(f"[KL] kl_loss={kl_loss.item():.4e}, scaled={scaled_kl.item():.4e}",
                        override_debug_mode=False)
        except Exception as e:
            print_rank_0(f"[KL] WARNING: KL computation failed: {e}")

    # Reset trajectory for next iteration (moves current layer_decisions → old_layer_decisions)
    reset_trajectory_tracker()

    # Run extra PPO epochs on the just-stored trajectory (now in old_layer_decisions)
    if trajectory_tracker.ppo_epochs > 1 and trajectory_tracker.ppo_reeval:
        trajectory_tracker.run_extra_ppo_epochs(
            rl_loss_coeff=rl_loss_coeff,
            discount_factor=rl_discount_factor,
            clip_ratio=getattr(trajectory_tracker, 'ppo_clip_ratio', 0.2),
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
    with open("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/.cursor/debug-63a0ae.log","a") as _f: _f.write(json.dumps(_dbg)+"\n")
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

def _init_kl_state(model):
    """One-time initialization: snapshot router weights and register logit capture hook."""
    if _kl_state['initialized']:
        return
    
    # Snapshot all router weights (frozen reference)
    ref_weights = {}
    for name, param in model.named_parameters():
        if 'router' in name and 'weight' in name:
            ref_weights[name] = param.data.clone().detach()
    _kl_state['ref_router_weights'] = ref_weights
    
    # Register forward hook on output_layer to capture logits
    if hasattr(model, 'output_layer'):
        model.output_layer.register_forward_hook(_kl_capture_logits_hook)
        print_rank_0(f"[KL] Registered logit capture hook on output_layer, "
                     f"snapshotted {len(ref_weights)} router weight tensors")
    elif hasattr(model, 'module') and hasattr(model.module, 'output_layer'):
        # Handle DDP/wrapped models
        model.module.output_layer.register_forward_hook(_kl_capture_logits_hook)
        print_rank_0(f"[KL] Registered logit capture hook on module.output_layer, "
                     f"snapshotted {len(ref_weights)} router weight tensors")
    else:
        print_rank_0("[KL] WARNING: Could not find output_layer on model, KL constraint disabled")
    
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
    
    # Reference forward (no grad, labels=None to get logits)
    try:
        with torch.no_grad():
            ref_logits = model(tokens, position_ids, attention_mask,
                               labels=None, packed_seq_params=packed_seq_params)
            # ref_logits shape: [batch, seq, vocab] (transposed in _postprocess when labels=None)
            _kl_state['ref_logits'] = ref_logits.detach()
    except Exception as e:
        print_rank_0(f"[KL] WARNING: Reference forward failed: {e}")
        _kl_state['ref_logits'] = None
    finally:
        # Restore current router weights
        for name, param in model.named_parameters():
            if name in saved_weights:
                param.data.copy_(saved_weights[name])


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    kl_loss_coeff = getattr(args, 'kl_loss_coeff', 0.0)

    # One-time KL initialization (snapshot reference weights, register hook)
    if kl_loss_coeff > 0 and not _kl_state['initialized']:
        _init_kl_state(model)

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Clear previous logits
    _kl_state['current_logits'] = None

    if 'loss_mask' in inspect.signature(GPTModel.forward).parameters:
        # NOTE: MTP-head (since 0328) requires loss_mask to compute correct loss scale.
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params, loss_mask=loss_mask)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    # After normal forward, _kl_state['current_logits'] is populated by the hook (if KL enabled)

    # Run reference forward for KL constraint (only during training with grad)
    if kl_loss_coeff > 0 and torch.is_grad_enabled():
        _run_reference_forward(model, tokens, position_ids, attention_mask, packed_seq_params)

    # Choose loss function based on CLI arg parsed by Megatron
    # During eval (no grad), skip RL loss to avoid trajectory tracker issues and wasted compute
    use_rl_loss = getattr(args, 'use_rl_loss', False) and torch.is_grad_enabled()
    selected_loss_func = loss_func_with_rl if use_rl_loss else loss_func
    return output_tensor, partial(selected_loss_func, loss_mask, num_seqs)
