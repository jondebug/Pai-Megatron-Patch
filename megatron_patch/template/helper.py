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

    # NOTE: The grad will be scaled down by CP size later, should not remove this multilication factor
    # LINK: https://github.com/NVIDIA/Megatron-LM/issues/906
    # The issue is solved since 0926

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
        
        # Clear the tracker for next iteration
        clear_aux_losses_tracker()

    # RL loss computation (assumes tracker exists)
    from megatron_patch.model.qwen3_moe.moe.rl_trajectory import (
        get_trajectory_tracker,
        reset_trajectory_tracker,
    )

    trajectory_tracker = get_trajectory_tracker()
    rl_loss = torch.tensor(0.0, device=averaged_loss.device)

    print_rank_0(f"[RL DEBUG] Computing RL loss from {len(trajectory_tracker.layer_decisions)} layers")
    
    # Get RL loss coefficient
    rl_loss_coeff = getattr(args, 'rl_loss_coeff', 0.1)
    use_per_layer_loss = getattr(args, 'use_per_layer_loss', False)
    
    # Compute REINFORCE loss from the complete trajectory
    if use_per_layer_loss:
        rl_loss = trajectory_tracker.compute_reinforce_loss_per_layer(
            trajectory_tracker.layer_decisions,
            discount_factor=0.9
        )
    else:
        rl_loss = trajectory_tracker.compute_reinforce_loss(
            trajectory_tracker.layer_decisions,
            discount_factor=0.9
        )
    
    # ---------- RL DEBUG (prints + optional wandb) ----------
    rl_debug = True
    print_rank_0(f"[RL DEBUG] rl_debug={rl_debug}")
    from megatron.core import parallel_state as mpu
    is_main_rank = (mpu.get_data_parallel_rank() == 0)
    is_main_rank = True
    print_rank_0(f"[RL DEBUG] is_main_rank={is_main_rank}")

    # Predefine summary vars for potential zero-loss print below
    total_tokens_selected = None
    total_logprob = None
    total_layer_loss_mag = None
    reward_mean = None
    state_mean = None

    if rl_debug and is_main_rank:
        # Recompute per-layer debug stats (logprob, reward, state_value, layer_loss)
        sorted_layers = sorted(trajectory_tracker.layer_decisions.keys())
        layer_rewards = {}
        for layer_num in sorted_layers:
            _, _, _, reward = trajectory_tracker.layer_decisions[layer_num]
            layer_rewards[layer_num] = reward

        discount_factor = 0.9
        layer_values = {}
        accumulated_value = torch.tensor(0.0, device=averaged_loss.device)
        for layer_num in reversed(sorted_layers):
            accumulated_value = layer_rewards[layer_num] + discount_factor * accumulated_value
            layer_values[layer_num] = accumulated_value

        debug_metrics = {
            "debug/rl/num_layers": float(len(sorted_layers)),
            "debug/rl/coeff": float(rl_loss_coeff),
            "debug/rl/use_per_layer": float(1.0 if use_per_layer_loss else 0.0),
        }

        print_rank_0(f"[RL DEBUG] layers={len(sorted_layers)}, coeff={rl_loss_coeff}, per_layer={use_per_layer_loss}")
        total_tokens_selected = 0.0
        total_logprob = 0.0
        total_layer_loss_mag = 0.0
        reward_values = []
        state_values = []

        for layer_num in sorted_layers:
            logits, routing_map, _, reward = trajectory_tracker.layer_decisions[layer_num]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs * routing_map.float()
            layer_log_prob = chosen_log_probs.sum()
            state_value = layer_values[layer_num]
            layer_loss_dbg = -(layer_log_prob * (state_value if not use_per_layer_loss else reward))
            tokens_selected = float(routing_map.sum().item())

            print_rank_0(
                f"[RL DEBUG] L{layer_num}: tokens={int(tokens_selected)}, reward={reward.item():.6e}, logprob={layer_log_prob.item():.6e}, state_value={state_value.item():.6e}, layer_loss={layer_loss_dbg.item():.6e}"
            )

            # Accumulate summaries
            total_tokens_selected += tokens_selected
            total_logprob += float(layer_log_prob.item())
            total_layer_loss_mag += float(abs(layer_loss_dbg.item()))
            reward_values.append(reward)
            state_values.append(state_value)

            # Wandb debug metrics (compact)
            debug_metrics[f"debug/rl/l{layer_num}_tokens"] = float(tokens_selected)
            debug_metrics[f"debug/rl/l{layer_num}_reward"] = float(reward.item())
            debug_metrics[f"debug/rl/l{layer_num}_logprob"] = float(layer_log_prob.item())
            debug_metrics[f"debug/rl/l{layer_num}_state"] = float(state_value.item())
            debug_metrics[f"debug/rl/l{layer_num}_loss"] = float(layer_loss_dbg.item())

        # Log current unscaled/scaled rl losses as debug
        debug_metrics["debug/rl/loss_unscaled"] = float(rl_loss.item())

        # Add summary stats into loss_dict so they appear in training_log and wandb
        try:
            reward_mean = torch.stack(reward_values).mean()
            state_mean = torch.stack(state_values).mean()
        except Exception:
            reward_mean = torch.tensor(0.0, device=averaged_loss.device)
            state_mean = torch.tensor(0.0, device=averaged_loss.device)

        # Use magnitudes to ensure positive values for console printing filter
        loss_dict["debug_rl_num_layers"] = torch.tensor(float(len(sorted_layers)), device=averaged_loss.device)
        loss_dict["debug_rl_tokens_selected"] = torch.tensor(total_tokens_selected, device=averaged_loss.device)
        loss_dict["debug_rl_logprob_mag"] = torch.tensor(abs(total_logprob), device=averaged_loss.device)
        loss_dict["debug_rl_layer_loss_mag"] = torch.tensor(total_layer_loss_mag, device=averaged_loss.device)
        loss_dict["debug_rl_reward_mean"] = reward_mean
        loss_dict["debug_rl_state_mean"] = state_mean

    # Scale and add RL loss
    rl_loss = rl_loss * rl_loss_coeff

    # Always expose rl_loss for logging (even if zero)
    loss_dict["rl_loss"] = rl_loss.detach()

    # Add RL loss directly to the main loss only if non-zero
    if rl_loss.item() != 0.0:
        old_loss_value = averaged_loss.item()
        averaged_loss = averaged_loss + rl_loss
        print_rank_0(f"[RL DEBUG] Added RL loss: {old_loss_value:.6f} + {rl_loss.item():.6f} = {averaged_loss.item():.6f}")


    # Reset trajectory for next iteration
    reset_trajectory_tracker()

    if num_seqs is None:
        # average on token-level
        return loss[0] / loss[1] * args.context_parallel_size, loss_dict
    return loss[0] * args.context_parallel_size, num_seqs.sum(), loss_dict

def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()
    if 'loss_mask' in inspect.signature(GPTModel.forward).parameters:
        # NOTE: MTP-head (since 0328) requires loss_mask to compute correct loss scale.
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params, loss_mask=loss_mask)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)

    # Choose loss function based on CLI arg parsed by Megatron
    use_rl_loss = getattr(args, 'use_rl_loss', False)
    print_rank_0(f"[RL DEBUG] using {"loss_func_with_rl" if use_rl_loss else "loss_func"}")
    selected_loss_func = loss_func_with_rl if use_rl_loss else loss_func
    return output_tensor, partial(selected_loss_func, loss_mask, num_seqs)
