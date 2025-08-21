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

from typing import Union
from contextlib import nullcontext
import torch
import torch._dynamo
import inspect

from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron_patch.tokenizer import build_tokenizer
"""
from megatron_patch.model.qwen3_moe.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
"""
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import train_valid_test_datasets_provider
from megatron.training import get_args, pretrain, print_rank_0

torch._dynamo.config.suppress_errors = True


def freeze_router_parameters(model):
    """Freeze all parameters except MoE router weights."""
    router_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        # Check if this is a router/gate parameter - be more flexible with naming
        is_router = any(keyword in name.lower() for keyword in ['router', 'gate']) and 'weight' in name
        
        if is_router:
            param.requires_grad = True
            router_params += param.numel()
            print_rank_0(f"[TRAINABLE] {name}")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    total_params = router_params + frozen_params
    print_rank_0(f"Router-only training: {router_params:,} trainable / {total_params:,} total parameters ({router_params/total_params:.1%})")


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building QWen3 model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
        else:
            # Define the decoder layer spec
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                    normalization=args.normalization)
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

    # Apply router-only training if enabled
    if args.router_only_training:
        print_rank_0("ROUTER-ONLY TRAINING: Freezing all parameters except router weights")
        freeze_router_parameters(model)

    # Initialize RL loss if enabled
    if args.use_rl_loss:
        print_rank_0(f"RL LOSS ENABLED: Using {args.rl_algorithm.upper()} algorithm with coefficient {args.rl_loss_coeff}")
        if not args.router_only_training:
            print_rank_0("WARNING: RL loss is typically used with router-only training")
        
        # Enable trajectory tracking in MoE config
        for module in model.modules():
            if hasattr(module, 'config') and hasattr(module.config, 'moe_router_use_trajectory_tracking'):
                module.config.moe_router_use_trajectory_tracking = True
                print_rank_0(f"Enabled trajectory tracking for module: {module.__class__.__name__}")

    return model

def forward_step_with_rl(data_iterator, model):
    """Forward training step with RL loss integration."""
    from megatron_patch.template.helper import forward_step as base_forward_step
    from megatron_patch.model.qwen3_moe.moe.rl_trajectory import get_trajectory_tracker, reset_trajectory_tracker
    from megatron.core import parallel_state as mpu
    import torch
    
    args = get_args()
    
    # Reset trajectory tracker for new forward pass
    if args.use_rl_loss:
        reset_trajectory_tracker()
    
    # Call the base forward step to get LM loss
    output_tensor, loss_func = base_forward_step(data_iterator, model)
    
    # Compute RL loss if enabled
    if args.use_rl_loss:
        tracker = get_trajectory_tracker()
        
        if len(tracker.layer_decisions) > 0:
            # Compute RL loss based on algorithm choice
            if args.rl_algorithm == 'reinforce':
                rl_loss = tracker.compute_reinforce_loss(tracker.layer_decisions)
            elif args.rl_algorithm == 'ppo':
                old_trajectory = getattr(tracker, 'old_layer_decisions', {})
                rl_loss = tracker.compute_ppo_loss(tracker.layer_decisions, old_trajectory)
            else:
                rl_loss = torch.tensor(0.0, device=output_tensor.device)
            
            # Scale RL loss by coefficient
            scaled_rl_loss = rl_loss * args.rl_loss_coeff
            
            if mpu.get_tensor_model_parallel_rank() == 0:
                print_rank_0(f"RL Loss: {rl_loss:.6f}, Scaled: {scaled_rl_loss:.6f}")
        else:
            scaled_rl_loss = torch.tensor(0.0, device=output_tensor.device)
        
        # Create a wrapper loss function that adds RL loss
        def loss_func_with_rl(loss_mask, num_seqs, output_tensor):
            result = loss_func(loss_mask, num_seqs, output_tensor)
            
            if num_seqs is None:
                # Returns: (loss, losses_reduced_dict)
                lm_loss, losses_reduced = result
                total_loss = lm_loss + scaled_rl_loss
                
                # Add RL loss to reporting
                losses_reduced = losses_reduced.copy()
                losses_reduced["rl loss"] = scaled_rl_loss
                
                return total_loss, losses_reduced
            else:
                # Returns: (loss, num_seqs, losses_reduced_dict)  
                lm_loss, num_seqs_sum, losses_reduced = result
                total_loss = lm_loss + scaled_rl_loss
                
                # Add RL loss to reporting
                losses_reduced = losses_reduced.copy()
                losses_reduced["rl loss"] = scaled_rl_loss
                
                return total_loss, num_seqs_sum, losses_reduced
        
        return output_tensor, partial(loss_func_with_rl)
    
    # If no RL loss, return original
    return output_tensor, loss_func

if __name__ == "__main__":
    from functools import partial
    train_valid_test_datasets_provider.is_distributed = True

    # Use RL-enhanced forward step if RL loss is enabled
    args = get_args()
    if args.use_rl_loss:
        forward_step_func = forward_step_with_rl
    else:
        from megatron_patch.template.helper import forward_step as forward_step_func

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step_func,
        extra_args_provider=get_patch_args,
    )