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


def setup_wandb_logging():
    """Monkey patch the training_log function to add Wandb logging"""
    args = get_args()
    
    if getattr(args, 'enable_wandb_logging', False):
        try:
            import wandb
            from megatron.training.training import training_log as original_training_log
            from megatron.core import parallel_state as mpu
            import megatron.training.training as training_module  # Import at the top
            
            def enhanced_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, 
                                    iteration, loss_scale, report_memory_flag, skipped_iter,
                                    grad_norm, params_norm, num_zeros_in_grad):
                """Enhanced training_log with Wandb integration"""
                
                # Call original training_log first
                result = original_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
                                             iteration, loss_scale, report_memory_flag, skipped_iter,
                                             grad_norm, params_norm, num_zeros_in_grad)
                
                # Add Wandb logging
                try:
                    rank = mpu.get_data_parallel_rank()
                    
                    if rank == 0 and not skipped_iter:  # Only log from main process
                        metrics = {"iteration": iteration}
                        
                        # Log losses
                        for key, loss_value in loss_dict.items():
                            try:
                                if hasattr(loss_value, 'item'):
                                    metrics[f"train/{key}"] = loss_value.item()
                                else:
                                    metrics[f"train/{key}"] = float(loss_value)
                            except Exception as e:
                                raise e
                        
                        # Log additional metrics
                        if learning_rate is not None:
                            metrics["train/learning_rate"] = learning_rate
                        if grad_norm is not None:
                            metrics["train/grad_norm"] = grad_norm
                        if params_norm is not None:
                            metrics["train/params_norm"] = params_norm
                        if num_zeros_in_grad is not None:
                            metrics["train/num_zeros_in_grad"] = num_zeros_in_grad
                        if loss_scale is not None:
                            metrics["train/loss_scale"] = loss_scale
                        
                        wandb.log(metrics, step=iteration)
                        
                except Exception as e:
                    print_rank_0(f"ERROR: Failed to log to wandb: {e}")
                    raise e
                
                return result
            
            # Also enhance evaluate_and_print_results for evaluation metrics
            try:
                from megatron.training.training import evaluate_and_print_results as original_evaluate
                
                def enhanced_evaluate_and_print_results(prefix, forward_step_func, 
                                                       data_iterator, model, 
                                                       process_non_loss_data_func, config, 
                                                       verbose=False, write_to_tensorboard=True, 
                                                       iteration=0):
                    """Enhanced evaluate function with Wandb integration"""
                    
                    # Call original function first
                    result = original_evaluate(prefix, forward_step_func, data_iterator, model, 
                                             process_non_loss_data_func, config, verbose, 
                                             write_to_tensorboard, iteration)
                    
                    # Add Wandb logging for evaluation metrics
                    try:
                        rank = mpu.get_data_parallel_rank()
                        if rank == 0 and iteration > 0:
                            eval_metrics = {}
                            
                            # Log evaluation results if available
                            if hasattr(result, 'items'):
                                for key, value in result.items():
                                    try:
                                        if hasattr(value, 'item'):
                                            eval_metrics[f"eval/{prefix}_{key}"] = value.item()
                                        else:
                                            eval_metrics[f"eval/{prefix}_{key}"] = float(value)
                                    except:
                                        pass
                            
                            if eval_metrics:
                                wandb.log(eval_metrics, step=iteration)
                    
                    except Exception as e:
                        print_rank_0(f"ERROR: Failed to log eval metrics to wandb: {e}")
                    
                    return result
                
                # Now training_module is available
                training_module.evaluate_and_print_results = enhanced_evaluate_and_print_results
                
            except Exception as e:
                print_rank_0(f"WARNING: Failed to enhance evaluate function: {e}")
            
            # Replace the training_log function in the training module
            training_module.training_log = enhanced_training_log
            print_rank_0("WANDB: Enhanced training_log and evaluate functions installed")
            
        except ImportError:
            print_rank_0("WARNING: wandb not installed. Install with: pip install wandb")
            raise ImportError("wandb not installed")
        except Exception as e:
            print_rank_0(f"WARNING: Failed to setup wandb logging: {e}")
            raise e


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

    # Initialize wandb if enabled
    if args.enable_wandb_logging:
        from megatron.core import parallel_state as mpu
        if mpu.get_data_parallel_rank() == 0:  # Only log from main process
            try:
                import wandb
                import os
                
                # Set Wandb directories to lustre filesystem to avoid root quota issues
                wandb_base_dir = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/wandb_data"
                os.makedirs(wandb_base_dir, exist_ok=True)
                
                # Set environment variables for Wandb directories
                os.environ["WANDB_DIR"] = wandb_base_dir
                os.environ["WANDB_CONFIG_DIR"] = wandb_base_dir + "/config"
                os.environ["WANDB_CACHE_DIR"] = wandb_base_dir + "/cache"
                os.environ["WANDB_DATA_DIR"] = wandb_base_dir + "/data"
                
                # Create necessary directories
                for dir_path in [wandb_base_dir + "/config", wandb_base_dir + "/cache", wandb_base_dir + "/data"]:
                    os.makedirs(dir_path, exist_ok=True)
                
               
                run_name = args.wandb_run_name or f"qwen3-moe-{args.save.split('/')[-1]}"
                tags = args.wandb_run_tags.copy()
                if args.router_only_training:
                    tags.append("router-only")
                if args.use_rl_loss:
                    tags.append(f"rl-{args.rl_algorithm}")
                
                wandb.init(
                    project=args.wandb_project_name,
                    name=run_name,
                    tags=tags,
                    dir=wandb_base_dir,  # Explicit directory setting
                    config={
                        "model_size": getattr(args, 'hidden_size', 'unknown'),
                        "num_layers": getattr(args, 'num_layers', 'unknown'),
                        "learning_rate": args.lr,
                        "min_lr": args.min_lr,
                        "global_batch_size": args.global_batch_size,
                        "micro_batch_size": args.micro_batch_size,
                        "seq_length": args.seq_length,
                        "router_only": args.router_only_training,
                        "use_rl_loss": args.use_rl_loss,
                        "rl_algorithm": args.rl_algorithm if args.use_rl_loss else None,
                        "rl_loss_coeff": args.rl_loss_coeff if args.use_rl_loss else None,
                    }
                )
                print_rank_0(f"WANDB INITIALIZED: project={args.wandb_project_name}, name={run_name}")
                
                # Setup enhanced logging functions
                setup_wandb_logging()
                
            except ImportError:
                print_rank_0("WARNING: wandb not installed. Install with: pip install wandb")
                args.enable_wandb_logging = False
                raise ImportError("wandb not installed")
            except Exception as e:
                print_rank_0(f"WARNING: Failed to initialize wandb: {e}")
                args.enable_wandb_logging = False
                raise e

    return model

if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )