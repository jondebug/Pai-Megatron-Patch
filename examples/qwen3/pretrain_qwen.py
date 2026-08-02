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
            
            # --- Deterministic eval: always evaluate on the same data ---
            # Capture the validation dataloader so we can rebuild a fresh iterator
            # before each eval, ensuring the same samples are used every time.
            _eval_state = {"iteration": 0, "valid_dataloader": None}
            
            try:
                from megatron.training.training import (
                    evaluate as original_evaluate_fn,
                    evaluate_and_print_results as original_eval_and_print,
                    build_train_valid_test_data_loaders as original_build_loaders,
                )
                from megatron.core.rerun_state_machine import RerunDataIterator
                
                def capturing_build_loaders(build_train_valid_test_datasets_provider):
                    """Wrap build_train_valid_test_data_loaders to capture valid dataloader."""
                    train_dl, valid_dl, test_dl = original_build_loaders(
                        build_train_valid_test_datasets_provider
                    )
                    _eval_state["valid_dataloader"] = valid_dl
                    if valid_dl is not None:
                        print_rank_0(f"[EVAL] Captured validation dataloader "
                                     f"(dataset size={len(valid_dl.dataset)})")
                    return train_dl, valid_dl, test_dl
                
                training_module.build_train_valid_test_data_loaders = capturing_build_loaders
                
                def _make_fresh_valid_iterator():
                    """Build a fresh validation data iterator starting from sample 0."""
                    from megatron.legacy.data.data_samplers import build_pretraining_data_loader
                    valid_dl = _eval_state.get("valid_dataloader")
                    if valid_dl is None:
                        return None
                    # Rebuild the dataloader with consumed_samples=0
                    fresh_dl = build_pretraining_data_loader(valid_dl.dataset, consumed_samples=0)
                    return RerunDataIterator(iter(fresh_dl))
                
                def enhanced_evaluate(forward_step_func, data_iterator, model,
                                      process_non_loss_data_func, config, 
                                      verbose=False, non_loss_data_func=None):
                    """Wrap evaluate() to use fresh data iterator and log metrics to WandB."""
                    args = get_args()
                    
                    # Replace data_iterator with a fresh one starting from sample 0
                    # so the same eval data is used every time
                    fresh_iter = _make_fresh_valid_iterator()
                    if fresh_iter is not None:
                        data_iterator = fresh_iter
                    
                    # Save consumed_valid_samples so we always start from 0
                    saved_consumed = args.consumed_valid_samples
                    args.consumed_valid_samples = 0
                    
                    import time as _time
                    _eval_start = _time.monotonic()
                    total_loss_dict, collected_non_loss_data, timelimit = original_evaluate_fn(
                        forward_step_func, data_iterator, model,
                        process_non_loss_data_func, config, verbose, non_loss_data_func,
                    )
                    _eval_wall_time = _time.monotonic() - _eval_start
                    
                    # Restore consumed_valid_samples (don't let it accumulate)
                    args.consumed_valid_samples = saved_consumed
                    
                    if timelimit or total_loss_dict is None:
                        return total_loss_dict, collected_non_loss_data, timelimit
                    
                    # Log all eval metrics to WandB
                    try:
                        rank = mpu.get_data_parallel_rank()
                        if rank == 0:
                            eval_metrics = {}
                            for key, value in total_loss_dict.items():
                                try:
                                    v = value.item() if hasattr(value, 'item') else float(value)
                                    eval_metrics[f"eval/{key}"] = v
                                except Exception:
                                    pass
                            
                            # Critical eval metrics: curated subset with clean names
                            # for easy WandB dashboard viewing
                            _CRITICAL_EVAL_MAP = {
                                "lm loss": "critical_eval/lm_loss",
                                "num_tokens_on_critical_path": "critical_eval/critical_path",
                                # "aux_loss": "critical_eval/aux_loss",
                                # "load_balancing_entropy": "critical_eval/lb_entropy",
                                # "max_tokens_per_expert": "critical_eval/max_tokens_per_expert",
                                "locality_ratio": "critical_eval/locality_ratio",
                                # "locality_ratio_max": "critical_eval/locality_ratio_max",
                                # "locality_ratio_min": "critical_eval/locality_ratio_min",
                                "rl_loss": "critical_eval/rl_loss",
                                "rl_mean_reward": "critical_eval/rl_mean_reward",
                                "rl_value_loss": "critical_eval/rl_value_loss",
                                "rl_policy_loss": "critical_eval/rl_policy_loss",
                                "rl_entropy_bonus": "critical_eval/rl_entropy_bonus",
                                "rl_grad_norm_on_logits": "critical_eval/rl_grad_norm",
                            }
                            for raw_key, clean_key in _CRITICAL_EVAL_MAP.items():
                                if f"eval/{raw_key}" in eval_metrics:
                                    eval_metrics[clean_key] = eval_metrics[f"eval/{raw_key}"]
                            
                            eval_metrics["critical_eval/wall_time_sec"] = _eval_wall_time
                            
                            if eval_metrics:
                                iteration = _eval_state.get("iteration", 0)
                                wandb.log(eval_metrics, step=iteration)
                                print_rank_0(f"[EVAL] Logged {len(eval_metrics)} eval metrics "
                                             f"to WandB at step {iteration}")
                    except Exception as e:
                        print_rank_0(f"ERROR: Failed to log eval metrics to wandb: {e}")
                    
                    return total_loss_dict, collected_non_loss_data, timelimit
                
                def enhanced_evaluate_and_print_results(prefix, forward_step_func,
                                                       data_iterator, model, iteration,
                                                       process_non_loss_data_func, config,
                                                       verbose=False, write_to_tensorboard=True,
                                                       non_loss_data_func=None):
                    """Capture iteration for the evaluate wrapper, then call original."""
                    _eval_state["iteration"] = iteration
                    return original_eval_and_print(
                        prefix, forward_step_func, data_iterator, model, iteration,
                        process_non_loss_data_func, config, verbose, write_to_tensorboard,
                        non_loss_data_func,
                    )
                
                training_module.evaluate = enhanced_evaluate
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


def configure_router_only_training(model):
    """Configure model for router-only training.
    
    """
    args = get_args()
    if args.router_only_training:
        print_rank_0("\n" + "="*80)
        print_rank_0("CONFIGURING ROUTER-ONLY TRAINING (in model_provider)")
        print_rank_0("Setting requires_grad=False for non-router parameters")
        print_rank_0("This ensures only router params are in gradient buffers and optimizer")
        print_rank_0("="*80 + "\n")
        
        router_param_count = 0
        non_router_param_count = 0
        
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in ['router', 'gate']) and 'weight' in name:
                param.requires_grad = True
                router_param_count += 1
            else:
                param.requires_grad = False
                non_router_param_count += 1
        
        print_rank_0(f"\n[ROUTER-ONLY] Configuration complete:")
        print_rank_0(f"  - Router parameters (requires_grad=True): {router_param_count}")
        print_rank_0(f"  - Non-router parameters (requires_grad=False): {non_router_param_count}")
        print_rank_0("="*80 + "\n")
    else:
        print_rank_0("Router-only training not enabled")
    
    return model


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
        print_rank_0("ROUTER-ONLY TRAINING: Configuring model for router-only parameter updates")
        configure_router_only_training(model)

    # Topology-aware routing and RL loss are mutually exclusive
    topology_aware = getattr(args, 'moe_router_topology_aware', False)
    if topology_aware and args.use_rl_loss:
        raise ValueError("--moe-router-topology-aware and --use_rl_loss are mutually exclusive. "
                         "Topology bias is rank-specific and conflicts with RL gradient averaging.")

    # Initialize topology-aware routing if enabled
    if topology_aware:
        topo_lambda = getattr(args, 'moe_router_topology_lambda', 0.01)
        print_rank_0(f"TOPOLOGY-AWARE ROUTING: lambda={topo_lambda}")
        for module in model.modules():
            if 'router' in module.__class__.__name__.lower():
                module._topology_aware = True
                module._topology_lambda = topo_lambda

    # Initialize critical-path dynamic bias if enabled
    cp_bias = getattr(args, 'moe_router_critical_path_bias', False)
    if cp_bias:
        cp_topn = getattr(args, 'moe_router_critical_path_topn', 1)
        cp_alpha = getattr(args, 'moe_router_critical_path_alpha', 0.001)
        print_rank_0(f"CRITICAL-PATH BIAS: topn={cp_topn}, alpha={cp_alpha}")
        for module in model.modules():
            if 'router' in module.__class__.__name__.lower():
                module._critical_path_bias_enabled = True
                module._critical_path_topn = cp_topn
                module._critical_path_alpha = cp_alpha

    # Initialize RL loss if enabled
    print_rank_0(f"[RL DEBUG] Initializing training with use_rl_loss={args.use_rl_loss}")
    if args.use_rl_loss:
        print_rank_0(f"RL LOSS ENABLED: Using {args.rl_algorithm.upper()} algorithm with coefficient {args.rl_loss_coeff}")
        if not args.router_only_training:
            print_rank_0("WARNING: RL loss is typically used with router-only training")

        # --- P1: fail-fast H1-mismatch guard + prominent config banner ---------------------------
        # hard_gumbel_pl samples a stochastic action on a FIXED detached candidate pool and scores
        # it with an ORDERED Plackett-Luce log-prob (rl_ordered_logprob). Only the REINFORCE
        # per-token loss differentiates that PL log-prob; the PPO path re-scores with a summed
        # independent-softmax log-prob and NEVER touches the PL log-prob, so pairing PPO with
        # hard_gumbel_pl silently trains the wrong objective (the H1 mismatch). Assert it away.
        _rl_sampling = getattr(args, 'rl_sampling', 'argmax')
        _rl_algorithm = getattr(args, 'rl_algorithm', 'reinforce')
        if _rl_sampling == 'hard_gumbel_pl':
            assert _rl_algorithm == 'reinforce', (
                "hard_gumbel_pl requires --rl-algorithm reinforce (PPO bypasses the "
                "Plackett-Luce log-prob: it re-scores actions with a summed independent-softmax "
                "log-prob and never differentiates rl_ordered_logprob -> H1 mismatch)")
        # scoring_distribution := which log-prob the policy loss differentiates.
        if _rl_sampling == 'hard_gumbel_pl' and _rl_algorithm == 'reinforce':
            _scoring = 'ordered_plackett_luce(rl_ordered_logprob, per-token REINFORCE)'
        elif _rl_algorithm == 'reinforce':
            _scoring = 'summed_independent_softmax(log_softmax chosen, per-token REINFORCE)'
        else:
            _scoring = 'summed_independent_softmax(log_softmax chosen, PPO ratio)'
        _pool = int(getattr(args, 'rl_candidate_pool', 0)) or 'ALL_EXPERTS'
        _nepochs = int(getattr(args, 'rl_ppo_epochs', 1)) if _rl_algorithm == 'ppo' else 1
        print_rank_0(
            "[RL CONFIG BANNER] "
            f"sampling_distribution={_rl_sampling} | "
            f"scoring_distribution={_scoring} | "
            f"candidate_pool_size={_pool} | "
            f"algorithm={_rl_algorithm} | "
            f"num_policy_epochs={_nepochs} | "
            f"reward_type={getattr(args, 'rl_reward_type', 'expert0')} | "
            f"loo_beta={getattr(args, 'rl_loo_beta', 0.3)} | "
            f"global_loads={getattr(args, 'rl_global_loads', False)} | "
            f"perlayer_norm={getattr(args, 'rl_perlayer_norm', False)} | "
            f"rl_loss_coeff={getattr(args, 'rl_loss_coeff', 0.0)}")
        # -----------------------------------------------------------------------------------------

        from megatron_patch.model.qwen3_moe.moe.rl_trajectory import get_trajectory_tracker
        tracker = get_trajectory_tracker()
        # --- H1/H2 port: set sampling/reward config on the tracker BEFORE the first forward so the
        # router reads it during routing (helper.py re-affirms these at loss time). Defaults OFF. ---
        tracker.rl_sampling = getattr(args, 'rl_sampling', 'argmax')
        tracker.rl_candidate_pool = getattr(args, 'rl_candidate_pool', 0)
        tracker.rl_stochastic_temperature = getattr(args, 'rl_stochastic_temperature', 1.0)
        tracker.global_loads = getattr(args, 'rl_global_loads', False)
        tracker.loo_beta = getattr(args, 'rl_loo_beta', 0.3)
        tracker.reward_type = getattr(args, 'rl_reward_type', 'expert0')
        for module in model.modules():
            if 'router' in module.__class__.__name__.lower():
                module.config.moe_router_use_trajectory_tracking = True
                module._use_trajectory_tracking = True
                module._trajectory_tracker = tracker
                module._stochastic_routing = getattr(args, 'rl_stochastic_routing', False)
                module._stochastic_temperature = getattr(args, 'rl_stochastic_temperature', 1.0)
                # Store router module reference for multi-epoch PPO re-evaluation
                if hasattr(module, 'layer_number') and hasattr(module, 'gating'):
                    tracker._router_modules[module.layer_number] = module


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
                
                # Primary hyperparameters
                primary_config = {
                    "hidden_size": getattr(args, 'hidden_size', 'unknown'),
                    "num_layers": getattr(args, 'num_layers', 'unknown'),
                    "learning_rate": args.lr,
                    "min_lr": args.min_lr,
                    "moe_aux_loss_coeff": args.moe_aux_loss_coeff,
                    "global_batch_size": args.global_batch_size,
                    "micro_batch_size": args.micro_batch_size,
                    "seq_length": args.seq_length,
                    "router_only": args.router_only_training,
                    "use_rl_loss": args.use_rl_loss,
                }
                
                # Always log RL parameters so they're filterable/groupable in wandb
                # (even when RL is off, knowing the RL config is useful for sweep analysis)
                primary_config["rl_algorithm"] = getattr(args, 'rl_algorithm', None)
                primary_config["rl_loss_coeff"] = getattr(args, 'rl_loss_coeff', 0)
                primary_config["rl_per_token_rewards"] = getattr(args, 'rl_per_token_rewards', False)
                primary_config["rl_ppo_entropy_coeff"] = getattr(args, 'rl_ppo_entropy_coeff', 0.01)
                primary_config["rl_ppo_baseline_type"] = getattr(args, 'rl_ppo_baseline_type', 'mean')
                primary_config["rl_critic_hidden_dims"] = getattr(args, 'rl_critic_hidden_dims', [256])
                primary_config["rl_reward_type"] = getattr(args, 'rl_reward_type', None)
                primary_config["rl_reward_topn"] = getattr(args, 'rl_reward_topn', None)
                primary_config["rl_discount_factor"] = getattr(args, 'rl_discount_factor', None)
                primary_config["rl_normalize_rewards"] = getattr(args, 'rl_normalize_rewards', False)
                
                # Secondary/advanced hyperparameters - only include non-None values
                secondary_config = {}
                
                # MoE specific parameters
                moe_params = {
                    "moe_router_load_balancing_type": getattr(args, 'moe_router_load_balancing_type', None),
                    "moe_router_topk": getattr(args, 'moe_router_topk', None),
                    "moe_router_pre_softmax": getattr(args, 'moe_router_pre_softmax', None),
                    "moe_router_score_function": getattr(args, 'moe_router_score_function', None),
                    "moe_token_drop_policy": getattr(args, 'moe_token_drop_policy', None),
                    "moe_pad_expert_input_to_capacity": getattr(args, 'moe_pad_expert_input_to_capacity', None),
                    "moe_router_enable_expert_bias": getattr(args, 'moe_router_enable_expert_bias', None),
                    "moe_grouped_gemm": getattr(args, 'moe_grouped_gemm', None),
                }
                
                # Training parameters
                training_params = {
                    "train_iters": getattr(args, 'train_iters', None),
                    "weight_decay": getattr(args, 'weight_decay', None),
                    "adam_beta1": getattr(args, 'adam_beta1', None),
                    "adam_beta2": getattr(args, 'adam_beta2', None),
                    "adam_eps": getattr(args, 'adam_eps', None),
                    "lr_decay_style": getattr(args, 'lr_decay_style', None),
                    "lr_warmup_iters": getattr(args, 'lr_warmup_iters', None),
                    "lr_decay_iters": getattr(args, 'lr_decay_iters', None),
                    "optimizer": getattr(args, 'optimizer', None),
                    "use_distributed_optimizer": getattr(args, 'use_distributed_optimizer', None),
                }
                
                # Model architecture
                model_params = {
                    "num_attention_heads": getattr(args, 'num_attention_heads', None),
                    "num_query_groups": getattr(args, 'num_query_groups', None),
                    "ffn_hidden_size": getattr(args, 'ffn_hidden_size', None),
                }
                
                # Parallelism
                parallel_params = {
                    "tensor_model_parallel_size": getattr(args, 'tensor_model_parallel_size', None),
                    "pipeline_model_parallel_size": getattr(args, 'pipeline_model_parallel_size', None),
                    "expert_model_parallel_size": getattr(args, 'expert_model_parallel_size', None),
                    "sequence_parallel": getattr(args, 'sequence_parallel', None),
                    "context_parallel_size": getattr(args, 'context_parallel_size', None),
                    "overlap_p2p_comm": getattr(args, 'overlap_p2p_comm', None),
                }
                
                # Memory and performance
                perf_params = {
                    "external_cuda_graph": getattr(args, 'external_cuda_graph', None),
                    "cuda_graph_scope": getattr(args, 'cuda_graph_scope', None),
                    "recompute_granularity": getattr(args, 'recompute_granularity', None),
                    "recompute_modules": getattr(args, 'recompute_modules', None),
                }
                
                # Other training configs
                other_params = {
                    "seed": getattr(args, 'seed', None),
                    "bf16": getattr(args, 'bf16', None),
                    "fp16": getattr(args, 'fp16', None),
                    "gradient_clipping": getattr(args, 'clip_grad', None),
                    "log_interval": getattr(args, 'log_interval', None),
                    "eval_interval": getattr(args, 'eval_interval', None),
                    "eval_iters": getattr(args, 'eval_iters', None),
                    "timing_log_level": getattr(args, 'timing_log_level', None),
                    "deterministic_mode": getattr(args, 'deterministic_mode', None),
                    "calculate_per_token_loss": getattr(args, 'calculate_per_token_loss', None),
                }
                
                # Combine all parameters, filtering out None values
                for params in [moe_params, training_params, model_params, parallel_params, perf_params, other_params]:
                    secondary_config.update({k: v for k, v in params.items() if v is not None})
                
                # Combine configs - wandb will organize them nicely
                full_config = {
                    **primary_config,
                    "advanced": secondary_config  # Group secondary params under "advanced"
                }
                
                wandb.init(
                    project=args.wandb_project_name,
                    name=run_name,
                    tags=tags,
                    dir=wandb_base_dir,  # Explicit directory setting
                    config=full_config
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