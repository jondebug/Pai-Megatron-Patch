# Copyright (c) 2023 Alibaba PAI Team.
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
import argparse

def patch_if_not_exist(
        group_or_parser: Union[argparse._ArgumentGroup, argparse.ArgumentParser],
        keyname, type=None, default=None, choices=None, help=None
):
    has_keyname = False
    for action in vars(group_or_parser)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if keyname in action.option_strings:
                has_keyname = True

    if not has_keyname:
        return group_or_parser.add_argument(
            keyname,
            type=type,
            default=default,
            choices=choices,
            help=help,
        )
    return None


def get_patch_args(parser):
    group = parser.add_argument_group(title="patch")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--tokenizer-type" in action.option_strings:
                action.default = "NullTokenizer"

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--vocab-size" in action.option_strings:
                action.default = -1

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--optimizer" in action.option_strings:
                action.choices.append("hybridadam")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--position-embedding-type" in action.option_strings:
                action.choices.append("none")

    patch_if_not_exist(
        group,
        "--rotary-base",
        type=int,
        default=10000,
        help="Base to use for rotary positional embeddings, default 10000",
    )

    patch_if_not_exist(
        group,
        "--local-rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher",
    )

    patch_if_not_exist(
        group,
        "--spatial-merge-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--temporal-patch-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--patch-size",
        type=int,
        default=14,
    )

    patch_if_not_exist(
        group,
        "--rope-type",
        type=str,
        default='yarn',
        choices=['yarn', 'rope'],
        help="rope-type for MLA attn"
    )

    group.add_argument("--n-head-kv", type=int, default=None, help="n-head-kv")

    group.add_argument(
        "--transformer-type", type=str, default="megatron", help="transformer-type"
    )

    group.add_argument(
        "--max-padding-length", type=int, default=None, help="max-padding-length"
    )

    group.add_argument("--dataset", type=str, default=None, help="dataset")

    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of finetunning epochs. Zero results in " "evaluation only.",
    )

    group.add_argument(
        "--intermediate-size", type=int, default=None, help="--intermediate-size"
    )

    group.add_argument(
        "--extra-vocab-size", type=int, default=0, help="--extra-vocab-size"
    )

    group.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the last batch (maybe incomplete) in" "the data loader",
    )

    group.add_argument("--data-dir", default=None, help="data-dir")

    group.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Whitespace separated paths or corpora names " "for training.",
    )

    group.add_argument(
        "--valid-data", nargs="+", default=None, help="path(s) to the validation data."
    )

    group.add_argument("--patch-tokenizer-type", type=str, help="patch-tokenizer-type")

    group.add_argument(
        "--use-alibi-mask",
        action="store_true",
        help="use alibi mask for baichuan model",
    )

    group.add_argument("--use-normhead", action="store_true", help="use-normhead")

    group.add_argument("--glu-activation", type=str, help="GLU activations to use.")

    group.add_argument(
        "--attention-head-type",
        type=str,
        default=None,
        choices=["multihead", "multiquery"],
        help="Type of attention heads. `multihead` is the standard multi-head attention."
        "`multiquery` shares the values and keys across attention heads",
    )

    group.add_argument(
        "--transformer-timers",
        action="store_true",
        help="If set, activate the timers within the transformer layers."
        "Only for debugging, as this slows down the model.",
    )

    group.add_argument("--text-generate-input-file", type=str, default="")

    group.add_argument("--text-generate-output-file", type=str, default="")

    group.add_argument("--text-generate-gt-file", type=str, default="")

    group.add_argument(
        "--time",
        action="store_true",
        help="measure end to end text generation average time",
    )

    group.add_argument("--eval-dev", action="store_true")

    group.add_argument(
        "--input-len",
        type=int,
        default=1,
        help="input lenth for measure end to end text generation average time",
    )

    group.add_argument(
        "--generation-length", type=int, default=None, help="generation-seq-len"
    )

    group.add_argument("--top-p", type=float, default=0.0, help="Top p sampling.")

    group.add_argument("--top-k", type=int, default=0, help="Top k sampling.")

    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )

    group.add_argument(
        "--repetition_penalty", type=float, default=1.1, help="Repetition_penalty."
    )

    group.add_argument(
        "--embed-layernorm", action="store_true", help="use layernorm for embedding"
    )

    group.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition_penalty."
    )

    group.add_argument(
        "--source-seq-len", type=int, default=None, help="source-seq-len"
    )

    group.add_argument(
        "--target-seq-len", type=int, default=None, help="target-seq-len"
    )

    group.add_argument(
        "--position-encoding-2d", action="store_true", help="position-encoding-2d"
    )

    group.add_argument(
        "--z-loss-weight",
        type=float,
        default=0.0,
        help="the max-z weight for baichuan2",
    )

    group.add_argument(
        "--use-llama2-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument(
        "--use-mistral-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument("--mm-use-im-start-end", action="store_true")

    group.add_argument("--mm-use-im-patch-token", action="store_true")

    group.add_argument("--tune-mm-mlp-adapter", action="store_true")

    group.add_argument("--freeze-clip-vision-tower", action="store_true")

    group.add_argument("--freeze-llm", action="store_true")

    group.add_argument("--image-folder", type=str, default="")

    group.add_argument("--mm-vision-select-layer", type=int, default=None)

    group.add_argument("--vision-tower", type=str, default="")

    group.add_argument("--image-aspect-ratio", type=str, default="square")

    group.add_argument("--version", type=str, default="plain")

    group.add_argument("--mm-projector-type", type=str, default=None)

    group.add_argument("--image-size", type=int, default=None, help="image-size")


    group.add_argument("--sliding-window", type=int, default=None)

    group.add_argument("--rotary-scale-factor", type=int, default=1)

    group.add_argument("--cvcuda-image-processing", action="store_true")

    group.add_argument(
        "--expert-interval",
        type=int,
        default=2,
        help='Use experts in every "expert-interval" layers',
    )

    group.add_argument("--moe", action="store_true")

    group.add_argument("--moe-topk", type=int, default=1, help="moe-topk")

    group.add_argument(
        "--moe-expert-parallel-size",
        type=int,
        default=None,
        help="Degree of the MoE expert parallelism. By default, "
        "the size of this value will be automatically determined.",
    )

    group.add_argument(
        "--moe-train-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at training time",
    )

    group.add_argument(
        "--moe-eval-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at eval time.",
    )

    group.add_argument(
        "--moe-min-capacity",
        type=int,
        default=4,
        help="The minimum capacity per MoE expert regardless of the capacity_factor.",
    )

    group.add_argument(
        "--moe-loss-coeff",
        type=float,
        default=0.01,
        help="Scaling coefficient for adding MoE loss to model loss",
    )

    group.add_argument(
        "--use-tutel", action="store_true", help="Use Tutel optimization for MoE"
    )

    group.add_argument(
        "--router-type",
        type=str,
        default="topk",
        choices=["topk", "expert_choice"],
        help="Options for router type, support top1 & top2 and expert_choice",
    )

    group.add_argument(
        "--moe-input-feature-slicing",
        action="store_true",
        help="Enable moe all2all performance optimization.",
    )

    group.add_argument(
        "--disable-bias-linear-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_linear_fc",
    )

    group.add_argument(
        "--disable-bias-attn-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_attn_fc",
    )

    group.add_argument(
        "--disable-parallel-output",
        action="store_false",
        help="Disable parallel-output",
        dest="enable_parallel_output",
    )

    group.add_argument(
        "--task-list",
        type=str,
        default="all",
        help='Either "all" or comma separated list of tasks.',
    )

    group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )

    group.add_argument(
        "--adaptive-seq-len",
        default=False,
        action="store_true",
        help="Should the sequence length be adapted to the batch during evaluation,"
        " if in fp16 the results will be slightly different due to numerical"
        " errors but greatly speed up evaluation.",
    )

    group.add_argument(
        "--eval-fp32",
        default=False,
        action="store_true",
        help="Should the evaluation run in fp32",
    )

    group.add_argument("--num-fewshot", type=int, default=None, help="num fewshot")

    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    patch_if_not_exist(
        group,
        "--moe-ffn-hidden-size", type=int, default=None
    )

    group.add_argument("--shared-moe-ffn-hidden-size", type=int, default=None)

    group.add_argument(
        "--enable-shared-expert", action="store_true", help="enable-shared-expert"
    )

    patch_if_not_exist(
        group,
        "--q-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--kv-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--v-head-dim", type=int, default=None
    )

    group.add_argument("--qk-nope-head-dim", type=int, default=None)
    group.add_argument("--qk-rope-head-dim", type=int, default=None)
    group.add_argument("--num-shared-experts", type=int, default=None)

    patch_if_not_exist(
        group,
        "--moe-layer-freq", type=int, default=1
    )

    patch_if_not_exist(
        group,
        "--rotary-scaling-factor", type=int, default=1
    )

    group.add_argument(
        "--optimizer-offload-policy",
        default="static",
        type=str,
        help="Optimizer Offload Policy used by OffloadDistributedOptimizer, "
        "valid if base optimizer is HybridAdam.",
    )

    patch_if_not_exist(
        group,
        "--optimizer-offload-fraction", type=float, default=0.5
    )

    group.add_argument(
        "--train-mode", default="pretrain", type=str, help="pretrain or finetune"
    )

    group.add_argument(
        "--optimizer-offload-auto-threshold",
        type=int,
        default=2048 * 1024 * 1024,
        help="Optimizer Offload Threshold currently used by auto policy, "
        "tune larger if OOM occurs",
    )

    group.add_argument(
        "--optimizer-offload-chunk-size",
        type=int,
        default=32 * 1024 * 1024,
        help="Chunk size of Chunk Manager in Optimizer Offload,"
        "keep zero to search for a optimal size",
    )

    group.add_argument(
        "--cpu-offloading",
        default=False,
        action="store_true",
        help="Use activation checkpointing.",
    )

    group.add_argument(
        "--cpu-offloading-num-layers",
        type=int,
        default=0,
        help="The num of layers to be moved to CPU",
    )

    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=False)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument('--router-only-training', action='store_true', default=False,
                      help='Freeze all parameters except MoE router weights')
    group.add_argument('--use_rl_loss', action='store_true', default=False,
                      help='Enable reinforcement learning loss for router training')
    group.add_argument('--rl-algorithm', type=str, default='reinforce', choices=['reinforce', 'ppo'],
                      help='RL algorithm to use: reinforce or ppo (default: reinforce)')
    group.add_argument('--rl-loss-coeff', type=float, default=0.1,
                      help='Coefficient for RL loss (default: 0.1)')
    group.add_argument('--use-per-layer-loss', action='store_true', default=False,
                      help='Use per-layer discounted RL loss instead of independent layer losses')
    group.add_argument('--rl-per-token-rewards', action='store_true', default=False,
                      help='Use per-token rewards instead of scalar rewards for RL training. effectively this means the state is a single token instead of a sequence of tokens not a batch')
    group.add_argument('--rl-ppo-entropy-coeff', type=float, default=0.01,
                      help='Entropy coefficient for PPO loss (default: 0.01)')
    group.add_argument('--rl-ppo-baseline-type', type=str, default='mean', choices=['mean', 'critic'],
                      help='Type of baseline for PPO advantage calculation: mean (simple average) or critic (learned value function)')
    group.add_argument('--rl-critic-hidden-dims', type=int, nargs='+', default=[256],
                      help='Hidden dimensions for critic network layers (default: 256). Examples: "256" for 1 layer, "256 64 32" for 3 layers')
    group.add_argument('--rl-reward-type', type=str, default='expert0',
                      choices=['expert0', 'entropy', 'topn_load', 'critical_path',
                               'per_token_topn_binary', 'per_token_load_weighted'],
                      help='Reward function type: expert0 (focus on expert 0), entropy (load balance entropy), '
                           'topn_load (avg/topN load ratio), critical_path (directly targets max expert load), '
                           'per_token_topn_binary (per-token: -1 if hot expert, +1 otherwise), '
                           'per_token_load_weighted (per-token: continuous reward based on chosen expert load)')
    group.add_argument('--rl-reward-topn', type=int, default=12,
                      help='Number of top experts to consider for topn_load reward (default: 12)')
    group.add_argument('--rl-discount-factor', type=float, default=0.9,
                      help='Discount factor (gamma) for RL returns calculation (default: 0.9). Lower values weight immediate rewards more.')
    group.add_argument('--rl-normalize-rewards', action='store_true', default=False,
                      help='Enable running-mean/std reward normalization. Expands compressed reward ranges to [-1, +1] for stronger RL gradients.')
    group.add_argument('--rl-ppo-reeval', action='store_true', default=False,
                      help='Enable proper PPO: re-evaluate old states under current router weights '
                           'to compute correct importance ratios. Costs one extra linear layer per '
                           'MoE layer per step. Without this, falls back to REINFORCE (ratio=1.0).')
    group.add_argument('--rl-ppo-epochs', type=int, default=1,
                      help='Number of PPO epochs per training step (default: 1 = no extra epochs). '
                           'K>1 runs K-1 additional RL-only gradient steps on the stored trajectory '
                           'after each main training step. Requires --rl-ppo-reeval.')
    group.add_argument('--rl-lm-reward-coeff', type=float, default=0.0,
                      help='Coefficient (beta) for per-token LM cross-entropy reward. '
                           '0 = disabled. When > 0, adds -cross_entropy(token) as an additional '
                           'reward component to all layers, centered per-batch.')
    group.add_argument('--rl-ppo-clip-ratio', type=float, default=0.2,
                      help='PPO clipping ratio for policy updates (default: 0.2). '
                           'Lower values are more conservative, higher allow larger updates.')
    group.add_argument('--rl-use-ema-loads', action='store_true', default=False,
                      help='Use exponential moving average of expert loads for reward computation. '
                           'More stable signal across batches, less sensitive to per-batch noise.')
    group.add_argument('--rl-critic-layer-aware', action='store_true', default=False,
                      help='Give the critic network the layer index as an input feature. '
                           'Enables layer-conditional value predictions.')
    group.add_argument('--rl-replay-buffer-size', type=int, default=0,
                      help='Size of the replay buffer for PPO multi-epoch training. '
                           '0 = disabled (use only current trajectory). '
                           'When > 0, stores past trajectories and replays them during extra PPO epochs.')
    group.add_argument('--rl-ppo-extra-lr', type=float, default=1e-4,
                      help='Learning rate for the separate optimizer used in extra PPO epochs. '
                           'Automatically scaled by 1/(K-1) where K is rl_ppo_epochs.')
    group.add_argument('--rl-ppo-legacy-mode', action='store_true', default=False,
                      help='Enable legacy PPO behavior for A/B testing. '
                           'Legacy mode keeps inline extra-epoch updates in loss construction, '
                           'uses SGD for extra epochs, and uses REINFORCE-style main PPO update '
                           '(ratio fixed to 1). Default: False (new implementation).')
    group.add_argument('--kl-loss-coeff', type=float, default=0.0,
                      help='KL divergence loss coefficient. 0 = disabled. '
                           'Penalizes deviation of LM output distribution from pretrained reference. '
                           'Requires one extra no-grad forward pass per step (~1.5x wall time).')
    group.add_argument('--moe-router-topology-aware', action='store_true', default=False,
                      help='Enable topology-aware routing: adds a non-trainable bias to local '
                           'expert logits before top-k selection, reducing cross-GPU communication. '
                           'Mutually exclusive with --use_rl_loss.')
    group.add_argument('--moe-router-topology-lambda', type=float, default=0.01,
                      help='Locality bias strength for topology-aware routing. '
                           'Higher values favor local experts more aggressively (default: 0.01).')
    group.add_argument('--moe-router-critical-path-bias', action='store_true', default=False,
                      help='Enable critical-path dynamic bias: after each batch, reduce bias '
                           'for the top-N most loaded experts per layer. Only targets the peak, '
                           'leaving other experts undisturbed. Can combine with aux loss and topology bias.')
    group.add_argument('--moe-router-critical-path-topn', type=int, default=1,
                      help='Number of most-loaded experts to penalize per layer (default: 1). '
                           'N=1 targets only the single bottleneck expert.')
    group.add_argument('--moe-router-critical-path-alpha', type=float, default=0.001,
                      help='Bias update rate for critical-path bias (default: 0.001).')
    group.add_argument('--hellaswag-eval-interval', type=int, default=0,
                      help='Run HellaSwag accuracy check every N steps (0=disabled). Detects benchmark collapse during training.')
    group.add_argument('--hellaswag-eval-limit', type=int, default=100,
                      help='Number of HellaSwag samples per evaluation (default: 100). More samples = more accurate but slower.')
    group.add_argument('--eval-kl-tracking', action='store_true', default=False,
                      help='Track KL divergence from pretrained reference during eval.')
    group.add_argument('--log-expert-heatmap', action='store_true', default=False,
                      help='Log per-expert token load heatmaps to wandb every 50 steps.')
    group.add_argument('--enable-wandb-logging', action='store_true', default=False,
                      help='Enable wandb logging for training metrics')
    group.add_argument('--wandb-project-name', type=str, default='qwen3-moe-training',
                      help='Wandb project name (default: qwen3-moe-training)')
    group.add_argument('--wandb-run-name', type=str, default=None,
                      help='Wandb run name (default: auto-generated)')
    group.add_argument('--wandb-run-tags', type=str, nargs='*', default=[],
                      help='Wandb tags for the run')
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )
    group.add_argument(
        "--use-tiling", action="store_true", default=False, help="Use input image tiling"
    )
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles")
    group.add_argument(
        "--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile"
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to regularly sample from the video as input to the model.",
    )
    group.add_argument(
        "--online-evaluation-config", type=str, help="Config file for online evaluation."
    )

    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        choices=["mistral", "llama3", "chatml"],
        required=False,
        help="Prompt format to use with the tokenizer.",
    )

    group.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<image>"],
        help="Special tokens used in the multimodal model",
    )

    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",  # Default: Image tag not used.
        help="Surround image tokens with tags.",
    )

    return parser
