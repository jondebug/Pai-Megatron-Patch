#!/bin/bash
#SBATCH --job-name=ord_rayEP_prefill_pre_resub
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ep64_pre_prefill_resub_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
DIR=$BASE/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test
PRE=$BASE/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B
export MODEL=$PRE
export MODELNAME=pretrained_235b
export PLEN=8192 SWEEP_BS=1,2 PROFBS=1 NTRIALS=20
exec bash $DIR/run_ord_rayEP_prefill.sh
