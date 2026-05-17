#!/bin/bash
#SBATCH --job-name=vllm_install_test
#SBATCH --partition=interactive
#SBATCH --account=nvr_israel_rlop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_install_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs/vllm_install_%j.err

# =============================================================================
# vLLM install probe.
#
# The container ships PyTorch 2.7. vLLM 0.7-0.8 want torch ≤ 2.6, hence the
# pip dependency conflict we saw on job 27823348. This probe tries several
# vLLM version ranges and reports which one resolves cleanly.
# =============================================================================
set -euo pipefail

CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/jonathanp/containers/pai-megatron-patch_25.04.sqsh"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "vLLM INSTALL PROBE"
echo "============================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "============================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:$HOME,/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing:/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing" \
     bash -c '
         set +e  # We want to keep going past failed installs.

         echo
         echo "=== Container baseline ==="
         python3 -c "import sys; print(\"python:\", sys.version.split()[0])"
         python3 -c "import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.version.cuda)"
         echo

         echo "=== Probe 1: vllm latest (no constraint) ==="
         python3 -m pip install --quiet --user vllm 2>&1 | tail -5
         python3 -c "import vllm; print(\"vllm version:\", vllm.__version__)" 2>&1
         echo

         # If the above failed, try explicit version ranges.
         python3 -c "import vllm" 2>/dev/null && echo "Latest vLLM works \u2014 stopping" && exit 0

         for spec in "vllm>=0.10,<0.12" "vllm>=0.9,<0.10" "vllm==0.9.2" "vllm==0.10.0" "vllm==0.10.1"; do
             echo "=== Probe: $spec ==="
             python3 -m pip uninstall -y vllm 2>&1 | tail -2
             python3 -m pip install --quiet --user "$spec" 2>&1 | tail -5
             python3 -c "import vllm; print(\"vllm version:\", vllm.__version__)" 2>&1
             echo
         done
     '
EXIT=$?
echo "============================================================"
echo "End:   $(date)   Exit: $EXIT"
echo "============================================================"
exit $EXIT
