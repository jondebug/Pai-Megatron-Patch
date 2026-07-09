#!/bin/bash
# hsg_build_deci_handoff.sh — HSG adaptation of networking-insights
# dev-tools/dev-ops/vllm/build-vllm-deci-handoff.sh (their validated recipe).
# Rebuilds the FINAL deci-handoff container (Shahar vLLM pin + Sam's 3 HybridEP
# cherry-picks incl. the CG-replay fixes c95c94e/c249dbf that the registry tag
# 0.18_v2.29.7-1_arm64_hybridep predates) from public sources.
#
#SBATCH --account=nvr_israel_rlop
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --job-name=vllm_deci_build
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/deci_build_%j.out

set -euo pipefail

BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
OUTPUT_CONTAINER=$BASE/containers/vllm_deci_handoff.sqsh

VLLM_REMOTE=https://github.com/shaharmor98/smor-vllm.git
VLLM_BRANCH=smor/tomer-dp-ben-synthetic-acceptance
VLLM_PIN_SHA=010583efa6d2de7a16076edf0cd997cf7907fc22
DEEPEP_TONGLIU_COMMIT=9e94c34

echo "=== deci-handoff rebuild on HSG job=$SLURM_JOB_ID node=$(hostname) out=$OUTPUT_CONTAINER ==="

srun --nodes=1 --ntasks=1 \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="$BASE:$BASE" \
  --container-writable \
  --container-save="${OUTPUT_CONTAINER}" \
  bash -c "
    set -ex
    export DEBIAN_FRONTEND=noninteractive

    echo '===== LAYER 0: apt + uv + venv ====='
    apt-get update -y
    # tzdata postinst fails under enroot: /etc/localtime is bind-mounted from
    # the host (mv → EBUSY). Install tzdata alone, no-op its postinst, finish
    # configuration, then proceed with the real package set.
    apt-get install -y --no-install-recommends tzdata || true
    printf '#!/bin/sh\nexit 0\n' > /var/lib/dpkg/info/tzdata.postinst
    dpkg --configure -a
    apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      libibverbs-dev openssh-client \
      python3 python3-venv python3-dev \
      build-essential

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:\$PATH

    uv venv /opt/venv_vllm --python /usr/bin/python3 --seed --clear
    source /opt/venv_vllm/bin/activate

    echo '===== LAYER 1: vLLM (Shahar branch + 3 Sam HybridEP cherry-picks) ====='
    git clone https://github.com/vllm-project/vllm.git /opt/vllm
    cd /opt/vllm
    git config user.email 'jonathanp@nvidia.com'
    git config user.name  'Jonathan P (deci-handoff rebuild)'
    git remote add shahar ${VLLM_REMOTE}
    git remote add sam    https://github.com/samnordmann/vllm.git
    git fetch --no-tags shahar ${VLLM_BRANCH}
    git fetch --no-tags sam    hybrid_ep_integration_pr
    git checkout ${VLLM_PIN_SHA}
    git cherry-pick 3e6549755  # [Core] Add hybrid_ep backend
    git cherry-pick c95c94ec2  # Fix CUDA graph replay SIGSEGV
    git cherry-pick c249dbf22  # Reconstruct expert_topk_ids under CG capture
    git log --oneline -8

    export VLLM_USE_PRECOMPILED=1
    export UV_TORCH_BACKEND=cu130
    # Pin the exact cu130 aarch64 wheel for the pinned base commit (191e3fdaa =
    # merge-base of Shahar's branch with main). Without this, variant detection
    # inside uv build isolation falls back to the default-variant wheel, which is
    # a stable-libtorch-ABI-only build missing _C.abi3.so/_moe_C.abi3.so
    # (attempt-2 failure: ModuleNotFoundError vllm._C).
    export VLLM_PRECOMPILED_WHEEL_LOCATION='https://wheels.vllm.ai/191e3fdaa1fd3dd09441e7b22d4f2ddef51c012c/vllm-0.19.2rc1.dev47%2Bg191e3fdaa.cu130-cp38-abi3-manylinux_2_35_aarch64.whl'
    uv pip install -e /opt/vllm --torch-backend=cu130

    echo '===== LAYER 2: FlashInfer (auto-version match) ====='
    FI_VER=\$(uv pip show flashinfer-python | awk '/Version:/ {print \$2}')
    echo \"FlashInfer version: \$FI_VER\"
    uv pip install flashinfer-cubin==\"\$FI_VER\"
    uv pip install flashinfer-jit-cache==\"\$FI_VER\" --index-url https://flashinfer.ai/whl/cu130

    echo '===== LAYER 3: Ray + extras ====='
    uv pip install ray tblib pytest

    echo '===== LAYER 4: symlinks ====='
    for tool in vllm ray python python3 pip; do
        ln -sf /opt/venv_vllm/bin/\$tool /usr/local/bin/\$tool
    done

    echo '===== LAYER 5: HybridEP (NVSHMEM + Tongliu DeepEP) ====='
    mkdir -p /opt/vllm/ep_kernels_workspace
    git clone https://github.com/Autumn1998/DeepEP.git \
      /opt/vllm/ep_kernels_workspace/DeepEP
    cd /opt/vllm/ep_kernels_workspace/DeepEP
    git checkout ${DEEPEP_TONGLIU_COMMIT}
    cd /opt/vllm
    TORCH_CUDA_ARCH_LIST='10.0' \
      CPLUS_INCLUDE_PATH=\"/usr/local/cuda/include/nvtx3:\${CPLUS_INCLUDE_PATH:-}\" \
      bash tools/ep_kernels/install_python_libraries.sh

    echo '===== LAYER 6: bake CUDA_HOME (.pth + /etc/environment) ====='
    SITE_PKG=\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')
    cat > \"\$SITE_PKG/_zz_cuda_home.pth\" <<'PTH'
import os; os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
PTH
    echo 'CUDA_HOME=/usr/local/cuda' >> /etc/environment

    cd /
    python3 -c 'import vllm; print(\"vLLM:\", vllm.__version__)'
    python3 -c 'import deep_ep; print(\"deep_ep:\", deep_ep.__file__)'
    python3 -c 'from vllm.model_executor.layers.fused_moe.prepare_finalize.hybrid_ep import HybridEPPrepareAndFinalize; print(\"HybridEP backend OK\")'
    python3 -c 'import flashinfer; print(\"FlashInfer:\", flashinfer.__version__)'
    python3 -c 'import ray; print(\"Ray:\", ray.__version__)'
    python3 -c 'import torch; print(\"Torch:\", torch.__version__, \"CUDA:\", torch.version.cuda)'
    env -u CUDA_HOME python3 -c 'import os; assert os.environ[\"CUDA_HOME\"] == \"/usr/local/cuda\"; print(\"_zz_cuda_home.pth OK\")'
  "

echo "=== Build complete: ${OUTPUT_CONTAINER} ($(du -h ${OUTPUT_CONTAINER} 2>/dev/null | cut -f1)) ==="
