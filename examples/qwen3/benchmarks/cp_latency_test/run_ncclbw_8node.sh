#!/bin/bash
#SBATCH --job-name=ncclbw_8node
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/jonathanp/ncclbw_8node_%j.out
set -uo pipefail
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
CONTAINER=$BASE/containers/vllm-openai-latest.sqsh
MNT="$BASE:$BASE"
echo "=== nccl-tests on 8 nodes × 8 GPU (64 ranks total) ==="
# nccl-tests should be installed in the vllm-openai container (commonly via /opt/nccl-tests/build or pip nccl-tests).
# Try a couple of likely paths.
srun --nodes=8 --ntasks=64 --ntasks-per-node=8 --gpus-per-node=8 --mem=0 \
     --container-image="$CONTAINER" --container-mounts="$MNT" \
  bash -c '
   set -uo pipefail
   for P in /opt/nccl-tests/build /usr/local/bin /opt/nccl_tests/build /workspace/nccl-tests/build /root/nccl-tests/build; do
     if [ -x "$P/all_reduce_perf" ]; then export NTBIN=$P; break; fi
   done
   if [ -z "${NTBIN:-}" ]; then
     echo "RANK=$SLURM_PROCID NO nccl-tests binary found; install via apt or build manually"
     # Build it on the fly on rank 0 only
     if [ "$SLURM_PROCID" = "0" ]; then
       which nvcc
       # try install
       cd /tmp && (apt-get install -y --no-install-recommends git build-essential 2>/dev/null || true)
       [ ! -d nccl-tests ] && git clone --depth=1 https://github.com/NVIDIA/nccl-tests.git 2>&1 | tail -5
       cd nccl-tests && make MPI=0 -j 2>&1 | tail -10
       echo "build returned $?"
       ls -la build/
     fi
     exit 99
   fi
   # rank 0 prints; others just run
   if [ "$SLURM_PROCID" = "0" ]; then
     echo "=== AllReduce performance (1KB to 1GB) ==="
     $NTBIN/all_reduce_perf -b 1K -e 1G -f 2 -g 1 -n 5 -w 3 2>&1 | head -40
     echo
     echo "=== AllToAll performance (1KB to 1GB) ==="
     $NTBIN/alltoall_perf   -b 1K -e 1G -f 2 -g 1 -n 5 -w 3 2>&1 | head -40
     echo
     echo "=== AllToAllv (variable) performance ==="
     ls $NTBIN/alltoallv* 2>/dev/null && $NTBIN/alltoallv_perf -b 1K -e 1G -f 2 -g 1 -n 5 -w 3 2>&1 | head -40
   else
     $NTBIN/all_reduce_perf -b 1K -e 1G -f 2 -g 1 -n 5 -w 3 >/dev/null 2>&1
     $NTBIN/alltoall_perf   -b 1K -e 1G -f 2 -g 1 -n 5 -w 3 >/dev/null 2>&1
   fi
  '
echo "=== done $(date) ==="
