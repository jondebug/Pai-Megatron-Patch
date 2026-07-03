#!/bin/bash
# Given a shipped-in <ID>_routers_only.safetensors on HSG, this script:
#   1. Builds <ID>_router_swap/ by patching pretrained shards with the routers
#   2. Creates hsg_run_<ID>_ab.sh launcher
#   3. Submits the A/B bench (pretrained vs <ID>)
# Usage: ./hsg_setup_router_cell.sh <ID>
#   where <ID> matches the file /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/<ID>_routers_only.safetensors

set -uo pipefail

ID="${1:?Usage: $0 <ID>}"
BASE=/lustre/fsw/portfolios/nvr/users/jonathanp
ROUTER=$BASE/rl_token_routing/${ID}_routers_only.safetensors
SWAP_DIR=$BASE/rl_token_routing/${ID}_router_swap
LAUNCHER=$BASE/hsg_run_${ID}_ab.sh

[ -f "$ROUTER" ] || { echo "ERROR: router file $ROUTER not found"; exit 1; }
NR=$(python3 -c "from safetensors import safe_open
with safe_open('$ROUTER', framework='pt') as f:
    print(sum(1 for k in f.keys()))" 2>/dev/null || echo 0)
echo "Router file has $NR tensors ($(du -sh $ROUTER | cut -f1))"

# --- Step 1: Build swap dir if missing ---
if [ -d "$SWAP_DIR" ] && [ "$(ls $SWAP_DIR/*.safetensors 2>/dev/null | wc -l)" -ge 118 ]; then
  echo "[skip] $SWAP_DIR already has 118 shards"
else
  echo "Building $SWAP_DIR (5 min approx) ..."
  srun --account=nvr_israel_rlop --qos=interactive --partition=batch \
       --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --mem=64G --time=00:15:00 \
       --container-image=$BASE/containers/vllm-openai-arm.sqsh \
       --container-mounts=$BASE:$BASE,/lustre/fs1:/lustre/fs1 \
       python3 $BASE/build_r15_router_swap.py --routers $ROUTER --out $SWAP_DIR 2>&1 | tail -6
fi

# --- Step 2: Emit launcher ---
sed "s|r15_hf|${ID}_router_swap|g; s|r15ab|${ID}ab|g; s|r15_ep|${ID}_ep|g; s|hsg_r15ab|hsg_${ID}ab|g" \
    $BASE/hsg_run_r15_ab.sh > $LAUNCHER
chmod +x $LAUNCHER
echo "Launcher: $LAUNCHER"
grep -E "R15=|--models pre_ep|--job-name" $LAUNCHER | head -3

# --- Step 3: Submit ---
JID=$(sbatch --parsable $LAUNCHER 2>&1)
echo "Submitted job $JID"

# --- Step 4: Update manifest ---
MANIFEST=$BASE/rl_token_routing/CHECKPOINTS_MANIFEST.md
STAMP=$(date +%Y-%m-%dT%H:%M)
echo "- ${ID}: routers shipped $STAMP; router_swap built; bench job $JID submitted" >> $MANIFEST.log
echo "Manifest update log: $MANIFEST.log"
