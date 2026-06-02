#!/bin/bash
# dispatch.sh — submit an sbatch job spanning the best-available GPU partitions
# (interactive first) for the NRT/ORD clusters, to minimize queue wait.
# Run this ON the cluster login node.
#
# Usage:
#   bash dispatch.sh --cluster nrt|ord [--nodes N] [--jobname NAME] -- <sbatch_script> [args...]
#
# It picks a partition list by cluster + geometry (single vs multi-node), prepends
# interactive partitions, prints live capacity, and submits with `sbatch -p <list>`.
set -uo pipefail

CLUSTER=""; NODES=1; JOBNAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2 ;;
    --nodes)   NODES="$2"; shift 2 ;;
    --jobname) JOBNAME="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done
SCRIPT="${1:?need sbatch script after --}"; shift || true

# Partition lists (interactive first), by cluster + geometry.
if [ "$CLUSTER" = nrt ]; then
  if [ "$NODES" -le 1 ]; then PARTS="interactive,batch_short,batch_singlenode,batch_block1,batch_long"
  else PARTS="batch_block1,batch_long"; fi
elif [ "$CLUSTER" = ord ]; then
  if [ "$NODES" -le 1 ]; then PARTS="interactive_singlenode,polar4,polar3,backfill_singlenode"
  else PARTS="polar4,polar3,backfill_block1"; fi
else echo "ERROR: --cluster must be nrt or ord" >&2; exit 1; fi

echo "== capacity (idle/mix) for candidate partitions =="
for p in ${PARTS//,/ }; do
  sinfo -h -p "$p" -o "  %P %t %D" 2>/dev/null | grep -iE "idle|mix" || echo "  $p: (no idle/mix)"
done

JN=(); [ -n "$JOBNAME" ] && JN=(-J "$JOBNAME")
echo "== submitting $SCRIPT spanning: $PARTS =="
jid=$(sbatch --parsable -p "$PARTS" "${JN[@]}" "$SCRIPT" "$@" 2>&1 | tail -1)
echo "submitted job $jid"
squeue -j "$jid" -o "%i %P %T %r" 2>/dev/null
echo "tip: if PENDING on Priority, it's behind in queue — capacity will pick it up; add more partitions or use scontrol update jobid=$jid partition=<more>"
