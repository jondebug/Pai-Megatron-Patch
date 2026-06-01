#!/bin/bash
# KL coefficient sensitivity test.
# Runs three matched 80-iter A3B training runs at different kl_loss_coeff,
# then dumps a side-by-side comparison of router_weight_drift, lm_loss, CP.

set -e

REPO=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch
DATA=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-datasets/mmap_qwen3_datasets_text_document
CKPT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-30B-A3B-to-mcore
BASE_OUT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
TS=$(date +%y%m%d_%H%M%S)
TEST_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/kl_coeff_test_${TS}
mkdir -p "$TEST_DIR"

# Shared training args.
ITERS=80
COMMON_ARGS=(
  --router-only-training --use_rl_loss --rl-ppo-reeval --rl-algorithm ppo
  --rl-loss-coeff 0.5 --rl-ppo-baseline-type mean
  --rl-reward-type per_token_load_weighted
  --moe-aux-loss-coeff 0.003
  --train-iters "$ITERS" --eval-interval 1000 --eval-iters 0
  --hellaswag-eval-interval 0
)

run_one () {
  local NAME="$1"
  local KL="$2"
  local OUT="${BASE_OUT}/kl_coeff_test_${TS}_${NAME}"
  local LOG="${TEST_DIR}/${NAME}.log"
  mkdir -p "$OUT"
  echo ""
  echo "============================================================"
  echo "[$(date)] STARTING ${NAME} (kl_loss_coeff=${KL})"
  echo "  OUT=${OUT}"
  echo "  LOG=${LOG}"
  echo "============================================================"
  cd "$REPO/examples/qwen3"
  sh run_mcore_qwen3.sh dsw A3B 1 8 1e-4 1e-6 128 128 bf16 1 1 1 1 4 \
    true true true false sel false 100 \
    "$DATA" "$DATA" "$CKPT" 1024000 10240 "$OUT" \
    "${COMMON_ARGS[@]}" --kl-loss-coeff "$KL" 2>&1 | tee "$LOG" || true
  echo "[$(date)] FINISHED ${NAME}"
}

run_one "kl0"     "0"
run_one "kl1e-3"  "0.001"
run_one "kl1e2"   "100"

# Side-by-side comparison.
echo ""
echo "============================================================"
echo "RESULTS COMPARISON"
echo "============================================================"
for NAME in kl0 kl1e-3 kl1e2; do
  LOG="${TEST_DIR}/${NAME}.log"
  echo ""
  echo "--- ${NAME} ---"
  grep -oE "iteration\s+(1|20|40|60|80)/.*kl_loss[^|]*" "$LOG" 2>/dev/null | head -10 || echo "no iter lines"
  echo "  drift @ last iter:"
  grep -oE "router_weight_drift: [0-9.E+-]+" "$LOG" 2>/dev/null | tail -3 || echo "no drift"
done
echo ""
echo "============================================================"
echo "Test logs: ${TEST_DIR}/"
echo "============================================================"
