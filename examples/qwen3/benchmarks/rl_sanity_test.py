#!/usr/bin/env python3
# Sanity test for the paused-guard fix: does the RL policy gradient reach router.weight?
# Reproduces the real scenario: (1) grad-enabled MAIN forward populates the trajectory with in-graph
# logits, (2) the KL reference forward (paused=True) tries to overwrite with DETACHED logits, (3) RL
# loss is computed and we check requires_grad + gradient on the router weights + an actual weight step.
import torch, sys
torch.manual_seed(0)
from megatron_patch.model.qwen3_moe.moe.rl_trajectory import RouterTrajectoryTracker

E, seq, batch, topk, H = 8, 4, 2, 2, 16
def make_router_and_logits():
    gating = torch.nn.Linear(H, E)                       # the trainable router
    x = torch.randn(seq*batch, H)
    logits = gating(x).reshape(seq, batch, E)            # IN-GRAPH: requires_grad, grad_fn present
    _, idx = logits.topk(topk, dim=-1)
    rmap = torch.zeros_like(logits).scatter(-1, idx, 1.0)
    return gating, logits, rmap, x.reshape(seq, batch, H)

def new_tracker():
    tr = RouterTrajectoryTracker()
    tr.reward_type = "diff_lse_load"
    tr.per_token_rewards = True
    tr.baseline_type = "mean"
    tr.paused = False
    if hasattr(tr, "reset"): tr.reset()
    tr.paused = False
    return tr

def rl_grad_on_weight(tr, gating):
    loss = tr.compute_reinforce_loss(tr.layer_decisions, discount_factor=0.0)
    g = torch.autograd.grad(loss, gating.weight, retain_graph=True, allow_unused=True)[0]
    return loss, g

print("="*72)
print("TEST A — WITH the paused-guard fix (expected: CONNECTED)")
print("="*72)
tr = new_tracker()
gating, logits, rmap, latent = make_router_and_logits()
tr.paused = False
tr.add_layer_decision(1, latent, rmap, logits)                       # main forward: in-graph
sb = tr.layer_decisions[1][2]
print(f"  after MAIN forward: stored logits.requires_grad={sb.requires_grad} grad_fn={sb.grad_fn is not None}")
tr.paused = True                                                     # reference forward
tr.add_layer_decision(1, latent.detach(), rmap, logits.detach())     # tries to overwrite w/ detached
sa = tr.layer_decisions[1][2]
print(f"  after PAUSED ref forward: stored logits.requires_grad={sa.requires_grad}  (fix => still True)")
tr.paused = False
loss, g = rl_grad_on_weight(tr, gating)
gn = g.norm().item() if g is not None else None
print(f"  rl_loss.requires_grad={loss.requires_grad} value={float(loss):.6e}")
print(f"  ||d(rl_loss)/d(router.weight)|| = {gn}")
w0 = gating.weight.detach().clone()
with torch.no_grad(): gating.weight -= 0.1 * (g if g is not None else 0)
dw = (gating.weight - w0).norm().item()
print(f"  router.weight change after one RL step = {dw:.6e}")
A_ok = bool(sa.requires_grad) and bool(loss.requires_grad) and (g is not None) and gn > 0 and dw > 0

print()
print("="*72)
print("TEST B — CONTRAST: simulate the OLD bug (overwrite NOT guarded => DETACHED)")
print("="*72)
tr2 = new_tracker()
gating2, logits2, rmap2, latent2 = make_router_and_logits()
tr2.paused = False
tr2.add_layer_decision(1, latent2, rmap2, logits2)                  # main forward
tr2.paused = False                                                  # <-- guard NOT active (old behavior)
tr2.add_layer_decision(1, latent2.detach(), rmap2, logits2.detach())# overwrite WITH detached (the bug)
sa2 = tr2.layer_decisions[1][2]
print(f"  after unguarded overwrite: stored logits.requires_grad={sa2.requires_grad}  (bug => False)")
loss2 = tr2.compute_reinforce_loss(tr2.layer_decisions, discount_factor=0.0)
g2 = torch.autograd.grad(loss2, gating2.weight, retain_graph=True, allow_unused=True)[0]
print(f"  rl_loss.requires_grad={loss2.requires_grad}  grad_to_weight={'NONE (disconnected!)' if g2 is None else g2.norm().item()}")
B_shows_bug = (not sa2.requires_grad) and (g2 is None or not loss2.requires_grad)

print()
print("="*72)
print(f"RESULT: TEST A (fix connects RL) = {'PASS' if A_ok else 'FAIL'} | "
      f"TEST B (reproduces old disconnection) = {'PASS' if B_shows_bug else 'inconclusive'}")
print("="*72)
sys.exit(0 if A_ok else 1)
