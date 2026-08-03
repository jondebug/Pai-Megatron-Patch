# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# =============================================================================
# RL router MEASUREMENT INFRASTRUCTURE (reviewer-consolidated).
#
# P2 -- fixed deterministic critical-path (CP) probe. THE priority signal.
#   A small IMMUTABLE set of held-out probe batches (fixed token sequences +
#   masks, captured once from the first training steps) is re-routed with
#   DETERMINISTIC full-128 argmax top-k (NO Gumbel, NO candidate-pool
#   restriction, eval mode) and the exact critical path CP = sum_l max_e n_{l,e}
#   is measured, paired vs the INITIAL router theta0 (captured before any update
#   and applied via a weight swap). This is the noise-free "does RL lower the
#   deterministic CP" signal the on-policy training CP lacks.
#
# P3 -- frozen-rollout causal audit (added in a later commit) and
# P4 -- old-vs-new deterministic top-k churn (added in a later commit) live in
#   this same module; see run_audit()/churn helpers.
#
# EVERYTHING here is best-effort: any failure prints a one-line [PROBE]/[AUDIT]
# WARNING and disables the offending piece -- it must NEVER kill training. The
# probe forward mirrors the proven KL reference-forward guards in helper.py
# (pause the trajectory tracker, save/restore the aux-loss tracker, disable the
# KL logit-capture hook, eval mode) so it cannot contaminate training metrics.
# =============================================================================

import math
import torch

_PROBE = {
    'interval': 0,          # run the CP probe every N updates (0 = OFF)
    'n_target': 16,         # number of immutable probe microbatches to capture
    'batches': [],          # list of frozen (tokens, position_ids, attention_mask, packed_seq_params)
    'theta0_weights': None,  # dict name->tensor : router weights BEFORE any update (theta0)
    'baseline': None,        # theta0 probe result (computed once, immutable)
    'prev': None,            # previous probe result (for P4 old-vs-new churn)
    'last_iter': -1,         # last iteration a probe ran (dedup guard)
    'disabled': False,       # set True if the probe hits a fatal error
    # --- P3 audit ---
    'audit_interval': 0,
    'audit_frozen': None,    # snapshot of the just-completed rollout (pre optimizer.step)
    'audit_disabled': False,
}


def _is_rank0():
    try:
        import torch.distributed as d
        return (not d.is_available()) or (not d.is_initialized()) or d.get_rank() == 0
    except Exception:
        return True


def _print0(msg):
    if _is_rank0():
        print(msg, flush=True)


def configure(args):
    """Read the probe/audit cadence from args (called once from forward_step)."""
    _PROBE['interval'] = int(getattr(args, 'rl_probe_interval', 0) or 0)
    _PROBE['n_target'] = max(1, int(getattr(args, 'rl_probe_batches', 16) or 16))
    _PROBE['audit_interval'] = int(getattr(args, 'rl_audit_interval', 0) or 0)


def probe_enabled():
    return _PROBE['interval'] > 0 and not _PROBE['disabled']


def audit_enabled():
    return _PROBE['audit_interval'] > 0 and not _PROBE['audit_disabled']


def banner_once(args):
    """P1 banner emitted from the TRAINING path (reliably flushed to the .out, unlike
    model_provider's early-setup stdout). Names the sampling vs scoring distribution so the
    H1 mismatch is visible. The fail-fast assert itself lives in model_provider."""
    if _PROBE.get('_banner_done'):
        return
    _PROBE['_banner_done'] = True
    try:
        if not getattr(args, 'use_rl_loss', False):
            return
        samp = getattr(args, 'rl_sampling', 'argmax')
        algo = getattr(args, 'rl_algorithm', 'reinforce')
        if samp == 'hard_gumbel_pl' and algo == 'reinforce':
            scoring = 'ordered_plackett_luce(rl_ordered_logprob, per-token REINFORCE)'
        elif samp == 'hard_gumbel_pl' and algo == 'ppo':
            scoring = 'ordered_plackett_luce(rl_ordered_logprob, PPO clipped ratio)'
        elif algo == 'reinforce':
            scoring = 'summed_independent_softmax(log_softmax chosen, per-token REINFORCE)'
        else:
            scoring = 'summed_independent_softmax(log_softmax chosen, PPO ratio)'
        pool = int(getattr(args, 'rl_candidate_pool', 0)) or 'ALL_EXPERTS'
        nepochs = int(getattr(args, 'rl_ppo_epochs', 1)) if algo == 'ppo' else 1
        _print0(
            "[RL CONFIG BANNER] "
            f"sampling_distribution={samp} | scoring_distribution={scoring} | "
            f"candidate_pool_size={pool} | algorithm={algo} | num_policy_epochs={nepochs} | "
            f"reward_type={getattr(args, 'rl_reward_type', 'expert0')} | "
            f"loo_beta={getattr(args, 'rl_loo_beta', 0.3)} | "
            f"global_loads={getattr(args, 'rl_global_loads', False)} | "
            f"perlayer_norm={getattr(args, 'rl_perlayer_norm', False)} | "
            f"rl_loss_coeff={getattr(args, 'rl_loss_coeff', 0.0)}")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Capture: freeze the first n_target training microbatches as the immutable probe
# set, and snapshot theta0 router weights before the first optimizer step.
# -----------------------------------------------------------------------------
def snapshot_theta0(model):
    if _PROBE['theta0_weights'] is not None:
        return
    try:
        w = {}
        for name, p in model.named_parameters():
            if 'router' in name and 'weight' in name:
                w[name] = p.data.clone().detach()
        _PROBE['theta0_weights'] = w
        _print0(f"[PROBE] snapshotted theta0 router weights: {len(w)} tensors")
    except Exception as e:
        _print0(f"[PROBE] WARNING: theta0 snapshot failed: {e}")
        _PROBE['theta0_weights'] = {}


def maybe_capture(batch):
    """Freeze an immutable clone of a training microbatch until n_target reached."""
    if not (probe_enabled() or audit_enabled()):
        return
    if len(_PROBE['batches']) >= _PROBE['n_target']:
        return
    try:
        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = batch
        frozen = (
            tokens.detach().clone(),
            position_ids.detach().clone() if position_ids is not None else None,
            attention_mask.detach().clone() if isinstance(attention_mask, torch.Tensor) else attention_mask,
            packed_seq_params,  # small metadata object; reused read-only
        )
        _PROBE['batches'].append(frozen)
        if len(_PROBE['batches']) == _PROBE['n_target']:
            _print0(f"[PROBE] captured {_PROBE['n_target']} immutable probe batches")
    except Exception as e:
        _print0(f"[PROBE] WARNING: probe-batch capture failed: {e}")


def buffer_ready():
    return len(_PROBE['batches']) >= _PROBE['n_target']


def should_probe(iteration):
    if not probe_enabled() or not buffer_ready():
        return False
    if iteration == _PROBE['last_iter']:
        return False
    return (iteration % _PROBE['interval']) == 0


# -----------------------------------------------------------------------------
# The deterministic probe forward.
# -----------------------------------------------------------------------------
def _router_geometry(tracker):
    mod0 = next(iter(tracker._router_modules.values()))
    k = int(mod0.topk)
    E = int(mod0.config.num_moe_experts)
    return k, E


def _dp_group():
    try:
        from megatron.core import parallel_state
        return parallel_state.get_data_parallel_group()
    except Exception:
        return None


def _register_hooks(router_modules, captured, capture_logits=False):
    handles = []
    for ln, mod in router_modules.items():
        def hook(m, inp, out, _ln=ln):
            try:
                # router.forward returns (scores, routing_map); routing_map is [T, E] bool.
                rm = out[1].detach()
                lg = None
                # P4: rank-0 only, recompute dense router logits for the categorical-KL churn
                # measure (the router policy distribution over the fixed pool of experts).
                if capture_logits and _is_rank0():
                    try:
                        with torch.no_grad():
                            lg = m.gating(inp[0]).detach().float().view(-1, rm.shape[-1])
                    except Exception:
                        lg = None
                captured[_ln] = {'rm': rm, 'logits': lg}
            except Exception:
                pass
        handles.append(mod.register_forward_hook(hook))
    return handles


def _allreduce_counts(counts, group):
    try:
        import torch.distributed as d
        if group is not None and d.is_available() and d.is_initialized() and d.get_world_size(group=group) > 1:
            # sorted() => identical collective order on every rank (avoids a mismatch/deadlock)
            for ln in sorted(counts):
                d.all_reduce(counts[ln], op=d.ReduceOp.SUM, group=group)
    except Exception:
        pass


def _probe_once(model, tracker, group, k, E):
    """Run the deterministic probe over ALL frozen batches; return CP + per-layer
    GLOBAL (DP-all-reduced) expert counts, plus (rank-0) per-token top-k SETS and dense
    router logits for the P4 churn/KL metrics.

    per_batch_cp : list of per-batch global CP (for mean +/- CI across batches).
    counts       : {layer -> [E]} global counts SUMMED over all probe batches
                   (stable per-layer structure: max load, hot expert, load CV).
    topk         : {layer -> [T_total, k]} rank-0 deterministic top-k sets (fixed states).
    logits       : {layer -> [T_total, E]} rank-0 dense router logits (for categorical KL)."""
    router_modules = tracker._router_modules
    layer_ids = sorted(router_modules.keys())  # IDENTICAL on every rank => lockstep all-reduce
    per_batch_cp = []
    agg = {}
    r0 = _is_rank0()
    topk = {}
    logits = {}
    dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    for (tokens, pos, attn, pack) in _PROBE['batches']:
        captured = {}
        handles = _register_hooks(router_modules, captured, capture_logits=r0)
        try:
            with torch.no_grad():
                # eval + no_grad => the router's Gumbel path is bypassed
                # (_maybe_rl_gumbel_route requires training and grad) => deterministic argmax top-k.
                model(tokens, pos, attn, labels=None, packed_seq_params=pack)
        finally:
            for h in handles:
                h.remove()
        # Key over the FULL sorted layer set (zeros for any layer a hook missed) so every rank
        # all-reduces the same layers in the same order.
        batch_counts = {}
        for ln in layer_ids:
            cap = captured.get(ln)
            if cap is not None:
                batch_counts[ln] = cap['rm'].float().sum(dim=0)                 # [E] local
            else:
                batch_counts[ln] = torch.zeros(E, device=dev, dtype=torch.float32)
        _allreduce_counts(batch_counts, group)                                  # [E] GLOBAL
        per_batch_cp.append(sum(float(batch_counts[ln].max().item()) for ln in layer_ids))
        for ln in layer_ids:
            agg[ln] = batch_counts[ln].clone() if ln not in agg else (agg[ln] + batch_counts[ln])
        if r0:
            for ln, cap in captured.items():
                # deterministic top-k SET on this rank's fixed tokens (the actual routing decision)
                tk = cap['rm'].float().topk(k, dim=-1).indices.sort(dim=-1).values.to(torch.int16)
                topk[ln] = tk if ln not in topk else torch.cat([topk[ln], tk], dim=0)
                if cap['logits'] is not None:
                    lg = cap['logits'].half()
                    logits[ln] = lg if ln not in logits else torch.cat([logits[ln], lg], dim=0)
    return {'per_batch_cp': per_batch_cp, 'counts': agg, 'topk': topk, 'logits': logits, 'k': k}


def _probe_once_with_theta0(model, tracker, group, k, E):
    """Deterministic probe with the INITIAL router theta0 swapped in (weight swap
    exactly like _run_reference_forward), then restore current weights."""
    theta0 = _PROBE['theta0_weights']
    if not theta0:
        return _probe_once(model, tracker, group, k, E)
    saved = {}
    try:
        for name, p in model.named_parameters():
            if name in theta0:
                saved[name] = p.data.clone()
                p.data.copy_(theta0[name])
        return _probe_once(model, tracker, group, k, E)
    finally:
        for name, p in model.named_parameters():
            if name in saved:
                p.data.copy_(saved[name])


def _cp_agg(res):
    return sum(float(c.max().item()) for c in res['counts'].values())


def _mean_ci(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    ci = 1.96 * math.sqrt(var) / math.sqrt(n)
    return m, ci, n


def _emit(iteration, base, cur, k, E):
    layers = sorted(cur['counts'].keys())
    cp0 = _cp_agg(base)
    cpc = _cp_agg(cur)
    r_probe = (cp0 - cpc) / cp0 if cp0 > 0 else 0.0
    mcp, ci, n = _mean_ci(cur['per_batch_cp'])
    mcp0, _, _ = _mean_ci(base['per_batch_cp'])

    dmax = {}         # per-layer change in max load (cur - base)  (>0 = regressed)
    improving = 0
    cv_cur = []
    cv_base = []
    for ln in layers:
        cc = cur['counts'][ln]
        bc = base['counts'].get(ln)
        cm = float(cc.max().item())
        bm = float(bc.max().item()) if bc is not None else cm
        dmax[ln] = cm - bm
        if cm < bm:
            improving += 1
        mu = float(cc.mean().item())
        if mu > 0:
            cv_cur.append(float(cc.std().item()) / mu)
        if bc is not None:
            mub = float(bc.mean().item())
            if mub > 0:
                cv_base.append(float(bc.std().item()) / mub)
    frac_impr = improving / max(1, len(layers))
    cvc = sum(cv_cur) / max(1, len(cv_cur))
    cvb = sum(cv_base) / max(1, len(cv_base))

    # hot-expert identity for the 3 layers with the highest current max load
    hot = sorted(layers, key=lambda l: -float(cur['counts'][l].max().item()))[:3]
    hot_s = ", ".join(
        f"L{l}:e{int(cur['counts'][l].argmax().item())}(n={float(cur['counts'][l].max().item()):.0f})"
        for l in hot)
    # worst-regressing layers (largest positive dmax)
    worst = sorted(layers, key=lambda l: -dmax[l])[:3]
    worst_s = ", ".join(f"L{l}:{dmax[l]:+.0f}" for l in worst)

    _print0(
        f"[PROBE] iter={iteration} CP0={cp0:.1f} CPtheta={cpc:.1f} dCP={cpc-cp0:+.1f} "
        f"R_probe={r_probe:+.4f} | meanCP_batch theta={mcp:.1f}+/-{ci:.1f} (n={n}) base={mcp0:.1f} "
        f"| k={k} E={E}")
    _print0(
        f"[PROBE] iter={iteration} frac_layers_improving={frac_impr:.3f} "
        f"load_CV theta={cvc:.4f} base={cvb:.4f} | hot[{hot_s}] | worst_regress[{worst_s}]")


def run_probe(model, tracker, iteration):
    """Run the deterministic-CP probe and emit [PROBE] rank-0 lines. Best-effort."""
    if not probe_enabled() or not buffer_ready():
        return
    if not getattr(tracker, '_router_modules', None):
        return
    _PROBE['last_iter'] = iteration
    was_training = bool(getattr(model, 'training', True))
    prev_paused = getattr(tracker, 'paused', False)
    aux_saved = None
    kl_prev = None
    try:
        k, E = _router_geometry(tracker)
        group = _dp_group()
        # --- guards (mirror _run_reference_forward) ---
        tracker.paused = True
        try:
            model.eval()
        except Exception:
            pass
        aux_saved = _save_aux()
        kl_prev = _set_kl_capture(True)

        if _PROBE['baseline'] is None:
            _PROBE['baseline'] = _probe_once_with_theta0(model, tracker, group, k, E)
        cur = _probe_once(model, tracker, group, k, E)
        _emit(iteration, _PROBE['baseline'], cur, k, E)
        _run_churn(iteration, _PROBE['baseline'], cur, k)  # P4: old-vs-new deterministic top-k churn
        _PROBE['prev'] = cur
    except Exception as e:
        _print0(f"[PROBE] WARNING: probe failed at iter {iteration}: {e}; disabling probe")
        _PROBE['disabled'] = True
    finally:
        if aux_saved is not None:
            _restore_aux(aux_saved)
        if kl_prev is not None:
            _set_kl_capture(kl_prev)
        tracker.paused = prev_paused
        if was_training:
            try:
                model.train()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Guard helpers (aux-loss tracker save/restore + KL logit-capture disable).
# -----------------------------------------------------------------------------
def _save_aux():
    try:
        from megatron.core.transformer.moe.moe_utils import (
            get_moe_layer_wise_logging_tracker, clear_aux_losses_tracker)
        t = get_moe_layer_wise_logging_tracker()
        saved = {name: {kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
                        for kk, vv in entry.items()}
                 for name, entry in t.items()}
        clear_aux_losses_tracker()
        return saved
    except Exception:
        return None


def _restore_aux(saved):
    try:
        from megatron.core.transformer.moe.moe_utils import (
            get_moe_layer_wise_logging_tracker, clear_aux_losses_tracker)
        t = get_moe_layer_wise_logging_tracker()
        clear_aux_losses_tracker()
        if saved is not None:
            for name, entry in saved.items():
                t[name] = entry
    except Exception:
        pass


def _set_kl_capture(disabled):
    """Disable/restore the helper's KL output-layer logit-capture hook so the probe
    forward does not clobber the training forward's captured logits. Returns the
    previous value. Lazy import avoids a circular dependency with helper.py."""
    try:
        from megatron_patch.template import helper
        prev = helper._kl_state.get('capture_disabled', False)
        helper._kl_state['capture_disabled'] = disabled
        return prev
    except Exception:
        return None


# -----------------------------------------------------------------------------
# P4 churn / P3 audit -- filled in by later commits; safe no-ops for now.
# -----------------------------------------------------------------------------
def _set_metrics(old_idx, new_idx, k):
    """old_idx, new_idx: [T, k] int expert-id sets (per token). Returns mean set_churn and
    mean top-k Jaccard over the T tokens. set_churn = 1 - |old ∩ new|/k."""
    o = old_idx.long()
    n = new_idx.long()
    inter = (n.unsqueeze(-1) == o.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()  # [T] |new ∩ old|
    churn = (1.0 - inter / k).mean().item()
    union = (2 * k - inter).clamp(min=1.0)
    jacc = (inter / union).mean().item()
    return churn, jacc


def _cat_kl(logits_ref, logits_cur):
    """Mean per-token categorical KL(softmax(ref) || softmax(cur)) over the expert axis."""
    pr = torch.log_softmax(logits_ref.float(), dim=-1)
    pc = torch.log_softmax(logits_cur.float(), dim=-1)
    p = pr.exp()
    kl = (p * (pr - pc)).sum(dim=-1)  # [T]
    return kl.mean().item()


def _run_churn(iteration, base, cur, k):
    """P4: replace the saturated sampled-vs-argmax flip_rate with OLD-vs-NEW DETERMINISTIC
    top-k SET replacement on the identical fixed probe states. Reports set_churn / Jaccard
    vs the INITIAL router theta0 (cumulative) AND vs the PREVIOUS probe (incremental drift,
    the quantity the 0.1-1% guard watches), plus categorical KL vs theta0. Rank-0 only."""
    if not _is_rank0():
        return
    try:
        cur_tk = cur.get('topk', {}) or {}
        base_tk = base.get('topk', {}) or {}
        if not cur_tk:
            return
        prev = _PROBE.get('prev')
        prev_tk = (prev or {}).get('topk', {}) if prev else {}

        def _avg_sets(ref_tk):
            cs, js, nl = [], [], 0
            for ln, tk in cur_tk.items():
                r = ref_tk.get(ln)
                if r is None or r.shape != tk.shape:
                    continue
                c, j = _set_metrics(r, tk, k)
                cs.append(c)
                js.append(j)
                nl += 1
            if nl == 0:
                return None
            return sum(cs) / nl, sum(js) / nl, nl

        v0 = _avg_sets(base_tk)
        vp = _avg_sets(prev_tk) if prev_tk else None

        # categorical KL vs theta0 (dense router logits), averaged over layers
        cur_lg = cur.get('logits', {}) or {}
        base_lg = base.get('logits', {}) or {}
        kls = []
        for ln, lg in cur_lg.items():
            r = base_lg.get(ln)
            if r is not None and r.shape == lg.shape:
                kls.append(_cat_kl(r, lg))
        kl0 = (sum(kls) / len(kls)) if kls else float('nan')

        s0 = f"vs_theta0 set_churn={v0[0]:.4f} jaccard={v0[1]:.4f}" if v0 else "vs_theta0 n/a"
        sp = f"vs_prev set_churn={vp[0]:.4f} jaccard={vp[1]:.4f}" if vp else "vs_prev n/a(first)"
        _print0(f"[PROBE] iter={iteration} churn {s0} | {sp} | cat_KL_vs_theta0={kl0:.6e} k={k}")
    except Exception as e:
        _print0(f"[PROBE] WARNING: churn failed at iter {iteration}: {e}")


def snapshot_audit(tracker, iteration):
    """P3: snapshot the just-completed REINFORCE+hard_gumbel_pl rollout BEFORE the reset
    clears pl_decisions. Stores, per layer, the frozen latent (router input), the FIXED
    sampled action (pool_idx / pos), the old-policy ordered PL log-prob and the per-token
    advantages used in the loss. run_audit() recomputes the SAME action log-prob after the
    optimizer step. Best-effort; a no-op off the PL path or off the audit cadence."""
    if not audit_enabled():
        return
    if _PROBE['audit_interval'] <= 0 or (iteration % _PROBE['audit_interval']) != 0:
        return
    try:
        pl = getattr(tracker, 'pl_decisions', {}) or {}
        ld = getattr(tracker, 'layer_decisions', {}) or {}
        adv_list = getattr(tracker, '_dg_adv', []) or []
        if not pl or not ld:
            return  # not the hard_gumbel_pl per-token REINFORCE path -> nothing to audit
        tau = float(getattr(tracker, 'rl_stochastic_temperature', 1.0)) or 1.0
        sorted_layers = sorted(ld.keys())  # SAME order the loss used to fill _dg_adv
        layers = {}
        for i, ln in enumerate(sorted_layers):
            d = pl.get(ln)
            if d is None or i >= len(adv_list):
                continue
            old_ptlp = d['old_ptlp']
            adv = adv_list[i]
            try:
                adv2d = adv.view(old_ptlp.shape)
            except Exception:
                continue
            layers[ln] = {
                'latent': ld[ln][0].detach(),   # router input [seq, batch, hidden]
                'pool_idx': d['pool_idx'],       # FIXED detached pool [seq, batch, POOL]
                'pos': d['pos'],                 # ordered positions  [seq, batch, k]
                'old_ptlp': old_ptlp.detach(),   # old-policy PL log-prob [seq, batch]
                'adv': adv2d.detach(),           # per-token advantage    [seq, batch]
            }
        if layers:
            _PROBE['audit_frozen'] = {'iter': iteration, 'tau': tau, 'layers': layers}
    except Exception as e:
        _print0(f"[AUDIT] WARNING: snapshot failed: {e}")


def run_audit(tracker, iteration):
    """P3: recompute the FROZEN action's log-prob under the POST-STEP router and verify the
    causal chain: audit loss L=-E[A_old . logpi_theta(a_old)] decreased, E[dlogpi|A>0]>0,
    E[dlogpi|A<0]<0. Emits an [AUDIT] rank-0 line. Best-effort; never fatal."""
    fz = _PROBE.get('audit_frozen')
    if fz is None:
        return
    _PROBE['audit_frozen'] = None
    try:
        from .rl_trajectory import rl_ordered_logprob, rl_fp32
        rmods = getattr(tracker, '_router_modules', {}) or {}
        tau = fz['tau']
        num_before = num_after = denom = 0.0
        dpos_sum = dneg_sum = 0.0
        dpos_n = dneg_n = 0
        for ln, d in fz['layers'].items():
            router = rmods.get(ln)
            if router is None:
                continue
            with torch.no_grad():
                new_logits = router.gating(d['latent'])                                  # post-step logits
                new_ptlp = rl_ordered_logprob(rl_fp32(new_logits), d['pool_idx'], d['pos'], tau)
            A = d['adv']
            old = d['old_ptlp']
            new_ptlp = new_ptlp.view_as(old)
            num_before += float((A * old).sum().item())
            num_after += float((A * new_ptlp).sum().item())
            denom += float(A.numel())
            dlp = (new_ptlp - old)
            pos = A > 0
            neg = A < 0
            if bool(pos.any()):
                dpos_sum += float(dlp[pos].sum().item())
                dpos_n += int(pos.sum().item())
            if bool(neg.any()):
                dneg_sum += float(dlp[neg].sum().item())
                dneg_n += int(neg.sum().item())
        if denom <= 0:
            return
        L_before = -num_before / denom
        L_after = -num_after / denom
        d_pos = dpos_sum / max(1, dpos_n)
        d_neg = dneg_sum / max(1, dneg_n)
        ok_loss = L_after < L_before
        ok_pos = d_pos > 0
        ok_neg = d_neg < 0
        agree = ok_loss and ok_pos and ok_neg
        # predicted objective change = first-order descent (dL<0 expected after a grad step);
        # realized = the actual audit-loss change on the frozen actions.
        _print0(
            f"[AUDIT] iter={fz['iter']} audit_loss_before={L_before:+.6e} after={L_after:+.6e} "
            f"realized_dL={L_after - L_before:+.6e} predicted=descent(<0) loss_decreased={int(ok_loss)} | "
            f"E[dlogpi|A>0]={d_pos:+.6e}(>0:{int(ok_pos)}) E[dlogpi|A<0]={d_neg:+.6e}(<0:{int(ok_neg)}) | "
            f"direction_agreement={int(agree)} n={int(denom)}")
    except Exception as e:
        _print0(f"[AUDIT] WARNING: audit recompute failed: {e}; disabling audit")
        _PROBE['audit_disabled'] = True
