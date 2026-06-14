"""Canonical derivation of cell metadata from the run_name. Single source of truth,
imported by both ingest_evals.py (future rows) and backfill_metadata.py (existing rows)."""
import re

def _num(pat, n):
    m = re.search(pat, n)
    return m.group(1) if m else ""

def parse_meta(name):
    n = name or ""
    rlc = _num(r"rlc([0-9]+(?:\.[0-9]+)?)", n)
    aux = _num(r"aux([0-9]+(?:\.[0-9]+)?)", n)
    kl  = _num(r"(?:^|_)kl([0-9]+(?:\.[0-9]+)?)", n)
    lm  = _num(r"(?:^|_)lm([0-9]+(?:\.[0-9]+)?)", n)
    norl = "norl" in n
    pretrained = n.startswith("pretrained") or "pretrained_baseline" in n
    # RL is signalled by an rlc coeff (new convention) OR ppo/critic tokens (old convention)
    rl_token = ("ppo" in n) or ("c256" in n) or ("rladv" in n) or n.startswith("crit")
    rlc_on = rlc not in ("", "0", "0.0", "0.00")
    rl_en  = (not norl) and (rlc_on or rl_token)
    aux_en = aux not in ("", "0", "0.0", "0.00")
    # reward type (match canonical names already used in the CSV/plot)
    rt = ""
    if   re.search(r"(rwd)?critical_path", n): rt = "critical_path"
    elif re.search(r"(rwd)?entropy", n):       rt = "entropy"
    elif re.search(r"(rwd)?topn", n):          rt = "per_token_topn_binary"
    elif re.search(r"(rwd)?per_token_load_weighted", n): rt = "per_token_load_weighted"
    elif rl_en:                                 rt = "per_token_load_weighted"  # sweep default
    if   pretrained:        cat = "pretrained"
    elif rl_en and aux_en:  cat = "rl+aux"
    elif rl_en:             cat = "rl_only"
    elif aux_en:            cat = "aux_only"
    else:                   cat = "other"
    return {
        "category": cat,
        "rl_enabled": "True" if rl_en else "False",
        "aux_enabled": "True" if aux_en else "False",
        "rl_reward_type": rt,
        "rl_loss_coeff": rlc,
        "aux_loss_coeff": aux,
        "kl_loss_coeff": kl,
        "lm_reward_coeff": lm,
    }
