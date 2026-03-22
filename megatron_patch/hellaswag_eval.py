"""Lightweight in-process HellaSwag evaluation for tracking benchmark accuracy during training.

Runs directly on the Megatron model without checkpoint conversion.
Designed to be fast (~30-60s for 100 samples) and called periodically during training.
"""
import torch
import os
import json
import time

_hellaswag_cache = None


def _load_hellaswag_data(limit=100):
    """Load HellaSwag validation set, cached after first call."""
    global _hellaswag_cache
    if _hellaswag_cache is not None and len(_hellaswag_cache) >= limit:
        return _hellaswag_cache[:limit]

    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")
        samples = []
        for item in ds:
            samples.append({
                'ctx': item['ctx'],
                'endings': item['endings'],
                'label': int(item['label']),
            })
        _hellaswag_cache = samples
        return samples[:limit]
    except Exception as e:
        cache_path = os.path.join(os.path.dirname(__file__), 'hellaswag_val.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                _hellaswag_cache = json.load(f)
            return _hellaswag_cache[:limit]
        print(f"[HELLASWAG] Could not load dataset: {e}", flush=True)
        return []


def _get_tokenizer():
    """Get the Megatron tokenizer."""
    try:
        from megatron.training import get_tokenizer
        return get_tokenizer()
    except Exception:
        return None


def _compute_log_likelihood(model, input_ids, target_start_idx, device):
    """Compute average log-likelihood of tokens from target_start_idx onward."""
    tokens = input_ids[:, :-1].to(device)
    labels = input_ids[:, 1:].to(device)
    seq_len = tokens.shape[1]

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = None

    with torch.no_grad():
        output = model(tokens, position_ids, attention_mask, labels=labels)

    if isinstance(output, torch.Tensor):
        if output.dim() == 0:
            return -output.item()
        losses = output.float()
        if losses.dim() >= 2:
            target_losses = losses.view(-1)[max(0, target_start_idx - 1):]
            return -target_losses.mean().item()
        return -losses.mean().item()

    return 0.0


def run_hellaswag_eval(model, limit=100):
    """Run HellaSwag evaluation on the current model.
    
    Args:
        model: Megatron GPT model
        limit: Number of samples to evaluate
        
    Returns:
        dict with 'hellaswag_accuracy', 'hellaswag_correct', 'hellaswag_total', 'hellaswag_time_sec'
    """
    from megatron.core import parallel_state as mpu
    if mpu.get_data_parallel_rank() != 0:
        return None

    samples = _load_hellaswag_data(limit)
    if not samples:
        return None

    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return None

    tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    device = next(model.parameters()).device

    model.eval()
    start_time = time.time()
    correct = 0
    total = 0

    for sample in samples:
        ctx = sample['ctx']
        endings = sample['endings']
        label = sample['label']

        ctx_tokens = tok.encode(ctx)
        best_score = float('-inf')
        best_idx = -1

        for i, ending in enumerate(endings):
            full_text = ctx + " " + ending
            full_tokens = tok.encode(full_text)

            if len(full_tokens) > 126:
                full_tokens = full_tokens[:126]

            input_ids = torch.tensor([full_tokens], dtype=torch.long)
            score = _compute_log_likelihood(model, input_ids, len(ctx_tokens), device)

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == label:
            correct += 1
        total += 1

    model.train()
    elapsed = time.time() - start_time

    accuracy = correct / max(1, total) * 100.0
    print(f"[HELLASWAG] {correct}/{total} = {accuracy:.1f}% ({elapsed:.1f}s)", flush=True)

    return {
        'hellaswag_accuracy': accuracy,
        'hellaswag_correct': correct,
        'hellaswag_total': total,
        'hellaswag_time_sec': elapsed,
    }
