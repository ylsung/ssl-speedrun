"""Representation-geometry callbacks (hypothesis 2 instrumentation).

effective_rank: exp(entropy of normalized singular values) of the hidden-state
matrix — the standard rank-collapse / anisotropy diagnostic (NITP's headline
representation metric). Higher = less degenerate geometry.
"""
import torch


@torch.no_grad()
def effective_rank(h: torch.Tensor, max_rows: int = 4096) -> float:
    """h: (N, D) matrix of hidden states (rows already masked to real tokens)."""
    if h.size(0) > max_rows:
        idx = torch.randperm(h.size(0), device=h.device)[:max_rows]
        h = h[idx]
    h = h.float()
    h = h - h.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(h)
    p = s.pow(2)
    p = p / p.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    return float(ent.exp())


@torch.no_grad()
def layer_effective_ranks(model, batch) -> dict:
    """Effective rank per layer on one batch, keyed 'erank_<layer>'; rows are
    positions where valid_mask is True. Layer indices follow model hiddens:
    0 = embeddings, i = output of block i."""
    _, hiddens = model(batch["tokens"])
    m = batch["valid_mask"]
    return {f"erank_{i}": effective_rank(h[m]) for i, h in enumerate(hiddens)}
