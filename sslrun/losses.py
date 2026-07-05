"""Loss plugins. Every method in the benchmark = a list of these in a YAML config.

Contract: plugin(logits, hiddens, batch) -> scalar loss (already weighted).
batch dict: tokens (B,T) long, target_mask (B,T) bool - True at positions whose
*token is a supervised target* (answer region incl. EOS), pad excluded.

Conventions:
- NTP: logits at t-1 predict token t where target_mask[t].
- Plugins may own parameters (heads); LossStack registers them for the optimizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTPLoss(nn.Module):
    name = "ntp"

    def __init__(self, weight: float = 1.0, **kw):
        super().__init__()
        self.weight = weight

    def forward(self, logits, hiddens, batch):
        tokens, mask = batch["tokens"], batch["target_mask"]
        # predict token t from position t-1
        lg = logits[:, :-1]
        tgt = tokens[:, 1:]
        m = mask[:, 1:]
        if m.sum() == 0:
            return logits.new_zeros(())
        loss = F.cross_entropy(lg[m], tgt[m])
        return self.weight * loss


class MTPLoss(nn.Module):
    """Discrete multi-token prediction: extra linear heads predict tokens t+2..t+k
    from the last hidden state at position t (offset 1 is the ordinary LM head)."""
    name = "mtp"

    def __init__(self, d_model: int, vocab_size: int, k: int = 4,
                 weight: float = 1.0, src_layer: int = -1, **kw):
        super().__init__()
        assert k >= 2
        self.k = k
        self.weight = weight
        self.src_layer = src_layer
        self.heads = nn.ModuleList(
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(k - 1))

    def forward(self, logits, hiddens, batch):
        tokens, mask = batch["tokens"], batch["target_mask"]
        h = hiddens[self.src_layer]
        total, n_terms = h.new_zeros(()), 0
        for i, head in enumerate(self.heads):
            off = i + 2  # predict token t+off from position t
            if h.size(1) <= off:
                continue
            lg = head(h[:, :-off])
            tgt = tokens[:, off:]
            m = mask[:, off:]
            if m.sum() == 0:
                continue
            total = total + F.cross_entropy(lg[m], tgt[m])
            n_terms += 1
        if n_terms == 0:
            return h.new_zeros(())
        return self.weight * total / n_terms


class JepaPooledLoss(nn.Module):
    """Position-invariant JEPA arm: from hidden at src_layer, position t, predict
    the mean-pooled (stop-grad) tgt_layer representation of tokens t+1..t+k.

    Loss = 1 - cosine similarity, averaged over positions whose full window is
    real (non-pad) tokens. Predictor head is a 2-layer MLP (asymmetry vs target).
    """
    name = "jepa_pooled"

    def __init__(self, d_model: int, k: int = 8, weight: float = 1.0,
                 src_layer: int = -1, tgt_layer: int = 1,
                 answer_only: bool = False, **kw):
        super().__init__()
        self.k = k
        self.weight = weight
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.answer_only = answer_only
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

    def forward(self, logits, hiddens, batch):
        k = self.k
        h_src = hiddens[self.src_layer]
        B, T, D = h_src.shape
        if T <= k:
            return h_src.new_zeros(())
        tgt = hiddens[self.tgt_layer].detach()

        # mean over window t+1..t+k via cumsum: pooled[t] = (cs[t+k]-cs[t])/k
        cs = tgt.float().cumsum(dim=1)
        pooled = (cs[:, k:] - cs[:, :-k]) / k        # (B, T-k, D), aligned with t=0..T-k-1
        pred = self.predictor(h_src[:, :T - k])       # (B, T-k, D)

        # validity: every token in the window must be a real (non-pad) token;
        # optionally restrict to windows fully inside the answer region.
        vmask = batch["target_mask"] if self.answer_only else batch["valid_mask"]
        vm = vmask.float().cumsum(dim=1)
        full = (vm[:, k:] - vm[:, :-k]) >= (k - 0.5)  # all k window positions valid
        if full.sum() == 0:
            return h_src.new_zeros(())

        cos = F.cosine_similarity(pred.float(), pooled, dim=-1)  # (B, T-k)
        loss = (1.0 - cos)[full].mean()
        return self.weight * loss


LOSS_REGISTRY = {c.name: c for c in (NTPLoss, MTPLoss, JepaPooledLoss)}


class LossStack(nn.Module):
    """Owns the model + all loss plugins so a single optimizer covers everything."""

    def __init__(self, model, loss_cfgs: list):
        super().__init__()
        self.model = model
        plugins = []
        for cfg in loss_cfgs:
            cfg = dict(cfg)
            kind = cfg.pop("kind")
            cls = LOSS_REGISTRY[kind]
            plugins.append(cls(d_model=model.cfg.d_model,
                               vocab_size=model.cfg.vocab_size, **cfg))
        self.plugins = nn.ModuleList(plugins)

    def forward(self, batch):
        logits, hiddens = self.model(batch["tokens"])
        parts = {}
        total = logits.new_zeros(())
        for p in self.plugins:
            l = p(logits, hiddens, batch)
            parts[p.name] = float(l.detach())
            total = total + l
        return total, parts
