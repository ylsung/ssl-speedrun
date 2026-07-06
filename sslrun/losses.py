"""Loss plugins. Every method in the benchmark = a list of these in a YAML config.

Contract: plugin(logits, hiddens, batch, tgt_hiddens) -> scalar loss (already
weighted). tgt_hiddens are the hidden states latent targets are drawn from:
the same-pass hiddens by default, or an EMA teacher's if the plugin sets
target: ema (LossStack then maintains the EMA copy; trainer calls update_ema()
after each optimizer step).

batch dict: tokens (B,T) long, target_mask (B,T) bool - True at positions whose
*token is a supervised target* (answer region incl. EOS), pad excluded.

Conventions:
- NTP: logits at t-1 predict token t where target_mask[t].
- Plugins may own parameters (heads); LossStack registers them for the optimizer.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTPLoss(nn.Module):
    name = "ntp"

    def __init__(self, weight: float = 1.0, **kw):
        super().__init__()
        self.weight = weight

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
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

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
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
                 answer_only: bool = False, target: str = "same", **kw):
        super().__init__()
        self.k = k
        self.weight = weight
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.answer_only = answer_only
        self.target = target
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        k = self.k
        h_src = hiddens[self.src_layer]
        B, T, D = h_src.shape
        if T <= k:
            return h_src.new_zeros(())
        src_of_tgt = tgt_hiddens if self.target == "ema" else hiddens
        tgt = src_of_tgt[self.tgt_layer].detach()

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


class JepaPerTokenLoss(nn.Module):
    """Position-indexed JEPA arm: from hidden at src_layer, position t, predict
    the (stop-grad) tgt_layer representation of each of tokens t+1..t+M.
    M=1 is the NITP repro; M>1 is the short-horizon rung of the pyramid.

    Predictor: shared expansion + per-offset output heads (asymmetry vs target,
    parameters O(M) in the cheap output layer only).
    """
    name = "jepa_pertoken"

    def __init__(self, d_model: int, m: int = 1, weight: float = 1.0,
                 src_layer: int = -1, tgt_layer: int = 1,
                 answer_only: bool = False, target: str = "same", **kw):
        super().__init__()
        assert m >= 1
        self.m = m
        self.weight = weight
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.answer_only = answer_only
        self.target = target
        self.shared = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False), nn.GELU())
        self.heads = nn.ModuleList(
            nn.Linear(2 * d_model, d_model, bias=False) for _ in range(m))

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        h_src = hiddens[self.src_layer]
        T = h_src.size(1)
        src_of_tgt = tgt_hiddens if self.target == "ema" else hiddens
        tgt = src_of_tgt[self.tgt_layer].detach().float()
        vmask = batch["target_mask"] if self.answer_only else batch["valid_mask"]

        z = self.shared(h_src)
        total, n_terms = h_src.new_zeros(()), 0
        for i, head in enumerate(self.heads):
            off = i + 1  # predict z_{t+off} from position t
            if T <= off:
                continue
            m = vmask[:, :-off] & vmask[:, off:]  # src and target both real
            if m.sum() == 0:
                continue
            pred = head(z[:, :-off])
            cos = F.cosine_similarity(pred.float(), tgt[:, off:], dim=-1)
            total = total + (1.0 - cos)[m].mean()
            n_terms += 1
        if n_terms == 0:
            return h_src.new_zeros(())
        return self.weight * total / n_terms


LOSS_REGISTRY = {c.name: c for c in (NTPLoss, MTPLoss, JepaPooledLoss,
                                     JepaPerTokenLoss)}


class LossStack(nn.Module):
    """Owns the model + all loss plugins so a single optimizer covers everything.

    If any plugin sets target: ema, an EMA copy of the model provides target
    hiddens (one extra no-grad forward per step); trainer must call
    update_ema() after each optimizer step.
    """

    def __init__(self, model, loss_cfgs: list, ema_decay: float = 0.999):
        super().__init__()
        self.model = model
        self.ema_decay = ema_decay
        plugins = []
        for cfg in loss_cfgs:
            cfg = dict(cfg)
            kind = cfg.pop("kind")
            cls = LOSS_REGISTRY[kind]
            plugins.append(cls(d_model=model.cfg.d_model,
                               vocab_size=model.cfg.vocab_size, **cfg))
        self.plugins = nn.ModuleList(plugins)
        self.ema_model = None
        if any(getattr(p, "target", "same") == "ema" for p in plugins):
            self.ema_model = copy.deepcopy(model).requires_grad_(False)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_model is None:
            return
        d = self.ema_decay
        for pe, pm in zip(self.ema_model.parameters(), self.model.parameters()):
            pe.lerp_(pm, 1.0 - d)
        for be, bm in zip(self.ema_model.buffers(), self.model.buffers()):
            be.copy_(bm)

    def forward(self, batch):
        logits, hiddens = self.model(batch["tokens"])
        tgt_hiddens = None
        if self.ema_model is not None:
            with torch.no_grad():
                _, tgt_hiddens = self.ema_model(batch["tokens"])
        parts = {}
        total = logits.new_zeros(())
        for p in self.plugins:
            l = p(logits, hiddens, batch, tgt_hiddens)
            parts[p.name] = float(l.detach())
            total = total + l
        return total, parts
