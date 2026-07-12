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

    def __init__(self, weight: float = 1.0, mask_corrupt_src: bool = False, **kw):
        super().__init__()
        self.weight = weight
        # skip loss where the predicting position's input token was corrupted
        # (prediction is ill-posed without the immediate predecessor)
        self.mask_corrupt_src = mask_corrupt_src

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        tokens, mask = batch["tokens"], batch["target_mask"]
        # predict token t from position t-1
        lg = logits[:, :-1]
        tgt = tokens[:, 1:]
        m = mask[:, 1:]
        if self.mask_corrupt_src and batch.get("drop_mask") is not None:
            m = m & ~batch["drop_mask"][:, :-1]
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


def _subsample(x, n=4096):
    if x.size(0) <= n:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:n]
    return x[idx]


def _vicreg_reg(p):
    """VICReg variance hinge + covariance penalty on predictions (N, D).
    Coefficients follow the paper's 25/25/1 ratio relative to the alignment
    term: variance at 1x, covariance at 0.04x."""
    p = p - p.mean(dim=0, keepdim=True)
    std = p.var(dim=0).clamp_min(1e-6).sqrt()
    v = F.relu(1.0 - std).mean()
    n, d = p.shape
    cov = (p.T @ p) / max(n - 1, 1)
    c = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
    return v + 0.04 * c


_SIGREG_T = torch.linspace(0.5, 3.5, 7)


def _sigreg_reg(p, n_dirs=32):
    """SIGReg (LeJEPA): push predictions (N, D) toward an isotropic standard
    Gaussian. Sketch with fresh random unit directions; match the empirical
    characteristic function of each 1-D projection to N(0,1)'s (Epps-Pulley
    style), on a fixed frequency grid weighted by the N(0,1) pdf."""
    N, D = p.shape
    u = torch.randn(D, n_dirs, device=p.device, dtype=p.dtype)
    u = u / u.norm(dim=0, keepdim=True).clamp_min(1e-6)
    proj = p @ u                                     # (N, n_dirs)
    t = _SIGREG_T.to(p.device, p.dtype)              # (T,)
    w = torch.exp(-0.5 * t * t)
    w = w / w.sum()
    tp = proj.unsqueeze(-1) * t                      # (N, n_dirs, T)
    c_emp = torch.cos(tp).mean(dim=0)                # (n_dirs, T)
    s_emp = torch.sin(tp).mean(dim=0)
    c_gauss = torch.exp(-0.5 * t * t)
    err = (c_emp - c_gauss).pow(2) + s_emp.pow(2)
    return (err * w).sum(dim=-1).mean()


class JepaPooledLoss(nn.Module):
    """Position-invariant JEPA arm: from hidden at src_layer, position t
    (optionally mean-pooled over the trailing src_pool tokens — chunk-to-chunk),
    predict the mean-pooled (stop-grad) tgt_layer representation of tokens
    t+1..t+k.

    alignment objective ∈ {cosine, l2, infonce}; optional distribution
    regularizer on predictor outputs ∈ {none, sigreg, vicreg} (LeJEPA-style).
    Predictor head is a 2-layer MLP (asymmetry vs target).
    """
    name = "jepa_pooled"

    def __init__(self, d_model: int, k: int = 8, weight: float = 1.0,
                 src_layer: int = -1, tgt_layer: int = 1,
                 answer_only: bool = False, target: str = "same",
                 objective: str = "cosine", reg: str = "none",
                 reg_weight: float = 1.0, src_pool: int = 1,
                 pool: str = "mean", temperature: float = 0.1,
                 gap: int = 0, n_bits: int = 64,
                 weight_by: str = "none", **kw):
        super().__init__()
        assert objective in ("cosine", "l2", "infonce", "infonce_hard", "lsh")
        assert reg in ("none", "sigreg", "vicreg")
        assert pool in ("mean", "attn")
        assert weight_by in ("none", "entropy")
        self.k = k
        self.weight = weight
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.answer_only = answer_only
        self.target = target
        self.objective = objective
        self.reg = reg
        self.reg_weight = reg_weight
        self.src_pool = src_pool
        self.pool = pool
        self.temperature = temperature
        self.gap = gap
        self.weight_by = weight_by
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        # attention pooling: scorer trains through the loss; pooled *values*
        # stay stop-grad. Shared by target window and (if src_pool>1) source.
        self.scorer = nn.Linear(d_model, 1, bias=False) if pool == "attn" else None
        if objective == "lsh":
            # fixed random hyperplanes; same projection on both sides
            self.register_buffer("lsh_proj",
                                 torch.randn(d_model, n_bits) / d_model ** 0.5)

    def _attn_pool_windows(self, z, start, length):
        """Attention-pool sliding windows of `length` starting at `start`.
        z: (B, T, D) -> (B, T-start-length+1, D)."""
        w = z[:, start:].unfold(1, length, 1)          # (B, n, D, length)
        s = self.scorer(z[:, start:]).squeeze(-1)      # (B, T-start)
        a = s.unfold(1, length, 1).softmax(dim=-1)     # (B, n, length)
        return torch.einsum("bnl,bndl->bnd", a, w)

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        k, c, g = self.k, self.src_pool, self.gap
        src_hiddens = batch.get("src_hiddens") or hiddens
        h_src = src_hiddens[self.src_layer]
        B, T, D = h_src.shape
        # pairing: source at position t (pooled over t-c+1..t when c>1)
        # predicts the pooled target window t+g+1 .. t+g+k.
        n_pairs = T - k - g - (c - 1)
        if n_pairs <= 0:
            return h_src.new_zeros(())
        src_of_tgt = tgt_hiddens if self.target == "ema" else hiddens
        tgt = src_of_tgt[self.tgt_layer].detach()

        # pooled_all[i] = pool of tgt[i+1..i+k], i = 0..T-k-1
        if self.pool == "attn":
            pooled_all = self._attn_pool_windows(tgt.float(), 1, k)
        else:
            cs = tgt.float().cumsum(dim=1)
            pooled_all = (cs[:, k:] - cs[:, :-k]) / k
        # src_arr[m] = source at t = m + c - 1
        if c > 1:
            if self.pool == "attn":
                src_arr = self._attn_pool_windows(h_src, 0, c).to(h_src.dtype)
            else:
                ss = h_src.float().cumsum(dim=1)
                src_arr = torch.cat([ss[:, c - 1:c], ss[:, c:] - ss[:, :-c]], dim=1) / c
                src_arr = src_arr.to(h_src.dtype)
        else:
            src_arr = h_src
        src = src_arr[:, :n_pairs]                        # t = c-1 .. c-1+n_pairs-1
        pooled = pooled_all[:, c - 1 + g: c - 1 + g + n_pairs]
        pred = self.predictor(src)

        # validity: full target window (and, for c>1, full source window) real
        vmask = batch["target_mask"] if self.answer_only else batch["valid_mask"]
        vm = vmask.float().cumsum(dim=1)
        full_all = (vm[:, k:] - vm[:, :-k]) >= (k - 0.5)  # aligned with pooled_all
        full = full_all[:, c - 1 + g: c - 1 + g + n_pairs]
        if c > 1:
            vsrc = torch.cat([vm[:, c - 1:c], vm[:, c:] - vm[:, :-c]], dim=1) >= (c - 0.5)
            full = full & vsrc[:, :n_pairs]
        if full.sum() == 0:
            return h_src.new_zeros(())

        # optional per-position difficulty weights: detached NTP entropy at t
        w = None
        if self.weight_by == "entropy":
            lp = F.log_softmax(logits.detach().float(), dim=-1)
            ent = -(lp.exp() * lp).sum(-1)                # (B, T)
            w = ent[:, c - 1: c - 1 + n_pairs][full]
            w = w / w.mean().clamp_min(1e-6)

        if self.objective == "infonce_hard":
            # negatives = other target windows of the SAME sequence
            pn = F.normalize(pred.float(), dim=-1)
            zn = F.normalize(pooled, dim=-1)
            sim = torch.einsum("bnd,bmd->bnm", pn, zn) / self.temperature
            sim = sim.masked_fill(~full.unsqueeze(1), float("-inf"))
            lbl = torch.arange(n_pairs, device=sim.device).expand(B, -1)
            ce = F.cross_entropy(sim.flatten(0, 1), lbl.flatten(), reduction="none")
            ce = ce.view(B, n_pairs)[full]
            align = (ce * w).mean() / w.mean() if w is not None else ce.mean()
            p = pred.float()[full]
        else:
            p = pred.float()[full]                        # (N, D)
            z = pooled[full]                              # (N, D)
            if self.objective == "cosine":
                per = 1.0 - F.cosine_similarity(p, z, dim=-1)
                align = (per * w).mean() / w.mean() if w is not None else per.mean()
            elif self.objective == "l2":
                per = (p - z).pow(2).mean(-1)
                align = (per * w).mean() / w.mean() if w is not None else per.mean()
            elif self.objective == "lsh":
                # discrete latent codes: sign bits of centered random projections
                mu = z.mean(dim=0, keepdim=True)
                bits = ((z - mu) @ self.lsh_proj > 0).float()
                bit_logits = (p - mu) @ self.lsh_proj
                per = F.binary_cross_entropy_with_logits(
                    bit_logits, bits, reduction="none").mean(-1)
                align = (per * w).mean() / w.mean() if w is not None else per.mean()
            else:  # infonce: in-batch negatives over subsampled pairs
                n = min(p.size(0), 1024)
                idx = torch.randperm(p.size(0), device=p.device)[:n]
                pn = F.normalize(p[idx], dim=-1)
                zn = F.normalize(z[idx], dim=-1)
                sim = pn @ zn.T / self.temperature
                labels = torch.arange(n, device=p.device)
                align = F.cross_entropy(sim, labels)

        loss = align
        if self.reg == "vicreg":
            loss = loss + self.reg_weight * _vicreg_reg(_subsample(p))
        elif self.reg == "sigreg":
            loss = loss + self.reg_weight * _sigreg_reg(_subsample(p))
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
        src_hiddens = batch.get("src_hiddens") or hiddens
        h_src = src_hiddens[self.src_layer]
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


class JepaBeliefLoss(nn.Module):
    """Future-summary latent target from the SAME decoder run right-to-left
    (FSP, arXiv:2510.14751 "learned summaries"; Belief State Transformer,
    arXiv:2410.23506). Round-8 rationale: teacher-forced forward hiddens are
    shaped by NTP and carry mostly short-range features, so predicting them
    (jepa_pooled) inherits NTP's myopia; a reversed-pass state at reversed
    index L-2-t is a causal summary of *only* the suffix x_{t+1..L-1}.

    LossStack runs the shared trunk on the length-reversed sequence (RoPE:
    positions 0..T-1, direction identifiable from content) and supplies
    rev_* tensors. Two terms:
      - ground_w: plain NTP on the reversed student pass (backward LM), so
        reversed inputs are in-distribution and the EMA teacher's reversed
        states are grounded rather than garbage. ground_w>0 with weight=0 is
        the attribution control (backward LM alone, no latent loss).
      - weight: forward hidden at t (predictor MLP) matches the stop-grad
        (default EMA) reversed-pass tgt_layer state summarizing x_{t+1..}.
        answer_only restricts source positions t to those whose suffix
        starts in the answer region (incl. the pre-answer planning token).
    """
    name = "belief"

    def __init__(self, d_model: int, weight: float = 0.5,
                 ground_w: float = 0.5, src_layer: int = -1,
                 tgt_layer: int = 4, answer_only: bool = True,
                 target: str = "ema", objective: str = "cosine",
                 temperature: float = 0.1, **kw):
        super().__init__()
        assert objective in ("cosine", "l2", "infonce")
        self.weight = weight
        self.ground_w = ground_w
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.answer_only = answer_only
        self.target = target
        self.objective = objective
        self.temperature = temperature
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        rev_tokens = batch["rev_tokens"]
        lengths = batch["lengths"]
        B, T = rev_tokens.shape
        ar = torch.arange(T, device=rev_tokens.device)
        rev_valid = ar.unsqueeze(0) < lengths.unsqueeze(1)
        total = logits.new_zeros(())

        if self.ground_w > 0:
            rev_logits = batch["rev_logits"]
            m = rev_valid[:, 1:]
            if m.sum() > 0:
                total = total + self.ground_w * F.cross_entropy(
                    rev_logits[:, :-1][m], rev_tokens[:, 1:][m])

        if self.weight > 0:
            src_hiddens = batch.get("src_hiddens") or hiddens
            h = src_hiddens[self.src_layer]
            rev_h = (batch["rev_tgt_hiddens"] if self.target == "ema"
                     else batch["rev_hiddens"])
            tgt = rev_h[self.tgt_layer].detach().float()
            D = h.size(-1)
            # belief state for source position t lives at reversed index
            # L-2-t (causal summary of x_{t+1..L-1}); needs t <= L-2
            jmap = lengths.unsqueeze(1) - 2 - ar.unsqueeze(0)  # (B, T)
            ok = (jmap >= 0) & batch["valid_mask"]
            if self.answer_only:
                am = torch.zeros_like(ok)
                am[:, :-1] = batch["target_mask"][:, 1:]
                ok = ok & am
            if ok.sum() == 0:
                return total
            z_all = tgt.gather(
                1, jmap.clamp_min(0).unsqueeze(-1).expand(-1, -1, D))
            p = self.predictor(h).float()[ok]
            z = z_all[ok]
            if self.objective == "cosine":
                align = (1.0 - F.cosine_similarity(p, z, dim=-1)).mean()
            elif self.objective == "l2":
                align = (p - z).pow(2).mean()
            else:  # infonce, in-batch negatives over subsampled pairs
                n = min(p.size(0), 1024)
                idx = torch.randperm(p.size(0), device=p.device)[:n]
                pn = F.normalize(p[idx], dim=-1)
                zn = F.normalize(z[idx], dim=-1)
                sim = pn @ zn.T / self.temperature
                align = F.cross_entropy(
                    sim, torch.arange(n, device=p.device))
            total = total + self.weight * align
        return total


class BackchainLoss(nn.Module):
    """Round 9: backward-chaining belief targets (design: ABLATIONS.md §R9).

    Teacher view x̃ = x with the (contiguous) target_mask span reversed in
    place — question forward, answer reversed — so every reversed-answer
    token is predictable by easy local steps (goal copy, then unique-parent
    lookups) and the trunk *computes* the plan while scanning it. LossStack
    runs the student (grad) and EMA teacher (no grad) on x̃ and supplies
    bc_* tensors. Three terms:

      ground_w: NTP on the reversed span (targets x̃_{s+1..e}; the first
        flipped token x̃_s is excluded — position s−1 has an identical
        prefix in both views, so training its prediction toward x̃_s would
        conflict with forward NTP at the planning position).
      mtp_w:    k−1 linear heads read single reversed-pass states and
        predict the next k reversed tokens — the bottleneck that forces
        each state to *contain* the remaining chain (attention bypass
        closed). Learnable via the chain curriculum: late states have
        walked most steps already.
      weight:   forward hidden at t (predictor MLP, cosine/InfoNCE) matches
        the stop-grad EMA reversed-pass tgt_layer state at j(t) = s+e−1−t —
        the state that has just finished back-chaining through exactly the
        suffix x_{t+1..e} that position t still has to generate. Source
        positions t ∈ [s−1, e−1] (answer region incl. the planning token).

    Per-term values are exposed in last_parts (bc_g / bc_m / bc_a) — bc_m
    is the health metric: if it stays at ~log|V| the bottleneck never
    filled. Heads + predictor are train-only; inference is a plain decoder.
    """
    name = "backchain"

    def __init__(self, d_model: int, vocab_size: int, k: int = 8,
                 weight: float = 0.5, ground_w: float = 0.5,
                 mtp_w: float = 0.5, src_layer: int = -1, tgt_layer: int = 4,
                 target: str = "ema", objective: str = "cosine",
                 temperature: float = 0.1, **kw):
        super().__init__()
        assert objective in ("cosine", "infonce")
        self.k = k
        self.weight = weight
        self.ground_w = ground_w
        self.mtp_w = mtp_w
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.target = target
        self.objective = objective
        self.temperature = temperature
        self.heads = nn.ModuleList(
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(k - 1))
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        self.last_parts = {}

    def forward(self, logits, hiddens, batch, tgt_hiddens=None):
        bc_tokens, bc_mask = batch["bc_tokens"], batch["bc_mask"]
        s, e = batch["bc_span"]                        # (B,), (B,) inclusive
        total = logits.new_zeros(())
        self.last_parts = {}

        if self.ground_w > 0:
            bc_logits = batch["bc_logits"]
            m = bc_mask[:, 1:]
            if m.sum() > 0:
                g = F.cross_entropy(bc_logits[:, :-1][m], bc_tokens[:, 1:][m])
                total = total + self.ground_w * g
                self.last_parts["bc_g"] = float(g.detach())

        if self.mtp_w > 0:
            h = batch["bc_hiddens"][-1]
            mtp, n_terms = h.new_zeros(()), 0
            for i, head in enumerate(self.heads):
                off = i + 2
                if h.size(1) <= off:
                    continue
                m = bc_mask[:, off:]
                if m.sum() == 0:
                    continue
                lg = head(h[:, :-off])
                mtp = mtp + F.cross_entropy(lg[m], bc_tokens[:, off:][m])
                n_terms += 1
            if n_terms > 0:
                mtp = mtp / n_terms
                total = total + self.mtp_w * mtp
                self.last_parts["bc_m"] = float(mtp.detach())

        if self.weight > 0:
            h_src = hiddens[self.src_layer]
            B, T, D = h_src.shape
            ar = torch.arange(T, device=h_src.device)
            bc_h = (batch["bc_tgt_hiddens"] if self.target == "ema"
                    else batch["bc_hiddens"])
            tgt = bc_h[self.tgt_layer].detach().float()
            # source positions t with t+1 inside the span
            ok = torch.zeros_like(batch["target_mask"])
            ok[:, :-1] = batch["target_mask"][:, 1:]
            jmap = (s + e - 1).unsqueeze(1) - ar.unsqueeze(0)  # (B, T)
            jmap = jmap.clamp(0, T - 1)
            z_all = tgt.gather(1, jmap.unsqueeze(-1).expand(-1, -1, D))
            if ok.sum() > 0:
                p = self.predictor(h_src).float()[ok]
                z = z_all[ok]
                if self.objective == "cosine":
                    a = (1.0 - F.cosine_similarity(p, z, dim=-1)).mean()
                else:
                    n = min(p.size(0), 1024)
                    idx = torch.randperm(p.size(0), device=p.device)[:n]
                    pn = F.normalize(p[idx], dim=-1)
                    zn = F.normalize(z[idx], dim=-1)
                    sim = pn @ zn.T / self.temperature
                    a = F.cross_entropy(sim, torch.arange(n, device=p.device))
                total = total + self.weight * a
                self.last_parts["bc_a"] = float(a.detach())
        return total


LOSS_REGISTRY = {c.name: c for c in (NTPLoss, MTPLoss, JepaPooledLoss,
                                     JepaPerTokenLoss, JepaBeliefLoss,
                                     BackchainLoss)}


class LossStack(nn.Module):
    """Owns the model + all loss plugins so a single optimizer covers everything.

    If any plugin sets target: ema, an EMA copy of the model provides target
    hiddens (one extra no-grad forward per step); trainer must call
    update_ema() after each optimizer step.
    """

    def __init__(self, model, loss_cfgs: list, ema_decay: float = 0.999,
                 corrupt_p: float = 0.0, corrupt_side: str = "student",
                 teacherless: bool = False):
        super().__init__()
        # student: one corrupted pass, NTP rides it (2 passes total)
        # ema:     corrupt the teacher's input (noisy-target control)
        # latent:  clean pass for NTP + separate corrupted pass that supplies
        #          the latent-loss source hiddens (3 passes, clean attribution)
        assert corrupt_side in ("student", "ema", "latent")
        # teacherless (Bachmann & Nagarajan): replace the *answer-region*
        # inputs with the mask embedding — removes the teacher-forcing crutch
        # while every target stays determined by the clean question. Mutually
        # exclusive with random corruption; eval must decode in parallel.
        assert not (teacherless and corrupt_p > 0)
        self.model = model
        self.ema_decay = ema_decay
        self.corrupt_p = corrupt_p
        self.corrupt_side = corrupt_side
        self.teacherless = teacherless
        plugins = []
        for cfg in loss_cfgs:
            cfg = dict(cfg)
            kind = cfg.pop("kind")
            cls = LOSS_REGISTRY[kind]
            plugins.append(cls(d_model=model.cfg.d_model,
                               vocab_size=model.cfg.vocab_size, **cfg))
        self.plugins = nn.ModuleList(plugins)
        self.needs_reverse = any(p.name == "belief" for p in plugins)
        self.needs_backchain = any(p.name == "backchain" for p in plugins)
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
        drop = None
        if self.teacherless:
            drop = batch["target_mask"] & batch["valid_mask"]
        elif self.corrupt_p > 0 and self.training:
            drop = (torch.rand_like(batch["tokens"], dtype=torch.float)
                    < self.corrupt_p) & batch["valid_mask"]
            drop[:, 0] = False  # never corrupt BOS
        logits, hiddens = self.model(
            batch["tokens"], drop if self.corrupt_side == "student" else None)
        src_hiddens = None
        if drop is not None and self.corrupt_side == "latent":
            _, src_hiddens = self.model(batch["tokens"], drop)
        tgt_hiddens = None
        if self.ema_model is not None:
            with torch.no_grad():
                _, tgt_hiddens = self.ema_model(
                    batch["tokens"],
                    drop if self.corrupt_side == "ema" else None)
        batch = {**batch, "drop_mask": drop, "src_hiddens": src_hiddens}
        if self.needs_reverse:
            # per-row flip of the valid region (PAD tail stays in place);
            # reversed tokens sit at RoPE positions 0..T-1 — only relative
            # offsets exist, direction is identifiable from content
            tokens = batch["tokens"]
            lengths = batch["valid_mask"].sum(dim=1)
            ar = torch.arange(tokens.size(1), device=tokens.device)
            ridx = torch.where(ar.unsqueeze(0) < lengths.unsqueeze(1),
                               (lengths.unsqueeze(1) - 1 - ar).clamp_min(0),
                               ar.unsqueeze(0).expand(tokens.size(0), -1))
            rev_tokens = tokens.gather(1, ridx)
            rev_logits, rev_hiddens = self.model(rev_tokens)
            rev_tgt_hiddens = None
            if self.ema_model is not None:
                with torch.no_grad():
                    _, rev_tgt_hiddens = self.ema_model(rev_tokens)
            batch.update(rev_tokens=rev_tokens, rev_logits=rev_logits,
                         rev_hiddens=rev_hiddens,
                         rev_tgt_hiddens=rev_tgt_hiddens, lengths=lengths)
        if self.needs_backchain:
            # teacher view x̃: flip the contiguous target_mask span in place
            # (question forward, answer reversed); see BackchainLoss docstring
            tokens, tmask = batch["tokens"], batch["target_mask"]
            B, T = tokens.shape
            ar = torch.arange(T, device=tokens.device)
            s = tmask.int().argmax(dim=1)                       # first True
            e = s + tmask.sum(dim=1) - 1                        # last True
            midx = torch.where(tmask, (s + e).unsqueeze(1) - ar.unsqueeze(0),
                               ar.unsqueeze(0).expand(B, -1))
            bc_tokens = tokens.gather(1, midx)
            # CE/MTP targets exclude the first flipped token x̃_s (position
            # s−1 has an identical prefix in both views — see docstring)
            bc_mask = tmask & (ar.unsqueeze(0) > s.unsqueeze(1))
            bc_logits, bc_hiddens = self.model(bc_tokens)
            bc_tgt_hiddens = None
            if self.ema_model is not None:
                with torch.no_grad():
                    _, bc_tgt_hiddens = self.ema_model(bc_tokens)
            batch.update(bc_tokens=bc_tokens, bc_mask=bc_mask,
                         bc_logits=bc_logits, bc_hiddens=bc_hiddens,
                         bc_tgt_hiddens=bc_tgt_hiddens, bc_span=(s, e))
        parts = {}
        total = logits.new_zeros(())
        for p in self.plugins:
            l = p(logits, hiddens, batch, tgt_hiddens)
            parts[p.name] = float(l.detach())
            if getattr(p, "last_parts", None):
                parts.update(p.last_parts)
            total = total + l
        return total, parts
