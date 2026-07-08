"""Compact GPT-style decoder that exposes all layer hidden states.

Kept deliberately simple (AdamW-friendly, no fused tricks) for the MVP;
speedrun optimizations (Muon etc.) come later, behind the same interface.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    max_seq_len: int = 256
    tie_weights: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.mask_emb = nn.Parameter(0.02 * torch.randn(cfg.d_model))  # corruption token
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = _rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, drop_mask=None):
        """Returns (logits, hiddens). hiddens[0] = embedding output,
        hiddens[i] = output of block i (pre final norm), len = n_layer + 1.
        drop_mask (B, T) bool: replace those positions' input embeddings with
        the learned mask embedding (corruption for denoising-style
        objectives; a zero-out was numerically unstable through RMSNorm)."""
        x = self.tok_emb(idx)
        if drop_mask is not None:
            m = drop_mask.unsqueeze(-1)
            x = torch.where(m, self.mask_emb.to(x.dtype).expand_as(x), x)
        hiddens = [x]
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
            hiddens.append(x)
        logits = self.lm_head(self.norm_f(x))
        return logits, hiddens

    @torch.no_grad()
    def generate_greedy(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 50):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            lg = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                kth = lg.topk(min(top_k, lg.size(-1)), dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
