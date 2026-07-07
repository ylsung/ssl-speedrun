"""Generic memmap language-modeling task (Tier 1: TinyStories; later FineWeb).

Expects a directory produced by scripts/prepare_tinystories.py:
    train.bin, val.bin   uint16 token streams
    meta.json            {"vocab_size": ..., ...}

batch(): random contiguous windows from train.bin. All real positions are
supervised (target_mask True except position 0), so the loss plugins behave
exactly as on the synthetic tasks.
evaluate(): mean NTP cross-entropy on fixed windows from val.bin.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LMConfig:
    data_dir: str = "data/tinystories"
    seq_len: int = 512
    val_batches: int = 8
    val_batch_size: int = 32
    seed: int = 0

    @property
    def vocab_size(self):
        with open(os.path.join(self.data_dir, "meta.json")) as f:
            return json.load(f)["vocab_size"]


class LMTask:
    def __init__(self, cfg: LMConfig):
        self.cfg = cfg
        self.train = np.memmap(os.path.join(cfg.data_dir, "train.bin"),
                               dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(cfg.data_dir, "val.bin"),
                             dtype=np.uint16, mode="r")

    def _windows(self, data, batch_size, rng):
        T = self.cfg.seq_len
        starts = rng.integers(0, len(data) - T - 1, size=batch_size)
        return np.stack([np.asarray(data[s:s + T]) for s in starts]).astype(np.int64)

    def batch(self, batch_size: int, rng: np.random.Generator, device="cpu"):
        toks = torch.from_numpy(self._windows(self.train, batch_size, rng)).to(device)
        tmask = torch.ones_like(toks, dtype=torch.bool)
        tmask[:, 0] = False  # position 0 has no predecessor
        return {
            "tokens": toks,
            "target_mask": tmask,
            "valid_mask": torch.ones_like(toks, dtype=torch.bool),
        }

    @torch.no_grad()
    def evaluate(self, model, rng: np.random.Generator, device="cpu"):
        c = self.cfg
        tot, n = 0.0, 0
        for _ in range(c.val_batches):
            toks = torch.from_numpy(
                self._windows(self.val, c.val_batch_size, rng)).to(device)
            logits, _ = model(toks)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1), toks[:, 1:].flatten())
            tot += float(loss) * toks.numel()
            n += toks.numel()
        vl = tot / n
        return {"val_loss": vl, "val_ppl": float(np.exp(vl))}

    @torch.no_grad()
    def sample_text(self, model, device="cpu", n_samples: int = 3,
                    max_new: int = 256):
        """Sample stories from <|endoftext|> (needs tokenizer.json + tokenizers)."""
        try:
            from tokenizers import Tokenizer, decoders
            tok = Tokenizer.from_file(
                os.path.join(self.cfg.data_dir, "tokenizer.json"))
            tok.decoder = decoders.ByteLevel()
        except Exception as e:
            return f"(no samples: {e})"
        with open(os.path.join(self.cfg.data_dir, "meta.json")) as f:
            eot = json.load(f)["eot_id"]
        idx = torch.full((n_samples, 1), eot, dtype=torch.long, device=device)
        out = model.generate(idx, max_new)
        texts = []
        for row in out[:, 1:].tolist():
            if eot in row:
                row = row[:row.index(eot)]
            texts.append(tok.decode(row))
        return "\n\n=====\n\n".join(texts)
