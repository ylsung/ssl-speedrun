"""Star-graph path-finding task (Bachmann & Nagarajan 2024, arXiv:2403.06963).

G(d, n): a center node with d arms of length n; exactly one arm ends at the goal.
Sequence:  BOS  (u v ESEP)*edges  QRY start goal EQ  path...  EOS  PAD*
The edge list is shuffled. Supervision (target_mask) covers only path+EOS.

Why NTP fails here: the only hard decision is the first node after the center
(requires planning down the arms); teacher forcing lets the model learn the
easy "follow the chain" behavior and shortcut the hard token. We therefore
report both full-path accuracy and *decision accuracy* (that first hard token).
"""
from dataclasses import dataclass

import numpy as np
import torch

# special tokens
PAD, BOS, ESEP, QRY, EQ, EOS = 0, 1, 2, 3, 4, 5
N_SPECIAL = 6


@dataclass
class StarGraphConfig:
    d: int = 5            # number of arms
    n: int = 5            # arm length (edges per arm)
    num_nodes: int = 50   # node-id pool size (>= d*n + 1)
    seed: int = 0
    # round-10 diagnostic: emit the answer goal→center (EOS stays last).
    # Every step is then deterministic-easy (goal is in the query; each next
    # node is the unique parent toward the center) — pure NTP should solve
    # this ~perfectly if backward order is learnable at all.
    reverse_path: bool = False

    @property
    def vocab_size(self):
        return N_SPECIAL + self.num_nodes

    @property
    def seq_len(self):
        # BOS + 3*edges + (QRY,start,goal,EQ) + path(n+1) + EOS
        return 1 + 3 * self.d * self.n + 4 + (self.n + 1) + 1


class StarGraphTask:
    def __init__(self, cfg: StarGraphConfig):
        assert cfg.num_nodes >= cfg.d * cfg.n + 1, "node pool too small"
        self.cfg = cfg

    def _sample(self, rng: np.random.Generator):
        c = self.cfg
        ids = rng.choice(c.num_nodes, size=c.d * c.n + 1, replace=False) + N_SPECIAL
        center, rest = ids[0], ids[1:].reshape(c.d, c.n)
        edges = []
        for a in range(c.d):
            prev = center
            for x in rest[a]:
                edges.append((prev, x))
                prev = x
        rng.shuffle(edges)
        goal_arm = rng.integers(c.d)
        path = [center] + list(rest[goal_arm])
        goal = path[-1]

        seq = [BOS]
        for u, v in edges:
            seq += [u, v, ESEP]
        seq += [QRY, center, goal, EQ]
        prefix_len = len(seq)
        ans = path[::-1] if self.cfg.reverse_path else path
        seq += ans + [EOS]
        return seq, prefix_len, path

    def batch(self, batch_size: int, rng: np.random.Generator, device="cpu"):
        c = self.cfg
        T = c.seq_len
        toks = np.full((batch_size, T), PAD, dtype=np.int64)
        tmask = np.zeros((batch_size, T), dtype=bool)
        vmask = np.zeros((batch_size, T), dtype=bool)
        prefix_lens = np.zeros(batch_size, dtype=np.int64)
        paths = []
        for b in range(batch_size):
            seq, plen, path = self._sample(rng)
            toks[b, :len(seq)] = seq
            vmask[b, :len(seq)] = True
            tmask[b, plen:len(seq)] = True   # path tokens + EOS
            prefix_lens[b] = plen
            paths.append(path)
        return {
            "tokens": torch.from_numpy(toks).to(device),
            "target_mask": torch.from_numpy(tmask).to(device),
            "valid_mask": torch.from_numpy(vmask).to(device),
            "prefix_len": torch.from_numpy(prefix_lens).to(device),
            "paths": paths,
        }

    @torch.no_grad()
    def _decode_with_thoughts(self, model, prefix, n_new, m, mode):
        """prefix + BOT + m thought slots + EOT, then greedy decode n_new
        tokens. BOT/EOT ids sit just past the task vocab (trainer contract).
        continuous: slots are filled by the norm_f(last-hidden) rollout and
        injected as input embeddings; pause: slots are literal BOT tokens."""
        BOT = self.cfg.vocab_size
        EOT = BOT + 1
        B = prefix.size(0)
        s = prefix.size(1)
        D = model.cfg.d_model
        cur = torch.cat([prefix, prefix.new_full((B, m + 2), BOT)], dim=1)
        cur[:, -1] = EOT
        th = []
        if mode == "continuous":
            for j in range(m):
                L = s + 1 + j
                inj = None
                if th:
                    ie = torch.zeros(B, L, D, device=cur.device,
                                     dtype=th[0].dtype)
                    ie[:, s + 1:s + 1 + len(th)] = torch.stack(th, dim=1)
                    im = torch.zeros(B, L, dtype=torch.bool, device=cur.device)
                    im[:, s + 1:s + 1 + len(th)] = True
                    inj = (im, ie)
                _, hid = model(cur[:, :L], inject=inj)
                th.append(model.norm_f(hid[-1][:, -1]))
        outs = []
        for _ in range(n_new):
            inj = None
            if th:
                T = cur.size(1)
                ie = torch.zeros(B, T, D, device=cur.device, dtype=th[0].dtype)
                ie[:, s + 1:s + 1 + m] = torch.stack(th, dim=1)
                im = torch.zeros(B, T, dtype=torch.bool, device=cur.device)
                im[:, s + 1:s + 1 + m] = True
                inj = (im, ie)
            logits, _ = model(cur, inject=inj)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def evaluate(self, model, rng: np.random.Generator, n_batches: int = 4,
                 batch_size: int = 64, device="cpu", teacherless: bool = False,
                 thoughts: int = 0, thought_mode: str = "continuous"):
        """Greedy-decode from the prefix; fixed-length task so prefix_len is
        constant and we can decode the whole batch at once. teacherless:
        parallel decode instead — one forward pass with the answer-region
        inputs replaced by the mask embedding (matching training; gold answer
        tokens never enter the model), predictions read off positions
        plen-1 .. plen+n_new-2."""
        c = self.cfg
        n_new = c.n + 2  # path (n+1 tokens) + EOS
        tot = full_ok = dec_ok = 0
        for _ in range(n_batches):
            batch = self.batch(batch_size, rng, device)
            plen = int(batch["prefix_len"][0])
            if thoughts:
                out = self._decode_with_thoughts(
                    model, batch["tokens"][:, :plen], n_new, thoughts,
                    thought_mode)
            elif teacherless:
                drop = batch["target_mask"] & batch["valid_mask"]
                logits, _ = model(batch["tokens"], drop)
                out = logits[:, plen - 1:plen - 1 + n_new].argmax(dim=-1)
            else:
                prefix = batch["tokens"][:, :plen]
                out = model.generate_greedy(prefix, n_new)[:, plen:]
            gold = batch["tokens"][:, plen:plen + n_new]
            full_ok += (out == gold).all(dim=1).sum().item()
            # decision token = first node after the center = 2nd path token
            dec_ok += (out[:, 1] == gold[:, 1]).sum().item()
            tot += batch_size
        return {"path_acc": full_ok / tot, "decision_acc": dec_ok / tot}


if __name__ == "__main__":
    task = StarGraphTask(
        StarGraphConfig(n=5, d=5, num_nodes=50, seed=0)
    )
    
    task._sample(np.random.default_rng(0))
    
    breakpoint()
    