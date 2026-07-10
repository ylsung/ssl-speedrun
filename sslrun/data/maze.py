"""Maze path-generation task (Tier 0).

A perfect maze on a W x H grid (randomized-DFS spanning tree), so the path
between any two cells is unique. Sequence layout mirrors stargraph:

    BOS  (u v ESEP)*open_edges  QRY start goal EQ  path...  EOS  PAD*

The edge list is shuffled; supervision (target_mask) covers path + EOS.
Hard tokens are the junction decisions: cells where >1 open neighbor
(excluding the cell we came from) is available. `decision_acc` scores the
first such junction on each path; forced corridor steps are trivial NTP.

Synonym rendering (round 5): with synonyms=s > 1 each cell has s surface
tokens and every occurrence samples one i.i.d., so the emission is
many-to-one — token-space CE has an irreducible floor of log(s) per content
token while the latent cell sequence stays deterministic. syn_scope="answer"
restricts sampling to the supervised region (prefix stays canonical).
Accuracy is scored on equivalence classes (any synonym of the right cell
counts). With synonyms=1 the task and rng stream are bit-identical to the
original, so old runs remain valid s=1 anchors.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .stargraph import PAD, BOS, ESEP, QRY, EQ, EOS, N_SPECIAL


@dataclass
class MazeConfig:
    width: int = 6
    height: int = 6
    min_path: int = 6     # resample start/goal until path has >= this many cells
    seed: int = 0
    synonyms: int = 1     # surface tokens per cell; 1 = original task
    syn_scope: str = "all"  # "all" | "answer" (prefix stays canonical)

    @property
    def n_cells(self):
        return self.width * self.height

    @property
    def vocab_size(self):
        return N_SPECIAL + self.n_cells * self.synonyms

    @property
    def max_path(self):
        return self.n_cells  # unique simple path can visit every cell at most once

    @property
    def seq_len(self):
        n_edges = self.n_cells - 1  # spanning tree
        return 1 + 3 * n_edges + 4 + self.max_path + 1

    @property
    def prefix_len(self):
        return 1 + 3 * (self.n_cells - 1) + 4  # constant: BOS + edges + query


class MazeTask:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg

    def _neighbors(self, c):
        W, H = self.cfg.width, self.cfg.height
        r, col = divmod(c, W)
        if r > 0:
            yield c - W
        if r < H - 1:
            yield c + W
        if col > 0:
            yield c - 1
        if col < W - 1:
            yield c + 1

    def _gen_maze(self, rng: np.random.Generator):
        """Randomized DFS spanning tree; returns adjacency sets over cells."""
        n = self.cfg.n_cells
        adj = [set() for _ in range(n)]
        visited = np.zeros(n, dtype=bool)
        stack = [int(rng.integers(n))]
        visited[stack[0]] = True
        while stack:
            c = stack[-1]
            nbrs = [x for x in self._neighbors(c) if not visited[x]]
            if not nbrs:
                stack.pop()
                continue
            x = nbrs[int(rng.integers(len(nbrs)))]
            adj[c].add(x)
            adj[x].add(c)
            visited[x] = True
            stack.append(x)
        return adj

    def _path(self, adj, start, goal):
        """Unique tree path start -> goal via DFS parent pointers."""
        n = self.cfg.n_cells
        parent = np.full(n, -1, dtype=np.int64)
        stack, seen = [start], {start}
        while stack:
            c = stack.pop()
            if c == goal:
                break
            for x in adj[c]:
                if x not in seen:
                    seen.add(x)
                    parent[x] = c
                    stack.append(x)
        path = [goal]
        while path[-1] != start:
            path.append(int(parent[path[-1]]))
        return path[::-1]

    def _sample(self, rng: np.random.Generator):
        c = self.cfg
        adj = self._gen_maze(rng)
        while True:
            start, goal = rng.choice(c.n_cells, size=2, replace=False)
            path = self._path(adj, int(start), int(goal))
            if len(path) >= c.min_path:
                break
        # first junction decision: >1 open next option (excluding where we came from)
        dec_idx = None
        for i in range(len(path) - 1):
            options = adj[path[i]] - ({path[i - 1]} if i > 0 else set())
            if len(options) > 1:
                dec_idx = i + 1  # position (within path) of the chosen next cell
                break

        edges = []
        for u in range(c.n_cells):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        rng.shuffle(edges)

        s = c.synonyms

        def tok(x, ans=False):
            j = 0
            if s > 1 and (ans or c.syn_scope == "all"):
                j = int(rng.integers(s))
            return N_SPECIAL + x * s + j

        seq = [BOS]
        for u, v in edges:
            seq += [tok(u), tok(v), ESEP]
        seq += [QRY, tok(path[0]), tok(path[-1]), EQ]
        seq += [tok(x, ans=True) for x in path] + [EOS]
        return seq, len(path), dec_idx

    def batch(self, batch_size: int, rng: np.random.Generator, device="cpu"):
        c = self.cfg
        T, plen = c.seq_len, c.prefix_len
        toks = np.full((batch_size, T), PAD, dtype=np.int64)
        tmask = np.zeros((batch_size, T), dtype=bool)
        vmask = np.zeros((batch_size, T), dtype=bool)
        plens = np.full(batch_size, plen, dtype=np.int64)
        path_lens = np.zeros(batch_size, dtype=np.int64)
        dec_idxs = np.full(batch_size, -1, dtype=np.int64)
        for b in range(batch_size):
            seq, plen_path, dec = self._sample(rng)
            toks[b, :len(seq)] = seq
            vmask[b, :len(seq)] = True
            tmask[b, plen:len(seq)] = True
            path_lens[b] = plen_path
            if dec is not None:
                dec_idxs[b] = dec
        return {
            "tokens": torch.from_numpy(toks).to(device),
            "target_mask": torch.from_numpy(tmask).to(device),
            "valid_mask": torch.from_numpy(vmask).to(device),
            "prefix_len": torch.from_numpy(plens).to(device),
            "path_len": torch.from_numpy(path_lens).to(device),
            "decision_idx": torch.from_numpy(dec_idxs).to(device),
        }

    def _classes(self, t):
        """Map surface tokens to equivalence classes (identity for specials)."""
        s = self.cfg.synonyms
        if s == 1:
            return t
        return torch.where(t >= N_SPECIAL,
                           torch.div(t - N_SPECIAL, s, rounding_mode="floor")
                           + N_SPECIAL, t)

    @torch.no_grad()
    def evaluate(self, model, rng: np.random.Generator, n_batches: int = 4,
                 batch_size: int = 64, device="cpu"):
        c = self.cfg
        n_new = c.max_path + 1
        tot = full_ok = 0
        dec_tot = dec_ok = 0
        for _ in range(n_batches):
            batch = self.batch(batch_size, rng, device)
            plen = c.prefix_len
            out = model.generate_greedy(batch["tokens"][:, :plen], n_new)[:, plen:]
            gold = batch["tokens"][:, plen:plen + n_new]
            out, gold = self._classes(out), self._classes(gold)
            L = batch["path_len"]  # (B,) path cells; +1 for EOS
            pos = torch.arange(n_new, device=out.device).unsqueeze(0)
            m = pos < (L + 1).unsqueeze(1)
            full_ok += ((out == gold) | ~m).all(dim=1).sum().item()
            d = batch["decision_idx"]
            has_dec = d >= 0
            di = d.clamp_min(0).unsqueeze(1)
            dec_ok += (out.gather(1, di) == gold.gather(1, di)).squeeze(1)[has_dec].sum().item()
            dec_tot += has_dec.sum().item()
            tot += batch_size
        res = {"path_acc": full_ok / tot,
               "decision_acc": dec_ok / max(dec_tot, 1)}
        if c.synonyms > 1:
            # invariance probe: are synonyms of a cell closer in embedding
            # space than tokens of different cells?
            E = model.tok_emb.weight[N_SPECIAL:N_SPECIAL + c.n_cells * c.synonyms]
            E = torch.nn.functional.normalize(E, dim=-1)
            G = E @ E.T
            cls = torch.arange(c.n_cells, device=G.device
                               ).repeat_interleave(c.synonyms)
            same = cls.unsqueeze(0) == cls.unsqueeze(1)
            eye = torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
            res["syn_cos_within"] = G[same & ~eye].mean().item()
            res["syn_cos_between"] = G[~same].mean().item()
        return res
