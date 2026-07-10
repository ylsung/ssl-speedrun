"""Grid-embedding task (round 7): inverse maze layout.

Input: the shuffled edge list of a random maze tree whose vertices carry
random labels (fresh permutation per sample, so labels leak no coordinates).
Output: the full maze drawing as a (2H-1) x (2W-1) token map — vertex labels
at (2r, 2c), E(dge)/W(all) at the slots between them, C(orner) filler —
emitted row-major:

    BOS  (u v ESEP)*edges  QRY EQ  map...  EOS

The target layout is rendered from a random D4 perspective (4 rotations x
reflection) each sample, so the same input admits 8 shown answers (and many
more valid re-folds the data never shows). Eval is by *verification*, not
gold-matching: vertex slots must form a permutation, the stated E set must
equal the input edge set exactly (edges present AND no additional edges),
and structural slots must be C. Metrics: perm_valid, edge_f1, embed_acc.

Solving this is global constraint satisfaction — no copy shortcut from the
input, and locally consistent placements can be globally unsatisfiable, so
instances can require lookahead. Eval mazes are a held-out set: training
resamples any maze whose (geometric) edge set collides with the eval set.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .stargraph import PAD, BOS, ESEP, QRY, EQ, EOS

E_TOK = 6   # edge
W_TOK = 7   # wall (adjacent but no edge)
C_TOK = 8   # corner filler
N_SPECIAL = 9


@dataclass
class GridEmbedConfig:
    width: int = 4
    height: int = 4
    seed: int = 0
    n_eval: int = 512   # held-out mazes for evaluation

    @property
    def n_cells(self):
        return self.width * self.height

    @property
    def vocab_size(self):
        return N_SPECIAL + self.n_cells

    @property
    def map_h(self):
        return 2 * self.height - 1

    @property
    def map_w(self):
        return 2 * self.width - 1

    @property
    def map_len(self):
        return self.map_h * self.map_w

    @property
    def prefix_len(self):
        return 1 + 3 * (self.n_cells - 1) + 2  # BOS + edges + QRY EQ

    @property
    def seq_len(self):
        return self.prefix_len + self.map_len + 1  # + EOS


class GridEmbedTask:
    def __init__(self, cfg: GridEmbedConfig):
        self.cfg = cfg
        # held-out eval mazes, generated from a stream training never uses
        rng = np.random.default_rng(cfg.seed + 77_000)
        self.eval_adjs, self.eval_hashes = [], set()
        attempts = 0
        while len(self.eval_adjs) < cfg.n_eval and attempts < 50 * cfg.n_eval:
            adj = self._gen_maze(rng)
            h = self._hash(adj)
            attempts += 1
            if h not in self.eval_hashes:
                self.eval_hashes.add(h)
                self.eval_adjs.append(adj)

    # -- maze generation (same randomized-DFS tree as maze.py) --------------
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

    def _hash(self, adj):
        return tuple(sorted((u, v) for u in range(self.cfg.n_cells)
                            for v in adj[u] if u < v))

    # -- rendering -----------------------------------------------------------
    def _render(self, adj, rng: np.random.Generator):
        """One training sequence: random labels, shuffled edges, random D4
        perspective. Returns (seq,)."""
        c = self.cfg
        W, H, n = c.width, c.height, c.n_cells
        lab = rng.permutation(n)  # cell -> label
        edges = [(int(lab[u]), int(lab[v])) for u in range(n)
                 for v in adj[u] if u < v]
        edges = [(b, a) if rng.integers(2) else (a, b) for a, b in edges]
        rng.shuffle(edges)
        eset = {frozenset(e) for e in edges}
        # random perspective on the label grid
        g = np.array([int(lab[i]) for i in range(n)]).reshape(H, W)
        g = np.rot90(g, int(rng.integers(4)))
        if rng.integers(2):
            g = np.fliplr(g)
        g = np.ascontiguousarray(g)  # square grids: shape preserved

        tok = lambda x: N_SPECIAL + x
        seq = [BOS]
        for u, v in edges:
            seq += [tok(u), tok(v), ESEP]
        seq += [QRY, EQ]
        gh, gw = g.shape
        for r in range(2 * gh - 1):
            for col in range(2 * gw - 1):
                if r % 2 == 0 and col % 2 == 0:
                    seq.append(tok(int(g[r // 2, col // 2])))
                elif r % 2 == 1 and col % 2 == 1:
                    seq.append(C_TOK)
                else:
                    if r % 2 == 0:  # horizontal slot
                        a, b = g[r // 2, (col - 1) // 2], g[r // 2, (col + 1) // 2]
                    else:           # vertical slot
                        a, b = g[(r - 1) // 2, col // 2], g[(r + 1) // 2, col // 2]
                    seq.append(E_TOK if frozenset((int(a), int(b))) in eset
                               else W_TOK)
        seq.append(EOS)
        return seq

    def _train_adj(self, rng):
        while True:
            adj = self._gen_maze(rng)
            if self._hash(adj) not in self.eval_hashes:
                return adj

    def batch(self, batch_size: int, rng: np.random.Generator, device="cpu",
              eval_mazes=False):
        c = self.cfg
        T, plen = c.seq_len, c.prefix_len
        toks = np.full((batch_size, T), PAD, dtype=np.int64)
        tmask = np.zeros((batch_size, T), dtype=bool)
        vmask = np.zeros((batch_size, T), dtype=bool)
        for b in range(batch_size):
            if eval_mazes:
                adj = self.eval_adjs[int(rng.integers(len(self.eval_adjs)))]
            else:
                adj = self._train_adj(rng)
            seq = self._render(adj, rng)
            toks[b, :len(seq)] = seq
            vmask[b, :len(seq)] = True
            tmask[b, plen:len(seq)] = True
        return {
            "tokens": torch.from_numpy(toks).to(device),
            "target_mask": torch.from_numpy(tmask).to(device),
            "valid_mask": torch.from_numpy(vmask).to(device),
        }

    # -- verification-based evaluation ---------------------------------------
    def _parse_edges(self, row):
        """Input edge set (as label frozensets) from the prefix tokens."""
        eset = set()
        for i in range(1, self.cfg.prefix_len - 2, 3):
            eset.add(frozenset((int(row[i]) - N_SPECIAL,
                                int(row[i + 1]) - N_SPECIAL)))
        return eset

    @torch.no_grad()
    def evaluate(self, model, rng: np.random.Generator, n_batches: int = 4,
                 batch_size: int = 64, device="cpu"):
        c = self.cfg
        n = c.n_cells
        tot = perm_ok = full_ok = 0
        f1_sum = 0.0
        for _ in range(n_batches):
            batch = self.batch(batch_size, rng, device, eval_mazes=True)
            plen = c.prefix_len
            out = model.generate_greedy(batch["tokens"][:, :plen],
                                        c.map_len)[:, plen:].cpu().numpy()
            toks = batch["tokens"].cpu().numpy()
            for b in range(batch_size):
                eset = self._parse_edges(toks[b])
                m = out[b].reshape(c.map_h, c.map_w)
                verts = m[::2, ::2]
                labels = verts.flatten() - N_SPECIAL
                pv = (labels.min() >= 0 and labels.max() < n
                      and len(set(labels.tolist())) == n)
                perm_ok += pv
                stated, structural_ok = set(), True
                for r in range(c.map_h):
                    for col in range(c.map_w):
                        t = m[r, col]
                        if r % 2 == 1 and col % 2 == 1:
                            structural_ok &= (t == C_TOK)
                        elif r % 2 != col % 2:
                            if t not in (E_TOK, W_TOK):
                                structural_ok = False
                            elif t == E_TOK:
                                if r % 2 == 0:
                                    a = m[r, col - 1] - N_SPECIAL
                                    bb = m[r, col + 1] - N_SPECIAL
                                else:
                                    a = m[r - 1, col] - N_SPECIAL
                                    bb = m[r + 1, col] - N_SPECIAL
                                stated.add(frozenset((int(a), int(bb))))
                inter = len(stated & eset)
                prec = inter / max(len(stated), 1)
                rec = inter / max(len(eset), 1)
                f1_sum += 2 * prec * rec / max(prec + rec, 1e-9)
                full_ok += int(pv and structural_ok and stated == eset)
                tot += 1
        return {"embed_acc": full_ok / tot, "perm_valid": perm_ok / tot,
                "edge_f1": f1_sum / tot}
