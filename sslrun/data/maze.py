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

Top-paths mode (round 6): mode="toppaths" asks for the n_paths longest
paths in the tree instead of one start->goal path, layout

    BOS  (u v ESEP)*edges  QRY EQ  path1 ESEP path2 ... EOS  PAD*

The answer is order-invariant at the *sequence* level: paths are emitted in
random order and random direction (a path reversed is the same path), so
which token is correct next depends on a global latent assignment, not a
per-token lookup. Ground truth: the maze is a spanning tree, so the k
longest paths are the k endpoint pairs with maximal tree distance — one BFS
per node gives all-pairs distances exactly. Eval accepts any n_paths
distinct valid simple paths whose sorted lengths match the gold top-k
length multiset (ties handled for free). Metrics: set_acc, path_valid,
len_ratio.
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
    mode: str = "path"    # "path" (start->goal) | "toppaths" (k longest)
    n_paths: int = 1      # toppaths only: how many longest paths

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
        if self.mode == "toppaths":
            k = self.n_paths
            return 1 + 3 * n_edges + 2 + k * self.max_path + (k - 1) + 1
        return 1 + 3 * n_edges + 4 + self.max_path + 1

    @property
    def prefix_len(self):
        frame = 2 if self.mode == "toppaths" else 4  # QRY [start goal] EQ
        return 1 + 3 * (self.n_cells - 1) + frame


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

    def _tok_fn(self, rng):
        c = self.cfg
        s = c.synonyms

        def tok(x, ans=False):
            j = 0
            if s > 1 and (ans or c.syn_scope == "all"):
                j = int(rng.integers(s))
            return N_SPECIAL + x * s + j

        return tok

    def _edge_seq(self, adj, rng, tok):
        edges = []
        for u in range(self.cfg.n_cells):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        rng.shuffle(edges)
        seq = [BOS]
        for u, v in edges:
            seq += [tok(u), tok(v), ESEP]
        return seq

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

        tok = self._tok_fn(rng)
        seq = self._edge_seq(adj, rng, tok)
        seq += [QRY, tok(path[0]), tok(path[-1]), EQ]
        seq += [tok(x, ans=True) for x in path] + [EOS]
        return seq, len(path), dec_idx

    def _tree_dists(self, adj, root):
        """Distances from root over the spanning tree (single traversal)."""
        dist = np.full(self.cfg.n_cells, -1, dtype=np.int64)
        dist[root] = 0
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    stack.append(v)
        return dist

    def _sample_toppaths(self, rng: np.random.Generator):
        c = self.cfg
        adj = self._gen_maze(rng)
        D = np.stack([self._tree_dists(adj, r) for r in range(c.n_cells)])
        iu, iv = np.triu_indices(c.n_cells, k=1)
        lens = D[iu, iv] + 1  # path length in cells
        k = c.n_paths
        # top-k pairs by length, random tie-break so tied answers vary
        perm = rng.permutation(len(lens))
        order = perm[np.argsort(-lens[perm], kind="stable")]
        top = order[:k]
        gold_lens = np.sort(lens[top])[::-1].copy()
        paths = [self._path(adj, int(iu[i]), int(iv[i])) for i in top]
        # nuisance: random direction per path, random emission order
        paths = [p[::-1] if rng.integers(2) else p for p in paths]
        emit = [paths[j] for j in rng.permutation(k)]
        tok = self._tok_fn(rng)
        seq = self._edge_seq(adj, rng, tok)
        seq += [QRY, EQ]
        for j, p in enumerate(emit):
            if j:
                seq.append(ESEP)
            seq += [tok(x, ans=True) for x in p]
        seq += [EOS]
        return seq, gold_lens

    def batch(self, batch_size: int, rng: np.random.Generator, device="cpu"):
        c = self.cfg
        T, plen = c.seq_len, c.prefix_len
        if c.mode == "toppaths":
            toks = np.full((batch_size, T), PAD, dtype=np.int64)
            tmask = np.zeros((batch_size, T), dtype=bool)
            vmask = np.zeros((batch_size, T), dtype=bool)
            top_lens = np.zeros((batch_size, c.n_paths), dtype=np.int64)
            for b in range(batch_size):
                seq, gold_lens = self._sample_toppaths(rng)
                toks[b, :len(seq)] = seq
                vmask[b, :len(seq)] = True
                tmask[b, plen:len(seq)] = True
                top_lens[b] = gold_lens
            return {
                "tokens": torch.from_numpy(toks).to(device),
                "target_mask": torch.from_numpy(tmask).to(device),
                "valid_mask": torch.from_numpy(vmask).to(device),
                "top_lens": torch.from_numpy(top_lens).to(device),
            }
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

    def _syn_probe(self, model, res):
        """Embedding-space synonym invariance: within- vs between-class cosine."""
        c = self.cfg
        E = model.tok_emb.weight[N_SPECIAL:N_SPECIAL + c.n_cells * c.synonyms]
        E = torch.nn.functional.normalize(E, dim=-1)
        G = E @ E.T
        cls = torch.arange(c.n_cells, device=G.device
                           ).repeat_interleave(c.synonyms)
        same = cls.unsqueeze(0) == cls.unsqueeze(1)
        eye = torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
        res["syn_cos_within"] = G[same & ~eye].mean().item()
        res["syn_cos_between"] = G[~same].mean().item()

    def _parse_adj(self, prefix_cls):
        """Rebuild tree adjacency from the (class-mapped) edge-list prefix."""
        adj = [set() for _ in range(self.cfg.n_cells)]
        for i in range(1, self.cfg.prefix_len - (2 if self.cfg.mode == "toppaths" else 4), 3):
            u, v = prefix_cls[i] - N_SPECIAL, prefix_cls[i + 1] - N_SPECIAL
            adj[u].add(v)
            adj[v].add(u)
        return adj

    @torch.no_grad()
    def _evaluate_toppaths(self, model, rng, n_batches, batch_size, device):
        c = self.cfg
        k = c.n_paths
        n_new = k * c.max_path + k  # paths + separators + EOS
        tot = set_ok = 0
        valid_frac = len_ratio = 0.0
        for _ in range(n_batches):
            batch = self.batch(batch_size, rng, device)
            plen = c.prefix_len
            out = model.generate_greedy(batch["tokens"][:, :plen], n_new)
            out_cls = self._classes(out).cpu().numpy()
            gold_lens = batch["top_lens"].cpu().numpy()
            for b in range(batch_size):
                adj = self._parse_adj(out_cls[b, :plen])
                row = out_cls[b, plen:].tolist()
                row = row[:row.index(EOS)] if EOS in row else row
                segs, cur, bad = [], [], False
                for t in row:
                    if t == ESEP:
                        segs.append(cur)
                        cur = []
                    elif t >= N_SPECIAL:
                        cur.append(t - N_SPECIAL)
                    else:
                        bad = True  # stray special token
                segs.append(cur)

                def path_ok(p):
                    return (len(p) >= 2 and len(set(p)) == len(p)
                            and all(v in adj[u] for u, v in zip(p, p[1:])))

                valid = [p for p in segs if path_ok(p)]
                # distinct up to reversal
                seen, distinct = set(), []
                for p in valid:
                    key = min(tuple(p), tuple(p[::-1]))
                    if key not in seen:
                        seen.add(key)
                        distinct.append(p)
                gl = sorted(gold_lens[b].tolist(), reverse=True)
                el = sorted((len(p) for p in distinct), reverse=True)[:k]
                set_ok += int(not bad and len(segs) == k and el == gl)
                valid_frac += len(distinct) / k
                len_ratio += (max(el) / gl[0]) if el else 0.0
                tot += 1
        res = {"set_acc": set_ok / tot, "path_valid": valid_frac / tot,
               "len_ratio": len_ratio / tot}
        if c.synonyms > 1:
            self._syn_probe(model, res)
        return res

    @torch.no_grad()
    def evaluate(self, model, rng: np.random.Generator, n_batches: int = 4,
                 batch_size: int = 64, device="cpu"):
        c = self.cfg
        if c.mode == "toppaths":
            return self._evaluate_toppaths(model, rng, n_batches, batch_size,
                                           device)
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
            self._syn_probe(model, res)
        return res
