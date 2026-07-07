# Methods & data reference (as of 2026-07-08)

Everything tried so far, with exact hyperparameters. Configs in `configs/` are
the ground truth; this file is the readable index to them.

---

## 1. Model

One decoder-only GPT for everything (`sslrun/model.py`): RMSNorm (pre-norm),
RoPE, causal SDPA attention, GELU MLP (4× expansion), no biases, tied
input/output embeddings, init N(0, 0.02). `forward()` returns
`(logits, hiddens)` where `hiddens[0]` = embedding output and `hiddens[i]` =
output of block *i* (pre final-norm), so any loss/probe can tap any depth.

| | Tier 0 (stargraph, maze, maze2) | Tier 1 (TinyStories) |
|---|---|---|
| layers | **6** | **8** |
| d_model / heads | 256 / 4 | 512 / 8 |
| context | 87 (stargraph) / 147 (maze) | 512 |
| params | 4.7M (ntp) – 5.7M (pyramid) | ~27M |

The param spread within a tier is the loss plugins' predictor heads; they are
optimized jointly with the model and discarded at eval (eval = plain LM).

## 2. Trainer (identical for all methods)

AdamW β=(0.9, 0.95), weight decay 0.01, grad-clip 1.0. LR: linear warmup then
cosine to 10% of peak. Fresh batch every step (Tier 0 generates data on the
fly — effectively infinite data, no memorization confound; Tier 1 samples
random 512-token windows from the token stream). Seeds 0–2
(`torch.manual_seed` + numpy generator; GPU SDPA is nondeterministic, so
same-seed runs still vary slightly).

| | Tier 0 | Tier 1 |
|---|---|---|
| steps | 4,000 | 20,000 |
| batch | 128 | 64 |
| peak LR / warmup | 1e-3 / 200 | 6e-4 / 500 |
| tokens/run | 44.5M (sg) / 75M (maze) | 655M (~1.35 epochs) |
| eval every | 200 | 500 |

**Supervision masks** (`target_mask`): Tier 0 — only the answer region
(path + EOS); the edge-list prompt is *not* language-modeled. Tier 1 — every
position except index 0. `valid_mask` = all real (non-PAD) tokens.

## 3. Method matrix

All losses are plugins over the same forward pass (`sslrun/losses.py`); total
loss = Σ weightᵢ · lossᵢ.

### Tier 0 first pass + TinyStories (6 methods)

| method | losses (weight) | method-specific hypers |
|---|---|---|
| `ntp` | NTP (1.0) | — |
| `mtp` | NTP (1.0) + MTP (0.5) | k=4 → 3 extra untied `Linear(d→V)` heads predicting tokens t+2, t+3, t+4 from the **last block's** hidden at t; CE per offset, averaged |
| `nitp` | NTP (1.0) + per-token latent (0.5) | M=1: predict hidden of token t+1 |
| `pertoken` | NTP (1.0) + per-token latent (0.5) | M=4: predict hiddens of t+1…t+4 |
| `jepa` | NTP (1.0) + pooled-window latent (0.5) | k = 5 (stargraph) / 8 (maze) / 16 (TinyStories): predict the *mean* of the next k hiddens |
| `pyramid` | NTP (1.0) + per-token (0.25) + pooled (0.25) | M=4 short-horizon + pooled long-horizon together |

**Shared JEPA-loss anatomy** (both latent plugins, all first-pass setups):

- **Source** (`src_layer=-1`): last block's output at position t
  (block 6 of 6 / block 8 of 8, pre final-norm).
- **Target** (`tgt_layer=1`): **block 1's output** from the *same* forward
  pass, `.detach()` (stop-grad). No EMA in the first pass.
- **Predictor** (the asymmetry that blocks the trivial solution):
  pooled → `Linear(d→2d) + GELU + Linear(2d→d)`;
  per-token → shared `Linear(d→2d)+GELU` trunk + one `Linear(2d→d)` per offset.
- **Distance**: 1 − cosine similarity (fp32), averaged over valid positions;
  per-token additionally averaged over offsets.
- **Validity**: per-token — source and target positions must both be real
  tokens; pooled — the entire k-token window must be real tokens.

### Maze second pass (4 variants, all on maze)

| variant | change vs `jepa`/`pyramid` above |
|---|---|
| `pyramid_hi` | pyramid with weights doubled (0.5 + 0.5) |
| `jepa_ans` | pooled k=8, loss only where the whole window lies **inside the answer** (`answer_only: true`) |
| `jepa_ema` | pooled k=8, target = **block 4 of 6** of an **EMA copy** of the model (decay 0.999) |
| `jepa_k16` | pooled k=16 (≈ a whole path), shallow same-pass target |

(`configs/stargraph_jepa_ema.yaml` also exists but has not been run.)

## 4. Direct answers to the six questions

1. **Layers**: 6 blocks (Tier 0), 8 blocks (Tier 1). Indexing: hidden 0 =
   embeddings, hidden i = block i output.
2. **JEPA source features**: always the **final block's** output at position t
   (`src_layer=-1`) in every setup tried so far. A mid-depth source (the D2′
   design) is planned but not yet run.
3. **JEPA target**: hidden states of *future* positions. First pass: **block 1**
   output, same forward pass, stop-grad. `jepa_ema` variant: **block 4** output
   of the EMA teacher. Per-token targets are individual future positions'
   hiddens; pooled targets are the mean over the next k positions' hiddens.
4. **EMA construction**: a `deepcopy` of the **entire model** (all 6 blocks +
   embeddings), gradients off, updated after *every* optimizer step:
   θ_ema ← 0.999·θ_ema + 0.001·θ. It is **not** truncated to the first M
   layers — though since only its block-4 hidden is consumed, blocks 5–6 of
   the teacher are computed and discarded (known inefficiency, harmless).
   Costs one extra no-grad forward per step (~15–20% step time).
5. **`answer_only`**: restricts *which positions produce JEPA loss terms*.
   False (default): every real-token position, including the shuffled
   edge-list prompt. True: only windows fully inside the supervised answer
   span (the path + EOS) — i.e., the latent loss is focused on plan tokens
   instead of mostly-random prompt tokens.
6. **Tokenizer**: identical across methods, differs per dataset. Tier 0 tasks
   have **no tokenizer** — sequences are raw integer ids (6 special tokens +
   node/cell ids). TinyStories uses a **custom ByteLevel BPE, vocab 4096**,
   trained on 200k stories, `<|endoftext|>` (id 0) as story separator; same
   tokenizer file for every method.

## 5. Datasets

### Star graph G(d=5, n=5) — `sslrun/data/stargraph.py`

Generated on the fly (infinite). A center node with 5 arms of 5 edges; node
ids drawn without replacement from a pool of 50 and offset by the 6 specials
→ **vocab 56**. Every sequence is exactly **87 tokens**, no padding:

- input/prompt: **80 tokens** = `BOS` + 25 shuffled edges (`u v ESEP` each) +
  `QRY center goal EQ`
- output/answer (supervised): **7 tokens** = 6 path nodes + `EOS`

Sample (n⟨i⟩ = node id):

```
prompt:  BOS n45 n29 ESEP n42 n1 ESEP n47 n25 ESEP n0 n42 ESEP n0 n32 ESEP
         n29 n22 ESEP n18 n12 ESEP ... (25 edges total) ... QRY n0 n12 EQ
answer:  n0 n13 n23 n5 n18 n12 EOS
```

The only *hard* token is the 2nd answer token (first node after the center —
requires resolving which arm reaches the goal); everything after is
edge-following. `decision_acc` scores exactly that token under greedy decode;
`path_acc` is exact match of the full 7-token answer. Eval: 256 fresh
sequences, fixed eval seed.

### Maze 6×6 — `sslrun/data/maze.py`

Generated on the fly. Perfect maze (randomized-DFS spanning tree over 36
cells → 35 open edges, unique path between any two cells), start/goal
resampled until path ≥ 6 cells. **Vocab 42**, fixed **seq_len 147** with PAD
tail:

- input/prompt: **110 tokens** = `BOS` + 35 shuffled open edges + `QRY start
  goal EQ`
- output/answer (supervised): **path + EOS, variable 7–33 tokens**; measured
  over 2,000 samples: path length mean **13.5**, median 13, p90 21, range
  6–32; 99.2% of paths contain ≥ 1 junction decision

Sample:

```
prompt:  BOS n31 n32 ESEP n14 n20 ESEP n25 n26 ESEP ... (35 edges) ... QRY n34 n7 EQ
answer:  n34 n28 n29 n23 n17 n11 n10 n16 n22 n21 n20 n14 n8 n7 EOS
```

`decision_acc` scores the first *junction* token (first step where >1 open
continuation exists, excluding the cell we came from); corridor steps are
forced and trivial. `path_acc` = exact decode of path+EOS.

### TinyStories — `sslrun/data/lm.py` + `scripts/prepare_tinystories.py`

`roneneldan/TinyStories` (HF), **2,119,719 stories**. Custom 4k BPE (above).
Encoded once on the GPU instance to uint16 memmaps (never stored locally):

- **train.bin: 485.9M tokens**, val.bin: 4.88M tokens
- tokens/story: mean **228**, median 196, p10 145, p90 365, max ~1.3k
- training example = random contiguous **512-token window** over the
  concatenated stream (stories separated by `<|endoftext|>`; windows cross
  story boundaries) — all positions supervised; no separate "output" span
- eval = cross-entropy on 8×32 fixed-seed windows (~131k tokens) from val.bin

Sample of the training stream (decoded):

> One day, a little girl named Lily found a needle in her room. She knew it
> was difficult to play with it because it was sharp. Lily wanted to share
> the needle with her mom, so she could sew a button on her shirt. …

## 6. Quirks & caveats worth knowing

- **Tier-0 NTP loss covers only the answer region.** The prompt (shuffled
  edge list) is conditioned on but never language-modeled. MTP heads are
  masked the same way. Latent losses cover all real tokens unless
  `answer_only: true`.
- **MTP heads read the pre-final-norm hidden** (same as the latent plugins);
  the main LM head reads the post-norm hidden.
- **The latent task is currently too easy** — the training curves
  (see the curves artifact / `results/`) show pooled-JEPA loss collapsing to
  ~0.01–0.02 nats in the first ~20% of training with shallow (block-1)
  targets, after which it supplies almost no gradient. The EMA/deep-target
  variant keeps the loss ~2× higher and is the best latent arm on maze —
  the next iteration should make targets harder (deeper layer, mid-depth
  source, EMA teacher, harder horizons) rather than reweighting.
- **Data is never reused** on Tier 0 (fresh generation each batch), so
  overfitting/memorization is not a factor; on TinyStories one run ≈ 1.35
  epochs.
- Predictor-head parameter counts differ across methods (ntp 4.74M …
  pyramid 5.65M at Tier 0). At these sizes the extra params sit in heads that
  are discarded at eval, but strictly speaking methods are not param-matched
  (they are data- and step-matched).
