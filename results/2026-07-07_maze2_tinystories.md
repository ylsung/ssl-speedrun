# Maze second pass + TinyStories first pass (2026-07-07)

RTX 5090. Raw logs in `runs_remote/`. Maze2 = 4 latent-arm variants × 3 seeds
(4000 steps, same budget as first pass). TinyStories = 6 methods × 1 seed,
20k steps × 32k tok = ~650M tokens, 27M params (8L/512d), 4k BPE, val on
held-out split.

## Maze second pass (reference: ntp 0.696 ± .043, mtp 0.821 ± .014)

```
method                decision_acc      path_acc    erank_last
maze2_jepa_ema       0.744 ±0.020   0.663 ±0.018        51.1
maze2_jepa_ans       0.736 ±0.040   0.645 ±0.031        24.9
maze2_pyramid_hi     0.720 ±0.024   0.639 ±0.037        55.1
maze2_jepa_k16       0.714 ±0.021   0.629 ±0.017        36.1
```

- **EMA + deep target is the best latent arm so far** (+0.048 over NTP vs
  +0.016 for the same-pass shallow version) — consistent with proposal H3.
  Still well short of MTP (+0.125). Kill-gate verdict unchanged: no latent
  loss beats the discrete control on planning at this scale — but the EMA
  direction is the one to push (deeper targets, horizon × EMA grid, D2′-style
  mid-depth source next).
- answer_only ≈ EMA within noise but 2× the seed variance; λ and k=16 don't
  help.

## TinyStories (single seed — CIs pending 2 more seeds)

```
method                    val_loss   val_ppl   erank_last
tinystories_pertoken       1.302      3.678        193.3
tinystories_jepa           1.306      3.690        191.1
tinystories_nitp           1.306      3.692        190.4
tinystories_pyramid        1.306      3.690        196.6
tinystories_ntp            1.309      3.701        192.8
tinystories_mtp            1.320      3.743        239.1
```

- Every latent arm ≤ NTP on val loss; per-token M=4 best (−0.007 nats,
  ~0.5%). Direction matches NITP's claims; magnitude is small and this is
  ONE seed — do not over-read until the 3-seed table lands.
- MTP *hurts* LM loss (+0.011) while showing the highest erank here —
  opposite of its Tier-0 erank collapse; worth understanding later.
- Story samples (runs_remote/*/samples.txt): fluent, coherent multi-sentence
  stories from all methods — the hypothesis-1 eval surface (judge scoring)
  is viable at this scale.

## Next

1. TinyStories seeds 1–2 (running) → CIs.
2. Frozen linear probes + judge eval to test whether the erank/geometry
   advantage buys anything (hypothesis 2).
3. Maze third pass around the EMA arm.
