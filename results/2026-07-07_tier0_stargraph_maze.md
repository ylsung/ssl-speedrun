# Tier 0 — star graph + maze, 6 methods × 3 seeds (2026-07-07)

RTX 5090, 4000 steps, batch 128, 6L/256d (~4.7–5.7M params). Raw logs in
`runs_remote/` (synced; instance survived a stop/start). Chance on
star-graph decision = 1/d = 0.2.

```
method                        n     decision_acc         path_acc  erank_last
maze_jepa                     3    0.712 ±0.009    0.639 ±0.013        39.7
maze_mtp                      3    0.821 ±0.014    0.780 ±0.016        16.3
maze_nitp                     3    0.709 ±0.025    0.637 ±0.025        43.7
maze_ntp                      3    0.696 ±0.043    0.605 ±0.031        27.4
maze_pertoken                 3    0.712 ±0.029    0.643 ±0.023        37.1
maze_pyramid                  3    0.731 ±0.025    0.656 ±0.016        44.6
stargraph_jepa                3    0.094 ±0.072    0.065 ±0.092        40.3
stargraph_mtp                 3    0.185 ±0.040    0.185 ±0.040         3.4
stargraph_nitp                3    0.049 ±0.013    0.000 ±0.000        24.1
stargraph_ntp                 3    0.076 ±0.057    0.052 ±0.074        17.7
stargraph_pertoken            3    0.134 ±0.079    0.125 ±0.091        17.6
stargraph_pyramid             3    0.109 ±0.081    0.074 ±0.105        34.2
```

## Findings

1. **Star graph: universal failure.** Every method is at/below the 0.2 chance
   level on the first-decision token — the Bachmann & Nagarajan NTP failure
   replicates, and at this budget no auxiliary loss (discrete or latent)
   fixes it. Their paper's fix was teacherless/multi-forward training, not an
   aux loss; this is consistent.
2. **Maze: MTP wins the planning metric, JEPA arms trail.** MTP +0.125
   decision over NTP; pyramid (best latent arm) +0.035. Against the Week-1
   kill-gate ("a latent loss must beat NTP+MTP on planning"), latent losses
   currently FAIL. Caveats before calling it: λ=0.5/0.25 and tgt_layer=1
   untuned, no answer_only ablation, no EMA/deep targets, 4000 steps may
   undertrain, and horizons (k=5/8, M=4) were guesses.
3. **Representation story goes the other way (hypothesis 2).** JEPA arms
   raise last-layer effective rank (34–45 vs NTP 18); MTP *collapses* it
   (3.4 on star graph). NITP-style geometry claims reproduce; the open
   question is whether that geometry buys anything downstream — exactly what
   the TinyStories probes are for.

## Next

- Tier-0 second pass (cheap): λ / tgt_layer / answer_only / EMA-deep grid +
  2–4× longer training on maze, where there is signal to differentiate.
- Tier 1 TinyStories (running next): val loss, story samples, erank; probes
  later. Watch whether the erank advantage translates into anything.
