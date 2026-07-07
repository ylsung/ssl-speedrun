# Star graph d=5 n=5 — partial results (2026-07-06)

Vast.ai RTX 5090 instance died before the full 6-method × 3-seed table was
synced back; only the NTP/MTP rows were retrieved. JEPA-family star-graph rows
and the whole maze suite need a rerun (configs + seeds are in the repo; the
suite runner skips completed runs, ~2 GPU-hours total).

4000 steps, batch 128, 6L/256d (4.7M params), 3 seeds, final eval:

| method | decision_acc | path_acc | erank_last | note |
|---|---|---|---|---|
| ntp | 0.076 ± 0.057 | 0.052 ± 0.074 | 17.7 | ≤ chance (1/d = 0.2) — the expected planning failure |
| ntp+mtp (k=4) | 0.185 ± 0.040 | 0.185 ± 0.040 | 3.4 | ≈ chance; last-layer effective rank collapses |

Observations:

- NTP fails the first-decision token exactly as Bachmann & Nagarajan predict;
  high seed variance (one seed hit 0.195, another 0.031).
- MTP does **not** fix the planning decision at this scale and its extra heads
  crush last-layer effective rank (3.4 vs 17.7) — consistent with the
  proposal's §2.4 degeneration story, worth tracking as its own finding.
- Lesson encoded in scripts/sync_results.sh: pull runs/ off the instance
  after every suite.
