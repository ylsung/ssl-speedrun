# ssl-speedrun

Compact benchmark for JEPA-decoder hybrid losses (see the vault notes
[[JEPA Decoder Proposal]] and [[SSL Speedrun]]). Every method = a list of
loss plugins in a YAML config; one trainer runs everything.

## Quickstart

```bash
pip install -r requirements.txt
python -m sslrun.train configs/stargraph_ntp.yaml       # one run
bash scripts/run_suite.sh stargraph 3                   # all 6 methods x 3 seeds
bash scripts/run_suite.sh maze 3
python scripts/collect_results.py                       # mean ± std per method
```

Logs: `runs/<name>/log.csv`, `runs/<name>/summary.json` (incl. per-layer
effective ranks).

## Methods (each = a `losses:` list in YAML)

| config suffix | losses | role |
|---|---|---|
| `ntp` | NTP | control |
| `mtp` | NTP + multi-token heads (k=4) | discrete control arm |
| `nitp` | NTP + per-token latent, M=1 | NITP repro / sanity check |
| `pertoken` | NTP + per-token latents, M=4 | position-indexed arm |
| `jepa` | NTP + pooled-window latent (k tokens ahead) | position-invariant arm |
| `pyramid` | NTP + per-token M=4 + pooled window | resolution pyramid (default bet) |

Latent targets are same-pass shallow-layer (block 1) with stop-grad; predictor
heads are asymmetric MLPs (anti-collapse per proposal §2.3). EMA teacher comes
later, only where the free version shows signal.

## Tier 0 tasks

**Star graph** (Bachmann & Nagarajan 2024). G(d,n): center node, d arms of
length n, find the path to the goal. The only *hard* token is the first node
after the center (requires planning); teacher-forced NTP is known to shortcut
it. `decision_acc` ≈ 1/d means NTP failed as expected; the JEPA/MTP question
is whether they fix it.

**Maze** (perfect maze on a W×H grid, DFS spanning tree → unique paths).
Prompt = shuffled open-edge list + query; output = the cell path.
`decision_acc` scores the first junction (>1 open continuation) on each path;
corridor steps are trivial.

Both report `path_acc` (exact decode of path+EOS) and `decision_acc`.

## Metrics beyond accuracy

- `erank_<layer>`: effective rank (exp of singular-value entropy) of hidden
  states per layer — the representation-degeneration diagnostic (hypothesis 2;
  NITP's headline metric). Logged at every eval into CSV + summary.

## Layout

```
sslrun/model.py      GPT (RoPE, RMSNorm), returns all layer hiddens
sslrun/losses.py     plugins: ntp | mtp | jepa_pooled | jepa_pertoken + LossStack
sslrun/metrics.py    effective-rank callbacks
sslrun/data/         task generators (stargraph, maze; othello/tinystories next)
sslrun/train.py      config-driven trainer, CSV + JSON summaries
configs/             one YAML per method x task
scripts/             GPU provisioning, suite runner, results aggregation
```

## Adding a method

Write a plugin in `losses.py` with `name`, `__init__(d_model, vocab_size, **cfg)`,
`forward(logits, hiddens, batch) -> scalar`; register it in `LOSS_REGISTRY`;
reference it in a config. Nothing else changes.

## Adding a task

Module in `sslrun/data/` with a `Config` dataclass (`vocab_size`, `seq_len`
properties) and a `Task` class with `batch()` (returns `tokens`,
`target_mask`, `valid_mask`, …) and `evaluate()` (returns the metric dict);
register in `TASKS` in `train.py`.

## Roadmap (from [[SSL Speedrun]])

- [x] Tier 0: star graph, maze
- [x] Per-token latent loss (M=1..8) + resolution pyramid
- [x] Effective-rank callback
- [ ] Tier 0: ProsQA, Othello board-state probes
- [ ] Per-layer linear-probe callbacks
- [ ] EMA target encoder variant
- [ ] Tier 1: TinyStories + LLM-judge eval
- [ ] Tier 2: single-GPU FineWeb speedrun fork
