# ssl-speedrun

Compact benchmark for JEPA-decoder hybrid losses (see the vault notes
[[JEPA Decoder Proposal]] and [[SSL Speedrun]]). Every method = a list of
loss plugins in a YAML config; one trainer runs everything.

## Quickstart

```bash
pip install -r requirements.txt
python -m sslrun.train configs/stargraph_ntp.yaml            # NTP control
python -m sslrun.train configs/stargraph_mtp.yaml            # + multi-token prediction
python -m sslrun.train configs/stargraph_jepa.yaml           # + pooled-window JEPA
```

Logs: `runs/<name>/log.csv`, `runs/<name>/summary.json`.

## Tier 0 task: star graph (Bachmann & Nagarajan 2024)

G(d,n): center node, d arms of length n, find the path to the goal.
The only *hard* token is the first node after the center (requires planning);
teacher-forced NTP is known to shortcut it. Metrics:

- `path_acc` — exact-match of the full decoded path
- `decision_acc` — accuracy on that first hard token (≈ 1/d = random guessing
  means NTP failed as expected; the JEPA/MTP question is whether they fix it)

## Layout

```
sslrun/model.py      GPT (RoPE, RMSNorm), returns all layer hiddens
sslrun/losses.py     loss plugins: ntp | mtp | jepa_pooled + LossStack
sslrun/data/         task generators (stargraph; maze/othello/tinystories next)
sslrun/train.py      config-driven trainer, CSV + JSON summaries
configs/             one YAML per method x task
scripts/             GPU provisioning + run suites
```

## Adding a method

Write a plugin in `losses.py` with `name`, `__init__(d_model, vocab_size, **cfg)`,
`forward(logits, hiddens, batch) -> scalar`; register it in `LOSS_REGISTRY`;
reference it in a config. Nothing else changes.

## Roadmap (from [[SSL Speedrun]])

- [ ] Tier 0: maze, ProsQA, Othello board-state probes
- [ ] Per-token latent loss (M=1..8) + resolution pyramid
- [ ] Effective-rank / probe callbacks
- [ ] Tier 1: TinyStories + LLM-judge eval
- [ ] Tier 2: single-GPU FineWeb speedrun fork
