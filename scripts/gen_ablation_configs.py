"""Generate the maze ablation grid (see ABLATIONS.md). One YAML per cell,
named abl_<cell>.yaml; every run shares the maze first-pass training setup.
Usage: python scripts/gen_ablation_configs.py
"""
import os

HEAD = """task: {kind: maze, width: 6, height: 6, min_path: 6, seed: 0}
model: {n_layer: 6, n_head: 4, d_model: 256}
losses:
  - {kind: ntp, weight: 1.0}
"""
TRAIN = """train:
  steps: 4000
  batch_size: 128
  lr: 1.0e-3
  warmup: 200
  eval_every: 200
  seed: 0
"""


def pooled(**kw):
    base = dict(kind="jepa_pooled", k=8, weight=0.5, src_layer=-1, tgt_layer=1)
    base.update(kw)
    items = ", ".join(f"{k}: {v}" for k, v in base.items())
    return "  - {%s}\n" % items


def pertoken(**kw):
    base = dict(kind="jepa_pertoken", m=4, weight=0.5, src_layer=-1, tgt_layer=1)
    base.update(kw)
    items = ", ".join(f"{k}: {v}" for k, v in base.items())
    return "  - {%s}\n" % items


CELLS = {
    # Q1/Q2: teacher x target-depth factorial (k=8, src=-1)
    # done: same-pass tgt1 (maze_jepa), ema tgt4 (maze2_jepa_ema)
    "same_tgt2": pooled(tgt_layer=2),
    "same_tgt4": pooled(tgt_layer=4),
    "same_tgt6": pooled(tgt_layer=6),
    "ema_tgt1": pooled(tgt_layer=1, target="ema"),
    "ema_tgt2": pooled(tgt_layer=2, target="ema"),
    "ema_tgt6": pooled(tgt_layer=6, target="ema"),
    # Q3: source depth with best teacher (ema tgt4); src6(=-1) done
    "src2_ema_tgt4": pooled(src_layer=2, tgt_layer=4, target="ema"),
    "src4_ema_tgt4": pooled(src_layer=4, tgt_layer=4, target="ema"),
    # Q4: horizon / hardness with ema tgt4 (k8 done)
    "k4_ema_tgt4": pooled(k=4, tgt_layer=4, target="ema"),
    "k16_ema_tgt4": pooled(k=16, tgt_layer=4, target="ema"),
    "k32_ema_tgt4": pooled(k=32, tgt_layer=4, target="ema"),
    "m8_ema_tgt4": pertoken(m=8, tgt_layer=4, target="ema"),
    "m16_ema_tgt4": pertoken(m=16, tgt_layer=4, target="ema"),
    "mtp_k8": "  - {kind: mtp, k: 8, weight: 0.5}\n",
    # Q5: chunk-to-chunk (source pooled over trailing c tokens) + attention pooling
    "c4_ema_tgt4": pooled(src_pool=4, tgt_layer=4, target="ema"),
    "c8_ema_tgt4": pooled(src_pool=8, tgt_layer=4, target="ema"),
    "attn_ema_tgt4": pooled(tgt_layer=4, target="ema", pool="attn"),
    "attnc8_ema_tgt4": pooled(src_pool=8, tgt_layer=4, target="ema", pool="attn"),
    # Q6: objective family on the reference arm (ema tgt4, k8)
    "l2_ema_tgt4": pooled(tgt_layer=4, target="ema", objective="l2"),
    "sigreg_ema_tgt4": pooled(tgt_layer=4, target="ema", objective="l2",
                              reg="sigreg", reg_weight=1.0),
    "vicreg_ema_tgt4": pooled(tgt_layer=4, target="ema", objective="l2",
                              reg="vicreg", reg_weight=1.0),
    "infonce_ema_tgt4": pooled(tgt_layer=4, target="ema", objective="infonce"),
}

# Round 3 — hardness program (ABLATIONS.md). value = (loss_lines, train_extra)
EMA4 = dict(tgt_layer=4, target="ema")
HARD = {
    # 0: causal data2vec — corrupted student pass (NTP rides it), clean EMA targets
    "h_d2v25": (pooled(**EMA4), "  corrupt_p: 0.25\n"),
    "h_d2v50": (pooled(**EMA4), "  corrupt_p: 0.5\n"),
    "h_d2vctl50": ("", "  corrupt_p: 0.5\n"),          # corrupted-NTP-only control
    "h_d2vctl25": ("", "  corrupt_p: 0.25\n"),         # attribution control for d2v25
    "h_noisytgt50": (pooled(**EMA4),                    # noisy-target control (expect fail)
                     "  corrupt_p: 0.5\n  corrupt_side: ema\n"),
    # 1: gap targets — drop the easy near future
    "h_gap8": (pooled(gap=8, **EMA4), ""),
    "h_gap16": (pooled(gap=16, **EMA4), ""),
    # 2: discrete latent codes (sign-LSH bits, BCE)
    "h_lsh": (pooled(objective="lsh", n_bits=64, **EMA4), ""),
    # 3: hard-negative InfoNCE (same-sequence negatives)
    "h_nceh": (pooled(objective="infonce_hard", **EMA4), ""),
    # 4: difficulty-weighted (detached NTP entropy)
    "h_entw": (pooled(weight_by="entropy", **EMA4), ""),
}

os.makedirs("configs/ablation", exist_ok=True)
for name, loss_line in CELLS.items():
    with open(f"configs/ablation/abl_{name}.yaml", "w") as f:
        f.write(f"# ablation cell {name} (see ABLATIONS.md)\n"
                + HEAD + loss_line + TRAIN)
for name, (loss_line, extra) in HARD.items():
    with open(f"configs/ablation/abl_{name}.yaml", "w") as f:
        f.write(f"# hardness cell {name} (see ABLATIONS.md round 3)\n"
                + HEAD + loss_line + TRAIN + extra)
print(f"wrote {len(CELLS) + len(HARD)} configs to configs/ablation/")
