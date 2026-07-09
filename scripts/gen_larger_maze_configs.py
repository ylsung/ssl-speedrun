"""Generate the maze ablation grid (see ABLATIONS.md). One YAML per cell,
named abl_<cell>.yaml; every run shares the maze first-pass training setup.
Usage: python scripts/gen_ablation_configs.py
"""
import os

HEAD = """model: {n_layer: 6, n_head: 4, d_model: 256}
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


TASKS = {
    "maze10": "task: {kind: maze, width: 10, height: 10, min_path: 10, seed: 0}\n",
    "maze20": "task: {kind: maze, width: 20, height: 20, min_path: 20, seed: 0}\n",
}

METHODS = {
    "ntp": "",
    "mtp_k8": "  - {kind: mtp, k: 8, weight: 0.5}\n",
    "infonce_ema_tgt4": pooled(tgt_layer=4, target="ema", objective="infonce"),
}

os.makedirs("configs/larger_maze", exist_ok=True)
for name, loss_line in METHODS.items():
    for task_name, task_line in TASKS.items():
        with open(f"configs/larger_maze/{task_name}_{name}.yaml", "w") as f:
            f.write(f"# {task_name} {name}\n"
                    + task_line + HEAD + loss_line + TRAIN)
    with open(f"configs/ablation/abl_{name}.yaml", "w") as f:
        f.write(f"# ablation cell {name} (see ABLATIONS.md)\n"
                + HEAD + loss_line + TRAIN)
