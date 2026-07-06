"""Config-driven trainer. Usage:
    python -m sslrun.train configs/stargraph_ntp.yaml [--device auto] [--steps N]
Writes CSV logs + final summary JSON to runs/<run_name>/.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from .model import GPT, ModelConfig
from .losses import LossStack
from .metrics import layer_effective_ranks
from .data.stargraph import StarGraphConfig, StarGraphTask
from .data.maze import MazeConfig, MazeTask

TASKS = {"stargraph": (StarGraphConfig, StarGraphTask),
         "maze": (MazeConfig, MazeTask)}


def pick_device(arg):
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_task(cfg):
    tcfg = cfg["task"]
    kind = tcfg["kind"]
    if kind not in TASKS:
        raise ValueError(f"unknown task {kind}")
    cfg_cls, task_cls = TASKS[kind]
    c = cfg_cls(**{k: v for k, v in tcfg.items() if k != "kind"})
    return task_cls(c), c.vocab_size, c.seq_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--seed", type=int, default=None, help="override train.seed")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["train"]
    if args.steps is not None:
        tr["steps"] = args.steps
    if args.seed is not None:
        tr["seed"] = args.seed
    device = pick_device(args.device)
    seed = tr.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    eval_rng_seed = seed + 10_000

    task, vocab_size, seq_len = build_task(cfg)
    mcfg = ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len, **cfg["model"])
    model = GPT(mcfg).to(device)
    stack = LossStack(model, cfg["losses"]).to(device)
    n_params = sum(p.numel() for p in stack.parameters())

    opt = torch.optim.AdamW(stack.parameters(), lr=tr["lr"],
                            weight_decay=tr.get("weight_decay", 0.01),
                            betas=(0.9, 0.95))
    steps, warmup = tr["steps"], tr.get("warmup", 100)

    def lr_at(s):
        if s < warmup:
            return s / max(warmup, 1)
        p = (s - warmup) / max(steps - warmup, 1)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * p))  # cosine to 10%

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0] + f"_s{seed}"
    run_dir = os.path.join(cfg.get("run_root", "runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    logger = csv.writer(log_f)
    logger.writerow(["step", "loss", "loss_parts", "path_acc", "decision_acc",
                     "erank_last", "lr", "tok_per_s"])

    print(f"[{run_name}] device={device} params={n_params/1e6:.2f}M "
          f"vocab={vocab_size} seq_len={seq_len}")

    bs = tr["batch_size"]
    eval_every = tr.get("eval_every", 200)
    t0, toks_seen = time.time(), 0
    best = {"path_acc": 0.0, "decision_acc": 0.0}

    for step in range(1, steps + 1):
        batch = task.batch(bs, rng, device)
        loss, parts = stack(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stack.parameters(), 1.0)
        opt.step()
        sched.step()
        toks_seen += bs * seq_len

        if step % eval_every == 0 or step == steps:
            model.eval()
            ev = task.evaluate(model, np.random.default_rng(eval_rng_seed),
                               device=device)
            eranks = layer_effective_ranks(
                model, task.batch(256, np.random.default_rng(eval_rng_seed), device))
            model.train()
            best = {k: max(best[k], ev[k]) for k in best}
            tps = toks_seen / (time.time() - t0)
            lv = float(loss.detach())
            er_last = eranks[f"erank_{len(model.blocks)}"]
            parts_s = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
            print(f"step {step:6d} loss {lv:.4f} [{parts_s}] "
                  f"path_acc {ev['path_acc']:.3f} dec_acc {ev['decision_acc']:.3f} "
                  f"erank {er_last:.1f} ({tps/1e3:.1f}k tok/s)")
            logger.writerow([step, f"{lv:.5f}", json.dumps(parts),
                             ev["path_acc"], ev["decision_acc"], f"{er_last:.2f}",
                             f"{sched.get_last_lr()[0]:.2e}", int(tps)])
            log_f.flush()

    summary = {"run": run_name, "config": args.config, "device": device,
               "params": n_params, "steps": steps, "final": ev, "best": best,
               "eranks": {k: round(v, 2) for k, v in eranks.items()},
               "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))
    log_f.close()


if __name__ == "__main__":
    main()
