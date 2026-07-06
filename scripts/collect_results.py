"""Aggregate runs/*/summary.json into a per-method table (mean ± std over seeds).
Usage: python scripts/collect_results.py [run_root]
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
by_method = defaultdict(list)
for p in sorted(glob.glob(f"{root}/*/summary.json")):
    s = json.load(open(p))
    method = re.sub(r"_seed\d+$|_s\d+$", "", s["run"])
    by_method[method].append(s)

if not by_method:
    sys.exit(f"no summaries under {root}/")


def last_erank(r):
    e = r.get("eranks") or {}
    if not e:
        return np.nan
    return e[max(e, key=lambda k: int(k.rsplit("_", 1)[1]))]


metrics = sorted({k for runs in by_method.values()
                  for r in runs for k in r["final"]})
hdr = f"{'method':28s} {'n':>2s}"
for k in metrics:
    hdr += f" {k:>16s}"
hdr += f" {'erank_last':>11s} {'wall_s':>7s}"
print(hdr)
for method, runs in sorted(by_method.items()):
    row = f"{method:28s} {len(runs):2d}"
    for k in metrics:
        v = np.array([r["final"][k] for r in runs if k in r["final"]], dtype=float)
        row += f"  {v.mean():7.3f} ±{v.std():.3f}" if len(v) else " " * 17
    er = np.array([last_erank(r) for r in runs], dtype=float)
    row += f" {np.nanmean(er):11.1f} {np.mean([r['wall_s'] for r in runs]):7.0f}"
    print(row)
