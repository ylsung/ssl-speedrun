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


print(f"{'method':28s} {'n':>2s} {'path_acc':>15s} {'decision_acc':>15s} "
      f"{'erank_last':>12s} {'wall_s':>7s}")
for method, runs in sorted(by_method.items()):
    pa = np.array([r["final"]["path_acc"] for r in runs])
    da = np.array([r["final"]["decision_acc"] for r in runs])
    er = np.array([last_erank(r) for r in runs], dtype=float)
    wall = np.mean([r["wall_s"] for r in runs])
    print(f"{method:28s} {len(runs):2d} "
          f"{pa.mean():7.3f} ±{pa.std():.3f} "
          f"{da.mean():7.3f} ±{da.std():.3f} "
          f"{np.nanmean(er):8.1f} {wall:7.0f}")
