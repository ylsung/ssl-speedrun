#!/usr/bin/env bash
# Run all 6 methods for one task, over several seeds.
# Usage: bash scripts/run_suite.sh <task: stargraph|maze> [n_seeds]
set -e -o pipefail
cd "$(dirname "$0")/.."
TASK=${1:?usage: run_suite.sh <task> [n_seeds]}
N_SEEDS=${2:-1}
for m in ntp mtp jepa nitp pertoken pyramid; do
  cfg="configs/${TASK}_${m}.yaml"
  for s in $(seq 0 $((N_SEEDS - 1))); do
    name="${TASK}_${m}_seed${s}"
    if [ -f "runs/$name/summary.json" ]; then
      echo "=== $name (done, skipping) ==="
      continue
    fi
    echo "=== $name ==="
    python -m sslrun.train "$cfg" --seed "$s" --run_name "$name" 2>&1 | tail -3
  done
done
echo "--- results ---"
python scripts/collect_results.py
