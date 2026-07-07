#!/usr/bin/env bash
# Run the maze ablation grid (ABLATIONS.md). Usage: bash scripts/run_ablation.sh [n_seeds]
set -e -o pipefail
cd "$(dirname "$0")/.."
N_SEEDS=${1:-3}
python scripts/gen_ablation_configs.py
for cfg in configs/ablation/abl_*.yaml; do
  base=$(basename "$cfg" .yaml)
  for s in $(seq 0 $((N_SEEDS - 1))); do
    name="${base}_seed${s}"
    if [ -f "runs/$name/summary.json" ]; then
      echo "=== $name (done, skipping) ==="
      continue
    fi
    echo "=== $name ==="
    python -m sslrun.train "$cfg" --seed "$s" --run_name "$name" 2>&1 | tail -2
  done
done
echo "--- results ---"
python scripts/collect_results.py
