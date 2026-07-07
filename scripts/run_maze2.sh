#!/usr/bin/env bash
# Tier-0 second pass on maze: hyperparameter sanity for the latent arms.
# Usage: bash scripts/run_maze2.sh [n_seeds]
set -e -o pipefail
cd "$(dirname "$0")/.."
N_SEEDS=${1:-3}
for m in pyramid_hi jepa_ans jepa_ema jepa_k16; do
  for s in $(seq 0 $((N_SEEDS - 1))); do
    name="maze2_${m}_seed${s}"
    if [ -f "runs/$name/summary.json" ]; then
      echo "=== $name (done, skipping) ==="
      continue
    fi
    echo "=== $name ==="
    python -m sslrun.train "configs/maze2_${m}.yaml" --seed "$s" --run_name "$name" 2>&1 | tail -3
  done
done
echo "--- results ---"
python scripts/collect_results.py
