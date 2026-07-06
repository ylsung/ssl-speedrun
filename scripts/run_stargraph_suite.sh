#!/usr/bin/env bash
# Run the star-graph method suite (optionally over several seeds).
# Usage: bash scripts/run_stargraph_suite.sh [n_seeds]
set -e
cd "$(dirname "$0")/.."
N_SEEDS=${1:-1}
for cfg in configs/stargraph_ntp.yaml configs/stargraph_mtp.yaml \
           configs/stargraph_jepa.yaml configs/stargraph_nitp.yaml \
           configs/stargraph_pertoken.yaml configs/stargraph_pyramid.yaml; do
  for s in $(seq 0 $((N_SEEDS - 1))); do
    name="$(basename "$cfg" .yaml)_seed${s}"
    echo "=== $name ==="
    python -m sslrun.train "$cfg" --seed "$s" --run_name "$name" 2>&1 | tail -3
  done
done
echo "--- results ---"
python scripts/collect_results.py
