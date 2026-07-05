#!/usr/bin/env bash
# Run the 3-method star-graph suite (optionally over several seeds).
# Usage: bash scripts/run_stargraph_suite.sh [n_seeds]
set -e
cd "$(dirname "$0")/.."
N_SEEDS=${1:-1}
for cfg in configs/stargraph_ntp.yaml configs/stargraph_mtp.yaml configs/stargraph_jepa.yaml; do
  for s in $(seq 0 $((N_SEEDS - 1))); do
    name="$(basename "$cfg" .yaml)_seed${s}"
    echo "=== $name ==="
    python -m sslrun.train "$cfg" --run_name "$name" 2>&1 | tail -3
  done
done
echo "--- results ---"
python - <<'EOF'
import glob, json
for p in sorted(glob.glob("runs/*/summary.json")):
    s = json.load(open(p))
    f = s["final"]
    print(f"{s['run']:32s} path_acc={f['path_acc']:.3f} dec_acc={f['decision_acc']:.3f} wall={s['wall_s']}s")
EOF
