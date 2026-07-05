#!/usr/bin/env bash
# One-shot setup on a fresh vast.ai instance (pytorch template recommended).
# Usage on the instance:  bash setup_vast.sh && cd ssl-speedrun && bash scripts/run_stargraph_suite.sh 3
set -e
pip install -q numpy pyyaml
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
