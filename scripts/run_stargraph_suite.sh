#!/usr/bin/env bash
# Back-compat wrapper. Usage: bash scripts/run_stargraph_suite.sh [n_seeds]
exec bash "$(dirname "$0")/run_suite.sh" stargraph "${1:-1}"
