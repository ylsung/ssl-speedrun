#!/usr/bin/env bash
# Pull runs/ + suite logs from the GPU instance to local ./runs_remote/.
# Run this after every suite — instance storage is ephemeral.
# Usage: bash scripts/sync_results.sh <ssh_port> <host> [remote_dir]
set -e -o pipefail
PORT=${1:?usage: sync_results.sh <ssh_port> <host> [remote_dir]}
HOST=${2:?usage: sync_results.sh <ssh_port> <host> [remote_dir]}
REMOTE=${3:-/workspace/ssl-speedrun}
mkdir -p runs_remote
rsync -az -e "ssh -p $PORT" "root@$HOST:$REMOTE/runs/" runs_remote/
rsync -az -e "ssh -p $PORT" --include='*.log' --exclude='*' \
      "root@$HOST:$REMOTE/" runs_remote/logs/
echo "synced to runs_remote/ ($(ls runs_remote | wc -l | tr -d ' ') entries)"
