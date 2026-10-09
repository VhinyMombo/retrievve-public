#!/usr/bin/env bash
# Run a GPU/ollama/mlx job under the shared GPU lock (retrievve.gpulock: kernel flock on the persistent
# ~/retrievve-runs/GPU.lock FILE) plus a resource watchdog. This script never creates or removes GPU.lock:
# the lock lives and dies with the gpulock process (deleting the file while held would break mutual exclusion).
#   watchdog: abort if swap grows > SWAP_MAX_GROWTH_GB (default 4) or free disk < MIN_FREE_GB (default 5)
# usage: scripts/gpu_job.sh <owner> <job-name> -- <command...>
set -euo pipefail
OWNER="$1"; JOB="$2"; shift 2; [ "${1:-}" = "--" ] && shift
PY="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"
SWAP_MAX_GROWTH_GB="${SWAP_MAX_GROWTH_GB:-4}"; MIN_FREE_GB="${MIN_FREE_GB:-5}"
swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }
free_gb() { df -g "$HOME" | awk 'NR==2 {print $4}'; }
SWAP0=$(swap_used_mb)
set -m                                   # job gets its own process group, so the watchdog can stop all of it
"$PY" -m retrievve.gpulock run --owner "$OWNER" --job "$JOB" -- "$@" &
PID=$!
stop() { kill -TERM -- "-$PID" 2>/dev/null || true; }
trap stop EXIT
while kill -0 "$PID" 2>/dev/null; do
  sleep 10
  growth=$(awk -v a="$(swap_used_mb)" -v b="$SWAP0" 'BEGIN {printf "%d", (a-b)/1024}')
  if [ "$growth" -gt "$SWAP_MAX_GROWTH_GB" ]; then echo "WATCHDOG: swap grew ${growth} GB — stopping $JOB"; exit 70; fi
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then echo "WATCHDOG: free disk < ${MIN_FREE_GB} GB — stopping $JOB"; exit 71; fi
done
trap - EXIT
wait "$PID"
