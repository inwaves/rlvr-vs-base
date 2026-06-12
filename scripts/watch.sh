#!/usr/bin/env bash
# One-command progress poll: heartbeat + latest log tail + GPU snapshot.
# Safe to run from any shell at any time; touches nothing.
cd "$(dirname "$0")/.."

python -m rlvr_vs_base.status --config configs/experiment.yaml || true

echo
echo "--- latest log ---"
latest=$(ls -t runs/logs/*.log 2>/dev/null | head -1 || true)
if [ -n "${latest:-}" ]; then
  echo "($latest)"
  tail -n 12 "$latest"
else
  echo "(no logs yet)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "--- GPU ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
fi
