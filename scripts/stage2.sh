#!/usr/bin/env bash
# Stage 2: adaptive extension on pre-registered unresolved problems only.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/experiment.yaml

mkdir -p runs/logs
LOG="runs/logs/$(basename "$0" .sh)-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader -l 30 >> runs/logs/gpu.csv &
  GPU_LOG_PID=$!
  trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT
fi

for model in base rl; do
  python -m rlvr_vs_base.generate --config "$CFG" --model "$model" \
    --benchmark aime24 --stage extension --subset-file runs/stage2/aime24.json
  python -m rlvr_vs_base.generate --config "$CFG" --model "$model" \
    --benchmark math500 --stage extension --subset-file runs/stage2/math500.json
done

for model in base rl; do
  python -m rlvr_vs_base.grade --config "$CFG" --model "$model" --benchmark math500,aime24
  python -m rlvr_vs_base.aggregate --config "$CFG" --model "$model" --benchmark math500,aime24
done

python -m rlvr_vs_base.analyze --config "$CFG" report --benchmark math500,aime24

for model in base rl; do
  for bench in math500 aime24; do
    python -m rlvr_vs_base.spotcheck --config "$CFG" --benchmark "$bench" --model "$model"
  done
done

cat <<'EOF'
=== STAGE 2 COMPLETE - final outputs ===
- runs/report/*.json + *.png : final pass@k curves, crossover-k, gaps with CIs
- runs/spotcheck/            : low-accuracy solves for MANUAL CoT review (guessing control)
Pull aggregates/report/spotcheck off the box, archive or discard raw shards, tear down.
EOF
