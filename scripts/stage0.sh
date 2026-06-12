#!/usr/bin/env bash
# Stage 0: pilot. All problems, small N. See README gates G1-G3.
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
    --benchmark math500,aime24,aime25 --stage pilot
done

for model in base rl; do
  python -m rlvr_vs_base.grade --config "$CFG" --model "$model" --benchmark math500,aime24,aime25
  python -m rlvr_vs_base.aggregate --config "$CFG" --model "$model" --benchmark math500,aime24,aime25
done

python -m rlvr_vs_base.analyze --config "$CFG" pilot-report
python -m rlvr_vs_base.analyze --config "$CFG" select-math500-hard

cat <<'EOF'
=== STAGE 0 COMPLETE - review gates before stage 1 ===
G1 sanity: RL pass@1 > base pass@1 on aime24. If not, STOP: debug prompts/grading.
G2 caps:   math500 cap-hit < 3% for both models; else set math500 max_tokens: 16384
           in configs/experiment.yaml and re-run this script (resumes, fills deficit).
G3 budget: pilot-report forecast within budget; else trim stage-1 targets in config.
Next: bash scripts/stage1.sh
EOF
