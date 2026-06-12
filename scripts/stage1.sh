#!/usr/bin/env bash
# Stage 1: main run on MATH500-hard subset + AIME24, plus prompt ablation.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/experiment.yaml

# Main generation (math500 auto-loads the math500_hard subset at stage main).
for model in base rl; do
  python -m rlvr_vs_base.generate --config "$CFG" --model "$model" \
    --benchmark math500,aime24 --stage main
done

# Prompt-variant ablation: boxed template on the math500-hard subset, small N.
for model in base rl; do
  python -m rlvr_vs_base.generate --config "$CFG" --model "$model" \
    --benchmark math500 --stage main --template boxed --target-n 32
done

for model in base rl; do
  python -m rlvr_vs_base.grade --config "$CFG" --model "$model" --benchmark math500,aime24
  python -m rlvr_vs_base.aggregate --config "$CFG" --model "$model" --benchmark math500,aime24
  python -m rlvr_vs_base.grade --config "$CFG" --model "$model" --benchmark math500 --template boxed
  python -m rlvr_vs_base.aggregate --config "$CFG" --model "$model" --benchmark math500 --template boxed
done

python -m rlvr_vs_base.analyze --config "$CFG" report --benchmark math500,aime24
python -m rlvr_vs_base.analyze --config "$CFG" --template boxed report --benchmark math500
python -m rlvr_vs_base.analyze --config "$CFG" stage2-candidates --benchmark aime24
python -m rlvr_vs_base.analyze --config "$CFG" stage2-candidates --benchmark math500

cat <<'EOF'
=== STAGE 1 COMPLETE ===
Review runs/report/*.png (interim curves; ablation curve should tell the same
story as the main prompt) and runs/stage2/*.json (candidate lists, capped).
Next: bash scripts/stage2.sh
EOF
