#!/usr/bin/env bash
# One-time GPU box setup + preflight. Safe to re-run. See README runbook.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p runs/logs
LOG="runs/logs/setup_box-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

python -m pip install -U pip
pip install -e ".[dev]"
pip install vllm
python -c "import vllm; print('vllm', vllm.__version__)"
pip freeze > box_env_freeze.txt

echo "=== Unit tests (CPU) ==="
pytest -q

echo "=== Dataset checks ==="
python -m rlvr_vs_base.data --config configs/experiment.yaml --check

echo "=== Prompt dry-run (compare against README golden) ==="
python -m rlvr_vs_base.generate --config configs/experiment.yaml \
  --model base --benchmark aime24 --stage pilot --dry-run

echo "=== Model prefetch ==="
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3.1-7B-RL-Zero-Math"):
    print("downloading", m)
    snapshot_download(m)
PY

echo "=== Throughput probe (~2 min GPU; samples count toward the pilot) ==="
python -m rlvr_vs_base.generate --config configs/experiment.yaml \
  --model base --benchmark math500 --stage pilot --limit-problems 8 --target-n 4 --seed 999

echo "GATE G0: tests green, prompt matches golden, probe tokens/s sane."
echo "Next: bash scripts/stage0.sh"
