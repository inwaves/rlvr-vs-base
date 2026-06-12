"""Heartbeat: machine-readable progress for remote monitoring.

generate/grade write runs/status.json after every group/shard. Polling
`python -m rlvr_vs_base.status` (or `bash scripts/watch.sh`) from any shell
shows progress, throughput, quality canaries, and heartbeat age. A stale
heartbeat during generation means the engine is wedged: kill the process
and re-run the stage script — generation is resumable by design, so at
most one group of samples is lost.
"""

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path

STALE_AFTER_SECONDS = 20 * 60  # generous: one stage-1 group is ~5-10 min


def write_status(runs_root: Path, payload: dict) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(runs_root)
    payload = dict(payload)
    payload["disk_free_gb"] = round(du.free / 1e9, 1)
    payload["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = runs_root / ".tmp-status.json"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, runs_root / "status.json")


def main() -> None:
    from .config import load_config, runs_root as runs_root_fn

    p = argparse.ArgumentParser(description="Show the current heartbeat")
    p.add_argument("--config", default="configs/experiment.yaml")
    args = p.parse_args()

    path = runs_root_fn(load_config(args.config)) / "status.json"
    if not path.exists():
        print("No status.json yet — nothing has run.")
        return
    with open(path) as f:
        s = json.load(f)
    print(json.dumps(s, indent=1))
    updated = datetime.datetime.fromisoformat(s["updated"])
    age = (datetime.datetime.now(datetime.timezone.utc) - updated).total_seconds()
    if s.get("phase") in ("generate", "engine_loading", "grade") and age > STALE_AFTER_SECONDS:
        print(f"\nHeartbeat age: {age:.0f}s — STALE. Engine likely wedged: kill and re-run the stage (resumable).")
    else:
        print(f"\nHeartbeat age: {age:.0f}s (fresh)")


if __name__ == "__main__":
    main()
