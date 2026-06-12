"""Aggregate graded shards into per-problem (n, c) counts."""

import argparse
import gzip
import json
from pathlib import Path

from .config import aggregate_path, graded_dir, load_config


def aggregate_dir(graded: Path) -> dict:
    problems: dict[str, dict] = {}
    for f in sorted(graded.glob("shard-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                d = problems.setdefault(
                    r["problem_id"], {"n": 0, "c": 0, "cap_hits": 0, "sum_tokens": 0}
                )
                d["n"] += 1
                d["c"] += int(r["correct"])
                capped = r["finish_reason"] == "length" or (
                    r["finish_reason"] is None and r["completion_tokens"] >= r["max_tokens"]
                )
                d["cap_hits"] += int(capped)
                d["sum_tokens"] += r["completion_tokens"]

    n_total = sum(d["n"] for d in problems.values())
    summary = {
        "problems": len(problems),
        "samples": n_total,
        "mean_tokens": (sum(d["sum_tokens"] for d in problems.values()) / n_total) if n_total else 0.0,
        "cap_hit_rate": (sum(d["cap_hits"] for d in problems.values()) / n_total) if n_total else 0.0,
        "pass_at_1": (sum(d["c"] / d["n"] for d in problems.values()) / len(problems)) if problems else 0.0,
    }
    return {"problems": problems, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate graded shards")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", required=True, help="benchmark name, or comma-separated list")
    p.add_argument("--template", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    template = args.template or cfg["prompt"]["template"]

    for benchmark in [b.strip() for b in args.benchmark.split(",")]:
        gdir = graded_dir(cfg, benchmark, args.model, template)
        agg = aggregate_dir(gdir) if gdir.exists() else {"problems": {}, "summary": {}}
        out = aggregate_path(cfg, benchmark, args.model, template)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=1)
        print(f"{benchmark}/{args.model}/{template}: {agg['summary']}")


if __name__ == "__main__":
    main()
