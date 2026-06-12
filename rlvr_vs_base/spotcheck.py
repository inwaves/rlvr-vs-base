"""Export traces for the manual CoT spot check (guessing control).

Mirrors the paper's control: for problems with per-problem accuracy below a
threshold (default 5%) that were nonetheless solved, dump every correct
trace for manual inspection. AIME answers are integers in 0-999, so at
N=1024 a random guesser has a substantial chance of a lucky hit — low-rate
solves MUST pass this check before they count in the headline narrative.
"""

import argparse
import gzip
import json

from .config import gens_dir, graded_dir, load_config, runs_root


def main() -> None:
    p = argparse.ArgumentParser(description="Dump low-accuracy solved traces for manual CoT review")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--template", default=None)
    p.add_argument("--accuracy-below", type=float, default=0.05)
    args = p.parse_args()

    cfg = load_config(args.config)
    template = args.template or cfg["prompt"]["template"]

    # Per-problem counts and the uids of correct samples.
    counts: dict[str, dict] = {}
    correct_uids: dict[str, set] = {}
    gdir = graded_dir(cfg, args.benchmark, args.model, template)
    for f in sorted(gdir.glob("shard-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                d = counts.setdefault(r["problem_id"], {"n": 0, "c": 0})
                d["n"] += 1
                d["c"] += int(r["correct"])
                if r["correct"]:
                    correct_uids.setdefault(r["problem_id"], set()).add(r["sample_uid"])

    targets = {
        pid
        for pid, d in counts.items()
        if d["c"] > 0 and d["c"] / d["n"] < args.accuracy_below
    }
    if not targets:
        print("No problems below the accuracy threshold with solves — nothing to check.")
        return

    outdir = runs_root(cfg) / "spotcheck" / args.benchmark / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    dumped = 0
    src = gens_dir(cfg, args.benchmark, args.model, template)
    for f in sorted(src.glob("shard-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                pid = r["problem_id"]
                if pid in targets and r["sample_uid"] in correct_uids.get(pid, set()):
                    safe = pid.replace("/", "_")
                    out = outdir / f"{safe}.{r['sample_uid'][:8]}.txt"
                    out.write_text(r["text"])
                    dumped += 1

    index = outdir / "INDEX.txt"
    lines = [
        f"{pid}: {counts[pid]['c']}/{counts[pid]['n']} correct"
        for pid in sorted(targets)
    ]
    index.write_text("\n".join(lines) + "\n")
    print(f"{len(targets)} problems flagged, {dumped} correct traces dumped to {outdir}")


if __name__ == "__main__":
    main()
