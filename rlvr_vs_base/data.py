"""Benchmark loading and normalization.

Each benchmark is normalized to a list of records:
    {problem_id: str, problem: str, answer: str, level: str}

Dataset IDs and expected counts live in configs/experiment.yaml and are
asserted here so a silently changed upstream dataset fails loudly.
"""

import argparse


def load_benchmark(cfg: dict, name: str) -> list[dict]:
    from datasets import load_dataset  # lazy: not needed for offline tests

    bcfg = cfg["benchmarks"][name]
    ds = load_dataset(bcfg["dataset"], split=bcfg["split"])

    rows = []
    for r in ds:
        if name == "math500":
            pid = r["unique_id"]
            level = str(r.get("level", ""))
        elif name == "aime24":
            pid = f"aime24-{r['id']}"
            level = ""
        elif name == "aime25":
            pid = f"aime25-{r['id']}"
            level = ""
        else:
            raise ValueError(f"Unknown benchmark: {name}")
        rows.append(
            {
                "problem_id": str(pid),
                "problem": r["problem"].strip(),
                "answer": str(r["answer"]).strip(),
                "level": level,
            }
        )

    expected = bcfg["expected_count"]
    if len(rows) != expected:
        raise AssertionError(f"{name}: expected {expected} problems, got {len(rows)}")
    ids = [r["problem_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{name}: duplicate problem_ids")
    return rows


def main() -> None:
    from .config import load_config

    p = argparse.ArgumentParser(description="Sanity-check benchmark datasets")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    for name in cfg["benchmarks"]:
        rows = load_benchmark(cfg, name)
        golds = sum(1 for r in rows if r["answer"])
        print(f"{name}: {len(rows)} problems, {golds} with gold answers — OK")


if __name__ == "__main__":
    main()
