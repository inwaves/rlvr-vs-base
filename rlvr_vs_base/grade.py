"""Grading: extraction + math-verify equivalence, with per-item timeouts.

Extraction policy (matches the RL Zero training format first, then falls
back; the SAME policy is applied to both models, which is what makes the
comparison fair):
  1. last line-anchored 'Answer: <expr>' (the RL Zero trained format)
  2. last markdown-decorated answer line ('**Answer:** <expr>' etc.)
  3. last \\boxed{...} (brace-balanced)
  4. math-verify parse of the trailing text (last resort)

Verification uses math-verify equivalence (gold first, candidate second).
A pebble process pool gives per-item timeouts so one pathological parse
cannot hang a shard.
"""

import argparse
import gzip
import json
import os
import re
from pathlib import Path

from .config import gens_dir, graded_dir, load_config
from .data import load_benchmark

ANSWER_LINE_RE = re.compile(r"(?im)^[ \t]*Answer[ \t]*:[ \t]*(.+?)[ \t]*$")
ANSWER_DECOR_RE = re.compile(r"(?im)^[ \t>#*_]*Answer[ \t]*\**[ \t]*:[ \t]*\**[ \t]*(.+?)[ \t]*$")


def extract_last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    i = start + len("\\boxed")
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_candidate(text: str) -> tuple[str | None, str]:
    m = ANSWER_LINE_RE.findall(text)
    if m:
        return m[-1].strip(), "answer_line"
    m = ANSWER_DECOR_RE.findall(text)
    if m:
        return m[-1].strip(), "answer_line_decorated"
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip(), "boxed"
    return None, "none"


def grade_text(text: str, gold: str) -> dict:
    """Pure grading function (runs inside a worker process)."""
    from math_verify import parse, verify  # heavy import, cached per process

    candidate, method = extract_candidate(text)
    try:
        gold_parsed = parse(f"${gold}$")
        if not gold_parsed:
            gold_parsed = parse(gold)
        if candidate is not None:
            for wrapped in (f"${candidate}$", candidate):
                pred = parse(wrapped)
                if pred and verify(gold_parsed, pred):
                    return {"correct": True, "method": method}
            return {"correct": False, "method": method}
        # Last resort: let math-verify extract from the trailing text.
        pred = parse(text[-2000:])
        ok = bool(pred) and verify(gold_parsed, pred)
        return {"correct": bool(ok), "method": "fulltext"}
    except Exception:
        return {"correct": False, "method": f"{method}:error"}


def _worker(payload: tuple[str, str]) -> dict:
    text, gold = payload
    return grade_text(text, gold)


def grade_payloads(payloads: list, timeout: float, workers: int, worker=_worker) -> list[dict]:
    """Grade payloads with per-item timeouts, preserving input order.

    One pool.schedule() per payload with indexed result collection keeps
    outputs aligned with inputs by construction: a timed-out or crashed
    worker fills its own slot and cannot shift later results.
    Regression test: tests/test_grade_alignment.py.
    """
    from pebble import ProcessPool

    results: list[dict | None] = [None] * len(payloads)
    with ProcessPool(max_workers=workers) as pool:
        futures = [pool.schedule(worker, args=(p,), timeout=timeout) for p in payloads]
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result()
            except Exception as exc:  # TimeoutError, ProcessExpired, worker error
                results[i] = {"correct": False, "method": f"worker_error:{type(exc).__name__}"}
    return results  # type: ignore[return-value]


def grade_shard(shard: Path, out: Path, golds: dict[str, str], timeout: float, workers: int) -> dict:
    with gzip.open(shard, "rt") as f:
        recs = [json.loads(line) for line in f]

    payloads = [(r["text"], golds[r["problem_id"]]) for r in recs]
    results = grade_payloads(payloads, timeout, workers)

    assert len(results) == len(recs), f"grading count mismatch on {shard.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".tmp-{out.name}"
    n_correct = 0
    with gzip.open(tmp, "wt") as f:
        for r, g in zip(recs, results):
            n_correct += int(g["correct"])
            f.write(
                json.dumps(
                    {
                        "sample_uid": r["sample_uid"],
                        "problem_id": r["problem_id"],
                        "correct": g["correct"],
                        "method": g["method"],
                        "finish_reason": r["finish_reason"],
                        "completion_tokens": r["completion_tokens"],
                        "max_tokens": r["max_tokens"],
                    }
                )
                + "\n"
            )
    os.replace(tmp, out)
    return {"samples": len(recs), "correct": n_correct}


def main() -> None:
    p = argparse.ArgumentParser(description="Grade generated samples")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", required=True, help="benchmark name, or comma-separated list")
    p.add_argument("--template", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    template = args.template or cfg["prompt"]["template"]
    timeout = cfg["grading"]["timeout_seconds"]
    workers = cfg["grading"]["workers"] or os.cpu_count()

    for benchmark in [b.strip() for b in args.benchmark.split(",")]:
        golds = {r["problem_id"]: r["answer"] for r in load_benchmark(cfg, benchmark)}
        gdir = gens_dir(cfg, benchmark, args.model, template)
        odir = graded_dir(cfg, benchmark, args.model, template)
        shards = sorted(gdir.glob("shard-*.jsonl.gz")) if gdir.exists() else []
        if not shards:
            print(f"{benchmark}/{args.model}: no shards to grade")
            continue
        done = skipped = 0
        for shard in shards:
            out = odir / shard.name
            if out.exists():
                skipped += 1
                continue
            stats = grade_shard(shard, out, golds, timeout, workers)
            done += 1
            print(
                f"{benchmark}/{args.model}: graded {shard.name} "
                f"({stats['correct']}/{stats['samples']} correct)"
            )
        print(f"{benchmark}/{args.model}: {done} shards graded, {skipped} already done")


if __name__ == "__main__":
    main()
