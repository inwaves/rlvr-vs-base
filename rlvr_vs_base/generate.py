"""vLLM batch generation with resume support.

Design:
- Output shards: runs/{benchmark}/{model_key}/gens/{template}/shard-*.jsonl.gz,
  one record per sample. Append-only; atomic rename on completion.
- Resume: existing samples per problem are counted across all shards; only
  the deficit (target N minus existing) is generated. All stages accumulate
  into the same pool — every sample comes from the same sampling distribution
  (same prompt, temperature, top-p; engine seed varies per invocation, which
  mirrors the original repo's cross-run seed mechanism).
- Sampling diversity: engine-level seed per invocation; NO per-request seed,
  so vLLM's internal RNG state diversifies samples within and across rounds
  (this is the mechanism the limit-of-RLVR README documents).
"""

import argparse
import datetime
import gzip
import json
import os
import time
import uuid
from pathlib import Path

from .config import gens_dir, load_config, runs_root, subset_path
from .data import load_benchmark
from .prompts import render
from .status import write_status


def count_existing(dirpath: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not dirpath.exists():
        return counts
    for f in sorted(dirpath.glob("shard-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                counts[rec["problem_id"]] = counts.get(rec["problem_id"], 0) + 1
    return counts


def load_subset_ids(path: Path) -> set[str]:
    with open(path) as f:
        data = json.load(f)
    ids = data["problem_ids"] if isinstance(data, dict) else data
    return set(ids)


def resolve_subset(cfg, bcfg, benchmark, stage, subset_file) -> set[str] | None:
    """Explicit --subset-file wins; 'main' auto-loads the configured subset;
    'extension' REQUIRES an explicit file (the stage-2 candidate list) so a
    typo cannot silently launch a full-benchmark deep run."""
    if subset_file:
        return load_subset_ids(Path(subset_file))
    if stage == "extension":
        raise SystemExit(
            f"{benchmark}: stage 'extension' requires --subset-file "
            "(runs/stage2/<benchmark>.json from 'analyze stage2-candidates')"
        )
    if stage != "pilot" and bcfg.get("main_subset"):
        p = subset_path(cfg, bcfg["main_subset"])
        if not p.exists():
            raise SystemExit(
                f"{benchmark}: subset file {p} not found — run "
                "'python -m rlvr_vs_base.analyze select-math500-hard' first"
            )
        return load_subset_ids(p)
    return None


def plan_benchmark(cfg, benchmark, model_key, template, stage, target_n, subset_file, limit):
    bcfg = cfg["benchmarks"][benchmark]
    if target_n is None:
        stages = bcfg.get("stages", {})
        if stage not in stages:
            raise SystemExit(
                f"{benchmark}: stage '{stage}' is not budgeted in config "
                f"(available: {list(stages)}); pass --target-n to override"
            )
        target_n = stages[stage]

    problems = load_benchmark(cfg, benchmark)
    subset = resolve_subset(cfg, bcfg, benchmark, stage, subset_file)
    if subset is not None:
        missing = subset - {p["problem_id"] for p in problems}
        if missing:
            raise SystemExit(f"{benchmark}: subset ids not in benchmark: {sorted(missing)[:5]} ...")
        problems = [p for p in problems if p["problem_id"] in subset]
    if limit:
        problems = problems[:limit]

    outdir = gens_dir(cfg, benchmark, model_key, template)
    existing = count_existing(outdir)
    deficits = {
        p["problem_id"]: max(0, target_n - existing.get(p["problem_id"], 0)) for p in problems
    }
    return {
        "benchmark": benchmark,
        "bcfg": bcfg,
        "target_n": target_n,
        "problems": problems,
        "deficits": deficits,
        "outdir": outdir,
    }


def write_shard(outdir: Path, records: list[dict]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"shard-{uuid.uuid4().hex[:12]}.jsonl.gz"
    tmp = outdir / f".tmp-{name}"
    with gzip.open(tmp, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    final = outdir / name
    os.replace(tmp, final)
    return final


def run_generation(llm, plan, model_key, hf_id, template, scfg, engine_seed, rr):
    """Generate the deficit for one benchmark plan using an existing engine.

    Work is issued in groups of at most `prompts_per_call` prompts so that a
    shard lands and the heartbeat updates every few minutes — this bounds
    both data loss on preemption and heartbeat staleness for monitoring.
    """
    from vllm import SamplingParams

    bcfg = plan["bcfg"]
    benchmark = plan["benchmark"]
    deficits = dict(plan["deficits"])
    prompts_by_pid = {p["problem_id"]: render(template, p["problem"]) for p in plan["problems"]}
    chunk_n = scfg["chunk_n"]
    group_size = scfg.get("prompts_per_call", 64)
    total_new = sum(deficits.values())
    done = 0
    t0 = time.time()

    while True:
        pending = [pid for pid, d in deficits.items() if d > 0]
        if not pending:
            break
        # Largest deficits first so coverage stays even across problems.
        pending.sort(key=lambda pid: -deficits[pid])
        group = pending[:group_size]
        t_grp = time.time()
        batch_params = [
            SamplingParams(
                n=min(chunk_n, deficits[pid]),
                temperature=scfg["temperature"],
                top_p=scfg["top_p"],
                max_tokens=bcfg["max_tokens"],
            )
            for pid in group
        ]
        outs = llm.generate([prompts_by_pid[pid] for pid in group], batch_params)
        records = []
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for pid, out in zip(group, outs):
            for comp in out.outputs:
                records.append(
                    {
                        "sample_uid": uuid.uuid4().hex,
                        "problem_id": pid,
                        "text": comp.text,
                        "finish_reason": comp.finish_reason,
                        "completion_tokens": len(comp.token_ids),
                        "model": hf_id,
                        "model_key": model_key,
                        "benchmark": benchmark,
                        "template": template,
                        "engine_seed": engine_seed,
                        "temperature": scfg["temperature"],
                        "top_p": scfg["top_p"],
                        "max_tokens": bcfg["max_tokens"],
                        "created": now,
                    }
                )
            deficits[pid] -= len(out.outputs)
        write_shard(plan["outdir"], records)
        done += len(records)

        # Telemetry + quality canaries for this group.
        grp_toks = sum(r["completion_tokens"] for r in records)
        grp_dt = max(time.time() - t_grp, 1e-9)
        cap_pct = 100 * sum(1 for r in records if r["finish_reason"] == "length") / len(records)
        empty_pct = 100 * sum(1 for r in records if len(r["text"].strip()) < 5) / len(records)
        elapsed = max(time.time() - t0, 1e-9)
        eta_min = (total_new - done) / max(done / elapsed, 1e-9) / 60
        print(
            f"[{benchmark}/{model_key}] {done}/{total_new} samples | "
            f"{grp_toks / grp_dt:,.0f} tok/s | cap-hit {cap_pct:.1f}% | ETA {eta_min:.0f} min"
        )
        if cap_pct > 10:
            print(f"WARN [{benchmark}/{model_key}] cap-hit {cap_pct:.1f}% this group — check max_tokens (gate G2)")
        if empty_pct > 2:
            print(f"WARN [{benchmark}/{model_key}] {empty_pct:.1f}% near-empty completions — check prompt/template")
        write_status(
            rr,
            {
                "phase": "generate",
                "benchmark": benchmark,
                "model": model_key,
                "template": template,
                "samples_done": done,
                "samples_total": total_new,
                "group_tok_per_s": round(grp_toks / grp_dt),
                "cap_hit_pct_group": round(cap_pct, 1),
                "empty_pct_group": round(empty_pct, 1),
                "eta_min": round(eta_min),
                "engine_seed": engine_seed,
            },
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch generation with resume")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True, help="model key from config (base|rl)")
    p.add_argument("--benchmark", required=True, help="benchmark name, or comma-separated list")
    p.add_argument("--stage", required=True, help="pilot|main|extension (sets target N from config)")
    p.add_argument("--template", default=None, help="prompt template (default: config prompt.template)")
    p.add_argument("--subset-file", default=None, help="JSON file restricting problem ids")
    p.add_argument("--target-n", type=int, default=None, help="override stage target N")
    p.add_argument("--limit-problems", type=int, default=None, help="smoke-test cap")
    p.add_argument("--seed", type=int, default=None, help="engine seed (default: time-derived)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--dry-run", action="store_true", help="print plan + first prompt; no GPU needed")
    args = p.parse_args()

    cfg = load_config(args.config)
    template = args.template or cfg["prompt"]["template"]
    scfg = cfg["sampling"]
    hf_id = cfg["models"][args.model]["hf_id"]
    benchmarks = [b.strip() for b in args.benchmark.split(",")]

    plans = [
        plan_benchmark(
            cfg, b, args.model, template, args.stage, args.target_n,
            args.subset_file, args.limit_problems,
        )
        for b in benchmarks
    ]

    total_deficit = sum(sum(pl["deficits"].values()) for pl in plans)
    for pl in plans:
        n_pending = sum(1 for v in pl["deficits"].values() if v > 0)
        print(
            f"PLAN {pl['benchmark']}/{args.model}/{template} stage={args.stage}: "
            f"{len(pl['problems'])} problems, target N={pl['target_n']}, "
            f"{sum(pl['deficits'].values())} new samples across {n_pending} problems "
            f"(max_tokens={pl['bcfg']['max_tokens']})"
        )
    if args.dry_run:
        if plans[0]["problems"]:
            sample = plans[0]["problems"][0]
            print("\n--- RENDERED PROMPT (first problem, verbatim between markers) ---")
            print(render(template, sample["problem"]))
            print("--- END PROMPT ---")
        return
    if total_deficit == 0:
        print("Nothing to do: all targets already met.")
        return

    engine_seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    max_len = max(pl["bcfg"]["max_tokens"] for pl in plans) + 1024
    rr = runs_root(cfg)
    write_status(rr, {"phase": "engine_loading", "model": args.model, "hf_id": hf_id, "benchmarks": benchmarks})

    from vllm import LLM

    llm = LLM(
        model=hf_id,
        dtype="bfloat16",
        seed=engine_seed,
        max_model_len=max_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    for pl in plans:
        run_generation(llm, pl, args.model, hf_id, template, scfg, engine_seed, rr)
    write_status(rr, {"phase": "idle", "note": "generation complete", "model": args.model, "benchmarks": benchmarks})
    print("Generation complete.")


if __name__ == "__main__":
    main()
