"""Analysis CLI: subset selection, stage-2 candidates, pilot report, final report.

All selection rules here are pre-registered in configs/experiment.yaml and
described in the kb replication plan. They are symmetric in the two models.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import aggregate_path, load_config, runs_root, stage2_path, subset_path


def load_agg(cfg, benchmark, model_key, template) -> dict:
    path = aggregate_path(cfg, benchmark, model_key, template)
    if not path.exists():
        raise SystemExit(f"Missing aggregate {path} — run grade + aggregate first")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- selection

def select_hard_subset(
    problems: list[dict],
    agg_base: dict,
    agg_rl: dict,
    threshold: float,
    target_size: int,
    seed: int,
) -> dict:
    """Symmetric rule: keep every problem where EITHER model's pilot accuracy
    is below `threshold`; top up to `target_size` with a stratified-by-level
    random draw from the remainder. If the rule alone exceeds target_size,
    keep all rule-selected problems (the rule is primary, the size is a
    target)."""
    hard, rest = [], []
    for p in problems:
        pid = p["problem_id"]
        db, dr = agg_base.get(pid), agg_rl.get(pid)
        if db is None or dr is None or db["n"] == 0 or dr["n"] == 0:
            hard.append(pid)  # no pilot data -> keep, never silently drop
            continue
        if (db["c"] / db["n"] < threshold) or (dr["c"] / dr["n"] < threshold):
            hard.append(pid)
        else:
            rest.append(p)

    topup: list[str] = []
    if len(hard) < target_size and rest:
        need = target_size - len(hard)
        by_level: dict[str, list[str]] = defaultdict(list)
        for p in rest:
            by_level[p["level"]].append(p["problem_id"])
        rng = random.Random(seed)
        levels = sorted(by_level)
        total = sum(len(v) for v in by_level.values())
        alloc = {lvl: int(round(need * len(by_level[lvl]) / total)) for lvl in levels}
        # fix rounding drift deterministically
        drift = need - sum(alloc.values())
        for lvl in levels:
            if drift == 0:
                break
            step = 1 if drift > 0 else -1
            if alloc[lvl] + step >= 0 and alloc[lvl] + step <= len(by_level[lvl]):
                alloc[lvl] += step
                drift -= step
        for lvl in levels:
            take = min(alloc[lvl], len(by_level[lvl]))
            topup.extend(rng.sample(sorted(by_level[lvl]), take))

    ids = sorted(set(hard) | set(topup))
    return {
        "problem_ids": ids,
        "composition": {
            "rule_selected": len(hard),
            "topup": len(topup),
            "total": len(ids),
        },
        "params": {"threshold": threshold, "target_size": target_size, "seed": seed},
    }


def stage2_candidates(
    agg_base: dict, agg_rl: dict, min_successes: int, cap: int
) -> list[dict]:
    """Pre-registered rule: a problem is unresolved if one model solves it
    while the other has fewer than `min_successes` successes — i.e. the
    trailing model's curve is not yet saturated at the current N. Ranked by
    accuracy gap, capped at `cap`."""
    out = []
    for pid in sorted(set(agg_base) | set(agg_rl)):
        db, dr = agg_base.get(pid), agg_rl.get(pid)
        if not db or not dr or db["n"] == 0 or dr["n"] == 0:
            continue
        cb, cr = db["c"], dr["c"]
        unresolved = (cr > 0 and cb < min_successes) or (cb > 0 and cr < min_successes)
        if unresolved:
            gap = abs(cb / db["n"] - cr / dr["n"])
            out.append(
                {"problem_id": pid, "gap": gap, "base": {"n": db["n"], "c": cb}, "rl": {"n": dr["n"], "c": cr}}
            )
    out.sort(key=lambda d: (-d["gap"], d["problem_id"]))
    return out[:cap]


# ---------------------------------------------------------------- commands

def cmd_select_math500_hard(cfg, template) -> None:
    from .data import load_benchmark

    scfg = cfg["subset"]["math500_hard"]
    problems = load_benchmark(cfg, "math500")
    agg_b = load_agg(cfg, "math500", "base", template)["problems"]
    agg_r = load_agg(cfg, "math500", "rl", template)["problems"]
    sel = select_hard_subset(
        problems, agg_b, agg_r,
        scfg["accuracy_threshold"], scfg["target_size"], scfg["seed"],
    )
    out = subset_path(cfg, "math500_hard")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sel, f, indent=1)
    print(f"math500_hard: {sel['composition']} -> {out}")


def cmd_stage2(cfg, benchmark, template) -> None:
    s2 = cfg["stage2"]
    agg_b = load_agg(cfg, benchmark, "base", template)["problems"]
    agg_r = load_agg(cfg, benchmark, "rl", template)["problems"]
    cands = stage2_candidates(
        agg_b, agg_r, s2["min_successes_resolved"], s2["max_problems_per_benchmark"]
    )
    out = stage2_path(cfg, benchmark)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"problem_ids": [c["problem_id"] for c in cands], "candidates": cands}, f, indent=1)
    print(f"{benchmark}: {len(cands)} stage-2 candidates -> {out}")
    for c in cands:
        print(f"  {c['problem_id']}: base {c['base']['c']}/{c['base']['n']}, rl {c['rl']['c']}/{c['rl']['n']}")


def cmd_pilot_report(cfg, template, usd_per_btoken: float) -> None:
    print(f"{'benchmark':<10} {'model':<5} {'samples':>8} {'pass@1':>7} {'meanTok':>8} {'capHit%':>8}")
    means = {}
    for benchmark in cfg["benchmarks"]:
        for model_key in cfg["models"]:
            try:
                s = load_agg(cfg, benchmark, model_key, template)["summary"]
            except SystemExit:
                continue
            means[(benchmark, model_key)] = s["mean_tokens"]
            print(
                f"{benchmark:<10} {model_key:<5} {s['samples']:>8} {s['pass_at_1']:>7.3f} "
                f"{s['mean_tokens']:>8.0f} {100 * s['cap_hit_rate']:>7.1f}%"
            )
    # Forecast stage 1 + 2 token volume from measured means.
    fc = []
    sub = cfg["subset"]["math500_hard"]["target_size"]
    for model_key in cfg["models"]:
        t = 0.0
        b = cfg["benchmarks"]["math500"]["stages"]
        t += sub * (b["main"] - b["pilot"]) * means.get(("math500", model_key), 3000)
        a = cfg["benchmarks"]["aime24"]["stages"]
        t += 30 * (a["main"] - a["pilot"]) * means.get(("aime24", model_key), 6000)
        s2cap = cfg["stage2"]["max_problems_per_benchmark"]
        t += s2cap * (a["extension"] - a["main"]) * means.get(("aime24", model_key), 6000)
        fc.append((model_key, t))
    total = sum(t for _, t in fc)
    print("\nForecast (stage 1 + capped stage 2, output tokens):")
    for model_key, t in fc:
        print(f"  {model_key}: {t / 1e6:,.0f}M tokens")
    print(f"  TOTAL: {total / 1e9:.2f}B tokens  (~${total / 1e9 * usd_per_btoken:.0f} at ${usd_per_btoken:.0f}/B)")


def cmd_report(cfg, benchmark, template, bootstrap_iters: int) -> None:
    import numpy as np

    from .passk import bootstrap_curve, crossover_k, curve, k_grid

    agg_b = load_agg(cfg, benchmark, "base", template)["problems"]
    agg_r = load_agg(cfg, benchmark, "rl", template)["problems"]
    shared = sorted(set(agg_b) & set(agg_r))
    if not shared:
        raise SystemExit(f"{benchmark}: no shared problems with data")
    pb = {pid: agg_b[pid] for pid in shared}
    pr = {pid: agg_r[pid] for pid in shared}
    k_max = max(max(d["n"] for d in pb.values()), max(d["n"] for d in pr.values()))
    ks = k_grid(k_max)

    mb, mr = curve(pb, ks), curve(pr, ks)
    lb, ub = bootstrap_curve(pb, ks, iters=bootstrap_iters)
    lr, ur = bootstrap_curve(pr, ks, iters=bootstrap_iters)
    xk = crossover_k(ks, mb, mr)

    outdir = runs_root(cfg) / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": benchmark,
        "template": template,
        "problems": len(shared),
        "ks": ks,
        "base": {"mean": mb.tolist(), "lo": lb.tolist(), "hi": ub.tolist()},
        "rl": {"mean": mr.tolist(), "lo": lr.tolist(), "hi": ur.tolist()},
        "crossover_k": xk,
        "gap_at_k_max": float(mr[-1] - mb[-1]),
        "gap_at_1": float(mr[0] - mb[0]),
    }
    with open(outdir / f"{benchmark}.{template}.json", "w") as f:
        json.dump(payload, f, indent=1)

    from .plots import plot_passk  # lazy: matplotlib

    png = outdir / f"{benchmark}.{template}.png"
    plot_passk(payload, png)

    print(f"{benchmark} ({len(shared)} problems, k_max={k_max}):")
    print(f"  pass@1     base={mb[0]:.3f}  rl={mr[0]:.3f}  (rl-base = {mr[0] - mb[0]:+.3f})")
    print(f"  pass@k_max base={mb[-1]:.3f}  rl={mr[-1]:.3f}  (rl-base = {mr[-1] - mb[-1]:+.3f})")
    print(f"  crossover_k = {xk}")
    print(f"  report -> {png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analysis commands")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--template", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("select-math500-hard")
    s2 = sub.add_parser("stage2-candidates")
    s2.add_argument("--benchmark", required=True)
    pr = sub.add_parser("pilot-report")
    pr.add_argument("--usd-per-btoken", type=float, default=60.0)
    rp = sub.add_parser("report")
    rp.add_argument("--benchmark", required=True, help="benchmark name, or comma-separated list")
    rp.add_argument("--bootstrap-iters", type=int, default=1000)

    args = p.parse_args()
    cfg = load_config(args.config)
    template = args.template or cfg["prompt"]["template"]

    if args.cmd == "select-math500-hard":
        cmd_select_math500_hard(cfg, template)
    elif args.cmd == "stage2-candidates":
        cmd_stage2(cfg, args.benchmark, template)
    elif args.cmd == "pilot-report":
        cmd_pilot_report(cfg, template, args.usd_per_btoken)
    elif args.cmd == "report":
        for benchmark in [b.strip() for b in args.benchmark.split(",")]:
            cmd_report(cfg, benchmark, template, args.bootstrap_iters)


if __name__ == "__main__":
    main()
