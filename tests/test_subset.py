"""Selection rules: symmetric hard-subset rule and stage-2 candidates."""

from rlvr_vs_base.analyze import select_hard_subset, stage2_candidates


def _problems(n=10):
    return [{"problem_id": f"p{i}", "level": str(i % 3)} for i in range(n)]


def _agg(values: dict[str, tuple[int, int]]) -> dict:
    return {pid: {"n": n, "c": c} for pid, (n, c) in values.items()}


def test_symmetric_rule_and_topup():
    problems = _problems(10)
    agg_b = _agg({f"p{i}": (16, 16) for i in range(10)})
    agg_r = _agg({f"p{i}": (16, 16) for i in range(10)})
    agg_b["p5"] = {"n": 16, "c": 0}  # hard for base only
    agg_r["p3"] = {"n": 16, "c": 0}  # hard for RL only

    sel = select_hard_subset(problems, agg_b, agg_r, threshold=0.5, target_size=4, seed=1)
    ids = set(sel["problem_ids"])
    assert {"p3", "p5"} <= ids, "rule must be symmetric in the two models"
    assert len(ids) == 4
    assert sel["composition"]["rule_selected"] == 2
    assert sel["composition"]["topup"] == 2

    sel2 = select_hard_subset(problems, agg_b, agg_r, threshold=0.5, target_size=4, seed=1)
    assert sel2["problem_ids"] == sel["problem_ids"], "selection must be deterministic"


def test_rule_overflow_keeps_all_hard():
    problems = _problems(10)
    agg_b = _agg({f"p{i}": (16, 0) for i in range(10)})
    agg_r = _agg({f"p{i}": (16, 16) for i in range(10)})
    sel = select_hard_subset(problems, agg_b, agg_r, threshold=0.5, target_size=3, seed=1)
    assert len(sel["problem_ids"]) == 10, "rule-selected problems are never dropped"


def test_missing_pilot_data_is_kept():
    problems = _problems(3)
    agg_b = _agg({"p0": (16, 16), "p1": (16, 16)})  # p2 missing
    agg_r = _agg({"p0": (16, 16), "p1": (16, 16), "p2": (16, 16)})
    sel = select_hard_subset(problems, agg_b, agg_r, threshold=0.5, target_size=1, seed=0)
    assert "p2" in sel["problem_ids"]


def test_stage2_rule():
    agg_b = _agg({"a": (256, 0), "b": (256, 100), "c": (256, 2), "d": (256, 0)})
    agg_r = _agg({"a": (256, 10), "b": (256, 120), "c": (256, 0), "d": (256, 0)})
    cands = stage2_candidates(agg_b, agg_r, min_successes=5, cap=12)
    ids = [c["problem_id"] for c in cands]
    # a: rl solves, base at 0 -> unresolved; c: base solves, rl at 0 -> unresolved
    # b: both saturated; d: both zero (no signal) -> excluded
    assert set(ids) == {"a", "c"}
    assert ids[0] == "a", "ranked by accuracy gap"

    capped = stage2_candidates(agg_b, agg_r, min_successes=5, cap=1)
    assert [c["problem_id"] for c in capped] == ["a"]
