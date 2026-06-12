"""Aggregation over graded shards."""

import gzip
import json

from rlvr_vs_base.aggregate import aggregate_dir


def write_shard(dirpath, name, recs):
    dirpath.mkdir(parents=True, exist_ok=True)
    with gzip.open(dirpath / name, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def rec(pid, correct, finish="stop", tokens=100, cap=8192, uid="u"):
    return {
        "sample_uid": uid,
        "problem_id": pid,
        "correct": correct,
        "method": "answer_line",
        "finish_reason": finish,
        "completion_tokens": tokens,
        "max_tokens": cap,
    }


def test_aggregate_counts(tmp_path):
    write_shard(
        tmp_path,
        "shard-aa.jsonl.gz",
        [rec("p1", True, uid="u1"), rec("p1", False, finish="length", tokens=8192, uid="u2")],
    )
    write_shard(tmp_path, "shard-bb.jsonl.gz", [rec("p2", False, tokens=50, uid="u3")])

    agg = aggregate_dir(tmp_path)
    assert agg["problems"]["p1"] == {"n": 2, "c": 1, "cap_hits": 1, "sum_tokens": 8292}
    assert agg["problems"]["p2"] == {"n": 1, "c": 0, "cap_hits": 0, "sum_tokens": 50}
    s = agg["summary"]
    assert s["problems"] == 2
    assert s["samples"] == 3
    assert abs(s["pass_at_1"] - 0.25) < 1e-9  # (1/2 + 0/1) / 2
    assert abs(s["cap_hit_rate"] - 1 / 3) < 1e-9


def test_aggregate_empty(tmp_path):
    agg = aggregate_dir(tmp_path)
    assert agg["problems"] == {}
