"""Regression test for grading alignment under worker timeout/crash.

Guards the correctness-critical invariant: grading results must stay
aligned one-to-one, in order, with input samples even when some workers
time out or crash (otherwise pass@k aggregates are silently corrupted).
"""

import time

from rlvr_vs_base.grade import grade_payloads


def flaky_worker(x):
    if x == "timeout":
        time.sleep(5)
        return {"correct": True, "method": "late"}
    if x == "crash":
        raise RuntimeError("boom")
    return {"correct": True, "method": f"ok-{x}"}


def test_alignment_with_timeout_and_crash():
    payloads = ["a", "timeout", "crash", "b", "c"]
    results = grade_payloads(payloads, timeout=1.0, workers=2, worker=flaky_worker)

    assert len(results) == len(payloads)
    assert results[0]["method"] == "ok-a"
    assert results[1]["correct"] is False and results[1]["method"].startswith("worker_error")
    assert results[2]["correct"] is False and results[2]["method"].startswith("worker_error")
    assert results[3]["method"] == "ok-b"
    assert results[4]["method"] == "ok-c"
