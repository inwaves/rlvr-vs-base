"""Estimator correctness: unbiased pass@k vs exact combinatorial formula."""

import math

from rlvr_vs_base.passk import crossover_k, curve, k_grid, pass_at_k


def ref(n: int, c: int, k: int) -> float:
    # pass@k = 1 - C(n-c, k) / C(n, k); C(n-c, k) = 0 when n-c < k.
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def test_matches_combinatorial_exactly():
    for n in (5, 12):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert abs(pass_at_k(n, c, k) - ref(n, c, k)) < 1e-9, (n, c, k)


def test_edges():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 1, 10) == 1.0  # n - c < k


def test_monotone_in_k():
    vals = [pass_at_k(64, 3, k) for k in (1, 2, 4, 8, 16, 32, 64)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_clamp_k_beyond_n():
    # Adaptive-N policy: k > n reports pass@n (conservative).
    assert pass_at_k(4, 1, 16) == pass_at_k(4, 1, 4)


def test_k_grid():
    assert k_grid(8) == [1, 2, 4, 8]
    assert k_grid(12) == [1, 2, 4, 8, 12]


def test_curve_and_crossover():
    base = {"a": {"n": 8, "c": 4}, "b": {"n": 8, "c": 1}}
    rl = {"a": {"n": 8, "c": 8}, "b": {"n": 8, "c": 0}}
    ks = [1, 2, 4, 8]
    cb, cr = curve(base, ks), curve(rl, ks)
    assert cr[0] > cb[0]  # RL ahead at k=1
    assert cb[-1] > cr[-1]  # base ahead at k=8
    assert crossover_k(ks, cb, cr) == 2


def test_no_crossover():
    # RL strictly dominates until both curves saturate at 1.0 (k = n forces
    # pass@k = 1.0 whenever c > 0). The ceiling tie must NOT count as a
    # crossover — it is benchmark saturation, not the base overtaking.
    base = {"a": {"n": 8, "c": 1}}
    rl = {"a": {"n": 8, "c": 8}}
    ks = [1, 2, 4, 8]
    assert crossover_k(ks, curve(base, ks), curve(rl, ks)) is None
