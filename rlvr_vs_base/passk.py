"""Unbiased pass@k estimation, curves, bootstrap CIs, crossover detection.

pass@k uses the Codex-paper unbiased estimator, the same formula as
pass@k.py in the official limit-of-RLVR repo:
    pass@k = 1 - C(n-c, k) / C(n, k)
computed in product form for numerical stability.

Mixed-N handling (adaptive budget): for a problem with n < k samples the
estimator is undefined; we report pass@n for that problem instead (a
conservative clamp). The pre-registered stage-2 rule extends exactly the
problems where this clamp could bite (unsaturated curves), so the residual
bias is confined to saturated problems where pass@k is pinned near its
asymptote. Disclosed in the write-up.
"""

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if k > n:
        k = n  # conservative clamp, see module docstring
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def k_grid(k_max: int) -> list[int]:
    ks = []
    k = 1
    while k <= k_max:
        ks.append(k)
        k *= 2
    if ks[-1] != k_max:
        ks.append(k_max)
    return ks


def curve(problems: dict[str, dict], ks: list[int]) -> np.ndarray:
    """Mean pass@k over problems; problems is {pid: {'n': int, 'c': int}}."""
    vals = np.empty((len(problems), len(ks)))
    for i, d in enumerate(problems.values()):
        for j, k in enumerate(ks):
            vals[i, j] = pass_at_k(d["n"], d["c"], k)
    return vals.mean(axis=0)


def bootstrap_curve(
    problems: dict[str, dict], ks: list[int], iters: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile CI over problem resampling."""
    rng = np.random.default_rng(seed)
    per_problem = np.empty((len(problems), len(ks)))
    for i, d in enumerate(problems.values()):
        for j, k in enumerate(ks):
            per_problem[i, j] = pass_at_k(d["n"], d["c"], k)
    n = len(problems)
    means = np.empty((iters, len(ks)))
    for b in range(iters):
        idx = rng.integers(0, n, size=n)
        means[b] = per_problem[idx].mean(axis=0)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def crossover_k(ks: list[int], base_mean: np.ndarray, rl_mean: np.ndarray) -> int | None:
    """Smallest k in the grid where the base curve meets or exceeds the RL curve.

    Ties at the saturation ceiling (both curves at 1.0) are excluded: when
    every problem is solved by both models at k, equality is an artifact of
    the estimator hitting its ceiling, evidence of benchmark saturation
    rather than of the base model overtaking. Convergence below the ceiling
    still counts, and the gap-at-k_max metric captures it regardless.
    """
    eps = 1e-12
    for k, b, r in zip(ks, base_mean, rl_mean):
        if b >= r and not (b >= 1.0 - eps and r >= 1.0 - eps):
            return k
    return None
