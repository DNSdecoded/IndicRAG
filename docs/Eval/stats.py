"""Bootstrap statistics for the eval gate (master plan §16.1).

Per-metric mean with a 95% bootstrap CI over queries, and a paired comparison
against a baseline: a regression is when the CI upper bound of (new - base) is
below 0. Resampling is by query and seeded, so the gate is deterministic.
Standard library only: CI must not grow a numpy/scipy dependency for this.
"""

import random
from statistics import fmean

RESAMPLES = 1000
SEED = 20261003


def _percentile(sorted_vals: list, q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list, q in [0, 1]."""
    if not sorted_vals:
        raise ValueError("empty sample")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(values: list, resamples: int = RESAMPLES, alpha: float = 0.05,
                 seed: int = SEED) -> dict:
    """Mean and (1 - alpha) percentile bootstrap CI of `values`."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(resamples))
    return {
        "mean": fmean(values),
        "lo": _percentile(means, alpha / 2),
        "hi": _percentile(means, 1 - alpha / 2),
        "n": n,
    }


def paired_diff_ci(new: dict, base: dict, resamples: int = RESAMPLES,
                   alpha: float = 0.05, seed: int = SEED) -> dict:
    """Bootstrap CI of mean(new[q] - base[q]) over the queries both runs scored.

    Paired by query id: the two runs share queries, and pairing removes the
    between-query variance that would otherwise swamp a real regression.
    """
    shared = sorted(set(new) & set(base))
    if not shared:
        raise ValueError("no shared query ids between runs")
    diffs = [new[q] - base[q] for q in shared]
    ci = bootstrap_ci(diffs, resamples, alpha, seed)
    ci["queries"] = shared
    return ci


def is_regression(new: dict, base: dict, **kw) -> tuple:
    """(regressed, ci): regressed when the diff CI's upper bound is below 0."""
    ci = paired_diff_ci(new, base, **kw)
    return ci["hi"] < 0, ci


def is_non_inferior(new: dict, base: dict, margin: float, **kw) -> tuple:
    """(ok, ci): non-inferior when the diff CI's lower bound is >= -margin (§F2, G-RET)."""
    ci = paired_diff_ci(new, base, **kw)
    return ci["lo"] >= -margin, ci
