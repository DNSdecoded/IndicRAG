import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stats  # noqa: E402


def test_ci_brackets_the_mean_and_is_deterministic():
    vals = [0.1, 0.4, 0.5, 0.9, 0.7, 0.3, 0.6]
    a = stats.bootstrap_ci(vals)
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert a == stats.bootstrap_ci(vals)


def test_constant_sample_has_zero_width_ci():
    ci = stats.bootstrap_ci([0.5] * 10)
    assert ci["lo"] == ci["hi"] == 0.5


def test_ci_covers_true_mean_most_of_the_time():
    rng = random.Random(1)
    hits = 0
    for trial in range(100):
        sample = [rng.gauss(0.6, 0.2) for _ in range(40)]
        ci = stats.bootstrap_ci(sample, resamples=500, seed=trial)
        hits += ci["lo"] <= 0.6 <= ci["hi"]
    assert hits >= 88   # nominal 95%; percentile bootstrap runs a little narrow at n=40


def test_identical_runs_are_not_a_regression():
    base = {f"q{i}": i / 20 for i in range(20)}
    regressed, ci = stats.is_regression(dict(base), base)
    assert not regressed and ci["mean"] == 0


def test_planted_drop_is_a_regression():
    rng = random.Random(2)
    base = {f"q{i}": rng.uniform(0.4, 0.9) for i in range(40)}
    new = {q: v - 0.15 + rng.uniform(-0.05, 0.05) for q, v in base.items()}
    regressed, ci = stats.is_regression(new, base)
    assert regressed and ci["hi"] < 0


def test_noise_is_not_a_regression():
    rng = random.Random(3)
    base = {f"q{i}": rng.uniform(0.4, 0.9) for i in range(40)}
    new = {q: v + rng.uniform(-0.05, 0.05) for q, v in base.items()}
    assert not stats.is_regression(new, base)[0]


def test_non_inferiority_margin():
    base = {f"q{i}": 0.5 for i in range(30)}
    assert stats.is_non_inferior({q: 0.49 for q in base}, base, margin=0.02)[0]
    assert not stats.is_non_inferior({q: 0.45 for q in base}, base, margin=0.02)[0]


def test_pairing_ignores_unshared_queries():
    ci = stats.paired_diff_ci({"a": 1.0, "b": 0.5}, {"a": 0.5, "c": 0.0})
    assert ci["queries"] == ["a"] and ci["mean"] == 0.5


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])
    with pytest.raises(ValueError):
        stats.paired_diff_ci({"a": 1}, {"b": 1})
