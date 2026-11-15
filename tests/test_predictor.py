"""Offline unit tests for the pure Rating-Space Elo-Survival model.

Everything here is synthetic: no network, no database, no Discord.  Only
``predictor`` is imported.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import predictor  # noqa: E402
from predictor import (  # noqa: E402
    AttemptRecord,
    CFPredictor,
    format_minutes,
    get_archetypes,
    get_era_factor,
    grad_hess,
    invert_matrix,
    log_posterior,
    solve_linear,
    walk_forward_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(
    rating: float = 1900.0,
    *,
    mins: float = 25.0,
    solved: bool = True,
    tags: list[str] | None = None,
    cid: int = 2050,
    days_ago: float = 1.0,
    was: int = 0,
    bootstrap: bool = False,
    contest_start_ts: int | None = None,
    budget_mins: float | None = None,
    weight_mult: float = 1.0,
) -> AttemptRecord:
    return AttemptRecord(
        rating=float(rating),
        contest_id=int(cid),
        tags=list(tags or []),
        actual_mins=float(mins),
        was_count=int(was),
        solved=bool(solved),
        days_ago=float(days_ago),
        is_contest_bootstrap=bool(bootstrap),
        contest_start_ts=contest_start_ts,
        budget_mins=budget_mins,
        weight_mult=float(weight_mult),
    )


def fit_model(history: list[AttemptRecord], base_prior: float = 1800.0) -> CFPredictor:
    model = CFPredictor(base_prior=base_prior)
    model.fit(history, base_prior=base_prior)
    return model


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


# ---------------------------------------------------------------------------
# 1. Tag / era mapping
# ---------------------------------------------------------------------------


def test_archetype_and_era_mapping() -> None:
    assert get_archetypes(["dp", "trees", "fft"]) == [3, 4]
    assert get_archetypes(["fft", "flows"]) == []

    assert get_era_factor(2050) == 0.0
    assert get_era_factor(850) == 1.0
    assert get_era_factor(1450) == 0.5

    now = 1_700_000_000.0
    three_and_three_quarters = int(3.75 * 365.25 * 86400)
    assert get_era_factor(1900, now - three_and_three_quarters, now) == pytest.approx(0.5)
    assert get_era_factor(1900, now + 86400, now) == 0.0
    assert get_era_factor(1900, now - int(20 * 365.25 * 86400), now) == 1.0


# ---------------------------------------------------------------------------
# 2. Overtime solves count as exact solves
# ---------------------------------------------------------------------------


def test_overtime_solve_vs_fast_solve() -> None:
    fast = [make_record(1900, mins=15.0) for _ in range(10)]
    fast_model = fit_model(fast, base_prior=1800.0)
    assert fast_model.s30 > 1950.0

    overtime = [make_record(1900, mins=32.0, solved=True) for _ in range(20)]
    overtime_model = fit_model(overtime, base_prior=1800.0)
    assert abs(overtime_model.s30 - 1888.0) < 40.0


# ---------------------------------------------------------------------------
# 3. Right-censored timeouts push skill down
# ---------------------------------------------------------------------------


def test_right_censored_timeouts_push_skill_down() -> None:
    baseline_records = [make_record(1800, mins=25.0) for _ in range(10)]
    baseline = fit_model(baseline_records, base_prior=1800.0)
    baseline_rating = baseline.predict_modern_rating(25.0)

    with_timeouts = baseline_records + [
        make_record(1800, mins=25.0, solved=False, cid=2050) for _ in range(5)
    ]
    lowered = fit_model(with_timeouts, base_prior=1800.0)
    assert lowered.s30 < baseline.s30
    assert lowered.predict_modern_rating(25.0) < baseline_rating

    easy_timeout = fit_model(
        [make_record(2400, mins=2.0, solved=False, cid=2050)],
        base_prior=1800.0,
    )
    assert abs(easy_timeout.s30 - 1800.0) < 1.0

    hard_timeout = fit_model(
        [make_record(1700, mins=40.0, solved=False, cid=2050)],
        base_prior=1800.0,
    )
    assert hard_timeout.s30 < 1775.0


# ---------------------------------------------------------------------------
# 4. Era discount
# ---------------------------------------------------------------------------


def test_era_discount_recommendation() -> None:
    model = CFPredictor()
    model.gamma = 200.0
    tags = ["dp", "graphs"]
    modern = model.predict_nominal_rating(25.0, 2050, tags)
    classic = model.predict_nominal_rating(25.0, 850, tags)
    assert classic - modern == pytest.approx(200.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Archetype handicaps diverge
# ---------------------------------------------------------------------------


def test_archetype_handicap_divergence() -> None:
    history = [make_record(1900, mins=12.0, tags=["dp"]) for _ in range(10)]
    history += [
        make_record(1900, mins=30.0, solved=False, tags=["greedy"])
        for _ in range(10)
    ]
    model = fit_model(history, base_prior=1800.0)
    assert model.delta[4] < -10.0
    assert model.delta[0] > 10.0
    assert model.predict_modern_rating(25.0, arch_idx=4) > model.predict_modern_rating(
        25.0, arch_idx=0
    )
    assert model.predict_solve_time(1900.0, 0.65, arch_idx=4) < model.predict_solve_time(
        1900.0, 0.65, arch_idx=0
    )


# ---------------------------------------------------------------------------
# 6. Time decay weights recent form
# ---------------------------------------------------------------------------


def test_time_decay_weights_recent_form() -> None:
    history = [
        make_record(1700, mins=30.0, solved=False, days_ago=120.0) for _ in range(15)
    ]
    history += [make_record(1900, mins=15.0, days_ago=1.0) for _ in range(15)]
    model = fit_model(history, base_prior=1800.0)
    assert model.s30 > 1900.0


# ---------------------------------------------------------------------------
# 8. Analytic gradient / Hessian vs. finite differences
# ---------------------------------------------------------------------------


def test_gradient_and_hessian_match_finite_differences() -> None:
    rng = random.Random(1234)
    tag_pool = [
        [],
        ["dp"],
        ["greedy"],
        ["trees", "dfs and similar"],
        ["math", "number theory"],
        ["data structures", "binary search"],
        ["bitmasks"],
    ]
    history = [
        make_record(
            rng.uniform(1200.0, 2600.0),
            mins=rng.uniform(5.0, 60.0),
            solved=rng.random() < 0.7,
            tags=rng.choice(tag_pool),
            cid=rng.choice([900, 1200, 1500, 1800, 2050]),
            days_ago=rng.uniform(0.0, 200.0),
            was=rng.randint(0, 3),
            bootstrap=rng.random() < 0.3,
        )
        for _ in range(7)
    ]
    theta = [1930.0, 170.0, 25.0, -35.0, 10.0, 0.0, -50.0, 20.0]
    gradient, hessian = grad_hess(theta, history, 1800.0, 180.0, 2.5)

    for index in range(len(theta)):
        h = 1e-4 * max(1.0, abs(theta[index]))
        up = list(theta)
        down = list(theta)
        up[index] += h
        down[index] -= h
        numeric = (
            log_posterior(up, history, 1800.0, 180.0, 2.5)
            - log_posterior(down, history, 1800.0, 180.0, 2.5)
        ) / (2.0 * h)
        assert abs(gradient[index] - numeric) <= 1e-8 + 1e-4 * abs(numeric)

    for row in range(len(theta)):
        h = 1e-3 * max(1.0, abs(theta[row]))
        up = list(theta)
        down = list(theta)
        up[row] += h
        down[row] -= h
        grad_up, _ = grad_hess(up, history, 1800.0, 180.0, 2.5)
        grad_down, _ = grad_hess(down, history, 1800.0, 180.0, 2.5)
        for column in range(len(theta)):
            numeric = (grad_up[column] - grad_down[column]) / (2.0 * h)
            assert abs(hessian[row][column] - numeric) <= 1e-8 + 1e-4 * abs(numeric)

    # A known 3x3 system.
    A = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    b = [8.0, -11.0, -3.0]
    assert solve_linear(A, b) == pytest.approx([2.0, 3.0, -1.0], abs=1e-12)

    inverse = invert_matrix(A)
    identity = matmul(A, inverse)
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            assert identity[row][column] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 9. Deterministic, convergent fit + standard errors
# ---------------------------------------------------------------------------


def _fit_history(count: int) -> list[AttemptRecord]:
    rng = random.Random(2024)
    tag_pool = [[], ["dp"], ["greedy"], ["trees"], ["math"]]
    records: list[AttemptRecord] = []
    for index in range(count):
        rating = rng.choice([1500, 1600, 1700, 1800, 1900, 2000])
        minutes = rng.uniform(8.0, 55.0)
        solved = rng.random() < 0.7
        records.append(
            make_record(
                rating,
                mins=minutes,
                solved=solved,
                tags=rng.choice(tag_pool),
                cid=rng.choice([1200, 1500, 1800, 2050]),
                days_ago=float(rng.uniform(0.0, 120.0)),
                was=rng.randint(0, 2),
            )
        )
    return records


def test_fit_is_deterministic_and_converges() -> None:
    small = _fit_history(10)
    large = _fit_history(40)

    first = CFPredictor(base_prior=1800.0)
    summary = first.fit(small, base_prior=1800.0)
    assert isinstance(summary.objective_trace, list)
    for earlier, later in zip(summary.objective_trace, summary.objective_trace[1:]):
        assert earlier <= later + 1e-12
    assert summary.converged is True

    second = CFPredictor(base_prior=1800.0)
    second.fit(small, base_prior=1800.0)
    assert first.s30 == second.s30
    assert first.gamma == second.gamma
    assert first.delta == second.delta
    assert first.contest_offset == second.contest_offset

    small_model = fit_model(small, base_prior=1800.0)
    large_model = fit_model(large, base_prior=1800.0)
    assert large_model.stderr["s30"] < small_model.stderr["s30"]

    prior_sds = {
        "s30": 150.0,
        "gamma": 75.0,
        "contest_offset": 50.0,
    }
    for model in (small_model, large_model):
        assert 0.0 < model.stderr["s30"] <= prior_sds["s30"] + 1e-9
        assert 0.0 < model.stderr["gamma"] <= prior_sds["gamma"] + 1e-9
        assert 0.0 < model.stderr["contest_offset"] <= prior_sds["contest_offset"] + 1e-9
        for value in model.stderr["delta"]:
            assert 0.0 < value <= 60.0 + 1e-9


# ---------------------------------------------------------------------------
# 10. WA penalty is not double counted
# ---------------------------------------------------------------------------


def test_wa_penalty_not_double_counted() -> None:
    clean = [make_record(1800, mins=25.0, was=0) for _ in range(12)]
    messy = [make_record(1800, mins=25.0, was=3) for _ in range(12)]
    clean_model = fit_model(clean, base_prior=1800.0)
    messy_model = fit_model(messy, base_prior=1800.0)
    assert clean_model.s30 == pytest.approx(messy_model.s30, abs=1e-12)

    record = make_record(1800, mins=21.5, was=4)
    assert predictor.effective_time(record, wa_penalty=5.0) == pytest.approx(41.5)
    assert predictor.effective_time(record) == pytest.approx(21.5)


# ---------------------------------------------------------------------------
# 11. Contest offset is learned
# ---------------------------------------------------------------------------


def test_contest_offset_learned() -> None:
    history = [make_record(1900, mins=30.0) for _ in range(20)]
    history += [
        make_record(1900, mins=45.0, bootstrap=True) for _ in range(20)
    ]
    model = fit_model(history, base_prior=1800.0)
    assert model.contest_offset > 15.0
    assert abs(model.s30 - 1900.0) < 40.0


# ---------------------------------------------------------------------------
# 14. Walk-forward backtest
# ---------------------------------------------------------------------------


def test_walk_forward_backtest() -> None:
    rng = random.Random(0)
    history: list[AttemptRecord] = []
    for index in range(80):
        rating = rng.uniform(1600.0, 2400.0)
        u = rng.uniform(0.01, 0.99)
        budget = 30.0
        days_ago = float(80 - index)
        tau = 30.0 * math.exp((rating - 2000.0) / 180.0)
        solve_time = tau * ((u / (1.0 - u)) ** (1.0 / 2.5))
        if solve_time <= 60.0:
            history.append(
                make_record(
                    rating,
                    mins=solve_time,
                    solved=True,
                    cid=2050,
                    days_ago=days_ago,
                    budget_mins=budget,
                )
            )
        else:
            history.append(
                make_record(
                    rating,
                    mins=60.0,
                    solved=False,
                    cid=2050,
                    days_ago=days_ago,
                    budget_mins=budget,
                )
            )

    result = walk_forward_backtest(history, min_train=10, max_eval=None)
    assert result.log_loss < 0.65
    assert result.n == 70
    assert sum(count for _, _, count in result.bins) == 70
    assert len(result.bins) == 10

    default_result = walk_forward_backtest(history, min_train=10)
    assert default_result.n == 60


# ---------------------------------------------------------------------------
# 15. Rating <-> time predictions are exact inverses
# ---------------------------------------------------------------------------


def test_rating_and_time_predictions_are_exact_inverses() -> None:
    model = CFPredictor()
    model.s30 = 1950.0
    model.gamma = 200.0
    model.kappa = 150.0
    model.k = 3.0
    model.delta = [40.0, -30.0, 10.0, 0.0, -55.0]
    model.contest_offset = 35.0

    for target_mins in [5.0, 25.0, 40.0, 90.0]:
        for confidence in [0.3, 0.5, 0.65, 0.8]:
            for arch_idx in [None, 0, 1, 4]:
                for contest in [False, True]:
                    rating = model.predict_modern_rating(
                        target_mins, confidence, arch_idx, contest
                    )
                    recovered = model.predict_solve_time(
                        rating,
                        confidence,
                        arch_idx=arch_idx,
                        era_y=0.0,
                        contest=contest,
                    )
                    assert recovered == pytest.approx(target_mins, abs=1e-9)
                    probability = model.predict_solve_probability(
                        rating,
                        target_mins=target_mins,
                        arch_idx=arch_idx,
                        era_y=0.0,
                        contest=contest,
                    )
                    assert probability == pytest.approx(confidence, abs=1e-9)


# ---------------------------------------------------------------------------
# 16. Duration formatting
# ---------------------------------------------------------------------------


def test_format_minutes() -> None:
    assert format_minutes(0.4) == "<1m"
    assert format_minutes(14.2) == "14m"
    assert format_minutes(59.6) == "1h00m"
    assert format_minutes(75.0) == "1h15m"
    assert format_minutes(179.6) == "3h00m"
    assert format_minutes(181.0) == ">3h"
