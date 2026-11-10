"""Unit tests for the pure estimation engine.

Every fixture in this file is a hardcoded, synthetic Python dict/list. These
tests never touch the network, never mock ``aiohttp`` and never import
``storage.py``; they exercise only the pure logic in ``estimator.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import estimator  # noqa: E402
from estimator import (  # noqa: E402
    SolveObservation,
    build_curve,
    curve_as_dict,
    extract_contest_deltas,
    recommend_rating,
    select_recent_participations,
    smooth_monotonic,
)

START = 1_600_000_000


def _submission(
    contest_id: int,
    index: str,
    relative_seconds: int,
    *,
    rating: int | None = 1500,
    participant: str = "CONTESTANT",
    verdict: str = "OK",
    submission_id: int = 1,
    start_time: int = START,
) -> dict:
    """Build one ``user.status``-shaped submission dict."""
    return {
        "id": submission_id,
        "contestId": contest_id,
        "creationTimeSeconds": start_time + max(relative_seconds, 0),
        "relativeTimeSeconds": relative_seconds,
        "problem": {
            "contestId": contest_id,
            "index": index,
            "name": f"Problem {index}",
            "rating": rating,
            "tags": ["math"],
        },
        "author": {
            "participantType": participant,
            "startTimeSeconds": start_time,
        },
        "verdict": verdict,
    }


def _observation(contest_id: int, rating: int, minutes: float) -> SolveObservation:
    return SolveObservation(
        contest_id=contest_id,
        problem_index="A",
        rating=rating,
        minutes=minutes,
        participant_type="CONTESTANT",
        start_time_seconds=START,
        submission_time=START + int(minutes * 60),
    )


# ---------------------------------------------------------------------------
# 1. Contest delta extraction
# ---------------------------------------------------------------------------


def test_deltas_from_contest_with_duplicate_ac_and_practice() -> None:
    submissions = [
        _submission(1000, "A", 600, submission_id=1),
        _submission(1000, "A", 900, submission_id=2),  # duplicate AC, later
        _submission(1000, "B", 1800, submission_id=3),
        _submission(1000, "C", 4500, submission_id=4),
        _submission(1000, "D", 300, participant="PRACTICE", submission_id=5),
    ]

    deltas = extract_contest_deltas(submissions)

    assert [delta.problem_index for delta in deltas] == ["A", "B", "C"]
    assert [delta.minutes for delta in deltas] == pytest.approx([10.0, 20.0, 45.0])


def test_earliest_ac_wins_regardless_of_api_order() -> None:
    submissions = [
        _submission(2000, "A", 900, submission_id=1),  # arrives first, not earliest
        _submission(2000, "A", 600, submission_id=2),
        _submission(2000, "B", 1800, submission_id=3),
    ]

    deltas = extract_contest_deltas(submissions)

    assert [delta.minutes for delta in deltas] == pytest.approx([10.0, 20.0])


def test_practice_and_invalid_submissions_are_ignored() -> None:
    submissions = [
        _submission(3000, "A", 300, participant="PRACTICE", submission_id=1),
        _submission(3000, "B", 600, verdict="WRONG_ANSWER", submission_id=2),
        _submission(3000, "C", 900, rating=None, submission_id=3),
        _submission(3000, "D", 0, submission_id=4),
        _submission(3000, "E", 18001, submission_id=5),  # over five hours
    ]

    assert extract_contest_deltas(submissions) == []


def test_virtual_participations_are_separate() -> None:
    submissions = [
        _submission(4000, "A", 600, participant="VIRTUAL", submission_id=1, start_time=START),
        _submission(4000, "B", 1800, participant="VIRTUAL", submission_id=2, start_time=START),
        _submission(4000, "A", 1200, participant="VIRTUAL", submission_id=3, start_time=START - 7200),
    ]

    deltas = extract_contest_deltas(submissions)

    # Two participations: (A@600, B@1800) -> 10m, 20m; and (A@1200) -> 20m.
    assert sorted(delta.minutes for delta in deltas) == pytest.approx([10.0, 20.0, 20.0])


def test_select_recent_participations_keeps_newest_35() -> None:
    observations = [
        _observation(contest_id, 1500, 30.0) for contest_id in range(1, 41)
    ]

    recent = select_recent_participations(observations, limit=35)

    assert {observation.contest_id for observation in recent} == set(range(6, 41))


# ---------------------------------------------------------------------------
# 2. Monotonic smoothing
# ---------------------------------------------------------------------------


def test_monotonic_smoothing_removes_higher_rating_dip() -> None:
    smoothed = smooth_monotonic({1600: 35.0, 1700: 55.0, 1800: 44.0})

    assert smoothed[1700] == pytest.approx(55.0)
    assert smoothed[1800] >= smoothed[1700]
    ratings = sorted(smoothed)
    assert all(
        smoothed[higher] >= smoothed[lower]
        for lower, higher in zip(ratings, ratings[1:])
    )


def test_build_curve_is_non_decreasing_with_rated_history() -> None:
    observations = [
        _observation(1, 1600, 35.0),
        _observation(2, 1600, 35.0),
        _observation(3, 1700, 55.0),
        _observation(4, 1700, 57.0),
        _observation(5, 1800, 44.0),
        _observation(6, 1800, 42.0),
    ]

    curve = curve_as_dict(
        build_curve(observations, current_rating=1700, rated_contest_count=6)
    )

    assert curve[1700] == pytest.approx(56.0)  # median of 55 and 57
    assert curve[1800] >= curve[1700]
    ratings = sorted(curve)
    assert all(
        curve[higher] >= curve[lower]
        for lower, higher in zip(ratings, ratings[1:])
    )


# ---------------------------------------------------------------------------
# 3. Reverse lookup
# ---------------------------------------------------------------------------


def test_recommend_rating_for_known_curve() -> None:
    curve = {
        800: 5.0,
        900: 8.0,
        1000: 12.0,
        1100: 17.0,
        1200: 23.0,
        1300: 30.0,
        1400: 38.0,
        1500: 48.0,
        1600: 60.0,
        1700: 75.0,
        1800: 95.0,
        1900: 120.0,
        2000: 150.0,
    }

    # target * 1.08 => 16.2, 32.4, 64.8 minutes
    assert recommend_rating(curve, 15, 1500) == 1000
    assert recommend_rating(curve, 30, 1500) == 1300
    assert recommend_rating(curve, 60, 1500) == 1600


def test_recommend_rating_clamps_to_personal_range() -> None:
    curve = {
        800: 5.0,
        900: 8.0,
        1000: 12.0,
        1100: 17.0,
        1200: 23.0,
        1300: 30.0,
        1400: 38.0,
        1500: 48.0,
        1600: 60.0,
        1700: 75.0,
        1800: 95.0,
        1900: 120.0,
        2000: 150.0,
    }

    # 5 minutes maps to 800, but the lower clamp for a 1500 user is 1000.
    assert recommend_rating(curve, 5, 1500) == 1000
    # 180 minutes maps to 2000, but the upper clamp is current + 300.
    assert recommend_rating(curve, 180, 1500) == 1800
    # A 1000-rated user is clamped to at most 1300.
    assert recommend_rating(curve, 60, 1000) == 1300


# ---------------------------------------------------------------------------
# 4. Cold-start fallback
# ---------------------------------------------------------------------------


def test_fallback_curve_for_user_with_zero_contests() -> None:
    points = build_curve([], current_rating=1500, rated_contest_count=0)
    curve = curve_as_dict(points)

    assert curve[1200] == pytest.approx(12.0)
    assert curve[1300] == pytest.approx(18.0)
    assert curve[1400] == pytest.approx(28.0)
    assert curve[1500] == pytest.approx(40.0)
    assert curve[1600] == pytest.approx(55.0)
    assert curve[1700] == pytest.approx(75.0)

    ratings = sorted(curve)
    assert all(
        curve[higher] >= curve[lower]
        for lower, higher in zip(ratings, ratings[1:])
    )
    assert recommend_rating(points, 30, 1500) == 1400


def test_fallback_is_used_below_five_rated_contests() -> None:
    observations = [
        _observation(1, 1600, 20.0),
        _observation(2, 1600, 20.0),
    ]

    curve = curve_as_dict(
        build_curve(observations, current_rating=1500, rated_contest_count=2)
    )

    # The two fast observations are ignored in favour of the anchored fallback.
    assert curve[1500] == pytest.approx(40.0)
    assert curve[1600] == pytest.approx(55.0)


def test_fallback_handles_extreme_current_ratings() -> None:
    for rating in (800, 1500, 2400, 3500):
        curve = curve_as_dict(build_curve([], current_rating=rating, rated_contest_count=0))
        ratings = sorted(curve)
        assert all(
            curve[higher] >= curve[lower]
            for lower, higher in zip(ratings, ratings[1:])
        )
        assert all(minutes >= 0 for minutes in curve.values())
        assert curve[rating] == pytest.approx(40.0)


def test_normalize_rating_snaps_to_grid() -> None:
    assert estimator.normalize_rating(1600) == 1600
    assert estimator.normalize_rating(1620) == 1600
    assert estimator.normalize_rating(1680) == 1700
    assert estimator.normalize_rating(1250) == 1300
    assert estimator.normalize_rating(100) == 800
    assert estimator.normalize_rating(9999) == 3500
    assert estimator.normalize_rating(None) == 1500
