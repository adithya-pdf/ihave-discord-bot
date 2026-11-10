"""Personal solve-time estimation for the ``;ihave`` Codeforces practice bot.

This module is deliberately free of Discord, networking and database code:
everything here is a deterministic, pure transformation of ``user.status``
payloads, which makes the math straightforward to unit test.

Pipeline
--------
1. :func:`extract_contest_deltas` keeps accepted contest/virtual submissions and
   converts them into per-problem minutes using the difference between
   consecutive ``relativeTimeSeconds`` values inside one participation.
2. :func:`select_recent_participations` keeps the 35 newest participations so
   the curve reflects current form instead of ancient history.
3. :func:`aggregate_raw_minutes` buckets those deltas by rating and uses a
   trimmed mean (or median for tiny buckets) to survive outliers.
4. :func:`smooth_monotonic` fills missing buckets with :func:`numpy.interp` and
   enforces a non-decreasing curve with a forward cumulative maximum.
5. :func:`build_curve` falls back to a rating-anchored cold-start curve when the
   user has fewer than five rated contests.
6. :func:`recommend_rating` reverses the curve for a requested time budget.

No scikit-learn, no pandas, no external services.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Problems are rated on a 100-point grid between these bounds.
RATING_MIN = 800
RATING_MAX = 3500
RATING_STEP = 100
RATING_GRID: tuple[int, ...] = tuple(range(RATING_MIN, RATING_MAX + 1, RATING_STEP))

#: An accepted contest submission more than five hours after the contest start
#: is not evidence about "time spent" and is discarded.
MAX_RELATIVE_SECONDS = 5 * 60 * 60

#: How many recent contest participations feed the curve.
RECENT_PARTICIPATIONS = 35

#: Below this many rated contests the estimator uses the cold-start fallback.
MIN_RATED_CONTESTS = 5

#: A rating bucket needs at least this many solves to produce a raw average.
MIN_SAMPLES_PER_BUCKET = 2

#: ``;ihave 30`` picks the highest rating that is solvable within 8% of 30 min.
TARGET_TOLERANCE = 1.08

#: Recommendation clamp relative to the user's current Codeforces rating.
RECOMMEND_LOW_OFFSET = 500
RECOMMEND_HIGH_OFFSET = 300

DEFAULT_RATING = 1500

#: Cold-start anchors: (rating offset from current rating, minutes).
_FALLBACK_ANCHORS: tuple[tuple[int, float], ...] = (
    (-300, 12.0),
    (-200, 18.0),
    (-100, 28.0),
    (0, 40.0),
    (100, 55.0),
    (200, 75.0),
)
_FALLBACK_FLOOR = 3.0
_FALLBACK_CEILING = 240.0


class EstimatorError(ValueError):
    """Raised when estimator input is structurally invalid."""


@dataclass(frozen=True)
class SolveObservation:
    """One solved problem inside a contest participation, with its time delta."""

    contest_id: int
    problem_index: str
    rating: int
    minutes: float
    participant_type: str
    start_time_seconds: int
    submission_time: int

    @property
    def participation_key(self) -> tuple[int, str, int]:
        """Identify a single contest participation (including virtual restarts)."""
        return (self.contest_id, self.participant_type, self.start_time_seconds)


@dataclass(frozen=True)
class RawStat:
    """Robust aggregate for one rating bucket."""

    minutes: float
    sample_size: int


@dataclass(frozen=True)
class CurvePoint:
    """One rating on the personal solve-time curve."""

    rating: int
    raw_minutes: float | None
    smoothed_minutes: float
    sample_size: int


def _as_int(value: Any) -> int | None:
    """Best-effort integer conversion for loosely typed API payloads."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_rating(rating: Any, default: int = DEFAULT_RATING) -> int:
    """Clamp an arbitrary rating onto the 800..3500 / step-100 grid.

    Uses conventional half-up rounding, so 1250 becomes 1300.
    """
    value = _as_int(rating)
    if value is None:
        value = default
    snapped = int(math.floor(value / RATING_STEP + 0.5)) * RATING_STEP
    return int(_clamp(snapped, RATING_MIN, RATING_MAX))


def _snap_rating(rating: int) -> int:
    return normalize_rating(rating)


def extract_contest_deltas(submissions: Iterable[Mapping[str, Any]]) -> list[SolveObservation]:
    """Turn raw ``user.status`` submissions into per-problem contest solve times.

    Only accepted submissions by ``CONTESTANT`` or ``VIRTUAL`` participants are
    considered. Within one participation the earliest accepted submission per
    problem index counts. Problems are then ordered by ``relativeTimeSeconds``
    and each solve time is the gap from the previous solve (the first problem
    counts from contest start, i.e. ``t / 60``).

    Practice submissions, unrated problems and solves after five hours are
    ignored.
    """
    participations: dict[tuple[int, str, int], dict[str, dict[str, int]]] = {}

    for submission in submissions or ():
        if not isinstance(submission, Mapping):
            continue
        if submission.get("verdict") != "OK":
            continue

        author = submission.get("author")
        if not isinstance(author, Mapping):
            continue
        participant_type = author.get("participantType")
        if participant_type not in ("CONTESTANT", "VIRTUAL"):
            continue

        problem = submission.get("problem")
        if not isinstance(problem, Mapping):
            continue
        rating = _as_int(problem.get("rating"))
        if rating is None or rating <= 0:
            continue
        problem_index = str(problem.get("index") or "").strip()
        if not problem_index:
            continue

        contest_id = _as_int(submission.get("contestId"))
        if contest_id is None:
            contest_id = _as_int(problem.get("contestId"))
        if contest_id is None:
            continue

        relative = _as_int(submission.get("relativeTimeSeconds"))
        if relative is None or relative <= 0 or relative > MAX_RELATIVE_SECONDS:
            continue

        start_time = _as_int(author.get("startTimeSeconds")) or 0
        submission_time = _as_int(submission.get("creationTimeSeconds")) or 0

        key = (contest_id, str(participant_type), start_time)
        solved_in_participation = participations.setdefault(key, {})
        previous = solved_in_participation.get(problem_index)
        if previous is None or relative < previous["relative"]:
            solved_in_participation[problem_index] = {
                "relative": relative,
                "rating": rating,
                "submission_time": submission_time,
            }

    observations: list[SolveObservation] = []
    for (contest_id, participant_type, start_time), solved in participations.items():
        ordered = sorted(
            solved.items(),
            key=lambda item: (item[1]["relative"], item[0]),
        )
        previous_relative = 0
        for problem_index, data in ordered:
            delta_seconds = data["relative"] - previous_relative
            previous_relative = data["relative"]
            if delta_seconds <= 0:
                # Two problems accepted in the same second: no usable delta.
                continue
            observations.append(
                SolveObservation(
                    contest_id=contest_id,
                    problem_index=problem_index,
                    rating=_snap_rating(data["rating"]),
                    minutes=round(delta_seconds / 60.0, 4),
                    participant_type=participant_type,
                    start_time_seconds=start_time,
                    submission_time=data["submission_time"],
                )
            )
    return observations


def select_recent_participations(
    observations: Iterable[SolveObservation],
    limit: int = RECENT_PARTICIPATIONS,
) -> list[SolveObservation]:
    """Keep only the ``limit`` most recent distinct contest participations.

    Recency is ordered by contest id (which increases over time) and then by
    participation start time, so virtual restarts of the same contest are
    counted as separate recent participations.
    """
    observations = list(observations)
    if limit <= 0 or not observations:
        return []
    keys = sorted(
        {observation.participation_key for observation in observations},
        key=lambda key: (key[0], key[2]),
        reverse=True,
    )
    keep = set(keys[:limit])
    return [observation for observation in observations if observation.participation_key in keep]


def aggregate_raw_minutes(observations: Iterable[SolveObservation]) -> dict[int, RawStat]:
    """Bucket solve times by rating, dropping buckets with fewer than 2 solves.

    Buckets with four or more solves use a 10% trimmed mean (at least one value
    trimmed from each end); smaller buckets use the median.
    """
    buckets: dict[int, list[float]] = {}
    for observation in observations:
        if observation.minutes <= 0:
            continue
        buckets.setdefault(int(observation.rating), []).append(float(observation.minutes))

    result: dict[int, RawStat] = {}
    for rating, values in buckets.items():
        if len(values) < MIN_SAMPLES_PER_BUCKET:
            continue
        ordered = sorted(values)
        count = len(ordered)
        if count >= 4:
            trim = max(1, int(count * 0.1))
            if count - 2 * trim < 1:
                trim = 0
            core = ordered[trim : count - trim] if trim else ordered
            minutes = float(np.mean(core))
        else:
            minutes = float(np.median(ordered))
        result[int(rating)] = RawStat(minutes=round(minutes, 4), sample_size=count)
    return result


def smooth_monotonic(raw_minutes: Mapping[int, float]) -> dict[int, float]:
    """Interpolate raw bucket averages onto the rating grid and force monotonicity.

    ``numpy.interp`` linearly fills the gaps between observed buckets (and
    clamps beyond the observed range); a forward cumulative maximum then removes
    survivorship-bias dips such as 1700: 55 min followed by 1800: 44 min.
    """
    cleaned: dict[int, float] = {}
    for rating, minutes in raw_minutes.items():
        rating_value = _as_int(rating)
        if rating_value is None:
            continue
        try:
            minute_value = float(minutes)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(minute_value) or minute_value <= 0:
            continue
        cleaned[int(rating_value)] = minute_value

    if not cleaned:
        return {}

    ratings = np.array(sorted(cleaned), dtype=float)
    values = np.array([cleaned[int(rating)] for rating in ratings], dtype=float)
    grid = np.asarray(RATING_GRID, dtype=float)

    interpolated = np.interp(grid, ratings, values)
    monotonic = np.maximum.accumulate(interpolated)

    return {
        int(rating): round(float(minutes), 2)
        for rating, minutes in zip(RATING_GRID, monotonic)
    }


def fallback_curve(current_rating: int | None = DEFAULT_RATING) -> dict[int, float]:
    """Cold-start curve anchored around the user's current Codeforces rating.

    ``R - 300 -> 12 min`` ... ``R + 200 -> 75 min``. Values outside the anchor
    range are linearly extrapolated with :func:`numpy.interp`, clamped to a
    sane 3..240 minute range and made non-decreasing.
    """
    rating = normalize_rating(current_rating)
    anchor_ratings = np.array([rating + offset for offset, _ in _FALLBACK_ANCHORS], dtype=float)
    anchor_minutes = np.array([minutes for _, minutes in _FALLBACK_ANCHORS], dtype=float)
    grid = np.asarray(RATING_GRID, dtype=float)

    values = np.interp(grid, anchor_ratings, anchor_minutes)
    values = np.clip(values, _FALLBACK_FLOOR, _FALLBACK_CEILING)
    values = np.maximum.accumulate(values)

    return {
        int(grid_rating): round(float(minutes), 2)
        for grid_rating, minutes in zip(RATING_GRID, values)
    }


def build_curve(
    observations: Iterable[SolveObservation],
    current_rating: int | None = DEFAULT_RATING,
    *,
    rated_contest_count: int | None = None,
) -> list[CurvePoint]:
    """Build the full 800..3500 personal curve.

    ``observations`` should already be limited to recent participations (see
    :func:`select_recent_participations`). ``rated_contest_count`` is the total
    number of rated contests in the user's history; when it is below
    :data:`MIN_RATED_CONTESTS` the cold-start fallback is used.
    """
    observations = list(observations)
    if rated_contest_count is None:
        rated_contest_count = len({observation.contest_id for observation in observations})

    raw_stats = aggregate_raw_minutes(observations)
    cold_start = rated_contest_count < MIN_RATED_CONTESTS or not raw_stats

    if cold_start:
        smoothed = fallback_curve(current_rating)
    else:
        smoothed = smooth_monotonic(
            {rating: stat.minutes for rating, stat in raw_stats.items()}
        )

    points: list[CurvePoint] = []
    for rating in RATING_GRID:
        stat = raw_stats.get(rating)
        points.append(
            CurvePoint(
                rating=rating,
                raw_minutes=stat.minutes if stat else None,
                smoothed_minutes=smoothed[rating],
                sample_size=stat.sample_size if stat else 0,
            )
        )
    return points


def curve_as_dict(curve: Iterable[CurvePoint] | Mapping[int, float]) -> dict[int, float]:
    """Normalize a curve (dataclass points or rating->minutes mapping) to a dict."""
    if isinstance(curve, Mapping):
        result: dict[int, float] = {}
        for rating, minutes in curve.items():
            rating_value = _as_int(rating)
            if rating_value is None:
                continue
            try:
                result[int(rating_value)] = float(minutes)
            except (TypeError, ValueError):
                continue
        return result

    result = {}
    for point in curve:
        if isinstance(point, CurvePoint):
            result[point.rating] = point.smoothed_minutes
            continue
        if isinstance(point, Mapping):
            rating_value = _as_int(point.get("rating"))
            minutes = point.get("smoothed_minutes")
        else:
            rating_value = _as_int(getattr(point, "rating", None))
            minutes = getattr(point, "smoothed_minutes", None)
        if rating_value is None or minutes is None:
            continue
        try:
            result[int(rating_value)] = float(minutes)
        except (TypeError, ValueError):
            continue
    return result


def recommend_rating(
    curve: Iterable[CurvePoint] | Mapping[int, float],
    target_minutes: float,
    current_rating: int | None = DEFAULT_RATING,
) -> int:
    """Reverse-lookup the best rating for a time budget.

    Picks the highest rating whose smoothed solve time is at most
    ``target_minutes * 1.08``; if every bucket is slower than that, the bucket
    closest to the target by absolute difference wins. The result is clamped to
    ``[current_rating - 500, current_rating + 300]`` (also 800..3500) and
    snapped to the 100-point grid.
    """
    minutes = curve_as_dict(curve)
    if not minutes:
        return normalize_rating(current_rating)

    try:
        target = float(target_minutes)
    except (TypeError, ValueError):
        target = 40.0
    if not math.isfinite(target):
        target = 40.0
    target = float(_clamp(target, 5.0, 180.0))
    threshold = target * TARGET_TOLERANCE

    eligible = [rating for rating, value in minutes.items() if value <= threshold]
    if eligible:
        chosen = max(eligible)
    else:
        chosen = min(minutes, key=lambda rating: (abs(minutes[rating] - target), rating))

    base_rating = _as_int(current_rating)
    if base_rating is None:
        base_rating = DEFAULT_RATING
    base_rating = int(_clamp(base_rating, RATING_MIN, RATING_MAX))
    low = int(math.ceil(max(RATING_MIN, base_rating - RECOMMEND_LOW_OFFSET) / RATING_STEP) * RATING_STEP)
    high = int(math.floor(min(RATING_MAX, base_rating + RECOMMEND_HIGH_OFFSET) / RATING_STEP) * RATING_STEP)
    if low > high:  # defensive; cannot happen with the offsets above
        low, high = RATING_MIN, RATING_MAX

    chosen = int(_clamp(int(chosen), low, high))
    chosen = normalize_rating(chosen)
    return int(_clamp(chosen, RATING_MIN, RATING_MAX))


def extract_solved_pairs(
    submissions: Iterable[Mapping[str, Any]],
) -> set[tuple[int, str]]:
    """Return every ``(contest_id, index)`` the user has an accepted submission for.

    This intentionally includes practice submissions: the recommender must not
    suggest a problem the user already solved outside a contest either.
    """
    solved: set[tuple[int, str]] = set()
    for submission in submissions or ():
        if not isinstance(submission, Mapping):
            continue
        if submission.get("verdict") != "OK":
            continue
        problem = submission.get("problem")
        if not isinstance(problem, Mapping):
            continue
        contest_id = _as_int(submission.get("contestId"))
        if contest_id is None:
            contest_id = _as_int(problem.get("contestId"))
        index_name = str(problem.get("index") or "").strip()
        if contest_id is None or not index_name:
            continue
        solved.add((contest_id, index_name))
    return solved


__all__ = [
    "CurvePoint",
    "DEFAULT_RATING",
    "EstimatorError",
    "MAX_RELATIVE_SECONDS",
    "MIN_RATED_CONTESTS",
    "MIN_SAMPLES_PER_BUCKET",
    "RATING_GRID",
    "RATING_MAX",
    "RATING_MIN",
    "RATING_STEP",
    "RECENT_PARTICIPATIONS",
    "RawStat",
    "SolveObservation",
    "TARGET_TOLERANCE",
    "aggregate_raw_minutes",
    "build_curve",
    "curve_as_dict",
    "extract_contest_deltas",
    "extract_solved_pairs",
    "fallback_curve",
    "normalize_rating",
    "recommend_rating",
    "select_recent_participations",
    "smooth_monotonic",
]
