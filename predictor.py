"""Rating-Space Elo-Survival model for Codeforces practice.

This module is pure standard-library Python and has no Discord, networking or
database imports, so it can be unit-tested and back-tested entirely offline.

The model
---------
A Codeforces problem has a nominal rating ``R`` (Elo-like difficulty).  A user
is summarised by ``s30``: the rating they solve with 50% probability in a
30-minute window on a *modern* problem.  Time and rating trade off through a
log-logistic survival curve.  Two modifiers sit on top of the rating axis:

* ``gamma`` discounts an old problem by up to ``gamma`` rating points
  (a 7.5-year-old problem is fully discounted);
* ``delta[a]`` is a per-archetype handicap (positive means the user is slower
  than the rating suggests on that topic).

``contest_offset`` adds extra effective difficulty for solves that happened in
a live contest, and ``kappa``/``k`` control the time/rating exchange rate and
the consistency (shape) of the log-logistic curve.

For one attempt let

    t      = max(actual_minutes, 1)
    y      = era factor in [0, 1]
    h      = mean(delta[a] for a in archetypes(tags))   (0 if none)
    r_eff  = rating - gamma * y + h + contest_offset * is_contest
    J      = k / kappa
    z      = k * ln(t / 30) + J * (s30 - r_eff)
    p      = sigmoid(z)

A solve contributes ``log k - log t + log sigmoid(z) + log sigmoid(-z)``; a
give-up / timeout contributes ``log sigmoid(-z)`` (right censoring).  For fixed
shape parameters ``z`` is affine in the eight parameters
``theta = [s30, gamma, d0..d4, contest_offset]``, so the weighted likelihood is
concave and Gaussian priors make the posterior strictly concave: Newton's
method with a backtracking line search finds the unique MAP estimate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Tag compression
# ---------------------------------------------------------------------------

#: Codeforces tag -> coarse archetype index.  Niche tags (fft, flows, graph
#: matchings, 2-sat, expression parsing, chinese remainder theorem, schedules,
#: communication, *special) are intentionally absent: they contribute no
#: handicap rather than inventing a bucket for a handful of problems.
ARCHETYPE_MAP: dict[str, int] = {
    # 0: Ad-Hoc, Greedy & Constructive
    "greedy": 0,
    "implementation": 0,
    "constructive algorithms": 0,
    "brute force": 0,
    "sortings": 0,
    "games": 0,
    "interactive": 0,
    # 1: Math & Number Theory
    "math": 1,
    "number theory": 1,
    "combinatorics": 1,
    "probabilities": 1,
    "matrices": 1,
    "geometry": 1,
    # 2: Data Structures, Strings & Search
    "data structures": 2,
    "binary search": 2,
    "two pointers": 2,
    "dsu": 2,
    "divide and conquer": 2,
    "ternary search": 2,
    "strings": 2,
    "hashing": 2,
    "string suffix structures": 2,
    # 3: Graphs & Trees
    "graphs": 3,
    "dfs and similar": 3,
    "trees": 3,
    "shortest paths": 3,
    # 4: Dynamic Programming & Bitmasks
    "dp": 4,
    "bitmasks": 4,
    "meet-in-the-middle": 4,
}

ARCHETYPE_NAMES: list[str] = [
    "Ad-Hoc / Greedy",
    "Math / NT",
    "DS / Search",
    "Graphs / Trees",
    "DP / Bitmasks",
]

N_ARCHETYPES = len(ARCHETYPE_NAMES)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Recency weighting: half-life of an observation, in days...
DECAY_HALF_LIFE_DAYS = 75.0
#: ...with a floor so ancient history never vanishes completely.
MIN_DECAY_WEIGHT = 0.05
#: Contest-derived rows are noisier than deliberate practice rows.
BOOTSTRAP_WEIGHT = 0.7
#: Extra minutes per wrong answer applied while *fitting*.
#:
#: Deliberately zero: ``actual_mins`` is wall-clock elapsed time, which already
#: contains the time lost to wrong submissions.  Adding five minutes per WA
#: here would count that time twice and make ``/times`` over-predict.  The
#: penalty is only used for the informational "contest-equivalent time"
#: display (see :data:`CONTEST_WA_PENALTY_MINS`).
FIT_WA_PENALTY_MINS = 0.0
#: Extra minutes per wrong answer shown as "contest-equivalent time".
CONTEST_WA_PENALTY_MINS = 5.0
#: A problem this old (or older) loses the full ``gamma`` era discount.
ERA_SPAN_YEARS = 7.5
#: Problem quality/era fallback: contest id at which a problem is "modern".
MODERN_CONTEST_ID = 2050.0
#: Contest-id span over which the fallback era discount ramps from 0 to 1.
ERA_CONTEST_SPAN = 1200.0

#: Parameter order of ``theta``: [s30, gamma, delta[0..4], contest_offset].
N_PARAMS = 2 + N_ARCHETYPES + 1

#: Gaussian prior standard deviations, in rating points.  Kept small enough to
#: regularise cold starts, large enough that real data dominates quickly.
S30_PRIOR_SD = 150.0
GAMMA_PRIOR_MEAN = 200.0
GAMMA_PRIOR_SD = 75.0
DELTA_PRIOR_MEAN = 0.0
DELTA_PRIOR_SD = 60.0
CONTEST_OFFSET_PRIOR_MEAN = 0.0
CONTEST_OFFSET_PRIOR_SD = 50.0
#: Log-space priors for the shape parameters.
KAPPA_PRIOR_MEAN = 180.0
KAPPA_PRIOR_SD_LOG = 0.25
K_PRIOR_MEAN = 2.5
K_PRIOR_SD_LOG = 0.25

#: Default shape.
DEFAULT_KAPPA = 180.0
DEFAULT_K = 2.5

#: Candidate grids used by the optional profile-likelihood shape refit.
KAPPA_GRID: tuple[float, ...] = (120.0, 150.0, 180.0, 220.0, 260.0)
K_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5)

#: Newton step component clip (rating points) and convergence tolerance.
MAX_STEP = 200.0
STEP_TOL = 1e-6

#: ``;practice``'s smallest allowable time budget, used by backtests.
MIN_BUDGET_MINS = 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def get_archetypes(cf_tags: Iterable[str] | None) -> list[int]:
    """Map Codeforces tags to sorted, unique archetype indices."""
    found: set[int] = set()
    for tag in cf_tags or ():
        if not isinstance(tag, str):
            continue
        archetype = ARCHETYPE_MAP.get(tag.strip().lower())
        if archetype is not None:
            found.add(archetype)
    return sorted(found)


def get_era_factor(
    contest_id: int,
    start_ts: int | None = None,
    now_ts: float | None = None,
) -> float:
    """Return the era discount factor in ``[0, 1]`` for a contest.

    With a start timestamp, the factor is the contest's age in years divided by
    :data:`ERA_SPAN_YEARS`, clamped to ``[0, 1]``.  "Modern" therefore always
    means "held today", with no hard-coded year.

    Without a timestamp (``contest.list`` unavailable, or synthetic tests) the
    factor falls back to the contest-id ramp ``(2050 - id) / 1200``, so
    ``2050 -> 0.0``, ``1450 -> 0.5`` and ``850 -> 1.0``.
    """
    if start_ts is not None:
        now = time.time() if now_ts is None else float(now_ts)
        age_years = (now - float(start_ts)) / (365.25 * 86400.0)
        return _clamp(age_years / ERA_SPAN_YEARS, 0.0, 1.0)
    try:
        cid = float(contest_id)
    except (TypeError, ValueError):
        return 0.0
    return _clamp((MODERN_CONTEST_ID - cid) / ERA_CONTEST_SPAN, 0.0, 1.0)


def format_minutes(m: float) -> str:
    """Format a predicted duration for compact human display.

    ``0.4 -> "<1m"``, ``14.2 -> "14m"``, ``59.6 -> "1h00m"``,
    ``75 -> "1h15m"``, ``179.6 -> "3h00m"``, ``181 -> ">3h"``.
    """
    try:
        value = float(m)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(value):
        return "?" if math.isnan(value) else (">3h" if value > 0 else "<1m")
    if value < 1.0:
        return "<1m"
    if value > 180.0:
        return ">3h"
    rounded = int(round(value))
    if rounded < 60:
        return f"{rounded}m"
    return f"{rounded // 60}h{rounded % 60:02d}m"


# ---------------------------------------------------------------------------
# Records and weighting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AttemptRecord:
    """One observation of the user attempting one problem.

    ``actual_mins`` is wall-clock time.  ``solved=False`` means the observation
    is right-censored: the user stopped trying without an AC, so all we know is
    that the solve time is greater than ``actual_mins``.
    """

    rating: float
    contest_id: int
    tags: list[str]
    actual_mins: float
    was_count: int
    solved: bool
    days_ago: float
    is_contest_bootstrap: bool = False
    contest_start_ts: int | None = None  # for timestamp-based era factor
    budget_mins: float | None = None  # practice budget, used by backtest
    weight_mult: float = 1.0  # e.g. 0.5 for auto-expired sessions


def effective_time(record: AttemptRecord, wa_penalty: float = FIT_WA_PENALTY_MINS) -> float:
    """Elapsed time used by the likelihood, floored at one minute."""
    try:
        raw = float(record.actual_mins) + float(wa_penalty) * float(record.was_count)
    except (TypeError, ValueError):
        raw = MIN_BUDGET_MINS
    return max(raw, MIN_BUDGET_MINS)


def record_weight(record: AttemptRecord) -> float:
    """Recency/decay/bootstrap weight of one observation."""
    days = max(0.0, float(record.days_ago))
    decay = max(0.5 ** (days / DECAY_HALF_LIFE_DAYS), MIN_DECAY_WEIGHT)
    bootstrap = BOOTSTRAP_WEIGHT if record.is_contest_bootstrap else 1.0
    try:
        multiplier = float(record.weight_mult)
    except (TypeError, ValueError):
        multiplier = 1.0
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        multiplier = 1.0
    return decay * bootstrap * multiplier


def _prior_means(base_prior: float) -> list[float]:
    means = [float(base_prior), GAMMA_PRIOR_MEAN]
    means.extend([DELTA_PRIOR_MEAN] * N_ARCHETYPES)
    means.append(CONTEST_OFFSET_PRIOR_MEAN)
    return means


def _prior_sds() -> list[float]:
    sds = [S30_PRIOR_SD, GAMMA_PRIOR_SD]
    sds.extend([DELTA_PRIOR_SD] * N_ARCHETYPES)
    sds.append(CONTEST_OFFSET_PRIOR_SD)
    return sds


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function without clamping the objective."""
    if z >= 0.0:
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:  # pragma: no cover - exp(-z) underflows anyway
            return 1.0
    try:
        e = math.exp(z)
    except OverflowError:  # pragma: no cover
        return 0.0
    return e / (1.0 + e)


def _log_sigmoid(z: float) -> float:
    """Numerically stable ``log(sigmoid(z))``."""
    if z >= 0.0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


# ---------------------------------------------------------------------------
# Design matrix / likelihood pieces
# ---------------------------------------------------------------------------


def _record_terms(
    record: AttemptRecord,
    now_ts: float,
    kappa: float,
    k: float,
) -> tuple[float, float, list[float], float]:
    """Return ``(z_offset, J, x, weight)`` for one record.

    ``z = z_offset + J * (theta . x)``.
    """
    t = effective_time(record)
    era = get_era_factor(record.contest_id, record.contest_start_ts, now_ts)
    archetypes = get_archetypes(record.tags)
    x = [0.0] * N_PARAMS
    x[0] = 1.0
    x[1] = era
    if archetypes:
        share = -1.0 / float(len(archetypes))
        for archetype in archetypes:
            x[2 + archetype] = share
    if record.is_contest_bootstrap:
        x[2 + N_ARCHETYPES] = -1.0
    j_value = k / kappa
    z_offset = k * math.log(t / 30.0) - j_value * float(record.rating)
    return z_offset, j_value, x, record_weight(record)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for left, right in zip(a, b):
        total += left * right
    return total


def _dot_into(a: Sequence[float], b: Sequence[float], out: list[float]) -> None:
    for row in range(len(a)):
        total = 0.0
        left = a[row]
        for column in range(len(b)):
            total += left[column] * b[column]
        out[row] = total


# ---------------------------------------------------------------------------
# Importable objective helpers
# ---------------------------------------------------------------------------


def log_posterior(
    theta: Sequence[float],
    history: Sequence[AttemptRecord],
    base_prior: float = 1800.0,
    kappa: float = DEFAULT_KAPPA,
    k: float = DEFAULT_K,
    now_ts: float | None = None,
    include_shape_prior: bool = False,
) -> float:
    """Penalised log-posterior of ``theta`` for ``history``.

    The Gaussian priors on ``s30``, ``gamma``, ``delta`` and ``contest_offset``
    are always included.  ``include_shape_prior`` additionally adds the
    ``ln k`` density term for every solved row plus the log-space priors on
    ``kappa`` and ``k``; this is what makes shape-parameter candidates
    comparable (a Newton solve for ``theta`` alone treats them as constants).
    """
    if len(theta) != N_PARAMS:
        raise ValueError(f"theta must have {N_PARAMS} entries, got {len(theta)}")
    now = time.time() if now_ts is None else float(now_ts)
    total = 0.0
    for record in history:
        z_offset, j_value, x, weight = _record_terms(record, now, kappa, k)
        z = z_offset + j_value * _dot(theta, x)
        if record.solved:
            t = effective_time(record)
            total += weight * (-math.log(t) + _log_sigmoid(z) + _log_sigmoid(-z))
            if include_shape_prior:
                total += weight * math.log(k)
        else:
            total += weight * _log_sigmoid(-z)

    means = _prior_means(base_prior)
    sds = _prior_sds()
    for value, mean, sd in zip(theta, means, sds):
        total -= 0.5 * ((value - mean) / sd) ** 2

    if include_shape_prior:
        total -= 0.5 * ((math.log(kappa) - math.log(KAPPA_PRIOR_MEAN)) / KAPPA_PRIOR_SD_LOG) ** 2
        total -= 0.5 * ((math.log(k) - math.log(K_PRIOR_MEAN)) / K_PRIOR_SD_LOG) ** 2
    return total


def grad_hess(
    theta: Sequence[float],
    history: Sequence[AttemptRecord],
    base_prior: float = 1800.0,
    kappa: float = DEFAULT_KAPPA,
    k: float = DEFAULT_K,
    now_ts: float | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Return ``(gradient, Hessian)`` of :func:`log_posterior` w.r.t. ``theta``.

    ``hessian`` is the (negative-definite) matrix of second derivatives.
    """
    if len(theta) != N_PARAMS:
        raise ValueError(f"theta must have {N_PARAMS} entries, got {len(theta)}")
    now = time.time() if now_ts is None else float(now_ts)
    gradient = [0.0] * N_PARAMS
    hessian = [[0.0] * N_PARAMS for _ in range(N_PARAMS)]

    for record in history:
        z_offset, j_value, x, weight = _record_terms(record, now, kappa, k)
        z = z_offset + j_value * _dot(theta, x)
        probability = _sigmoid(z)
        if record.solved:
            d1 = 1.0 - 2.0 * probability
            d2 = -2.0 * probability * (1.0 - probability)
        else:
            d1 = -probability
            d2 = -probability * (1.0 - probability)
        if d1 == 0.0 and d2 == 0.0:
            continue
        factor_grad = weight * d1 * j_value
        factor_hess = weight * d2 * j_value * j_value
        for row in range(N_PARAMS):
            if x[row] == 0.0:
                continue
            gradient[row] += factor_grad * x[row]
            if factor_hess == 0.0:
                continue
            row_value = factor_hess * x[row]
            hessian_row = hessian[row]
            for column in range(N_PARAMS):
                if x[column] != 0.0:
                    hessian_row[column] += row_value * x[column]

    means = _prior_means(base_prior)
    sds = _prior_sds()
    for index in range(N_PARAMS):
        inverse_variance = 1.0 / (sds[index] * sds[index])
        gradient[index] -= (theta[index] - means[index]) * inverse_variance
        hessian[index][index] -= inverse_variance
    return gradient, hessian


def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    size = len(A)
    if size == 0:
        return []
    if len(b) != size or any(len(row) != size for row in A):
        raise ValueError("solve_linear requires a square matrix and matching vector")
    matrix = [list(row) + [float(b[index])] for index, row in enumerate(A)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        pivot = matrix[pivot_row][column]
        if abs(pivot) < 1e-14:
            raise ValueError("matrix is singular")
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]
        pivot = matrix[column][column]
        for row in range(column + 1, size):
            factor = matrix[row][column] / pivot
            if factor == 0.0:
                continue
            matrix_row = matrix[row]
            pivot_values = matrix[column]
            for k_index in range(column, size + 1):
                matrix_row[k_index] -= factor * pivot_values[k_index]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = matrix[row][size]
        for column in range(row + 1, size):
            total -= matrix[row][column] * solution[column]
        pivot = matrix[row][row]
        if abs(pivot) < 1e-14:
            raise ValueError("matrix is singular")
        solution[row] = total / pivot
    return solution


def invert_matrix(A: list[list[float]]) -> list[list[float]]:
    """Invert a square matrix with Gauss-Jordan elimination."""
    size = len(A)
    if size == 0:
        return []
    if any(len(row) != size for row in A):
        raise ValueError("invert_matrix requires a square matrix")
    matrix = [list(row) + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(A)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot_row][column]) < 1e-14:
            raise ValueError("matrix is singular")
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]
        pivot = matrix[column][column]
        for k_index in range(2 * size):
            matrix[column][k_index] /= pivot
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor == 0.0:
                continue
            matrix_row = matrix[row]
            pivot_values = matrix[column]
            for k_index in range(2 * size):
                matrix_row[k_index] -= factor * pivot_values[k_index]

    return [row[size:] for row in matrix]


# ---------------------------------------------------------------------------
# Fit result / model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FitSummary:
    """What happened during one MAP fit."""

    n_records: int
    total_weight: float
    iterations: int
    converged: bool
    objective: float
    objective_trace: list[float]


@dataclass(slots=True)
class BacktestResult:
    """Walk-forward calibration metrics."""

    log_loss: float
    brier_score: float
    n: int
    bins: list[tuple[float, float, int]]


class CFPredictor:
    """Fitted Rating-Space Elo-Survival model.

    Parameters are stored as plain attributes in Codeforces rating points
    (``kappa`` is rating points per unit log-time, ``k`` is dimensionless).
    """

    def __init__(self, base_prior: float = 1800.0) -> None:
        self.base_prior = float(base_prior)
        self.s30 = float(base_prior)
        self.gamma = GAMMA_PRIOR_MEAN
        self.delta: list[float] = [DELTA_PRIOR_MEAN] * N_ARCHETYPES
        self.contest_offset = CONTEST_OFFSET_PRIOR_MEAN
        self.kappa = DEFAULT_KAPPA
        self.k = DEFAULT_K
        self.stderr: dict[str, Any] = {
            "s30": S30_PRIOR_SD,
            "gamma": GAMMA_PRIOR_SD,
            "delta": [DELTA_PRIOR_SD] * N_ARCHETYPES,
            "contest_offset": CONTEST_OFFSET_PRIOR_SD,
        }
        self.shape_fit_performed = False

    # -- state ---------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialise the fitted parameters for SQLite/JSON persistence."""
        return {
            "s30": float(self.s30),
            "gamma": float(self.gamma),
            "kappa": float(self.kappa),
            "k": float(self.k),
            "delta": [float(value) for value in self.delta],
            "contest_offset": float(self.contest_offset),
            "stderr": {
                "s30": float(self.stderr.get("s30", S30_PRIOR_SD)),
                "gamma": float(self.stderr.get("gamma", GAMMA_PRIOR_SD)),
                "delta": [float(value) for value in self.stderr.get("delta", [])][:N_ARCHETYPES],
                "contest_offset": float(
                    self.stderr.get("contest_offset", CONTEST_OFFSET_PRIOR_SD)
                ),
            },
        }

    def load_state(self, state: Mapping[str, Any] | None) -> None:
        """Apply a state produced by :meth:`to_state` (safe for partial dicts)."""
        if not state:
            return
        try:
            if state.get("s30") is not None:
                self.s30 = float(state["s30"])
            if state.get("gamma") is not None:
                self.gamma = float(state["gamma"])
            if state.get("kappa") is not None:
                self.kappa = float(state["kappa"])
            if state.get("k") is not None:
                self.k = float(state["k"])
            if state.get("contest_offset") is not None:
                self.contest_offset = float(state["contest_offset"])
        except (TypeError, ValueError):
            pass
        delta = state.get("delta")
        if isinstance(delta, (list, tuple)) and delta:
            parsed = [float(value) for value in delta[:N_ARCHETYPES]]
            while len(parsed) < N_ARCHETYPES:
                parsed.append(DELTA_PRIOR_MEAN)
            self.delta = parsed
        stderr = state.get("stderr")
        if isinstance(stderr, Mapping):
            parsed_stderr: dict[str, Any] = {}
            try:
                parsed_stderr["s30"] = float(stderr.get("s30", S30_PRIOR_SD))
                parsed_stderr["gamma"] = float(stderr.get("gamma", GAMMA_PRIOR_SD))
                parsed_stderr["contest_offset"] = float(
                    stderr.get("contest_offset", CONTEST_OFFSET_PRIOR_SD)
                )
            except (TypeError, ValueError):
                parsed_stderr = {}
            delta_sd = stderr.get("delta")
            if isinstance(delta_sd, (list, tuple)):
                parsed_delta = [float(value) for value in delta_sd[:N_ARCHETYPES]]
                while len(parsed_delta) < N_ARCHETYPES:
                    parsed_delta.append(DELTA_PRIOR_SD)
                parsed_stderr["delta"] = parsed_delta
            if parsed_stderr:
                self.stderr = {
                    "s30": parsed_stderr.get("s30", S30_PRIOR_SD),
                    "gamma": parsed_stderr.get("gamma", GAMMA_PRIOR_SD),
                    "delta": parsed_stderr.get("delta", [DELTA_PRIOR_SD] * N_ARCHETYPES),
                    "contest_offset": parsed_stderr.get(
                        "contest_offset", CONTEST_OFFSET_PRIOR_SD
                    ),
                }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any] | None,
        base_prior: float = 1800.0,
    ) -> "CFPredictor":
        predictor = cls(base_prior=base_prior)
        predictor.load_state(state)
        return predictor

    # -- Newton / MAP --------------------------------------------------------

    def _newton(
        self,
        history: Sequence[AttemptRecord],
        base_prior: float,
        kappa: float,
        k: float,
        now_ts: float,
        max_iters: int,
    ) -> tuple[list[float], list[float], int, bool]:
        theta = [float(base_prior), GAMMA_PRIOR_MEAN]
        theta.extend([DELTA_PRIOR_MEAN] * N_ARCHETYPES)
        theta.append(CONTEST_OFFSET_PRIOR_MEAN)

        objective = log_posterior(theta, history, base_prior, kappa, k, now_ts)
        trace = [objective]
        converged = False
        accepted = 0

        for _ in range(max(1, int(max_iters))):
            gradient, hessian = grad_hess(theta, history, base_prior, kappa, k, now_ts)
            negated = [[-value for value in row] for row in hessian]
            try:
                step = solve_linear(negated, gradient)
            except ValueError:
                converged = True
                break
            step = [_clamp(value, -MAX_STEP, MAX_STEP) for value in step]
            if max(abs(value) for value in step) < STEP_TOL:
                converged = True
                break

            alpha = 1.0
            improved = False
            for _line_search in range(40):
                candidate = [theta[i] + alpha * step[i] for i in range(N_PARAMS)]
                candidate_objective = log_posterior(
                    candidate, history, base_prior, kappa, k, now_ts
                )
                if candidate_objective >= objective:
                    theta = candidate
                    objective = candidate_objective
                    trace.append(objective)
                    accepted += 1
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                converged = True
                break
        return theta, trace, accepted, converged

    def _compute_stderr(
        self,
        theta: Sequence[float],
        history: Sequence[AttemptRecord],
        base_prior: float,
        kappa: float,
        k: float,
        now_ts: float,
    ) -> dict[str, Any]:
        _, hessian = grad_hess(theta, history, base_prior, kappa, k, now_ts)
        prior_sds = _prior_sds()
        values = list(prior_sds)
        try:
            covariance = invert_matrix([[-value for value in row] for row in hessian])
            diagonal = [covariance[index][index] for index in range(N_PARAMS)]
        except (ValueError, ZeroDivisionError):
            diagonal = [sd * sd for sd in prior_sds]
        for index in range(N_PARAMS):
            variance = diagonal[index]
            if not math.isfinite(variance) or variance <= 0.0:
                values[index] = prior_sds[index]
                continue
            value = math.sqrt(variance)
            if not math.isfinite(value) or value <= 0.0:
                value = prior_sds[index]
            values[index] = min(value, prior_sds[index])
        return {
            "s30": values[0],
            "gamma": values[1],
            "delta": values[2 : 2 + N_ARCHETYPES],
            "contest_offset": values[2 + N_ARCHETYPES],
        }

    def fit(
        self,
        history: Sequence[AttemptRecord],
        base_prior: float = 1800.0,
        now_ts: float | None = None,
        max_iters: int = 50,
        fit_shape: bool = False,
    ) -> FitSummary:
        """Fit the model by MAP estimation with Newton's method.

        Starts from deterministic warm defaults, so fitting the same history
        twice gives identical parameters.  Returns a :class:`FitSummary`; the
        fitted parameters live on ``self``.
        """
        records = list(history)
        now = time.time() if now_ts is None else float(now_ts)
        self.base_prior = float(base_prior)
        self.shape_fit_performed = False
        total_weight = sum(record_weight(record) for record in records)

        kappa = self.kappa if self.kappa > 0 else DEFAULT_KAPPA
        k = self.k if self.k > 0 else DEFAULT_K

        theta, trace, accepted, converged = self._newton(
            records, base_prior, kappa, k, now, max_iters
        )

        if fit_shape and total_weight >= 30.0 and records:
            shape_records = records[-400:]
            best_objective = -math.inf
            best_theta = theta
            best_kappa = kappa
            best_k = k
            best_trace = trace
            best_iterations = accepted

            for candidate_kappa in KAPPA_GRID:
                candidate_theta, candidate_trace, candidate_iterations, _ = self._newton(
                    shape_records, base_prior, candidate_kappa, k, now, max_iters
                )
                candidate_objective = log_posterior(
                    candidate_theta,
                    shape_records,
                    base_prior,
                    candidate_kappa,
                    k,
                    now,
                    include_shape_prior=True,
                )
                if candidate_objective > best_objective:
                    best_objective = candidate_objective
                    best_theta = candidate_theta
                    best_kappa = candidate_kappa
                    best_k = k
                    best_trace = candidate_trace
                    best_iterations = candidate_iterations

            for candidate_k in K_GRID:
                candidate_theta, candidate_trace, candidate_iterations, _ = self._newton(
                    shape_records, base_prior, best_kappa, candidate_k, now, max_iters
                )
                candidate_objective = log_posterior(
                    candidate_theta,
                    shape_records,
                    base_prior,
                    best_kappa,
                    candidate_k,
                    now,
                    include_shape_prior=True,
                )
                if candidate_objective > best_objective:
                    best_objective = candidate_objective
                    best_theta = candidate_theta
                    best_kappa = best_kappa
                    best_k = candidate_k
                    best_trace = candidate_trace
                    best_iterations = candidate_iterations

            theta = best_theta
            kappa = best_kappa
            k = best_k
            trace = best_trace
            accepted = best_iterations
            self.shape_fit_performed = True
            converged = True

        # Safety clamps.  The priors make these non-binding in practice.
        theta[1] = _clamp(theta[1], 0.0, 350.0)
        for archetype in range(N_ARCHETYPES):
            theta[2 + archetype] = _clamp(theta[2 + archetype], -300.0, 300.0)

        self.s30 = float(theta[0])
        self.gamma = float(theta[1])
        self.delta = [float(theta[2 + archetype]) for archetype in range(N_ARCHETYPES)]
        self.contest_offset = float(theta[2 + N_ARCHETYPES])
        self.kappa = float(kappa)
        self.k = float(k)
        self.stderr = self._compute_stderr(theta, records, base_prior, kappa, k, now)

        return FitSummary(
            n_records=len(records),
            total_weight=float(total_weight),
            iterations=int(accepted),
            converged=bool(converged),
            objective=float(trace[-1]) if trace else float("-inf"),
            objective_trace=[float(value) for value in trace],
        )

    # -- prediction: time -> rating -----------------------------------------

    def predict_modern_rating(
        self,
        target_mins: float,
        confidence: float = 0.65,
        arch_idx: int | None = None,
        contest: bool = False,
    ) -> float:
        """Rating achievable in ``target_mins`` at ``confidence`` on a modern problem.

        ``confidence`` is the probability that the solve happens within the
        window.  ``arch_idx`` applies that archetype's learned handicap;
        ``contest`` adds the learned live-contest offset.
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be strictly between 0 and 1")
        handicap = self.delta[arch_idx] if arch_idx is not None else 0.0
        offset = self.contest_offset if contest else 0.0
        time_bonus = self.kappa * math.log(max(float(target_mins), 1.0) / 30.0)
        confidence_offset = (self.kappa / self.k) * math.log(
            (1.0 - confidence) / confidence
        )
        return self.s30 - handicap - offset + time_bonus + confidence_offset

    def predict_nominal_rating(
        self,
        target_mins: float,
        contest_id: int,
        cf_tags: list[str] | tuple[str, ...],
        confidence: float = 0.65,
        start_ts: int | None = None,
        now_ts: float | None = None,
    ) -> float:
        """Nominal rating (with era and tag effects) for a time budget."""
        archetypes = get_archetypes(cf_tags)
        handicap = (
            sum(self.delta[archetype] for archetype in archetypes) / len(archetypes)
            if archetypes
            else 0.0
        )
        era = get_era_factor(contest_id, start_ts, now_ts)
        return (
            self.predict_modern_rating(target_mins, confidence, arch_idx=None, contest=False)
            - handicap
            + self.gamma * era
        )

    # -- prediction: rating -> time -----------------------------------------

    def predict_solve_time(
        self,
        rating: float,
        confidence: float = 0.50,
        arch_idx: int | None = None,
        cf_tags: list[str] | tuple[str, ...] | None = None,
        era_y: float = 0.0,
        contest: bool = False,
    ) -> float:
        """Exact inverse of the rating prediction: the ``confidence`` quantile.

        Returns minutes.  ``confidence=0.50`` is the median solve time,
        ``0.65`` a reliable estimate and ``0.80`` a safe one.
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be strictly between 0 and 1")
        if arch_idx is not None:
            handicap = self.delta[arch_idx]
        elif cf_tags:
            archetypes = get_archetypes(cf_tags)
            handicap = (
                sum(self.delta[archetype] for archetype in archetypes) / len(archetypes)
                if archetypes
                else 0.0
            )
        else:
            handicap = 0.0
        offset = self.contest_offset if contest else 0.0
        effective_rating = float(rating) - self.gamma * float(era_y) + handicap + offset
        log_relative_time = (effective_rating - self.s30) / self.kappa + (
            1.0 / self.k
        ) * math.log(confidence / (1.0 - confidence))
        return 30.0 * math.exp(_clamp(log_relative_time, -10.0, 10.0))

    def predict_solve_probability(
        self,
        rating: float,
        contest_id: int = 2050,
        cf_tags: list[str] | tuple[str, ...] = (),
        target_mins: float = 25.0,
        start_ts: int | None = None,
        now_ts: float | None = None,
        contest: bool = False,
        arch_idx: int | None = None,
        era_y: float | None = None,
    ) -> float:
        """``P(solve time <= target_mins)`` for a problem of nominal ``rating``."""
        if era_y is not None:
            era = float(era_y)
        else:
            era = get_era_factor(contest_id, start_ts, now_ts)
        if arch_idx is not None:
            handicap = self.delta[arch_idx]
        else:
            archetypes = get_archetypes(cf_tags)
            handicap = (
                sum(self.delta[archetype] for archetype in archetypes) / len(archetypes)
                if archetypes
                else 0.0
            )
        offset = self.contest_offset if contest else 0.0
        effective_rating = float(rating) - self.gamma * era + handicap + offset
        t = max(float(target_mins), MIN_BUDGET_MINS)
        j_value = self.k / self.kappa
        z = self.k * math.log(t / 30.0) + j_value * (self.s30 - effective_rating)
        return _sigmoid(z)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    history: Sequence[AttemptRecord],
    min_train: int = 10,
    base_prior: float = 1800.0,
    now_ts: float | None = None,
    max_eval: int | None = 60,
    max_train: int = 300,
) -> BacktestResult:
    """Time-ordered out-of-sample calibration test.

    For every eligible practice attempt ``n`` the model is refit on the
    (at most ``max_train``) preceding records with their ``days_ago`` values
    re-based relative to attempt ``n``, then asked for the probability of
    solving inside that attempt's own budget.  Reliability is reported in ten
    equal-width probability bins.
    """
    records = list(history)
    eligible: list[int] = []
    for index in range(max(0, int(min_train)), len(records)):
        record = records[index]
        if record.budget_mins is None or record.is_contest_bootstrap:
            continue
        if record.solved or record.actual_mins >= float(record.budget_mins):
            eligible.append(index)
    if max_eval is not None and len(eligible) > max_eval:
        eligible = eligible[-max_eval:]

    bins_acc = [[0.0, 0.0, 0] for _ in range(10)]
    if not eligible:
        return BacktestResult(
            log_loss=0.0,
            brier_score=0.0,
            n=0,
            bins=[(0.0, 0.0, 0) for _ in range(10)],
        )

    log_loss_total = 0.0
    brier_total = 0.0
    for index in eligible:
        evaluation = records[index]
        train_slice = records[max(0, index - max(0, int(max_train))) : index]
        rebased = [
            replace(record, days_ago=max(0.0, float(record.days_ago) - float(evaluation.days_ago)))
            for record in train_slice
        ]
        model = CFPredictor(base_prior=base_prior)
        model.fit(rebased, base_prior=base_prior, now_ts=now_ts)
        budget = float(evaluation.budget_mins or 0.0)
        probability = model.predict_solve_probability(
            evaluation.rating,
            evaluation.contest_id,
            evaluation.tags,
            budget,
            start_ts=evaluation.contest_start_ts,
            now_ts=now_ts,
            contest=evaluation.is_contest_bootstrap,
        )
        probability = _clamp(probability, 1e-6, 1.0 - 1e-6)
        outcome = 1.0 if (
            evaluation.solved and float(evaluation.actual_mins) <= budget
        ) else 0.0
        log_loss_total -= outcome * math.log(probability) + (1.0 - outcome) * math.log(
            1.0 - probability
        )
        brier_total += (probability - outcome) ** 2
        bin_index = min(int(probability * 10.0), 9)
        bins_acc[bin_index][0] += probability
        bins_acc[bin_index][1] += outcome
        bins_acc[bin_index][2] += 1

    n = len(eligible)
    bins: list[tuple[float, float, int]] = []
    for predicted_sum, outcome_sum, count in bins_acc:
        if count:
            bins.append((predicted_sum / count, outcome_sum / count, count))
        else:
            bins.append((0.0, 0.0, 0))
    return BacktestResult(
        log_loss=log_loss_total / n,
        brier_score=brier_total / n,
        n=n,
        bins=bins,
    )


__all__ = [
    "ARCHETYPE_MAP",
    "ARCHETYPE_NAMES",
    "AttemptRecord",
    "BOOTSTRAP_WEIGHT",
    "BacktestResult",
    "CFPredictor",
    "CONTEST_WA_PENALTY_MINS",
    "DECAY_HALF_LIFE_DAYS",
    "DELTA_PRIOR_SD",
    "ERA_SPAN_YEARS",
    "FIT_WA_PENALTY_MINS",
    "FitSummary",
    "GAMMA_PRIOR_SD",
    "KAPPA_GRID",
    "K_GRID",
    "MIN_DECAY_WEIGHT",
    "N_ARCHETYPES",
    "N_PARAMS",
    "S30_PRIOR_SD",
    "effective_time",
    "format_minutes",
    "get_archetypes",
    "get_era_factor",
    "grad_hess",
    "invert_matrix",
    "log_posterior",
    "record_weight",
    "solve_linear",
    "walk_forward_backtest",
]
