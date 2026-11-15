"""Codeforces data access, compact caching, problem selection and cold start.

Only the standard library (``urllib``) is used for HTTP so this module has no
Discord or third-party imports.  Every function that touches the network is
synchronous by design: ``bot.py`` calls it through ``asyncio.to_thread`` so the
Discord event loop never blocks.

Memory discipline
-----------------
* The raw ``problemset.problems`` payload is parsed immediately into
  :class:`CFProblem` (five fields) and a compact JSON cache is written before
  the raw response goes out of scope.
* ``user.status`` is consumed page by page; only compact solved tuples and the
  (small) live/virtual participation submissions needed for the 40 most recent
  participation sessions survive a page.
* One process-wide rate limiter serialises *all* Codeforces requests, whether
  they come from ``/link``, the background poller or a button handler.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import predictor as predictor_core

log = logging.getLogger(__name__)

API_BASE = "https://codeforces.com/api/"
USER_AGENT = "ihave-discord-bot/2.0 (Rating-Space Elo-Survival practice bot)"
MIN_REQUEST_INTERVAL = 2.05
REQUEST_TIMEOUT_SECONDS = 30
MAX_TRIES = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_JITTER_SECONDS = 0.75

PROBLEMSET_TTL_SECONDS = 24 * 60 * 60
CONTEST_TTL_SECONDS = 24 * 60 * 60

SCAN_PAGE_SIZE = 1000
MAX_SCAN_SUBMISSIONS = 20_000
RECENT_RATED_CONTESTS = 40
SYNC_PAGE_SIZE = 100
SYNC_MAX_PAGES = 10
OVERLAP_SECONDS = 900
POLLS_PER_TICK = 8

MIN_CONTEST_ID = 1200
BLIND_JITTER = (-100, 0, 100)
POOL_SIZE = 15
RECENT_POOL_PROBABILITY = 0.7

#: Explicit wrong-verdict set used for WA counts.  Compilation errors, skipped
#: submissions, rejected hacks, TESTING and missing verdicts never count.
WRONG_VERDICTS = frozenset(
    {
        "WRONG_ANSWER",
        "RUNTIME_ERROR",
        "TIME_LIMIT_EXCEEDED",
        "MEMORY_LIMIT_EXCEEDED",
        "IDLENESS_LIMIT_EXCEEDED",
        "PRESENTATION_ERROR",
        "CHALLENGED",
    }
)


class CFApiError(Exception):
    """A clear, user-facing Codeforces API failure."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


# Backwards-compatible alias used by older code/tests.
CFError = CFApiError


# ---------------------------------------------------------------------------
# Process-wide rate limiting
# ---------------------------------------------------------------------------

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _throttle() -> None:
    """Sleep until at least ``MIN_REQUEST_INTERVAL`` seconds have elapsed.

    The sleep happens while holding the lock, so concurrent worker threads are
    serialised in arrival order and the spacing is measured from the *start* of
    the previous request.
    """
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        wait_for = MIN_REQUEST_INTERVAL - (now - _LAST_REQUEST_AT)
        if wait_for > 0:
            time.sleep(wait_for)
        _LAST_REQUEST_AT = time.monotonic()


# ---------------------------------------------------------------------------
# Paths and compact caches
# ---------------------------------------------------------------------------


def data_dir() -> str:
    """Return ``BOT_DATA_DIR`` (default ``.``), creating it if necessary."""
    directory = (os.getenv("BOT_DATA_DIR") or ".").strip() or "."
    os.makedirs(directory, exist_ok=True)
    return directory


def cache_dir() -> str:
    directory = os.path.join(data_dir(), ".cache")
    os.makedirs(directory, exist_ok=True)
    return directory


def problemset_cache_path() -> str:
    return os.path.join(cache_dir(), "problemset_compact.json")


def contests_cache_path() -> str:
    return os.path.join(cache_dir(), "contests_compact.json")


def _write_json_atomic(path: str, payload: Any) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary, path)


def _file_age_seconds(path: str) -> float | None:
    try:
        return max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        return None


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CFProblem:
    """A Codeforces problem reduced to the five fields the bot needs."""

    contest_id: int
    index: str
    name: str
    rating: int | None
    tags: tuple[str, ...]


@dataclass(slots=True)
class BootstrapRow:
    """One contest-derived observation extracted from ``user.status``."""

    contest_id: int
    problem_index: str
    problem_name: str
    rating: int | None
    tags: list[str]
    actual_mins: float
    was_count: int
    solved: bool
    occurred_at: int
    contest_start_ts: int


@dataclass(slots=True)
class LinkScanResult:
    """Summary of a single-pass ``/link`` scan."""

    bootstrap_rows: list[BootstrapRow]
    scanned_submissions: int
    rated_contests: int
    last_status_ts: int
    problems: list[CFProblem]
    contest_starts: dict[int, int]
    virtual_contests: int = 0


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------


def _request(
    method: str,
    params: Mapping[str, Any] | None = None,
    *,
    attempts: int = MAX_TRIES,
) -> Any:
    """Call one Codeforces method with throttling, retries and backoff."""
    query = urllib.parse.urlencode(
        {key: value for key, value in (params or {}).items() if value is not None}
    )
    url = API_BASE + method + (f"?{query}" if query else "")
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    last_error: CFApiError | None = None

    for attempt in range(1, max(1, attempts) + 1):
        _throttle()
        status = 0
        raw = ""
        try:
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    status = int(response.status)
                    raw = response.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as error:
                status = int(error.code)
                try:
                    raw = error.read().decode("utf-8", "replace")
                except Exception:  # pragma: no cover - defensive
                    raw = ""
            payload: Any
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            comment = str(payload.get("comment") or "").strip()

            if status == 200 and payload.get("status") == "OK":
                return payload.get("result")

            call_limit = (
                status == 200
                and "limit" in comment.lower()
                and "exceed" in comment.lower()
            )
            if status == 429 or status >= 500 or call_limit:
                message = (
                    "Codeforces API call limit exceeded. Please wait a minute."
                    if call_limit
                    else f"Codeforces is temporarily unavailable (HTTP {status})."
                )
                last_error = CFApiError(message, status=status)
                log.warning(
                    "Codeforces %s transient failure (attempt %s/%s): HTTP %s %s",
                    method,
                    attempt,
                    attempts,
                    status,
                    comment,
                )
            else:
                raise CFApiError(
                    comment or f"Codeforces API error (HTTP {status}).", status=status
                )
        except CFApiError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = CFApiError(
                "Could not reach Codeforces. Check the bot's internet connection."
            )
            log.warning(
                "Codeforces %s network failure (attempt %s/%s): %s",
                method,
                attempt,
                attempts,
                error,
            )

        if attempt < attempts:
            delay = min(
                BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), 30.0
            ) + random.uniform(0.0, BACKOFF_JITTER_SECONDS)
            time.sleep(delay)

    raise last_error or CFApiError("Codeforces API request failed. Try again shortly.")


def _get(
    method: str,
    params: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> Any:
    if fetch_json is not None:
        return fetch_json(method, params or {})
    return _request(method, params)


# ---------------------------------------------------------------------------
# Problemset
# ---------------------------------------------------------------------------


def _parse_problemset_payload(payload: Any) -> list[CFProblem]:
    if not isinstance(payload, Mapping):
        raise CFApiError("Codeforces returned an unexpected problemset payload.")
    problems: list[CFProblem] = []
    seen: set[tuple[int, str]] = set()
    for item in payload.get("problems") or ():
        if not isinstance(item, Mapping):
            continue
        contest_id = _as_int(item.get("contestId"))
        index = str(item.get("index") or "").strip()
        if contest_id is None or not index:
            continue
        key = (contest_id, index)
        if key in seen:
            continue
        seen.add(key)
        name = str(item.get("name") or f"{contest_id}{index}")
        rating = _as_int(item.get("rating"))
        tags = tuple(str(tag) for tag in (item.get("tags") or ()) if isinstance(tag, str))
        problems.append(
            CFProblem(
                contest_id=contest_id,
                index=index,
                name=name,
                rating=rating,
                tags=tags,
            )
        )
    if not problems:
        raise CFApiError("Codeforces returned an empty problemset.")
    return problems


def write_compact_problemset(problems: Sequence[CFProblem]) -> None:
    """Write the slimmed 5-field cache immediately after parsing."""
    payload = [
        [problem.contest_id, problem.index, problem.name, problem.rating, list(problem.tags)]
        for problem in problems
    ]
    _write_json_atomic(problemset_cache_path(), payload)


def load_compact_problemset(max_age: float | None = PROBLEMSET_TTL_SECONDS) -> list[CFProblem] | None:
    """Load ``problemset_compact.json``; ``None`` when missing, stale or corrupt."""
    path = problemset_cache_path()
    if not os.path.exists(path):
        return None
    age = _file_age_seconds(path)
    if max_age is not None and (age is None or age > max_age):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not payload:
            return None
        problems: list[CFProblem] = []
        for entry in payload:
            if not isinstance(entry, (list, tuple)) or len(entry) < 5:
                continue
            contest_id = _as_int(entry[0])
            index = str(entry[1] or "").strip()
            if contest_id is None or not index:
                continue
            problems.append(
                CFProblem(
                    contest_id=contest_id,
                    index=index,
                    name=str(entry[2] or f"{contest_id}{index}"),
                    rating=_as_int(entry[3]),
                    tags=tuple(str(tag) for tag in (entry[4] or ()) if isinstance(tag, str)),
                )
            )
        return problems or None
    except (OSError, ValueError, json.JSONDecodeError):
        log.warning("Problemset cache at %s is corrupt; refetching.", path)
        return None


def fetch_problemset(
    *,
    force_refresh: bool = False,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> list[CFProblem]:
    """Return the problemset, preferring the compact 24-hour cache."""
    if not force_refresh:
        cached = load_compact_problemset()
        if cached is not None:
            return cached
    try:
        payload = _get("problemset.problems", {}, fetch_json)
        problems = _parse_problemset_payload(payload)
        write_compact_problemset(problems)
        return problems
    except CFApiError:
        stale = load_compact_problemset(max_age=None)
        if stale is not None:
            log.warning("Codeforces problemset unavailable; using the stale cache.")
            return stale
        raise


# ---------------------------------------------------------------------------
# Contests
# ---------------------------------------------------------------------------


def _parse_contests_payload(payload: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, list):
        raise CFApiError("Codeforces returned unexpected contest data.")
    contests: dict[int, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        contest_id = _as_int(item.get("id"))
        if contest_id is None:
            continue
        contests[contest_id] = {
            "startTimeSeconds": _as_int(item.get("startTimeSeconds")),
            "durationSeconds": _as_int(item.get("durationSeconds")),
            "name": str(item.get("name") or ""),
        }
    return contests


def _write_compact_contests(contests: Mapping[int, Mapping[str, Any]]) -> None:
    payload = [
        [
            int(contest_id),
            metadata.get("startTimeSeconds"),
            metadata.get("durationSeconds"),
            metadata.get("name") or "",
        ]
        for contest_id, metadata in contests.items()
    ]
    _write_json_atomic(contests_cache_path(), payload)


def _load_compact_contests(max_age: float | None = CONTEST_TTL_SECONDS) -> dict[int, dict[str, Any]] | None:
    path = contests_cache_path()
    if not os.path.exists(path):
        return None
    age = _file_age_seconds(path)
    if max_age is not None and (age is None or age > max_age):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not payload:
            return None
        contests: dict[int, dict[str, Any]] = {}
        for entry in payload:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            contest_id = _as_int(entry[0])
            if contest_id is None:
                continue
            contests[contest_id] = {
                "startTimeSeconds": _as_int(entry[1]),
                "durationSeconds": _as_int(entry[2]),
                "name": str(entry[3] if len(entry) > 3 else ""),
            }
        return contests or None
    except (OSError, ValueError, json.JSONDecodeError):
        log.warning("Contest cache at %s is corrupt; refetching.", path)
        return None


def fetch_contests(
    *,
    force_refresh: bool = False,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return ``{contest_id: {startTimeSeconds, durationSeconds, name}}``."""
    if not force_refresh:
        cached = _load_compact_contests()
        if cached is not None:
            return cached
    try:
        payload = _get("contest.list", {"gym": "false"}, fetch_json)
        contests = _parse_contests_payload(payload)
        _write_compact_contests(contests)
        return contests
    except CFApiError:
        stale = _load_compact_contests(max_age=None)
        if stale is not None:
            log.warning("Codeforces contest list unavailable; using the stale cache.")
            return stale
        raise


def fetch_contest_starts(
    *,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> dict[int, int]:
    """Return ``{contestId: startTimeSeconds}`` for contests with a known start."""
    contests = fetch_contests(fetch_json=fetch_json)
    return {
        contest_id: int(metadata["startTimeSeconds"])
        for contest_id, metadata in contests.items()
        if metadata.get("startTimeSeconds") is not None
    }


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


def normalize_handle(handle: str) -> str:
    cleaned = str(handle or "").strip().lstrip("@")
    if not cleaned:
        raise CFApiError("Please provide a Codeforces handle, e.g. `/link tourist`.")
    if len(cleaned) > 64 or any(character.isspace() for character in cleaned):
        raise CFApiError(f"`{handle}` does not look like a valid Codeforces handle.")
    return cleaned


def fetch_user_info(
    handle: str,
    *,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Return ``user.info`` for a handle, preserving the canonical spelling."""
    cleaned = normalize_handle(handle)
    try:
        result = _get("user.info", {"handles": cleaned}, fetch_json)
    except CFApiError as error:
        if "not found" in str(error).lower():
            raise CFApiError(f"Codeforces handle `{cleaned}` was not found.") from error
        raise
    if not isinstance(result, list) or not result or not isinstance(result[0], Mapping):
        raise CFApiError(f"Codeforces handle `{cleaned}` was not found.")
    return dict(result[0])


def fetch_user_rating(
    handle: str,
    *,
    info: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> int:
    """Current rating from ``user.info``; unrated users fall back to 1200."""
    payload = info if info is not None else fetch_user_info(handle, fetch_json=fetch_json)
    rating = _as_int(payload.get("rating"))
    return 1200 if rating is None else int(rating)


def fetch_user_ratings(
    handle: str,
    *,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the user's ``user.rating`` rows (rated contest changes)."""
    cleaned = normalize_handle(handle)
    try:
        result = _get("user.rating", {"handle": cleaned}, fetch_json)
    except CFApiError as error:
        message = str(error).lower()
        if (
            "not participated" in message
            or "has not" in message
            or "no rated" in message
        ):
            # Brand-new/unrated accounts have no rating history.
            return []
        raise
    if result is None:
        return []
    if not isinstance(result, list):
        raise CFApiError(f"Codeforces returned unexpected rating data for `{cleaned}`.")
    return [dict(item) for item in result if isinstance(item, Mapping)]


def fetch_user_status_page(
    handle: str,
    from_idx: int,
    count: int,
    *,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch one page of ``user.status`` (newest submissions first)."""
    cleaned = normalize_handle(handle)
    try:
        safe_count = max(1, min(int(count), 10_000))
    except (TypeError, ValueError):
        safe_count = 1000
    result = _get(
        "user.status",
        {"handle": cleaned, "from": max(1, int(from_idx)), "count": safe_count},
        fetch_json,
    )
    if result is None:
        return []
    if not isinstance(result, list):
        raise CFApiError(f"Codeforces returned unexpected submission data for `{cleaned}`.")
    return [dict(item) for item in result if isinstance(item, Mapping)]


# ---------------------------------------------------------------------------
# Contest bootstrap extractor (pure, unit-tested)
# ---------------------------------------------------------------------------


def _index_rank(index: str) -> tuple[int, int]:
    """Sort key for problem letters: A < B < C1 < C2 < D."""
    text = str(index or "").strip().upper()
    if not text:
        return (999, 0)
    letter = text[0]
    base = ord(letter) - 65 if "A" <= letter <= "Z" else 99
    suffix = text[1:]
    try:
        number = int(suffix) if suffix else 0
    except ValueError:
        number = 0
    return (base, number)


def _select_recent_rating_changes(
    rating_changes: Iterable[Any],
    max_contests: int,
) -> list[tuple[int, int | None, int]]:
    parsed: list[tuple[int, int | None, int]] = []
    for change in rating_changes or ():
        if not isinstance(change, Mapping):
            continue
        contest_id = _as_int(change.get("contestId"))
        if contest_id is None:
            continue
        old_rating = _as_int(change.get("oldRating"))
        updated = _as_int(change.get("ratingUpdateTimeSeconds"), 0) or 0
        parsed.append((contest_id, old_rating, updated))
    parsed.sort(key=lambda item: (item[2], item[0]), reverse=True)
    return parsed[: max(1, int(max_contests))]


def _problem_meta(
    problems_by_key: Mapping[tuple[int, str], Any] | None,
    contest_id: int,
    index: str,
    submission: Mapping[str, Any],
) -> tuple[str, int | None, list[str]]:
    entry: Any = None
    if problems_by_key:
        entry = problems_by_key.get((contest_id, index))
    if entry is not None:
        if isinstance(entry, Mapping):
            name = str(entry.get("name") or f"{contest_id}{index}")
            rating = _as_int(entry.get("rating"))
            tags = [str(tag) for tag in (entry.get("tags") or ())]
        else:
            name = str(getattr(entry, "name", None) or f"{contest_id}{index}")
            rating = _as_int(getattr(entry, "rating", None))
            tags = [str(tag) for tag in (getattr(entry, "tags", ()) or ())]
        return name, rating, tags
    raw = submission.get("problem")
    if isinstance(raw, Mapping):
        name = str(raw.get("name") or f"{contest_id}{index}")
        rating = _as_int(raw.get("rating"))
        tags = [str(tag) for tag in (raw.get("tags") or ())]
        return name, rating, tags
    return f"{contest_id}{index}", None, []


def _parse_rating_history(
    rating_changes: Iterable[Any],
) -> list[tuple[int, int | None, int | None, int]]:
    """Return ``(contest_id, old, new, update_ts)`` sorted oldest first."""
    parsed: list[tuple[int, int | None, int | None, int]] = []
    for change in rating_changes or ():
        if not isinstance(change, Mapping):
            continue
        contest_id = _as_int(change.get("contestId"))
        if contest_id is None:
            continue
        parsed.append(
            (
                contest_id,
                _as_int(change.get("oldRating")),
                _as_int(change.get("newRating")),
                _as_int(change.get("ratingUpdateTimeSeconds"), 0) or 0,
            )
        )
    parsed.sort(key=lambda item: (item[3], item[0]))
    return parsed


def _old_rating_for_participation(
    history: Sequence[tuple[int, int | None, int | None, int]],
    participant_type: str,
    contest_id: int,
    part_start_ts: int,
    old_by_contest: Mapping[int, int | None],
) -> int | None:
    """Rating immediately before a participation, for the -300 floor."""
    if participant_type == "CONTESTANT":
        old_rating = old_by_contest.get(contest_id)
        if old_rating is not None:
            return old_rating
    if not history:
        return None
    update_times = [item[3] for item in history]
    position = bisect.bisect_right(update_times, int(part_start_ts)) - 1
    if position >= 0:
        return history[position][2]  # newRating of the latest prior contest
    return history[0][1]  # before the first rated contest: its oldRating


def extract_contest_deltas_from_data(
    rating_changes: Iterable[Any],
    contests_by_id: Mapping[int, Mapping[str, Any]],
    submissions: Iterable[Mapping[str, Any]],
    problems_by_key: Mapping[tuple[int, str], Any] | None = None,
    now_ts: float | None = None,
    max_contests: int = RECENT_RATED_CONTESTS,
    first_ok_ts: Mapping[tuple[int, str], int] | None = None,
) -> list[BootstrapRow]:
    """Extract clean sequential solve times and end-of-contest timeouts.

    Live (``CONTESTANT``) and virtual (``VIRTUAL``) participations are grouped
    by ``(contest_id, participant_type, part_start_ts)``.  For virtuals a
    problem already solved before the virtual started is ignored, so re-solving
    an old problem does not create a fake fast delta.

    Pure function: all inputs are plain synthetic data in tests, and the same
    contest window logic is used in production from
    :func:`scan_user_history_for_link`.
    """
    del now_ts  # timestamps are stored on the rows; the model ages them later
    rating_history = _parse_rating_history(rating_changes)
    old_rating_by_contest = {
        contest_id: old_rating for contest_id, old_rating, _, _ in rating_history
    }

    first_ok: dict[tuple[int, str], int] = {}
    for key, value in (first_ok_ts or {}).items():
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            key_contest = _as_int(key[0])
            key_index = str(key[1] or "").strip()
            key_timestamp = _as_int(value)
            if key_contest is None or not key_index or key_timestamp is None:
                continue
            first_ok[(key_contest, key_index)] = key_timestamp

    sessions: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for submission in submissions or ():
        if not isinstance(submission, Mapping):
            continue
        problem = submission.get("problem")
        if not isinstance(problem, Mapping):
            continue
        contest_id = _as_int(submission.get("contestId"))
        if contest_id is None:
            contest_id = _as_int(problem.get("contestId"))
        index = str(problem.get("index") or "").strip()
        created = _as_int(submission.get("creationTimeSeconds"))
        verdict = str(submission.get("verdict") or "")

        if verdict == "OK" and contest_id is not None and index and created is not None:
            key = (contest_id, index)
            previous = first_ok.get(key)
            if previous is None or created < previous:
                first_ok[key] = created

        author = submission.get("author")
        if not isinstance(author, Mapping):
            continue
        participant_type = author.get("participantType")
        if participant_type not in ("CONTESTANT", "VIRTUAL"):
            continue
        if contest_id is None or contest_id not in contests_by_id:
            continue
        if not index or created is None:
            continue
        metadata = contests_by_id.get(contest_id)
        if not isinstance(metadata, Mapping):
            continue
        official_start = _as_int(metadata.get("startTimeSeconds"))
        duration = _as_int(metadata.get("durationSeconds"))
        if official_start is None or duration is None or duration <= 0:
            continue

        if participant_type == "CONTESTANT":
            part_start = official_start
        else:
            part_start = _as_int(author.get("startTimeSeconds"))
            if part_start is None or part_start <= 0:
                continue
        if not (part_start <= created <= part_start + duration):
            continue

        sessions.setdefault((contest_id, str(participant_type), part_start), []).append(
            {
                "index": index,
                "offset": (created - part_start) / 60.0,
                "verdict": verdict,
                "submission": submission,
            }
        )

    if not sessions:
        return []

    rows: list[BootstrapRow] = []
    ordered_sessions = sorted(
        sessions.keys(), key=lambda key: (key[2], key[0], key[1]), reverse=True
    )
    for session_key in ordered_sessions[: max(1, int(max_contests))]:
        contest_id, participant_type, part_start = session_key
        metadata = contests_by_id.get(contest_id) or {}
        official_start = int(_as_int(metadata.get("startTimeSeconds"), 0) or 0)
        duration_seconds = int(_as_int(metadata.get("durationSeconds"), 0) or 0)
        duration_mins = duration_seconds / 60.0
        contest_submissions = list(sessions[session_key])

        if participant_type == "VIRTUAL":
            pre_solved = {
                index
                for index in {str(item["index"]) for item in contest_submissions}
                if first_ok.get((contest_id, index)) is not None
                and int(first_ok[(contest_id, index)]) < part_start
            }
            if pre_solved:
                contest_submissions = [
                    item
                    for item in contest_submissions
                    if str(item["index"]) not in pre_solved
                ]
        if not contest_submissions:
            continue

        old_rating = _old_rating_for_participation(
            rating_history,
            participant_type,
            contest_id,
            part_start,
            old_rating_by_contest,
        )
        rating_floor = (old_rating - 300) if old_rating is not None else None

        by_problem: dict[str, list[dict[str, Any]]] = {}
        for item in contest_submissions:
            by_problem.setdefault(str(item["index"]), []).append(item)

        solved_offsets: dict[str, tuple[float, int]] = {}
        for index, items in by_problem.items():
            ok_items = [
                item
                for item in items
                if item["verdict"] == "OK" and float(item["offset"]) >= 0.0
            ]
            if not ok_items:
                continue
            first_ok_item = min(ok_items, key=lambda item: float(item["offset"]))
            accepted_offset = float(first_ok_item["offset"])
            wrong = sum(
                1
                for item in items
                if float(item["offset"]) < accepted_offset
                and item["verdict"] in WRONG_VERDICTS
            )
            solved_offsets[index] = (accepted_offset, wrong)

        solved_order = sorted(
            solved_offsets.items(), key=lambda item: (item[1][0], item[0])
        )
        previous_time = 0.0
        previous_rank: tuple[int, int] | None = None

        for index, (accepted_offset, wrong_count) in solved_order:
            rank = _index_rank(index)
            clean = previous_rank is None or rank >= previous_rank
            if clean:
                clean = not any(
                    str(item["index"]) != index
                    and previous_time < float(item["offset"]) < accepted_offset
                    for item in contest_submissions
                )
            problem_name, rating, tags = _problem_meta(
                problems_by_key, contest_id, index, by_problem[index][0]["submission"]
            )
            if (
                clean
                and rating is not None
                and (rating_floor is None or rating >= rating_floor)
            ):
                actual = max(1.0, accepted_offset - previous_time)
                rows.append(
                    BootstrapRow(
                        contest_id=contest_id,
                        problem_index=index,
                        problem_name=problem_name,
                        rating=rating,
                        tags=tags,
                        actual_mins=actual,
                        was_count=wrong_count,
                        solved=True,
                        occurred_at=part_start + int(round(accepted_offset * 60.0)),
                        contest_start_ts=official_start,
                    )
                )
            previous_time = accepted_offset
            previous_rank = rank

        last_ac = max(
            (offset for offset, _ in solved_offsets.values()), default=0.0
        )
        remaining = duration_mins - last_ac
        if remaining >= 15.0:
            post = [
                item
                for item in contest_submissions
                if float(item["offset"]) > last_ac
                and item["verdict"] != "OK"
                and str(item["index"]) not in solved_offsets
            ]
            distinct = {str(item["index"]) for item in post}
            if len(distinct) == 1:
                index = next(iter(distinct))
                problem_name, rating, tags = _problem_meta(
                    problems_by_key, contest_id, index, post[0]["submission"]
                )
                if rating is not None:
                    rows.append(
                        BootstrapRow(
                            contest_id=contest_id,
                            problem_index=index,
                            problem_name=problem_name,
                            rating=rating,
                            tags=tags,
                            actual_mins=max(1.0, remaining),
                            was_count=0,
                            solved=False,
                            occurred_at=part_start + duration_seconds,
                            contest_start_ts=official_start,
                        )
                    )

    rows.sort(key=lambda row: (row.occurred_at, row.contest_id, row.problem_index))
    return rows


# ---------------------------------------------------------------------------
# Single-pass /link scan
# ---------------------------------------------------------------------------


def _compact_participation_submission(
    submission: Mapping[str, Any],
    contest_id: int,
    participant_type: str,
    part_start_ts: int,
) -> dict[str, Any]:
    problem = submission.get("problem") if isinstance(submission.get("problem"), Mapping) else {}
    return {
        "contestId": contest_id,
        "creationTimeSeconds": _as_int(submission.get("creationTimeSeconds"), 0),
        "verdict": submission.get("verdict"),
        "author": {
            "participantType": participant_type,
            "startTimeSeconds": part_start_ts,
        },
        "problem": {
            "contestId": contest_id,
            "index": problem.get("index"),
            "name": problem.get("name"),
            "rating": problem.get("rating"),
            "tags": list(problem.get("tags") or ()),
        },
    }


def _compact_contestant_submission(
    submission: Mapping[str, Any],
    contest_id: int,
    start_ts: int,
) -> dict[str, Any]:
    """Backwards-compatible wrapper for the live-contest compactor."""
    return _compact_participation_submission(
        submission, contest_id, "CONTESTANT", start_ts
    )


def scan_user_history_for_link(
    db: Any,
    discord_id: int,
    handle: str,
    *,
    fetch_json: Callable[[str, Mapping[str, Any]], Any] | None = None,
    now_ts: float | None = None,
    max_submissions: int = MAX_SCAN_SUBMISSIONS,
    page_size: int = SCAN_PAGE_SIZE,
) -> LinkScanResult:
    """One-pass full history scan used by ``/link``.

    ``user.rating``, ``contest.list`` and ``problemset.problems`` are fetched
    first so the contest windows are known before submissions are scanned.
    Then ``user.status`` is paged 1000 rows at a time with no date cutoff; each
    page is reduced to compact solved rows and live/virtual participation
    submissions before the next page is requested.
    """
    cleaned = normalize_handle(handle)
    rating_changes = fetch_user_ratings(cleaned, fetch_json=fetch_json)
    contests_by_id = fetch_contests(fetch_json=fetch_json)
    problems = fetch_problemset(fetch_json=fetch_json)
    problems_by_key = {
        (problem.contest_id, problem.index): problem for problem in problems
    }

    selected = _select_recent_rating_changes(rating_changes, RECENT_RATED_CONTESTS)

    solved_rows: list[tuple[int, str, str]] = []
    participation_submissions: list[dict[str, Any]] = []
    first_ok_ts: dict[tuple[int, str], int] = {}
    virtual_participations: set[tuple[int, int]] = set()
    scanned = 0
    max_status_ts = 0
    from_idx = 1
    safe_page_size = max(1, int(page_size))

    while scanned < max_submissions:
        page = fetch_user_status_page(
            cleaned, from_idx, safe_page_size, fetch_json=fetch_json
        )
        if not page:
            break
        scanned += len(page)
        for submission in page:
            timestamp = _as_int(submission.get("creationTimeSeconds"), 0) or 0
            if timestamp > max_status_ts:
                max_status_ts = timestamp
            verdict = submission.get("verdict")
            problem = (
                submission.get("problem")
                if isinstance(submission.get("problem"), Mapping)
                else {}
            )
            contest_id = _as_int(submission.get("contestId"))
            if contest_id is None:
                contest_id = _as_int(problem.get("contestId"))
            index = str(problem.get("index") or "").strip()

            if verdict == "OK" and contest_id is not None and index:
                solved_rows.append(
                    (
                        int(contest_id),
                        index,
                        str(problem.get("name") or f"{contest_id}{index}"),
                    )
                )
                if timestamp > 0:
                    ok_key = (int(contest_id), index)
                    previous_ok = first_ok_ts.get(ok_key)
                    if previous_ok is None or timestamp < previous_ok:
                        first_ok_ts[ok_key] = timestamp

            author = submission.get("author")
            if not isinstance(author, Mapping):
                continue
            participant_type = author.get("participantType")
            if participant_type not in ("CONTESTANT", "VIRTUAL"):
                continue
            if contest_id is None or contest_id not in contests_by_id or not index:
                continue
            metadata = contests_by_id.get(contest_id)
            if not isinstance(metadata, Mapping):
                continue
            official_start = _as_int(metadata.get("startTimeSeconds"))
            duration = _as_int(metadata.get("durationSeconds"))
            if official_start is None or duration is None or duration <= 0:
                continue
            if participant_type == "CONTESTANT":
                part_start = official_start
            else:
                part_start = _as_int(author.get("startTimeSeconds"))
                if part_start is None or part_start <= 0:
                    continue
            if not (part_start <= timestamp <= part_start + duration):
                continue
            if participant_type == "VIRTUAL":
                virtual_participations.add((int(contest_id), int(part_start)))
            participation_submissions.append(
                _compact_participation_submission(
                    submission, contest_id, str(participant_type), part_start
                )
            )

        if len(page) < safe_page_size:
            break
        from_idx += safe_page_size

    db.add_cf_solved(discord_id, solved_rows, last_status_ts=max_status_ts)
    bootstrap_rows = extract_contest_deltas_from_data(
        rating_changes,
        contests_by_id,
        participation_submissions,
        problems_by_key=problems_by_key,
        now_ts=now_ts,
        max_contests=RECENT_RATED_CONTESTS,
        first_ok_ts=first_ok_ts,
    )
    contest_starts = {
        contest_id: int(metadata["startTimeSeconds"])
        for contest_id, metadata in contests_by_id.items()
        if metadata.get("startTimeSeconds") is not None
    }
    return LinkScanResult(
        bootstrap_rows=bootstrap_rows,
        scanned_submissions=scanned,
        rated_contests=len(selected),
        last_status_ts=max_status_ts,
        problems=problems,
        contest_starts=contest_starts,
        virtual_contests=len(virtual_participations),
    )


# ---------------------------------------------------------------------------
# Incremental status sync
# ---------------------------------------------------------------------------


def sync_recent_status(
    db: Any,
    discord_id: int,
    handle: str,
    last_status_ts: int | None = None,
    fetch_page: Callable[[str, int, int], list[dict[str, Any]]] | None = None,
    page_size: int = SYNC_PAGE_SIZE,
    max_pages: int = SYNC_MAX_PAGES,
) -> list[dict[str, Any]]:
    """Incrementally merge newly accepted problems, with a 15-minute overlap.

    Returns the first page of raw submissions so the background poller can
    inspect the user's active session without another API call.
    """
    cleaned = normalize_handle(handle)
    if last_status_ts is None:
        try:
            last_status_ts = int(db.get_last_status_ts(discord_id) or 0)
        except Exception:  # pragma: no cover - defensive for duck-typed stubs
            last_status_ts = 0
    cursor = int(last_status_ts or 0)
    fetcher = fetch_page or fetch_user_status_page
    safe_page_size = max(1, int(page_size))
    overlap_boundary = cursor - OVERLAP_SECONDS

    solved_rows: list[tuple[int, str, str]] = []
    first_page: list[dict[str, Any]] = []
    max_timestamp = 0
    reached_overlap = False

    for page_index in range(max(1, int(max_pages))):
        from_idx = 1 + page_index * safe_page_size
        page = fetcher(cleaned, from_idx, safe_page_size)
        page = list(page or [])
        if page_index == 0:
            first_page = page
        for submission in page:
            if not isinstance(submission, Mapping):
                continue
            timestamp = _as_int(submission.get("creationTimeSeconds"), 0) or 0
            if timestamp > max_timestamp:
                max_timestamp = timestamp
            # Overlap stop rule: verdicts on in-flight TESTING submissions can
            # turn OK after the cursor advanced, so re-scan the last 900s.
            if cursor > 0 and timestamp < overlap_boundary:
                reached_overlap = True
                break
            verdict = submission.get("verdict")
            if verdict is None or verdict == "TESTING":
                continue
            if verdict != "OK":
                continue
            problem = (
                submission.get("problem")
                if isinstance(submission.get("problem"), Mapping)
                else {}
            )
            contest_id = _as_int(submission.get("contestId"))
            if contest_id is None:
                contest_id = _as_int(problem.get("contestId"))
            index = str(problem.get("index") or "").strip()
            if contest_id is None or not index:
                continue
            solved_rows.append(
                (
                    int(contest_id),
                    index,
                    str(problem.get("name") or f"{contest_id}{index}"),
                )
            )
        if reached_overlap or len(page) < safe_page_size:
            break

    new_last_status_ts = max(cursor, max_timestamp)
    db.add_cf_solved(discord_id, solved_rows, last_status_ts=new_last_status_ts)
    return first_page


# ---------------------------------------------------------------------------
# Problem selection
# ---------------------------------------------------------------------------


def select_best_problem(
    predictor: Any,
    problems: list[CFProblem],
    solved_keys: set[tuple[int, str]],
    solved_names: set[str],
    target_mins: float,
    confidence: float = 0.65,
    target_archetype: int | None = None,
    min_contest_id: int | None = None,
    blind_jitter: bool = True,
    contest_starts: Mapping[int, int] | None = None,
    now_ts: float | None = None,
    rng: Any = None,
    allow_interactive: bool = False,
) -> CFProblem | None:
    """Pick a random unsolved problem whose effective rating matches the target.

    Blind jitter (``-100/0/+100``) keeps the user from inferring the rating.
    Candidates are ranked by effective rating (era- and tag-adjusted), the
    closest :data:`POOL_SIZE` form the pool, and 70% of the time the pool is
    restricted to contests ``>= 1200`` (modern, well-tested problems).
    """
    if not problems:
        return None
    generator = rng if rng is not None else random
    jitter = generator.choice(list(BLIND_JITTER)) if blind_jitter else 0
    target = (
        predictor.predict_modern_rating(
            float(target_mins), float(confidence), arch_idx=None
        )
        + float(jitter)
    )

    candidates: list[tuple[float, CFProblem, float]] = []
    for problem in problems:
        if not isinstance(problem, CFProblem):
            continue
        if problem.rating is None:
            continue
        key = (int(problem.contest_id), str(problem.index))
        if key in solved_keys or problem.name in solved_names:
            continue
        tags = [tag.strip().lower() for tag in problem.tags]
        if "*special" in tags:
            continue
        if "interactive" in tags and not allow_interactive:
            continue
        if min_contest_id is not None and problem.contest_id < min_contest_id:
            continue
        archetypes = predictor_core.get_archetypes(tags)
        if target_archetype is not None and target_archetype not in archetypes:
            continue
        handicap = (
            sum(predictor.delta[archetype] for archetype in archetypes)
            / len(archetypes)
            if archetypes
            else 0.0
        )
        start_ts = None
        if contest_starts is not None:
            start_ts = contest_starts.get(problem.contest_id)
        era = predictor_core.get_era_factor(problem.contest_id, start_ts, now_ts)
        effective_rating = float(problem.rating) - predictor.gamma * era + handicap
        candidates.append((abs(effective_rating - target), problem, effective_rating))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].contest_id, item[1].index))
    pool = [item[1] for item in candidates[:POOL_SIZE]]
    modern_pool = [problem for problem in pool if problem.contest_id >= MIN_CONTEST_ID]
    if modern_pool and generator.random() < RECENT_POOL_PROBABILITY:
        pool = modern_pool
    return generator.choice(pool)


__all__ = [
    "BootstrapRow",
    "CFApiError",
    "CFError",
    "CFProblem",
    "LinkScanResult",
    "MAX_SCAN_SUBMISSIONS",
    "MIN_REQUEST_INTERVAL",
    "OVERLAP_SECONDS",
    "POLLS_PER_TICK",
    "RECENT_RATED_CONTESTS",
    "SCAN_PAGE_SIZE",
    "SYNC_MAX_PAGES",
    "SYNC_PAGE_SIZE",
    "WRONG_VERDICTS",
    "cache_dir",
    "contests_cache_path",
    "data_dir",
    "extract_contest_deltas_from_data",
    "fetch_contest_starts",
    "fetch_contests",
    "fetch_problemset",
    "fetch_user_info",
    "fetch_user_rating",
    "fetch_user_ratings",
    "fetch_user_status_page",
    "load_compact_problemset",
    "problemset_cache_path",
    "scan_user_history_for_link",
    "select_best_problem",
    "sync_recent_status",
    "write_compact_problemset",
]
