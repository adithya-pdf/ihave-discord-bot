"""Asynchronous client for the public Codeforces API.

Only public endpoints are used (no API key, OAuth or paid service):

* ``problemset.problems``
* ``user.info``
* ``user.status``
* ``contest.list`` -- only to identify April Fools / Kotlin Heroes contests

Request pacing is a hard requirement: every call, including background session
polls, passes through one process-wide :class:`asyncio.Lock` and waits at least
:data:`MIN_REQUEST_INTERVAL` seconds since the previous request started. This
keeps the bot far below Codeforces' public call limit and avoids HTTP 429s.

Transient failures (timeouts, connection errors, HTTP 429 and 5xx) are retried
with exponential backoff; each attempt has a 30 second timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

API_BASE = "https://codeforces.com/api/"
USER_AGENT = "ihave-discord-bot/1.0 (Discord practice bot)"
MIN_REQUEST_INTERVAL = 2.05
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 3.0

#: Known April Fools contest ids (safety net when ``contest.list`` is unavailable).
DEFAULT_SPECIAL_CONTEST_IDS = frozenset(
    {409, 656, 784, 952, 1145, 1331, 1508, 1663, 1812, 1952}
)
SPECIAL_CONTEST_NAME_MARKERS = ("april fools", "kotlin")

# --- Process-wide rate limiter -------------------------------------------------
#
# Module-level state is intentional: command handlers, the problemset refresh
# and every session poller share the same lock, so even concurrent tasks cannot
# burst requests at Codeforces.

_RATE_LOCK: asyncio.Lock | None = None
_LAST_REQUEST_AT = 0.0


def _get_rate_lock() -> asyncio.Lock:
    """Lazily create the singleton lock (safe to import outside a running loop)."""
    global _RATE_LOCK
    if _RATE_LOCK is None:
        _RATE_LOCK = asyncio.Lock()
    return _RATE_LOCK


async def _throttle() -> None:
    """Wait until this process may issue another Codeforces request.

    The sleep intentionally happens *while holding* the lock so waiting callers
    are serialized in FIFO-ish order and the spacing is measured from the start
    of the previous request.
    """
    global _LAST_REQUEST_AT
    lock = _get_rate_lock()
    async with lock:
        now = time.monotonic()
        wait_for = MIN_REQUEST_INTERVAL - (now - _LAST_REQUEST_AT)
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        _LAST_REQUEST_AT = time.monotonic()


class CFError(Exception):
    """User-facing Codeforces error with a clear message."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def normalize_handle(handle: str) -> str:
    """Strip decoration from a handle and reject obviously invalid input."""
    cleaned = str(handle or "").strip().lstrip("@")
    if not cleaned:
        raise CFError("Please provide a Codeforces handle, e.g. `;link tourist`.")
    if len(cleaned) > 64 or any(character.isspace() for character in cleaned):
        raise CFError(f"`{handle}` does not look like a valid Codeforces handle.")
    return cleaned


def _api_message(status: int, comment: str) -> str:
    lowered = (comment or "").lower()
    if "not found" in lowered:
        return f"Codeforces could not find that handle ({comment})."
    if "limit" in lowered and "exceed" in lowered:
        return "Codeforces API call limit exceeded. Please wait a minute and try again."
    if comment:
        return f"Codeforces rejected the request: {comment}"
    if status == 400:
        return "Codeforces rejected the request (HTTP 400). Check the handle and try again."
    return f"Codeforces API error (HTTP {status})."


def _transient_message(status: int, comment: str) -> str:
    if status == 429:
        return "Codeforces rate limit reached (HTTP 429). Please wait a minute and try again."
    if comment:
        return f"Codeforces is temporarily unavailable ({comment}). Please try again shortly."
    return f"Codeforces is temporarily unavailable (HTTP {status}). Please try again shortly."


class CodeforcesClient:
    """Small async wrapper around the public Codeforces API."""

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None
        self._special_contest_ids: set[int] | None = None
        self._special_contest_ids_at = 0.0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> "CodeforcesClient":
        """Create the underlying aiohttp session if necessary."""
        await self._ensure_session()
        return self

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "CodeforcesClient":
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            self._owns_session = True
        return self._session

    # -- low level request handling -----------------------------------------

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        attempts: int = MAX_ATTEMPTS,
    ) -> Any:
        """Call one Codeforces method with throttling, retries and error mapping."""
        session = await self._ensure_session()
        url = API_BASE + method
        last_error: CFError | None = None

        for attempt in range(1, max(1, attempts) + 1):
            await _throttle()
            status = 0
            text = ""
            call_limit = False
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
                async with session.get(url, params=params, timeout=timeout) as response:
                    status = response.status
                    text = await response.text()
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                last_error = CFError(
                    "Could not reach Codeforces. Check the bot's internet connection "
                    "and try again.",
                    retryable=True,
                )
                log.warning(
                    "Codeforces %s request failed (attempt %s/%s): %s",
                    method,
                    attempt,
                    attempts,
                    exc,
                )
            else:
                try:
                    payload = json.loads(text) if text else {}
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
                    if call_limit:
                        message = (
                            "Codeforces API call limit exceeded. Please wait a "
                            "minute and try again."
                        )
                    else:
                        message = _transient_message(status, comment)
                    last_error = CFError(
                        message,
                        status=status,
                        retryable=True,
                    )
                    log.warning(
                        "Codeforces %s temporarily failed (attempt %s/%s): HTTP %s %s",
                        method,
                        attempt,
                        attempts,
                        status,
                        comment,
                    )
                else:
                    raise CFError(_api_message(status, comment), status=status)

            if attempt < attempts:
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                if status == 429 or call_limit:
                    delay *= 1.5
                log.info("Retrying Codeforces %s in %.1fs", method, delay)
                await asyncio.sleep(delay)

        raise last_error or CFError("Codeforces API request failed. Please try again.")

    # -- public API ----------------------------------------------------------

    async def get_user_info(self, handle: str) -> dict[str, Any]:
        """Return ``user.info`` for a handle (canonical spelling included)."""
        cleaned = normalize_handle(handle)
        try:
            result = await self._request("user.info", {"handles": cleaned})
        except CFError as exc:
            if "not found" in str(exc).lower():
                raise CFError(
                    f"Codeforces handle `{cleaned}` was not found. Handles are "
                    "case-insensitive, so check the spelling and try again."
                ) from exc
            raise
        if not isinstance(result, list) or not result:
            raise CFError(f"Codeforces handle `{cleaned}` was not found.")
        info = result[0]
        if not isinstance(info, dict):
            raise CFError(f"Codeforces returned unexpected data for `{cleaned}`.")
        return info

    async def get_user_status(self, handle: str, count: int = 10000) -> list[dict[str, Any]]:
        """Return the user's submissions (newest first, as Codeforces returns them)."""
        cleaned = normalize_handle(handle)
        try:
            safe_count = max(1, min(int(count), 10000))
        except (TypeError, ValueError):
            safe_count = 10000
        result = await self._request(
            "user.status",
            {"handle": cleaned, "from": 1, "count": safe_count},
        )
        if result is None:
            return []
        if not isinstance(result, list):
            raise CFError(f"Codeforces returned unexpected submission data for `{cleaned}`.")
        return [item for item in result if isinstance(item, dict)]

    async def get_problemset(self) -> list[dict[str, Any]]:
        """Fetch, normalize and filter ``problemset.problems``.

        Problems without a rating, outside 800..3500, or belonging to April
        Fools / Kotlin Heroes contests are excluded. Each returned dict has
        ``contest_id``, ``index_name``, ``name``, ``rating``, ``solved_count``
        and ``tags`` (a list of strings).
        """
        payload = await self._request("problemset.problems")
        if not isinstance(payload, dict):
            raise CFError("Codeforces returned an unexpected problemset payload.")

        statistics: dict[tuple[int, str], int] = {}
        for stat in payload.get("problemStatistics") or []:
            if not isinstance(stat, dict):
                continue
            key = (_as_optional_int(stat.get("contestId")), str(stat.get("index") or ""))
            if key[0] is None or not key[1]:
                continue
            statistics[key] = _as_optional_int(stat.get("solvedCount")) or 0

        special_contests = await self._get_special_contest_ids()
        problems: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()

        for problem in payload.get("problems") or []:
            if not isinstance(problem, dict):
                continue
            contest_id = _as_optional_int(problem.get("contestId"))
            index_name = str(problem.get("index") or "").strip()
            rating = _as_optional_int(problem.get("rating"))
            if contest_id is None or not index_name or rating is None:
                continue
            if contest_id in special_contests:
                continue
            if rating < 800 or rating > 3500:
                continue
            key = (contest_id, index_name)
            if key in seen:
                continue
            seen.add(key)
            problems.append(
                {
                    "contest_id": contest_id,
                    "index_name": index_name,
                    "name": str(problem.get("name") or f"{contest_id}{index_name}"),
                    "rating": int(rating),
                    "solved_count": int(statistics.get(key, 0)),
                    "tags": [str(tag) for tag in (problem.get("tags") or [])],
                }
            )

        if not problems:
            raise CFError("Codeforces returned an empty problemset. Please try again later.")
        return problems

    async def _get_special_contest_ids(self) -> set[int]:
        """Identify April Fools / Kotlin Heroes contests by name.

        Cached for six hours. If ``contest.list`` fails, a curated denylist is
        used so a refresh never fails just because of metadata.
        """
        now = time.monotonic()
        if self._special_contest_ids is not None and now - self._special_contest_ids_at < 6 * 3600:
            return self._special_contest_ids

        special = set(DEFAULT_SPECIAL_CONTEST_IDS)
        try:
            contests = await self._request("contest.list", {"gym": "false"}, attempts=2)
            for contest in contests or []:
                if not isinstance(contest, dict):
                    continue
                contest_id = _as_optional_int(contest.get("id"))
                name = str(contest.get("name") or "").lower()
                if contest_id is None:
                    continue
                if any(marker in name for marker in SPECIAL_CONTEST_NAME_MARKERS):
                    special.add(contest_id)
        except CFError as exc:
            log.warning("Could not fetch contest.list for special contest filtering: %s", exc)

        self._special_contest_ids = special
        self._special_contest_ids_at = now
        return special


def _as_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CFError",
    "CodeforcesClient",
    "MIN_REQUEST_INTERVAL",
    "normalize_handle",
]
