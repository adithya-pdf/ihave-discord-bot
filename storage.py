"""Serverless storage for the Vercel interactions bot.

Production uses an HTTP Redis (Upstash / Vercel KV) because Vercel Functions
have an ephemeral filesystem and no shared memory. Local development falls back
to an in-process memory store so the app can run without any cloud services.

The module keeps the same domain model as the old SQLite version:

* cached problemset, grouped by rating
* users (Discord id -> Codeforces handle / rating)
* solved problems per handle
* personal curves and durable ``/adjust`` overrides
* active practice sessions and button feedback idempotency

Redis is accessed through its REST API with ``aiohttp``; no connection pool is
needed, which is a good fit for serverless invocations.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

import aiohttp

import estimator

PROBLEMSET_TTL_SECONDS = 24 * 60 * 60
RECENT_CONTEST_WINDOW = 400
TOP_SOLVED_FRACTION = 0.70
FEEDBACK_TTL_SECONDS = 24 * 60 * 60
PROBLEMSET_LOCK_SECONDS = 180

_PREFIX = "ihave"
_ACTIVE_SESSIONS = f"{_PREFIX}:sessions:active"


class StorageError(RuntimeError):
    """Raised when the configured storage backend cannot complete an operation."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend:
    async def get(self, key: str) -> str | None:
        raise NotImplementedError

    async def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def set_members(self, key: str) -> list[str]:
        raise NotImplementedError

    async def set_add(self, key: str, members: Sequence[str]) -> int:
        raise NotImplementedError

    async def set_remove(self, key: str, members: Sequence[str]) -> int:
        raise NotImplementedError

    async def set_nx(self, key: str, value: str, *, ex: int = 0) -> bool:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class MemoryBackend(_Backend):
    """In-process fallback used by local development and tests.

    This is intentionally *not* a Vercel production backend: separate
    invocations do not share this dictionary.
    """

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        return self._strings.get(key)

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            self._strings[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._strings.pop(key, None)
            self._sets.pop(key, None)

    async def set_members(self, key: str) -> list[str]:
        return sorted(self._sets.get(key, set()))

    async def set_add(self, key: str, members: Sequence[str]) -> int:
        async with self._lock:
            current = self._sets.setdefault(key, set())
            before = len(current)
            current.update(str(member) for member in members)
            return len(current) - before

    async def set_remove(self, key: str, members: Sequence[str]) -> int:
        async with self._lock:
            current = self._sets.get(key, set())
            before = len(current)
            current.difference_update(str(member) for member in members)
            return before - len(current)

    async def set_nx(self, key: str, value: str, *, ex: int = 0) -> bool:
        async with self._lock:
            if key in self._strings:
                return False
            self._strings[key] = value
            return True


class RedisBackend(_Backend):
    """Upstash/Vercel KV Redis REST backend."""

    def __init__(self, url: str, token: str) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._session: aiohttp.ClientSession | None = None

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._token}"},
            )
        return self._session

    async def _command(self, *args: Any) -> Any:
        session = await self._session_or_create()
        body = [str(arg) for arg in args]
        try:
            async with session.post(
                self._url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    raise StorageError(
                        f"Redis returned HTTP {response.status}: {text[:200]}"
                    )
        except aiohttp.ClientError as exc:
            raise StorageError(f"Could not reach Redis: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise StorageError("Redis request timed out.") from exc

        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise StorageError("Redis returned invalid JSON.") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise StorageError(str(payload["error"]))
        return payload.get("result") if isinstance(payload, dict) else payload

    async def get(self, key: str) -> str | None:
        result = await self._command("GET", key)
        return None if result is None else str(result)

    async def set(self, key: str, value: str) -> None:
        await self._command("SET", key, value)

    async def delete(self, key: str) -> None:
        await self._command("DEL", key)

    async def set_members(self, key: str) -> list[str]:
        result = await self._command("SMEMBERS", key)
        if result is None:
            return []
        if not isinstance(result, list):
            raise StorageError("Redis SMEMBERS returned an unexpected value.")
        return [str(member) for member in result]

    async def set_add(self, key: str, members: Sequence[str]) -> int:
        if not members:
            return 0
        result = await self._command("SADD", key, *members)
        try:
            return int(result or 0)
        except (TypeError, ValueError):
            return 0

    async def set_remove(self, key: str, members: Sequence[str]) -> int:
        if not members:
            return 0
        result = await self._command("SREM", key, *members)
        try:
            return int(result or 0)
        except (TypeError, ValueError):
            return 0

    async def set_nx(self, key: str, value: str, *, ex: int = 0) -> bool:
        args: list[Any] = ["SET", key, value, "NX"]
        if ex > 0:
            args.extend(["EX", ex])
        result = await self._command(*args)
        return result == "OK"

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


_BACKEND: _Backend | None = None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def get_backend() -> _Backend:
    """Return the configured backend, preferring Upstash/Vercel KV on Vercel."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    url = _first_env("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL")
    token = _first_env("UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN")
    if url and token:
        _BACKEND = RedisBackend(url, token)
        return _BACKEND

    if os.getenv("VERCEL"):
        raise StorageError(
            "Redis is not configured. Set UPSTASH_REDIS_REST_URL and "
            "UPSTASH_REDIS_REST_TOKEN (or Vercel KV's KV_REST_API_URL and "
            "KV_REST_API_TOKEN) in the Vercel project."
        )

    _BACKEND = MemoryBackend()
    return _BACKEND


async def close() -> None:
    global _BACKEND
    if _BACKEND is not None:
        await _BACKEND.close()
        _BACKEND = None


# ---------------------------------------------------------------------------
# Keys and JSON helpers
# ---------------------------------------------------------------------------


def _casefold(handle: str) -> str:
    return str(handle or "").strip().casefold()


def _user_key(discord_id: int) -> str:
    return f"{_PREFIX}:user:{int(discord_id)}"


def _handle_key(handle: str) -> str:
    return f"{_PREFIX}:handle:{_casefold(handle)}"


def _curve_key(handle: str) -> str:
    return f"{_PREFIX}:curve:{_casefold(handle)}"


def _overrides_key(handle: str) -> str:
    return f"{_PREFIX}:overrides:{_casefold(handle)}"


def _solved_key(handle: str) -> str:
    return f"{_PREFIX}:solved:{_casefold(handle)}"


def _problems_key(rating: int) -> str:
    return f"{_PREFIX}:problems:{int(rating)}"


def _session_key(session_id: str) -> str:
    return f"{_PREFIX}:session:{session_id}"


def _user_session_key(discord_id: int) -> str:
    return f"{_PREFIX}:session:user:{int(discord_id)}"


_RECENT_CONTESTS_KEY = f"{_PREFIX}:problems:recent"
_PROBLEMSET_UPDATED_KEY = f"{_PREFIX}:problems:updated"
_PROBLEMSET_LOCK_KEY = f"{_PREFIX}:problems:lock"


async def _get_json(key: str) -> Any:
    raw = await get_backend().get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageError(f"Stored value for {key!r} is corrupted.") from exc


async def _set_json(key: str, value: Any) -> None:
    await get_backend().set(key, json.dumps(value, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Users and handle index
# ---------------------------------------------------------------------------


async def upsert_user(
    discord_id: int,
    cf_handle: str,
    current_rating: int | None,
    last_synced: int | None = None,
) -> None:
    user = await get_user(discord_id) or {}
    user.update(
        {
            "discord_id": int(discord_id),
            "cf_handle": cf_handle,
            "current_rating": int(current_rating or estimator.DEFAULT_RATING),
            "last_synced": int(last_synced if last_synced is not None else time.time()),
        }
    )
    await _set_json(_user_key(discord_id), user)
    await remember_handle(cf_handle)


async def get_user(discord_id: int) -> dict[str, Any] | None:
    data = await _get_json(_user_key(discord_id))
    return data if isinstance(data, dict) else None


async def remember_handle(handle: str) -> None:
    """Case-insensitive handle lookup (also serves as the canonical spelling)."""
    canonical = str(handle or "").strip()
    if canonical:
        await get_backend().set(_handle_key(canonical), canonical)


async def resolve_handle(name: str) -> str | None:
    cleaned = str(name or "").strip().lstrip("@")
    if not cleaned:
        return None
    raw = await get_backend().get(_handle_key(cleaned))
    return str(raw) if raw else None


# ---------------------------------------------------------------------------
# Solved problems
# ---------------------------------------------------------------------------


def _solved_member(contest_id: int, index_name: str) -> str:
    return f"{int(contest_id)}:{str(index_name)}"


async def add_solved_many(cf_handle: str, pairs: Iterable[tuple[int, str]]) -> int:
    members = [_solved_member(contest_id, index_name) for contest_id, index_name in pairs]
    if not members:
        return 0
    backend = get_backend()
    added = 0
    for start in range(0, len(members), 500):
        added += await backend.set_add(_solved_key(cf_handle), members[start : start + 500])
    return added


async def mark_solved(cf_handle: str, contest_id: int, index_name: str) -> None:
    await get_backend().set_add(
        _solved_key(cf_handle), [_solved_member(contest_id, index_name)]
    )


async def get_solved(cf_handle: str) -> set[str]:
    return set(await get_backend().set_members(_solved_key(cf_handle)))


async def get_solved_count(cf_handle: str) -> int:
    return len(await get_backend().set_members(_solved_key(cf_handle)))


# ---------------------------------------------------------------------------
# Curves and overrides
# ---------------------------------------------------------------------------


def _point_to_dict(point: estimator.CurvePoint | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(point, Mapping):
        return {
            "rating": int(point["rating"]),
            "raw_minutes": point.get("raw_minutes"),
            "smoothed_minutes": float(point["smoothed_minutes"]),
            "sample_size": int(point.get("sample_size") or 0),
        }
    return {
        "rating": int(point.rating),
        "raw_minutes": point.raw_minutes,
        "smoothed_minutes": float(point.smoothed_minutes),
        "sample_size": int(point.sample_size),
    }


def _fallback_points(current_rating: int) -> list[dict[str, Any]]:
    values = estimator.fallback_curve(current_rating)
    return [
        {
            "rating": rating,
            "raw_minutes": None,
            "smoothed_minutes": float(values[rating]),
            "sample_size": 0,
        }
        for rating in estimator.RATING_GRID
    ]


def _pin_values(values: Mapping[int, float], rating: int, minutes: float) -> dict[int, float]:
    """Project a non-decreasing curve through one exact point."""
    try:
        target = max(1.0, min(600.0, float(minutes)))
    except (TypeError, ValueError):
        target = 40.0
    pinned: dict[int, float] = {}
    for curve_rating in estimator.RATING_GRID:
        current = float(values.get(curve_rating, target))
        if curve_rating == rating:
            pinned[curve_rating] = target
        elif curve_rating < rating:
            pinned[curve_rating] = min(current, target)
        else:
            pinned[curve_rating] = max(current, target)
    return pinned


def _apply_overrides(
    values: Mapping[int, float], overrides: Mapping[int, float]
) -> dict[int, float]:
    result = {int(rating): float(minutes) for rating, minutes in values.items()}
    for rating in sorted(overrides):
        result = _pin_values(result, int(rating), float(overrides[rating]))
    return result


async def save_curve(
    cf_handle: str, points: Sequence[estimator.CurvePoint | Mapping[str, Any]]
) -> None:
    await _set_json(_curve_key(cf_handle), [_point_to_dict(point) for point in points])


async def get_curve(cf_handle: str) -> list[dict[str, Any]]:
    data = await _get_json(_curve_key(cf_handle))
    if not isinstance(data, list):
        return []
    points = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            points.append(
                {
                    "rating": int(row["rating"]),
                    "raw_minutes": (
                        None if row.get("raw_minutes") is None else float(row["raw_minutes"])
                    ),
                    "smoothed_minutes": float(row["smoothed_minutes"]),
                    "sample_size": int(row.get("sample_size") or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    points.sort(key=lambda row: row["rating"])
    return points


async def get_curve_map(cf_handle: str) -> dict[int, float]:
    return {
        int(row["rating"]): float(row["smoothed_minutes"])
        for row in await get_curve(cf_handle)
    }


async def save_overrides(cf_handle: str, overrides: Mapping[int, float]) -> None:
    await _set_json(
        _overrides_key(cf_handle),
        {str(int(rating)): float(minutes) for rating, minutes in overrides.items()},
    )


async def get_overrides(cf_handle: str) -> dict[int, float]:
    data = await _get_json(_overrides_key(cf_handle))
    if not isinstance(data, dict):
        return {}
    result: dict[int, float] = {}
    for rating, minutes in data.items():
        try:
            result[int(rating)] = float(minutes)
        except (TypeError, ValueError):
            continue
    return result


async def rebuild_curve(
    cf_handle: str,
    points: Sequence[estimator.CurvePoint | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Save a freshly computed curve and re-apply durable ``/adjust`` overrides."""
    rows = [_point_to_dict(point) for point in points]
    overrides = await get_overrides(cf_handle)
    if overrides:
        values = {int(row["rating"]): float(row["smoothed_minutes"]) for row in rows}
        values = _apply_overrides(values, overrides)
        for row in rows:
            row["smoothed_minutes"] = float(values[int(row["rating"])])
    await _set_json(_curve_key(cf_handle), rows)
    return rows


async def adjust_curve(
    cf_handle: str,
    rating: int,
    minutes: float,
    *,
    current_rating: int = estimator.DEFAULT_RATING,
) -> tuple[float, float]:
    """Pin the curve at ``(rating, minutes)``; return previous and new values."""
    rating = estimator.normalize_rating(rating)
    minutes = max(1.0, min(600.0, float(minutes)))
    points = await get_curve(cf_handle) or _fallback_points(current_rating)
    values = {int(row["rating"]): float(row["smoothed_minutes"]) for row in points}
    previous = float(values.get(rating, minutes))

    overrides = await get_overrides(cf_handle)
    overrides[rating] = minutes
    values = _apply_overrides(values, overrides)
    for row in points:
        row["smoothed_minutes"] = float(values[int(row["rating"])])
    await _set_json(_curve_key(cf_handle), points)
    await save_overrides(cf_handle, overrides)
    return previous, float(values[rating])


async def scale_curve_rating(
    cf_handle: str,
    rating: int,
    factor: float,
    *,
    current_rating: int = estimator.DEFAULT_RATING,
) -> tuple[float, float]:
    """Apply button feedback to one rating without making the curve non-monotonic."""
    rating = estimator.normalize_rating(rating)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        factor = 1.0
    factor = max(0.25, min(4.0, factor))

    points = await get_curve(cf_handle) or _fallback_points(current_rating)
    values = {int(row["rating"]): float(row["smoothed_minutes"]) for row in points}
    previous = float(values.get(rating, 40.0))
    target = max(1.0, min(600.0, previous * factor))
    values = _pin_values(values, rating, target)
    for row in points:
        row["smoothed_minutes"] = float(values[int(row["rating"])])
    await _set_json(_curve_key(cf_handle), points)
    return previous, float(values[rating])


# ---------------------------------------------------------------------------
# Problemset cache
# ---------------------------------------------------------------------------


async def problemset_age_seconds() -> float | None:
    raw = await get_backend().get(_PROBLEMSET_UPDATED_KEY)
    if raw is None:
        return None
    try:
        return max(0.0, time.time() - float(raw))
    except (TypeError, ValueError):
        return None


async def ensure_problemset(client: Any, *, force: bool = False) -> bool:
    """Refresh the cached problemset if it is older than 24 hours.

    The problemset is split into one JSON value per rating so no single Redis
    value becomes huge. A short NX lock prevents concurrent refreshes.
    """
    age = await problemset_age_seconds()
    recent = await _get_json(_RECENT_CONTESTS_KEY)
    if (
        not force
        and age is not None
        and age < PROBLEMSET_TTL_SECONDS
        and isinstance(recent, list)
    ):
        return False

    backend = get_backend()
    if not await backend.set_nx(_PROBLEMSET_LOCK_KEY, "1", ex=PROBLEMSET_LOCK_SECONDS):
        return False

    try:
        problems = await client.get_problemset()
        tiers: dict[int, list[dict[str, Any]]] = {}
        contest_ids: set[int] = set()
        for problem in problems:
            rating = int(problem["rating"])
            tiers.setdefault(rating, []).append(problem)
            contest_ids.add(int(problem["contest_id"]))

        for rating, rows in tiers.items():
            await _set_json(_problems_key(rating), rows)
        recent_ids = sorted(contest_ids, reverse=True)[:RECENT_CONTEST_WINDOW]
        await _set_json(_RECENT_CONTESTS_KEY, recent_ids)
        await backend.set(_PROBLEMSET_UPDATED_KEY, str(int(time.time())))
        return True
    finally:
        await backend.delete(_PROBLEMSET_LOCK_KEY)


async def _problem_tier(rating: int) -> list[dict[str, Any]]:
    data = await _get_json(_problems_key(rating))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


async def find_unsolved_problem(
    cf_handle: str,
    target_rating: int,
) -> dict[str, Any] | None:
    """Pick a random recent unsolved problem at or near the target rating."""
    target_rating = estimator.normalize_rating(target_rating)
    recent_raw = await _get_json(_RECENT_CONTESTS_KEY)
    recent = set(int(contest_id) for contest_id in (recent_raw or []))
    if not recent:
        return None
    solved = await get_solved(cf_handle)

    for offset in (0, 100, -100, 200, -200, 300, -300, 400, -400, 500, -500):
        rating = target_rating + offset
        if rating < estimator.RATING_MIN or rating > estimator.RATING_MAX:
            continue
        candidates = []
        for row in await _problem_tier(rating):
            try:
                contest_id = int(row["contest_id"])
                index_name = str(row["index_name"])
            except (KeyError, TypeError, ValueError):
                continue
            if contest_id not in recent:
                continue
            if _solved_member(contest_id, index_name) in solved:
                continue
            candidates.append(row)
        if not candidates:
            continue

        candidates.sort(
            key=lambda row: (
                -(int(row.get("solved_count") or 0)),
                -int(row.get("contest_id") or 0),
                str(row.get("index_name") or ""),
            )
        )
        pool_size = max(1, int(math.ceil(len(candidates) * TOP_SOLVED_FRACTION)))
        chosen = random.choice(candidates[:pool_size])
        return {
            "contest_id": int(chosen["contest_id"]),
            "index_name": str(chosen["index_name"]),
            "name": str(chosen["name"]),
            "rating": int(chosen["rating"]),
            "solved_count": int(chosen.get("solved_count") or 0),
            "tags": [str(tag) for tag in (chosen.get("tags") or [])],
            "requested_rating": target_rating,
        }
    return None


# ---------------------------------------------------------------------------
# Sessions and feedback idempotency
# ---------------------------------------------------------------------------


async def create_session(discord_id: int, **fields: Any) -> dict[str, Any]:
    old = await get_session_by_user(discord_id)
    if old is not None:
        await delete_session(str(old["id"]))

    session_id = uuid.uuid4().hex[:12]
    session = {
        "id": session_id,
        "user_id": int(discord_id),
        **fields,
    }
    await _set_json(_session_key(session_id), session)
    await get_backend().set(_user_session_key(discord_id), session_id)
    await get_backend().set_add(_ACTIVE_SESSIONS, [session_id])
    return session


async def get_session(session_id: str) -> dict[str, Any] | None:
    data = await _get_json(_session_key(session_id))
    return data if isinstance(data, dict) else None


async def get_session_by_user(discord_id: int) -> dict[str, Any] | None:
    session_id = await get_backend().get(_user_session_key(discord_id))
    if not session_id:
        return None
    session = await get_session(str(session_id))
    if session is None:
        await get_backend().delete(_user_session_key(discord_id))
        await get_backend().set_remove(_ACTIVE_SESSIONS, [str(session_id)])
        return None
    return session


async def update_session(session: Mapping[str, Any]) -> None:
    if "id" not in session:
        raise StorageError("Session is missing its id.")
    await _set_json(_session_key(str(session["id"])), dict(session))


async def delete_session(session_id: str) -> None:
    session = await get_session(session_id)
    backend = get_backend()
    await backend.delete(_session_key(session_id))
    await backend.set_remove(_ACTIVE_SESSIONS, [session_id])
    if session is not None:
        user_key = _user_session_key(int(session.get("user_id") or 0))
        if await backend.get(user_key) == session_id:
            await backend.delete(user_key)


async def remove_from_active(session_id: str) -> None:
    await get_backend().set_remove(_ACTIVE_SESSIONS, [session_id])


async def list_active_sessions() -> list[dict[str, Any]]:
    session_ids = await get_backend().set_members(_ACTIVE_SESSIONS)
    sessions: list[dict[str, Any]] = []
    for session_id in session_ids:
        session = await get_session(session_id)
        if session is None:
            await get_backend().set_remove(_ACTIVE_SESSIONS, [session_id])
            continue
        sessions.append(session)
    return sessions


async def claim_feedback(session_id: str) -> bool:
    """Atomically allow one feedback adjustment per session."""
    return await get_backend().set_nx(
        f"{_PREFIX}:feedback:{session_id}", "1", ex=FEEDBACK_TTL_SECONDS
    )


async def release_feedback(session_id: str) -> None:
    await get_backend().delete(f"{_PREFIX}:feedback:{session_id}")


__all__ = [
    "PROBLEMSET_TTL_SECONDS",
    "StorageError",
    "add_solved_many",
    "adjust_curve",
    "claim_feedback",
    "close",
    "create_session",
    "delete_session",
    "ensure_problemset",
    "find_unsolved_problem",
    "get_curve",
    "get_curve_map",
    "get_overrides",
    "get_session",
    "get_session_by_user",
    "get_solved",
    "get_solved_count",
    "get_user",
    "list_active_sessions",
    "mark_solved",
    "rebuild_curve",
    "release_feedback",
    "remember_handle",
    "remove_from_active",
    "resolve_handle",
    "save_curve",
    "save_overrides",
    "scale_curve_rating",
    "update_session",
    "upsert_user",
]
