"""Async SQLite persistence for the ``;ihave`` bot (aiosqlite only).

All statements go through ``aiosqlite`` so the Discord event loop never blocks
on disk I/O. The default database lives in ``data/bot.db``; the directory is
created automatically before the connection is opened, which keeps fresh
deployments from failing with ``FileNotFoundError``.

Tables
------
``problems``       cached ``problemset.problems`` rows (24 hour TTL)
``users``          Discord id -> Codeforces handle mapping
``user_solved``    accepted problems per handle (practice submissions included)
``user_curve``     raw / monotonically smoothed minutes per rating
``meta``           small key/value cache metadata
``curve_overrides`` manual ``;adjust`` calibrations that survive ``;sync``
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import aiosqlite

import estimator

DEFAULT_DB_PATH = os.path.join("data", "bot.db")
PROBLEMSET_TTL_SECONDS = 24 * 60 * 60
RECENT_CONTEST_WINDOW = 400
TOP_SOLVED_FRACTION = 0.70

_DB: aiosqlite.Connection | None = None
_DB_PATH = os.getenv("BOT_DB_PATH") or DEFAULT_DB_PATH
_INIT_LOCK: asyncio.Lock | None = None
_PROBLEMSET_LOCK: asyncio.Lock | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    contest_id   INTEGER NOT NULL,
    index_name   TEXT    NOT NULL,
    name         TEXT    NOT NULL,
    rating       INTEGER NOT NULL,
    solved_count INTEGER NOT NULL DEFAULT 0,
    tags         TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (contest_id, index_name)
);

CREATE INDEX IF NOT EXISTS idx_problems_rating
    ON problems (rating, contest_id DESC);

CREATE TABLE IF NOT EXISTS users (
    discord_id     INTEGER PRIMARY KEY,
    cf_handle      TEXT    NOT NULL,
    current_rating INTEGER,
    last_synced    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_users_handle ON users (cf_handle);

CREATE TABLE IF NOT EXISTS user_solved (
    cf_handle  TEXT    NOT NULL,
    contest_id INTEGER NOT NULL,
    index_name TEXT    NOT NULL,
    PRIMARY KEY (cf_handle, contest_id, index_name)
);

CREATE INDEX IF NOT EXISTS idx_user_solved_handle ON user_solved (cf_handle);

CREATE TABLE IF NOT EXISTS user_curve (
    cf_handle        TEXT    NOT NULL,
    rating           INTEGER NOT NULL,
    raw_minutes      REAL,
    smoothed_minutes REAL    NOT NULL,
    sample_size      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cf_handle, rating)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS curve_overrides (
    cf_handle  TEXT    NOT NULL,
    rating     INTEGER NOT NULL,
    minutes    REAL    NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (cf_handle, rating)
);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def _get_init_lock() -> asyncio.Lock:
    global _INIT_LOCK
    if _INIT_LOCK is None:
        _INIT_LOCK = asyncio.Lock()
    return _INIT_LOCK


def _get_problemset_lock() -> asyncio.Lock:
    global _PROBLEMSET_LOCK
    if _PROBLEMSET_LOCK is None:
        _PROBLEMSET_LOCK = asyncio.Lock()
    return _PROBLEMSET_LOCK


async def init_db(db_path: str | None = None) -> aiosqlite.Connection:
    """Open the database (creating ``data/`` and the schema when needed)."""
    global _DB, _DB_PATH

    async with _get_init_lock():
        if db_path:
            new_path = str(db_path)
            if _DB is not None and new_path != _DB_PATH:
                await _DB.close()
                _DB = None
            _DB_PATH = new_path
        if _DB is not None:
            return _DB

        # CRITICAL: never let a fresh deployment fail because data/ is missing.
        os.makedirs("data", exist_ok=True)
        directory = os.path.dirname(os.path.abspath(_DB_PATH))
        if directory:
            os.makedirs(directory, exist_ok=True)

        connection = await aiosqlite.connect(_DB_PATH)
        connection.row_factory = aiosqlite.Row
        try:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.executescript(_SCHEMA)
            await connection.commit()
        except Exception:
            await connection.close()
            raise
        _DB = connection
        return connection


async def get_db() -> aiosqlite.Connection:
    """Return the live connection, initializing it on first use."""
    if _DB is None:
        return await init_db()
    return _DB


async def close_db() -> None:
    global _DB
    if _DB is not None:
        connection, _DB = _DB, None
        await connection.close()


async def _fetchone(sql: str, params: Sequence[Any] = ()) -> aiosqlite.Row | None:
    database = await get_db()
    async with database.execute(sql, params) as cursor:
        return await cursor.fetchone()


async def _fetchall(sql: str, params: Sequence[Any] = ()) -> list[aiosqlite.Row]:
    database = await get_db()
    async with database.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


# ---------------------------------------------------------------------------
# Key/value meta
# ---------------------------------------------------------------------------


async def get_meta(key: str, default: str | None = None) -> str | None:
    row = await _fetchone("SELECT value FROM meta WHERE key = ?", (key,))
    return row["value"] if row is not None else default


async def set_meta(key: str, value: str) -> None:
    database = await get_db()
    await database.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    await database.commit()


# ---------------------------------------------------------------------------
# Problemset cache
# ---------------------------------------------------------------------------


async def problemset_age_seconds() -> float | None:
    value = await get_meta("problemset_refreshed_at")
    if value is None:
        return None
    try:
        return max(0.0, time.time() - float(value))
    except (TypeError, ValueError):
        return None


async def problem_count() -> int:
    row = await _fetchone("SELECT COUNT(*) AS count FROM problems")
    return int(row["count"]) if row is not None else 0


async def replace_problemset(problems: Sequence[dict[str, Any]]) -> int:
    """Replace the cached problemset in one transaction and stamp the time."""
    rows = []
    for problem in problems:
        rows.append(
            (
                int(problem["contest_id"]),
                str(problem["index_name"]),
                str(problem["name"]),
                int(problem["rating"]),
                int(problem.get("solved_count") or 0),
                ",".join(str(tag) for tag in (problem.get("tags") or [])),
            )
        )
    if not rows:
        raise ValueError("Refusing to replace the problemset with zero problems.")

    database = await get_db()
    try:
        await database.execute("DELETE FROM problems")
        await database.executemany(
            "INSERT OR REPLACE INTO problems "
            "(contest_id, index_name, name, rating, solved_count, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        await database.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("problemset_refreshed_at", str(int(time.time()))),
        )
        await database.commit()
    except Exception:
        await database.rollback()
        raise
    return len(rows)


async def ensure_problemset(client: Any, *, force: bool = False) -> bool:
    """Refresh ``problems`` if the cache is older than 24 hours.

    Returns ``True`` when a refresh happened. A lock prevents two commands from
    refreshing the same cache concurrently.
    """
    async with _get_problemset_lock():
        if not force:
            age = await problemset_age_seconds()
            count = await problem_count()
            if age is not None and age < PROBLEMSET_TTL_SECONDS and count > 0:
                return False
        problems = await client.get_problemset()
        await replace_problemset(problems)
        return True


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def upsert_user(
    discord_id: int,
    cf_handle: str,
    current_rating: int | None,
    last_synced: int | None = None,
) -> None:
    database = await get_db()
    await database.execute(
        "INSERT INTO users (discord_id, cf_handle, current_rating, last_synced) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(discord_id) DO UPDATE SET "
        "cf_handle = excluded.cf_handle, "
        "current_rating = excluded.current_rating, "
        "last_synced = excluded.last_synced",
        (int(discord_id), cf_handle, current_rating, last_synced),
    )
    await database.commit()


async def get_user(discord_id: int) -> dict[str, Any] | None:
    row = await _fetchone("SELECT * FROM users WHERE discord_id = ?", (int(discord_id),))
    return dict(row) if row is not None else None


async def get_user_by_handle(cf_handle: str) -> dict[str, Any] | None:
    row = await _fetchone(
        "SELECT * FROM users WHERE cf_handle = ? LIMIT 1", (cf_handle,)
    )
    return dict(row) if row is not None else None


async def resolve_handle(name: str) -> str | None:
    """Find the canonical spelling of a handle seen by this bot (case-insensitive)."""
    cleaned = str(name or "").strip().lstrip("@")
    if not cleaned:
        return None
    for table in ("users", "user_curve", "user_solved"):
        row = await _fetchone(
            f"SELECT cf_handle FROM {table} WHERE cf_handle = ? COLLATE NOCASE LIMIT 1",
            (cleaned,),
        )
        if row is not None:
            return str(row["cf_handle"])
    return None


# ---------------------------------------------------------------------------
# Solved problems
# ---------------------------------------------------------------------------


async def add_solved_many(cf_handle: str, pairs: Iterable[tuple[int, str]]) -> int:
    """Insert solved problems idempotently; returns the number of new rows."""
    rows = [(cf_handle, int(contest_id), str(index_name)) for contest_id, index_name in pairs]
    if not rows:
        return 0
    database = await get_db()
    async with database.execute("SELECT COUNT(*) AS count FROM user_solved WHERE cf_handle = ?", (cf_handle,)) as cursor:
        before_row = await cursor.fetchone()
    before = int(before_row["count"]) if before_row is not None else 0
    await database.executemany(
        "INSERT OR IGNORE INTO user_solved (cf_handle, contest_id, index_name) "
        "VALUES (?, ?, ?)",
        rows,
    )
    await database.commit()
    async with database.execute("SELECT COUNT(*) AS count FROM user_solved WHERE cf_handle = ?", (cf_handle,)) as cursor:
        after_row = await cursor.fetchone()
    after = int(after_row["count"]) if after_row is not None else 0
    return max(0, after - before)


async def mark_solved(cf_handle: str, contest_id: int, index_name: str) -> None:
    database = await get_db()
    await database.execute(
        "INSERT OR IGNORE INTO user_solved (cf_handle, contest_id, index_name) "
        "VALUES (?, ?, ?)",
        (cf_handle, int(contest_id), str(index_name)),
    )
    await database.commit()


async def get_solved_count(cf_handle: str) -> int:
    row = await _fetchone(
        "SELECT COUNT(*) AS count FROM user_solved WHERE cf_handle = ?", (cf_handle,)
    )
    return int(row["count"]) if row is not None else 0


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


async def save_curve(cf_handle: str, points: Sequence[estimator.CurvePoint]) -> None:
    """Upsert a full curve produced by :func:`estimator.build_curve`."""
    rows = [
        (
            cf_handle,
            int(point.rating),
            None if point.raw_minutes is None else float(point.raw_minutes),
            float(point.smoothed_minutes),
            int(point.sample_size),
        )
        for point in points
    ]
    database = await get_db()
    await database.executemany(
        "INSERT INTO user_curve "
        "(cf_handle, rating, raw_minutes, smoothed_minutes, sample_size) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(cf_handle, rating) DO UPDATE SET "
        "raw_minutes = excluded.raw_minutes, "
        "smoothed_minutes = excluded.smoothed_minutes, "
        "sample_size = excluded.sample_size",
        rows,
    )
    await database.commit()


async def get_curve(cf_handle: str) -> list[dict[str, Any]]:
    rows = await _fetchall(
        "SELECT cf_handle, rating, raw_minutes, smoothed_minutes, sample_size "
        "FROM user_curve WHERE cf_handle = ? ORDER BY rating",
        (cf_handle,),
    )
    return [dict(row) for row in rows]


async def get_curve_map(cf_handle: str) -> dict[int, float]:
    rows = await get_curve(cf_handle)
    return {
        int(row["rating"]): float(row["smoothed_minutes"])
        for row in rows
        if row["smoothed_minutes"] is not None
    }


async def _current_rating(cf_handle: str) -> int:
    row = await _fetchone(
        "SELECT current_rating FROM users WHERE cf_handle = ?", (cf_handle,)
    )
    if row is None or row["current_rating"] is None:
        return estimator.DEFAULT_RATING
    return estimator.normalize_rating(row["current_rating"])


def _clamp_curve_minutes(minutes: Any) -> float:
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        return 40.0
    if not math.isfinite(value):
        return 40.0
    return max(1.0, min(600.0, value))


def _pin_curve_value(
    values: Mapping[int, float],
    rating: int,
    minutes: float,
) -> dict[int, float]:
    """Project a non-decreasing curve through one pinned (rating, minutes) point.

    The pinned rating is set exactly to ``minutes``; lower ratings are capped at
    that value and higher ratings are floored at it. For a non-decreasing input
    this is the minimal change that preserves monotonicity, so ``;adjust`` and
    feedback take effect exactly even when the current curve disagrees with the
    new value.
    """
    target = _clamp_curve_minutes(minutes)
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
    values: Mapping[int, float],
    overrides: Mapping[int, float],
) -> dict[int, float]:
    result = {int(rating): float(minutes) for rating, minutes in values.items()}
    for rating in sorted(overrides):
        result = _pin_curve_value(result, int(rating), float(overrides[rating]))
    return result


async def _write_smoothed_values(
    cf_handle: str,
    values: Mapping[int, float],
) -> None:
    """Persist a full grid of smoothed minutes (raw observations are untouched)."""
    rows = await get_curve(cf_handle)
    existing = {int(row["rating"]): row for row in rows}
    database = await get_db()
    updates = []
    inserts = []
    for rating in estimator.RATING_GRID:
        value = float(values.get(rating, 40.0))
        if rating in existing:
            updates.append((value, cf_handle, rating))
        else:
            inserts.append((cf_handle, rating, value))

    if updates:
        await database.executemany(
            "UPDATE user_curve SET smoothed_minutes = ? "
            "WHERE cf_handle = ? AND rating = ?",
            updates,
        )
    if inserts:
        await database.executemany(
            "INSERT INTO user_curve "
            "(cf_handle, rating, raw_minutes, smoothed_minutes, sample_size) "
            "VALUES (?, ?, NULL, ?, 0)",
            inserts,
        )
    await database.commit()


async def recalculate_smoothed(cf_handle: str) -> None:
    """Recompute the displayed curve from raw observations plus overrides.

    Data-driven smoothing uses :func:`estimator.smooth_monotonic`; every
    ``;adjust`` override is then pinned into the result with
    :func:`_pin_curve_value`.
    """
    rows = await _fetchall(
        "SELECT rating, raw_minutes, sample_size FROM user_curve WHERE cf_handle = ?",
        (cf_handle,),
    )
    raw: dict[int, float] = {
        int(row["rating"]): float(row["raw_minutes"])
        for row in rows
        if row["raw_minutes"] is not None
    }
    if raw:
        base = estimator.smooth_monotonic(raw)
    else:
        base = estimator.fallback_curve(await _current_rating(cf_handle))

    overrides = await _get_overrides(cf_handle)
    values = _apply_overrides(base, overrides)
    await _write_smoothed_values(cf_handle, values)


async def rebuild_curve(
    cf_handle: str,
    points: Sequence[estimator.CurvePoint],
) -> list[dict[str, Any]]:
    """Save a freshly computed curve and re-apply durable ``;adjust`` overrides."""
    await save_curve(cf_handle, points)
    overrides = await _get_overrides(cf_handle)
    if overrides:
        rows = await get_curve(cf_handle)
        values = {
            int(row["rating"]): float(row["smoothed_minutes"])
            for row in rows
            if row["smoothed_minutes"] is not None
        }
        await _write_smoothed_values(cf_handle, _apply_overrides(values, overrides))
    return await get_curve(cf_handle)


# ---------------------------------------------------------------------------
# Manual overrides (`;adjust`) and feedback tweaks
# ---------------------------------------------------------------------------


async def _set_override(cf_handle: str, rating: int, minutes: float) -> None:
    database = await get_db()
    await database.execute(
        "INSERT INTO curve_overrides (cf_handle, rating, minutes, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(cf_handle, rating) DO UPDATE SET "
        "minutes = excluded.minutes, updated_at = excluded.updated_at",
        (cf_handle, int(rating), float(minutes), int(time.time())),
    )
    await database.commit()


async def _get_overrides(cf_handle: str) -> dict[int, float]:
    rows = await _fetchall(
        "SELECT rating, minutes FROM curve_overrides WHERE cf_handle = ? ORDER BY rating",
        (cf_handle,),
    )
    return {int(row["rating"]): float(row["minutes"]) for row in rows}


async def adjust_curve(
    cf_handle: str,
    rating: int,
    minutes: float,
) -> tuple[float, float]:
    """Apply a manual calibration (``;adjust <rating> <minutes>``).

    The requested point is pinned exactly (within 1–600 minutes) and the rest of
    the curve is adjusted minimally to stay non-decreasing. The override is
    stored separately so ``;sync`` re-applies it to fresh contest data.

    Returns ``(previous_smoothed, new_smoothed)`` for the adjusted rating.
    """
    rating = estimator.normalize_rating(rating)
    minutes = _clamp_curve_minutes(minutes)

    rows = await get_curve(cf_handle)
    if not rows:
        await recalculate_smoothed(cf_handle)
        rows = await get_curve(cf_handle)
    previous = next(
        (float(row["smoothed_minutes"]) for row in rows if int(row["rating"]) == rating),
        minutes,
    )

    await _set_override(cf_handle, rating, minutes)
    await recalculate_smoothed(cf_handle)

    rows = await get_curve(cf_handle)
    updated = next(
        (float(row["smoothed_minutes"]) for row in rows if int(row["rating"]) == rating),
        minutes,
    )
    return previous, updated


async def scale_curve_rating(
    cf_handle: str,
    rating: int,
    factor: float,
) -> tuple[float, float]:
    """Apply button feedback to the expected time at one rating.

    The current curve is pinned through ``old * factor``, so the displayed
    estimate at that rating changes by the requested amount and the curve stays
    non-decreasing. Feedback is a tuning signal: it is intentionally *not*
    stored as a durable ``;adjust`` override, so a later ``;sync`` rebuilds from
    fresh contest history.

    Returns ``(previous_smoothed, new_smoothed)``.
    """
    rating = estimator.normalize_rating(rating)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        factor = 1.0
    if not math.isfinite(factor):
        factor = 1.0
    factor = max(0.25, min(4.0, factor))

    rows = await get_curve(cf_handle)
    if not rows:
        await recalculate_smoothed(cf_handle)
        rows = await get_curve(cf_handle)

    values = {
        int(row["rating"]): float(row["smoothed_minutes"])
        for row in rows
        if row["smoothed_minutes"] is not None
    }
    if rating not in values:
        values[rating] = estimator.fallback_curve(await _current_rating(cf_handle)).get(
            rating, 40.0
        )
    previous = float(values[rating])
    target = _clamp_curve_minutes(previous * factor)
    updated = _pin_curve_value(values, rating, target)
    await _write_smoothed_values(cf_handle, updated)
    return previous, float(updated[rating])


# ---------------------------------------------------------------------------
# Problem selection
# ---------------------------------------------------------------------------


async def _candidates_for_rating(cf_handle: str, rating: int) -> list[aiosqlite.Row]:
    return await _fetchall(
        "SELECT p.contest_id, p.index_name, p.name, p.rating, p.solved_count, p.tags "
        "FROM problems AS p "
        "WHERE p.rating = ? "
        "  AND p.contest_id IN ("
        "      SELECT contest_id FROM ("
        "          SELECT DISTINCT contest_id FROM problems "
        "          ORDER BY contest_id DESC LIMIT ?"
        "      )"
        "  ) "
        "  AND NOT EXISTS ("
        "      SELECT 1 FROM user_solved AS s "
        "      WHERE s.cf_handle = ? "
        "        AND s.contest_id = p.contest_id "
        "        AND s.index_name = p.index_name"
        "  )",
        (int(rating), RECENT_CONTEST_WINDOW, cf_handle),
    )


async def find_unsolved_problem(
    cf_handle: str,
    target_rating: int,
) -> dict[str, Any] | None:
    """Pick a random unsolved, recent problem at (or near) ``target_rating``.

    Within the requested tier, the most-solved 70% of problems are preferred so
    recommendations are well-tested classics rather than obscure outliers.
    Nearby ratings are tried in expanding order if the exact tier has no
    eligible problem.
    """
    target_rating = estimator.normalize_rating(target_rating)
    for offset in (0, 100, -100, 200, -200, 300, -300, 400, -400, 500, -500):
        rating = target_rating + offset
        if rating < estimator.RATING_MIN or rating > estimator.RATING_MAX:
            continue
        rows = await _candidates_for_rating(cf_handle, rating)
        if not rows:
            continue

        ordered = sorted(
            rows,
            key=lambda row: (
                -(int(row["solved_count"] or 0)),
                -int(row["contest_id"]),
                str(row["index_name"]),
            ),
        )
        pool_size = max(1, int(math.ceil(len(ordered) * TOP_SOLVED_FRACTION)))
        chosen = random.choice(ordered[:pool_size])
        return {
            "contest_id": int(chosen["contest_id"]),
            "index_name": str(chosen["index_name"]),
            "name": str(chosen["name"]),
            "rating": int(chosen["rating"]),
            "solved_count": int(chosen["solved_count"] or 0),
            "tags": [tag for tag in str(chosen["tags"] or "").split(",") if tag],
            "requested_rating": target_rating,
        }
    return None


__all__ = [
    "PROBLEMSET_TTL_SECONDS",
    "add_solved_many",
    "adjust_curve",
    "close_db",
    "ensure_problemset",
    "find_unsolved_problem",
    "get_curve",
    "get_curve_map",
    "get_db",
    "get_meta",
    "get_solved_count",
    "get_user",
    "get_user_by_handle",
    "init_db",
    "mark_solved",
    "problem_count",
    "problemset_age_seconds",
    "rebuild_curve",
    "recalculate_smoothed",
    "replace_problemset",
    "resolve_handle",
    "save_curve",
    "scale_curve_rating",
    "set_meta",
    "upsert_user",
]
