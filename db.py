"""SQLite persistence for the IhHave bot.

One connection, guarded by a :class:`threading.Lock`, is shared by the Discord
event loop and worker threads (all Codeforces/model work runs inside
``asyncio.to_thread``).  ``check_same_thread=False`` plus the lock makes that
safe without shipping async SQLite plumbing.

The database file lives in ``BOT_DATA_DIR`` (default ``.``); ``BOT_DB_PATH`` is
still honoured for backwards compatibility with older deployments.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import predictor

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DEFAULT_DATA_DIR = "."
DEFAULT_DB_NAME = "bot.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    discord_id       INTEGER PRIMARY KEY,
    cf_handle        TEXT,
    base_prior       REAL,
    s30              REAL,
    gamma            REAL,
    kappa            REAL,
    k                REAL,
    delta_json       TEXT,
    contest_offset   REAL DEFAULT 0,
    stderr_json      TEXT,
    shape_fit_count  INTEGER DEFAULT 0,
    last_status_ts   INTEGER DEFAULT 0,
    updated_at       INTEGER
);

CREATE TABLE IF NOT EXISTS cf_solved (
    discord_id     INTEGER NOT NULL,
    contest_id     INTEGER NOT NULL,
    problem_index  TEXT    NOT NULL,
    problem_name   TEXT,
    PRIMARY KEY (discord_id, contest_id, problem_index)
);

CREATE TABLE IF NOT EXISTS attempts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_id        INTEGER NOT NULL,
    contest_id        INTEGER NOT NULL,
    problem_index     TEXT    NOT NULL,
    problem_name      TEXT,
    rating            INTEGER,
    tags_json         TEXT,
    budget_mins       REAL,
    actual_mins       REAL,
    was_count         INTEGER,
    solved            INTEGER,
    is_bootstrap      INTEGER,
    weight_mult       REAL DEFAULT 1.0,
    contest_start_ts  INTEGER,
    occurred_at       INTEGER NOT NULL,
    created_at        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_attempts_user_time
    ON attempts (discord_id, occurred_at, id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bootstrap
    ON attempts (discord_id, contest_id, problem_index)
    WHERE is_bootstrap = 1;

CREATE TABLE IF NOT EXISTS active_sessions (
    discord_id      INTEGER PRIMARY KEY,
    channel_id      INTEGER,
    message_id      INTEGER,
    contest_id      INTEGER,
    problem_index   TEXT,
    problem_name    TEXT,
    rating          INTEGER,
    tags_json       TEXT,
    budget_mins     REAL,
    confidence      REAL,
    blind_mode      INTEGER,
    started_at      INTEGER,
    timer_notified  INTEGER DEFAULT 0,
    last_polled_at  INTEGER DEFAULT 0
);
"""

#: Tables created by the pre-survival (v0) bot.  A legacy database has a
#: ``users`` table without the ``s30`` column; dropping these tables is the
#: only honest migration because the old schema stores a different model.
_LEGACY_TABLES = (
    "problems",
    "user_solved",
    "user_curve",
    "meta",
    "curve_overrides",
    "active_sessions",
    "cf_solved",
    "attempts",
    "users",
)


def default_db_path() -> str:
    """Resolve the SQLite path from the documented environment variables."""
    explicit = (os.getenv("BOT_DB_PATH") or "").strip()
    if explicit:
        return explicit
    data_dir = (os.getenv("BOT_DATA_DIR") or DEFAULT_DATA_DIR).strip() or DEFAULT_DATA_DIR
    return os.path.join(data_dir, DEFAULT_DB_NAME)


def _json_tags(tags: Iterable[str] | str | None) -> str:
    if tags is None:
        return "[]"
    if isinstance(tags, str):
        return json.dumps([tags])
    return json.dumps([str(tag) for tag in tags])


def _parse_tags(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(tag) for tag in value]
    try:
        data = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [str(tag) for tag in data]
    return []


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


def _field(source: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a mapping or object."""
    for name in names:
        if isinstance(source, Mapping):
            if name in source and source[name] is not None:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return default


class Database:
    """Thread-safe SQLite repository for users, attempts and live sessions."""

    def __init__(self, path: str | None = None) -> None:
        self.path = str(path) if path else default_db_path()
        if self.path != ":memory:":
            directory = os.path.dirname(os.path.abspath(self.path))
            if directory:
                os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.path, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # explicit transactions only
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._migrate()

    # -- lifecycle -----------------------------------------------------------

    def _migrate(self) -> None:
        assert self._conn is not None
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version < SCHEMA_VERSION:
            tables = {
                str(row[0])
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            legacy = False
            if "users" in tables:
                columns = {
                    str(row[1])
                    for row in self._conn.execute("PRAGMA table_info(users)").fetchall()
                }
                legacy = "s30" not in columns
            if legacy:
                log.warning("Migrating a pre-survival database at %s", self.path)
                for table in _LEGACY_TABLES:
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        else:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the connection; safe to call more than once."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("database is closed")
        return self._conn

    # -- users / model state -------------------------------------------------

    def upsert_user(
        self,
        discord_id: int,
        cf_handle: str,
        base_prior: float | None = None,
        updated_at: int | None = None,
    ) -> None:
        user_id = int(discord_id)
        now = int(time.time()) if updated_at is None else int(updated_at)
        initial_prior = 1800.0 if base_prior is None else float(base_prior)
        with self._lock:
            connection = self._connection()
            connection.execute(
                "INSERT INTO users (discord_id, cf_handle, base_prior, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(discord_id) DO UPDATE SET "
                "cf_handle = excluded.cf_handle, "
                "base_prior = COALESCE(?, users.base_prior), "
                "updated_at = excluded.updated_at",
                (user_id, str(cf_handle), initial_prior, now, base_prior),
            )

    def get_user(self, discord_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM users WHERE discord_id = ?", (int(discord_id),)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_by_handle(self, cf_handle: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM users WHERE cf_handle = ? COLLATE NOCASE LIMIT 1",
                (str(cf_handle).strip(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def save_model_state(
        self,
        discord_id: int,
        state: Mapping[str, Any],
        shape_fit_count: int | None = None,
        updated_at: int | None = None,
    ) -> None:
        user_id = int(discord_id)
        now = int(time.time()) if updated_at is None else int(updated_at)
        delta = state.get("delta") or [0.0] * predictor.N_ARCHETYPES
        stderr = state.get("stderr") or {}
        values = (
            _as_float(state.get("s30")),
            _as_float(state.get("gamma")),
            _as_float(state.get("kappa")),
            _as_float(state.get("k")),
            json.dumps([float(value) for value in delta]),
            _as_float(state.get("contest_offset"), 0.0),
            json.dumps(stderr),
            now,
        )
        with self._lock:
            connection = self._connection()
            cursor = connection.execute(
                "UPDATE users SET s30 = ?, gamma = ?, kappa = ?, k = ?, delta_json = ?, "
                "contest_offset = ?, stderr_json = ?, updated_at = ? "
                "WHERE discord_id = ?",
                values + (user_id,),
            )
            if cursor.rowcount == 0:
                connection.execute(
                    "INSERT INTO users (discord_id, s30, gamma, kappa, k, delta_json, "
                    "contest_offset, stderr_json, shape_fit_count, last_status_ts, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        user_id,
                        _as_float(state.get("s30"), 1800.0),
                        _as_float(state.get("gamma"), 200.0),
                        _as_float(state.get("kappa"), 180.0),
                        _as_float(state.get("k"), 2.5),
                        json.dumps([float(value) for value in delta]),
                        _as_float(state.get("contest_offset"), 0.0),
                        json.dumps(stderr),
                        int(shape_fit_count or 0),
                        now,
                    ),
                )
            elif shape_fit_count is not None:
                connection.execute(
                    "UPDATE users SET shape_fit_count = ? WHERE discord_id = ?",
                    (int(shape_fit_count), user_id),
                )

    def get_model_state(self, discord_id: int) -> dict[str, Any] | None:
        user = self.get_user(discord_id)
        if user is None or user.get("s30") is None:
            return None
        try:
            delta = json.loads(user.get("delta_json") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            delta = []
        try:
            stderr = json.loads(user.get("stderr_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            stderr = {}
        return {
            "s30": _as_float(user.get("s30")),
            "gamma": _as_float(user.get("gamma")),
            "kappa": _as_float(user.get("kappa")),
            "k": _as_float(user.get("k")),
            "delta": [float(value) for value in delta] if isinstance(delta, list) else [],
            "contest_offset": _as_float(user.get("contest_offset"), 0.0),
            "stderr": stderr if isinstance(stderr, dict) else {},
            "base_prior": _as_float(user.get("base_prior")),
            "cf_handle": user.get("cf_handle"),
            "shape_fit_count": int(user.get("shape_fit_count") or 0),
            "last_status_ts": int(user.get("last_status_ts") or 0),
        }

    def set_shape_fit_count(self, discord_id: int, count: int) -> None:
        with self._lock:
            self._connection().execute(
                "UPDATE users SET shape_fit_count = ? WHERE discord_id = ?",
                (int(count), int(discord_id)),
            )

    # -- Codeforces solved cache / status cursor -----------------------------

    def get_last_status_ts(self, discord_id: int) -> int:
        with self._lock:
            row = self._connection().execute(
                "SELECT last_status_ts FROM users WHERE discord_id = ?",
                (int(discord_id),),
            ).fetchone()
        return int(row["last_status_ts"] or 0) if row is not None else 0

    def set_last_status_ts(self, discord_id: int, timestamp: int) -> None:
        with self._lock:
            self._connection().execute(
                "UPDATE users SET last_status_ts = ? WHERE discord_id = ?",
                (int(timestamp), int(discord_id)),
            )

    @staticmethod
    def _normalise_solved_rows(
        solved_iterable: Iterable[Any],
    ) -> list[tuple[int, str, str]]:
        rows: list[tuple[int, str, str]] = []
        for item in solved_iterable or ():
            contest_id: int | None = None
            index: str | None = None
            name: str | None = None
            if isinstance(item, Mapping):
                contest_id = _as_int(
                    item.get("contest_id", item.get("contestId"))
                )
                index = item.get("problem_index", item.get("index"))
                name = item.get("problem_name", item.get("name"))
            elif isinstance(item, (tuple, list)):
                if len(item) >= 2:
                    contest_id = _as_int(item[0])
                    index = item[1]
                    if len(item) >= 3:
                        name = item[2]
            else:
                continue
            if contest_id is None or not index:
                continue
            clean_index = str(index)
            rows.append(
                (
                    int(contest_id),
                    clean_index,
                    str(name) if name else f"{contest_id}{clean_index}",
                )
            )
        return rows

    def add_cf_solved(
        self,
        discord_id: int,
        solved_iterable: Iterable[Any],
        last_status_ts: int | None = None,
    ) -> int:
        """Insert accepted problems idempotently and advance the status cursor."""
        user_id = int(discord_id)
        rows = [
            (user_id, contest_id, index, name)
            for contest_id, index, name in self._normalise_solved_rows(solved_iterable)
        ]
        with self._lock:
            connection = self._connection()
            before = int(
                connection.execute(
                    "SELECT COUNT(*) FROM cf_solved WHERE discord_id = ?", (user_id,)
                ).fetchone()[0]
            )
            if rows:
                connection.executemany(
                    "INSERT OR IGNORE INTO cf_solved "
                    "(discord_id, contest_id, problem_index, problem_name) "
                    "VALUES (?, ?, ?, ?)",
                    rows,
                )
            after = int(
                connection.execute(
                    "SELECT COUNT(*) FROM cf_solved WHERE discord_id = ?", (user_id,)
                ).fetchone()[0]
            )
            if last_status_ts is not None:
                connection.execute(
                    "UPDATE users SET last_status_ts = "
                    "MAX(COALESCE(last_status_ts, 0), ?) WHERE discord_id = ?",
                    (int(last_status_ts), user_id),
                )
            return max(0, after - before)

    def get_solved_count(self, discord_id: int) -> int:
        with self._lock:
            row = self._connection().execute(
                "SELECT COUNT(*) AS count FROM cf_solved WHERE discord_id = ?",
                (int(discord_id),),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def clear_user_cf_data(self, discord_id: int) -> None:
        """Forget a handle's contest data when the user re-links."""
        user_id = int(discord_id)
        with self._lock:
            connection = self._connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM attempts WHERE discord_id = ? AND is_bootstrap = 1",
                    (user_id,),
                )
                connection.execute(
                    "DELETE FROM cf_solved WHERE discord_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM active_sessions WHERE discord_id = ?", (user_id,)
                )
                connection.execute(
                    "UPDATE users SET last_status_ts = 0, shape_fit_count = 0 "
                    "WHERE discord_id = ?",
                    (user_id,),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def solved_keys_and_names(
        self,
        discord_id: int,
    ) -> tuple[set[tuple[int, str]], set[str]]:
        """Problems to avoid: synced solves plus accepted bot attempts."""
        user_id = int(discord_id)
        keys: set[tuple[int, str]] = set()
        names: set[str] = set()
        with self._lock:
            connection = self._connection()
            for row in connection.execute(
                "SELECT contest_id, problem_index, problem_name FROM cf_solved "
                "WHERE discord_id = ?",
                (user_id,),
            ):
                keys.add((int(row["contest_id"]), str(row["problem_index"])))
                if row["problem_name"]:
                    names.add(str(row["problem_name"]))
            for row in connection.execute(
                "SELECT contest_id, problem_index, problem_name FROM attempts "
                "WHERE discord_id = ? AND solved = 1",
                (user_id,),
            ):
                keys.add((int(row["contest_id"]), str(row["problem_index"])))
                if row["problem_name"]:
                    names.add(str(row["problem_name"]))
        return keys, names

    # -- attempts ------------------------------------------------------------

    def add_attempt(
        self,
        discord_id: int,
        *,
        rating: float | None,
        contest_id: int,
        problem_index: str,
        problem_name: str | None,
        tags: Iterable[str] | None,
        actual_mins: float,
        was_count: int = 0,
        solved: bool = False,
        is_bootstrap: bool = False,
        weight_mult: float = 1.0,
        contest_start_ts: int | None = None,
        budget_mins: float | None = None,
        occurred_at: int | None = None,
        created_at: int | None = None,
    ) -> int:
        now = int(time.time())
        values = (
            int(discord_id),
            int(contest_id),
            str(problem_index),
            str(problem_name) if problem_name else None,
            _as_int(rating),
            _json_tags(tags),
            _as_float(budget_mins),
            _as_float(actual_mins, 0.0),
            int(was_count),
            1 if solved else 0,
            1 if is_bootstrap else 0,
            float(weight_mult),
            _as_int(contest_start_ts),
            int(occurred_at) if occurred_at is not None else now,
            int(created_at) if created_at is not None else now,
        )
        with self._lock:
            cursor = self._connection().execute(
                "INSERT INTO attempts (discord_id, contest_id, problem_index, "
                "problem_name, rating, tags_json, budget_mins, actual_mins, was_count, "
                "solved, is_bootstrap, weight_mult, contest_start_ts, occurred_at, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            return int(cursor.lastrowid or 0)

    def add_bootstrap_rows(self, discord_id: int, rows: Sequence[Any]) -> int:
        """Insert contest-derived rows idempotently.

        The partial unique index allows one bootstrap row per
        ``(contest_id, problem_index)``.  Rows are ordered newest-first and an
        ``ON CONFLICT ... DO UPDATE`` only fires when the incoming participation
        happened after the stored one, so the most recent virtual/live session
        wins while re-running ``/link`` with the same data is a no-op.
        """
        user_id = int(discord_id)
        now = int(time.time())
        prepared: list[tuple[Any, ...]] = []
        for row in rows or ():
            contest_id = _as_int(_field(row, "contest_id", "contestId"))
            problem_index = _field(row, "problem_index", "problemIndex", "index")
            if contest_id is None or not problem_index:
                continue
            occurred_at = _as_int(
                _field(row, "occurred_at", "occurredAt"),
                _as_int(_field(row, "contest_start_ts", "contestStartTs"), now) or now,
            )
            prepared.append(
                (
                    user_id,
                    int(contest_id),
                    str(problem_index),
                    _field(row, "problem_name", "problemName", "name"),
                    _as_int(_field(row, "rating")),
                    _json_tags(_field(row, "tags")),
                    _as_float(_field(row, "actual_mins", "actualMins"), 0.0),
                    int(_field(row, "was_count", "wasCount", default=0) or 0),
                    1 if _field(row, "solved", default=False) else 0,
                    1,
                    float(_field(row, "weight_mult", "weightMult", default=1.0) or 1.0),
                    _as_int(_field(row, "contest_start_ts", "contestStartTs")),
                    int(occurred_at or now),
                    now,
                )
            )
        if not prepared:
            return 0
        prepared.sort(key=lambda item: int(item[12]), reverse=True)
        with self._lock:
            connection = self._connection()
            before = int(
                connection.execute(
                    "SELECT COUNT(*) FROM attempts "
                    "WHERE discord_id = ? AND is_bootstrap = 1",
                    (user_id,),
                ).fetchone()[0]
            )
            connection.executemany(
                "INSERT INTO attempts (discord_id, contest_id, problem_index, "
                "problem_name, rating, tags_json, actual_mins, was_count, solved, "
                "is_bootstrap, weight_mult, contest_start_ts, occurred_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(discord_id, contest_id, problem_index) "
                "WHERE is_bootstrap = 1 "
                "DO UPDATE SET "
                "problem_name = excluded.problem_name, "
                "rating = excluded.rating, "
                "tags_json = excluded.tags_json, "
                "actual_mins = excluded.actual_mins, "
                "was_count = excluded.was_count, "
                "solved = excluded.solved, "
                "weight_mult = excluded.weight_mult, "
                "contest_start_ts = excluded.contest_start_ts, "
                "occurred_at = excluded.occurred_at, "
                "created_at = excluded.created_at "
                "WHERE excluded.occurred_at > attempts.occurred_at",
                prepared,
            )
            after = int(
                connection.execute(
                    "SELECT COUNT(*) FROM attempts "
                    "WHERE discord_id = ? AND is_bootstrap = 1",
                    (user_id,),
                ).fetchone()[0]
            )
            return max(0, after - before)

    def count_attempts(self, discord_id: int) -> int:
        with self._lock:
            row = self._connection().execute(
                "SELECT COUNT(*) AS count FROM attempts WHERE discord_id = ?",
                (int(discord_id),),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def load_history(
        self,
        discord_id: int,
        now_ts: float | None = None,
    ) -> list[predictor.AttemptRecord]:
        """Return attempts oldest-first with ``days_ago`` from ``occurred_at``."""
        now = time.time() if now_ts is None else float(now_ts)
        with self._lock:
            rows = self._connection().execute(
                "SELECT * FROM attempts WHERE discord_id = ? "
                "ORDER BY occurred_at ASC, id ASC",
                (int(discord_id),),
            ).fetchall()
        history: list[predictor.AttemptRecord] = []
        for row in rows:
            occurred_at = int(row["occurred_at"] or 0)
            days_ago = max(0.0, (now - occurred_at) / 86400.0)
            history.append(
                predictor.AttemptRecord(
                    rating=float(row["rating"] or 0.0),
                    contest_id=int(row["contest_id"] or 0),
                    tags=_parse_tags(row["tags_json"]),
                    actual_mins=float(row["actual_mins"] or 0.0),
                    was_count=int(row["was_count"] or 0),
                    solved=bool(row["solved"]),
                    days_ago=days_ago,
                    is_contest_bootstrap=bool(row["is_bootstrap"]),
                    contest_start_ts=_as_int(row["contest_start_ts"]),
                    budget_mins=_as_float(row["budget_mins"]),
                    weight_mult=float(row["weight_mult"] or 1.0),
                )
            )
        return history

    def archetype_counts(self, discord_id: int) -> list[int]:
        """Number of logged attempts whose tags map to each archetype."""
        counts = [0] * predictor.N_ARCHETYPES
        with self._lock:
            rows = self._connection().execute(
                "SELECT tags_json FROM attempts WHERE discord_id = ?",
                (int(discord_id),),
            ).fetchall()
        for row in rows:
            for archetype in predictor.get_archetypes(_parse_tags(row["tags_json"])):
                counts[archetype] += 1
        return counts

    # -- active sessions -----------------------------------------------------

    @staticmethod
    def _session_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        session = dict(row)
        session["tags"] = _parse_tags(session.get("tags_json"))
        return session

    def create_session(
        self,
        discord_id: int,
        *,
        channel_id: int,
        message_id: int,
        contest_id: int,
        problem_index: str,
        problem_name: str | None,
        rating: int | None,
        tags: Iterable[str] | None,
        budget_mins: float,
        confidence: float,
        blind_mode: bool,
        started_at: int | None = None,
    ) -> None:
        now = int(time.time()) if started_at is None else int(started_at)
        values = (
            int(discord_id),
            _as_int(channel_id),
            _as_int(message_id),
            int(contest_id),
            str(problem_index),
            str(problem_name) if problem_name else None,
            _as_int(rating),
            _json_tags(tags),
            float(budget_mins),
            float(confidence),
            1 if blind_mode else 0,
            now,
        )
        with self._lock:
            connection = self._connection()
            connection.execute(
                "DELETE FROM active_sessions WHERE discord_id = ?", (int(discord_id),)
            )
            connection.execute(
                "INSERT INTO active_sessions (discord_id, channel_id, message_id, "
                "contest_id, problem_index, problem_name, rating, tags_json, "
                "budget_mins, confidence, blind_mode, started_at, timer_notified, "
                "last_polled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)",
                values,
            )

    def get_session(self, discord_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM active_sessions WHERE discord_id = ?",
                (int(discord_id),),
            ).fetchone()
        return self._session_dict(row)

    def get_session_by_message(self, message_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM active_sessions WHERE message_id = ?",
                (int(message_id),),
            ).fetchone()
        return self._session_dict(row)

    def list_sessions_oldest_polled_first(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT * FROM active_sessions "
                "ORDER BY last_polled_at ASC, started_at ASC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            session = self._session_dict(row)
            if session is not None:
                sessions.append(session)
        return sessions

    def touch_polled(self, discord_id: int, timestamp: int | None = None) -> None:
        now = int(time.time()) if timestamp is None else int(timestamp)
        with self._lock:
            self._connection().execute(
                "UPDATE active_sessions SET last_polled_at = ? WHERE discord_id = ?",
                (now, int(discord_id)),
            )

    def mark_timer_notified(self, discord_id: int) -> None:
        with self._lock:
            self._connection().execute(
                "UPDATE active_sessions SET timer_notified = 1 WHERE discord_id = ?",
                (int(discord_id),),
            )

    def cancel_session(self, discord_id: int) -> bool:
        with self._lock:
            cursor = self._connection().execute(
                "DELETE FROM active_sessions WHERE discord_id = ?",
                (int(discord_id),),
            )
            return cursor.rowcount > 0

    def complete_session(self, discord_id: int, attempt_row: Any) -> bool:
        """Atomically retire the session and log exactly one attempt.

        Returns ``False`` (and inserts nothing) if the session no longer exists,
        which makes a poller/button race harmless.
        """
        user_id = int(discord_id)
        now = int(time.time())
        contest_id = _as_int(_field(attempt_row, "contest_id", "contestId"))
        problem_index = _field(attempt_row, "problem_index", "problemIndex", "index")
        if contest_id is None or not problem_index:
            raise ValueError("attempt_row needs contest_id and problem_index")
        problem_name = _field(attempt_row, "problem_name", "problemName", "name")
        rating = _as_int(_field(attempt_row, "rating"))
        tags = _field(attempt_row, "tags", default=[])
        budget_mins = _as_float(_field(attempt_row, "budget_mins", "budgetMins"))
        actual_mins = _as_float(
            _field(attempt_row, "actual_mins", "actualMins"), 0.0
        )
        was_count = int(_field(attempt_row, "was_count", "wasCount", default=0) or 0)
        solved = bool(_field(attempt_row, "solved", default=False))
        weight_mult = float(
            _field(attempt_row, "weight_mult", "weightMult", default=1.0) or 1.0
        )
        contest_start_ts = _as_int(
            _field(attempt_row, "contest_start_ts", "contestStartTs")
        )
        occurred_at = _as_int(
            _field(attempt_row, "occurred_at", "occurredAt"), now
        )
        created_at = _as_int(_field(attempt_row, "created_at", "createdAt"), now)

        with self._lock:
            connection = self._connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM active_sessions WHERE discord_id = ?",
                    (user_id,),
                ).fetchone()
                if exists is None:
                    connection.execute("ROLLBACK")
                    return False
                connection.execute(
                    "DELETE FROM active_sessions WHERE discord_id = ?", (user_id,)
                )
                connection.execute(
                    "INSERT INTO attempts (discord_id, contest_id, problem_index, "
                    "problem_name, rating, tags_json, budget_mins, actual_mins, "
                    "was_count, solved, is_bootstrap, weight_mult, contest_start_ts, "
                    "occurred_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                    (
                        user_id,
                        int(contest_id),
                        str(problem_index),
                        str(problem_name) if problem_name else None,
                        rating,
                        _json_tags(tags),
                        budget_mins,
                        actual_mins,
                        was_count,
                        1 if solved else 0,
                        weight_mult,
                        contest_start_ts,
                        int(occurred_at or now),
                        int(created_at or now),
                    ),
                )
                if solved:
                    connection.execute(
                        "INSERT OR IGNORE INTO cf_solved "
                        "(discord_id, contest_id, problem_index, problem_name) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            user_id,
                            int(contest_id),
                            str(problem_index),
                            str(problem_name) if problem_name else None,
                        ),
                    )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise


__all__ = [
    "DEFAULT_DATA_DIR",
    "Database",
    "SCHEMA_VERSION",
    "default_db_path",
]
