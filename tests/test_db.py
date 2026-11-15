"""Offline tests for the SQLite persistence layer (Test 13)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db  # noqa: E402
import predictor  # noqa: E402

NOW = 1_700_000_000


def _attempt(
    *,
    contest_id: int = 1900,
    index: str = "A",
    name: str = "Problem A",
    rating: int = 1700,
    tags: list[str] | None = None,
    actual: float = 22.0,
    was: int = 1,
    solved: bool = True,
    budget: float = 25.0,
    occurred_at: int | None = None,
) -> dict:
    return {
        "contest_id": contest_id,
        "problem_index": index,
        "problem_name": name,
        "rating": rating,
        "tags": list(tags or ["dp"]),
        "budget_mins": budget,
        "actual_mins": actual,
        "was_count": was,
        "solved": solved,
        "weight_mult": 1.0,
        "contest_start_ts": None,
        "occurred_at": occurred_at,
    }


def test_db_semantics(tmp_path: Path) -> None:
    database = db.Database(str(tmp_path / "bot.db"))
    user_id = 42

    database.upsert_user(user_id, "Tourist", 2400.0)
    assert database.get_user(user_id)["cf_handle"] == "Tourist"

    # A practice problem can be attempted twice (no dedupe for practice).
    for offset in range(2):
        database.create_session(
            user_id,
            channel_id=1,
            message_id=100 + offset,
            contest_id=1900,
            problem_index="A",
            problem_name="Problem A",
            rating=1700,
            tags=["dp"],
            budget_mins=25.0,
            confidence=0.65,
            blind_mode=True,
            started_at=NOW,
        )
        assert database.complete_session(
            user_id,
            _attempt(occurred_at=NOW - 10 * 86400),
        )
    assert database.count_attempts(user_id) == 2

    # Bootstrap rows are idempotent thanks to the partial unique index.
    bootstrap_row = {
        "contest_id": 1500,
        "problem_index": "B",
        "problem_name": "Problem B",
        "rating": 1600,
        "tags": ["greedy"],
        "actual_mins": 20.0,
        "was_count": 1,
        "solved": True,
        "occurred_at": NOW - 5 * 86400,
        "contest_start_ts": NOW - 5 * 86400,
    }
    assert database.add_bootstrap_rows(user_id, [bootstrap_row]) == 1
    assert database.add_bootstrap_rows(user_id, [bootstrap_row]) == 0

    history = database.load_history(user_id, now_ts=float(NOW))
    assert len(history) == 3
    assert isinstance(history[0], predictor.AttemptRecord)
    assert history[0].is_contest_bootstrap is False
    assert history[0].days_ago == 10.0
    assert history[0].tags == ["dp"]
    assert history[-1].is_contest_bootstrap is True
    assert history[-1].days_ago == 5.0
    assert history[-1].tags == ["greedy"]

    # complete_session is atomic; the second call sees no session and logs nothing.
    database.create_session(
        user_id,
        channel_id=2,
        message_id=200,
        contest_id=2000,
        problem_index="C",
        problem_name="Problem C",
        rating=1800,
        tags=["trees"],
        budget_mins=30.0,
        confidence=0.5,
        blind_mode=False,
        started_at=NOW,
    )
    assert database.get_session(user_id)["problem_index"] == "C"
    assert database.complete_session(user_id, _attempt(contest_id=2000, index="C")) is True
    before = database.count_attempts(user_id)
    assert database.complete_session(user_id, _attempt(contest_id=2000, index="C")) is False
    assert database.count_attempts(user_id) == before

    # Incremental cf_solved sync: new rows only, cursor never moves backwards.
    assert database.add_cf_solved(
        user_id, [(111, "A", "Alpha")], last_status_ts=500
    ) == 1
    assert database.get_last_status_ts(user_id) == 500
    assert database.add_cf_solved(
        user_id, [(111, "A", "Alpha"), (112, "B", "Beta")], last_status_ts=400
    ) == 1
    assert database.get_last_status_ts(user_id) == 500
    assert database.add_cf_solved(user_id, [], last_status_ts=900) == 0
    assert database.get_last_status_ts(user_id) == 900

    keys, names = database.solved_keys_and_names(user_id)
    assert (111, "A") in keys and "Alpha" in names
    assert (112, "B") in keys and "Beta" in names
    # Union with accepted bot attempts (practice and bootstrap both count).
    assert (1900, "A") in keys
    assert (1500, "B") in keys
    assert (2000, "C") in keys
    assert "Problem B" in names

    # Model state round-trips through JSON columns.
    model = predictor.CFPredictor(base_prior=2400.0)
    model.s30 = 2111.5
    model.gamma = 175.0
    model.delta = [10.0, -20.0, 30.0, -40.0, 50.0]
    model.contest_offset = 12.0
    model.stderr = {
        "s30": 50.0,
        "gamma": 20.0,
        "delta": [5.0, 6.0, 7.0, 8.0, 9.0],
        "contest_offset": 4.0,
    }
    database.save_model_state(user_id, model.to_state(), shape_fit_count=3)
    state = database.get_model_state(user_id)
    assert state["s30"] == 2111.5
    assert state["delta"] == [10.0, -20.0, 30.0, -40.0, 50.0]
    assert state["shape_fit_count"] == 3

    # Re-linking a different handle clears contest data and sessions.
    database.create_session(
        user_id,
        channel_id=3,
        message_id=300,
        contest_id=2100,
        problem_index="D",
        problem_name="Problem D",
        rating=1900,
        tags=[],
        budget_mins=20.0,
        confidence=0.65,
        blind_mode=True,
        started_at=NOW,
    )
    database.clear_user_cf_data(user_id)
    assert database.get_session(user_id) is None
    assert database.get_last_status_ts(user_id) == 0
    assert database.get_solved_count(user_id) == 0
    remaining = database.load_history(user_id, now_ts=float(NOW))
    assert all(not record.is_contest_bootstrap for record in remaining)
    assert len(remaining) == 3

    database.close()
