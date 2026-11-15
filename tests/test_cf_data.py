"""Offline tests for Codeforces data handling (Tests 7, 12, 17 and 18)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cf_data  # noqa: E402
import db  # noqa: E402
import predictor  # noqa: E402

START = 1_600_000_000
CONTEST_ID = 100


# ---------------------------------------------------------------------------
# 7. Contest bootstrap extractor
# ---------------------------------------------------------------------------


def _contest_submission(
    index: str,
    offset_mins: float,
    *,
    verdict: str = "OK",
    rating: int | None = 1500,
    name: str | None = None,
    contest_id: int = CONTEST_ID,
    participant_type: str = "CONTESTANT",
    part_start: int = START,
) -> dict:
    return {
        "contestId": contest_id,
        "creationTimeSeconds": part_start + int(round(offset_mins * 60)),
        "verdict": verdict,
        "author": {"participantType": participant_type, "startTimeSeconds": part_start},
        "problem": {
            "contestId": contest_id,
            "index": index,
            "name": name or f"Problem {index}",
            "rating": rating,
            "tags": ["dp"],
        },
    }


def test_contest_bootstrap_extractor() -> None:
    contests_by_id = {
        CONTEST_ID: {
            "startTimeSeconds": START,
            "durationSeconds": 120 * 60,
            "name": "Test Round",
        }
    }
    rating_changes = [
        {
            "contestId": CONTEST_ID,
            "oldRating": 1500,
            "newRating": 1550,
            "ratingUpdateTimeSeconds": START + 120 * 60,
        }
    ]
    submissions = [
        # A is excluded by the rating floor (800 < 1500 - 300) but still
        # defines prev_time for B.
        _contest_submission("A", 5.0, rating=800),
        _contest_submission("B", 13.0, verdict="WRONG_ANSWER", rating=1400),
        _contest_submission("B", 18.0, rating=1400),
        _contest_submission("C", 30.0, verdict="WRONG_ANSWER", rating=1700),
        _contest_submission("C", 50.0, verdict="WRONG_ANSWER", rating=1700),
    ]

    rows = cf_data.extract_contest_deltas_from_data(
        rating_changes, contests_by_id, submissions
    )
    by_index = {row.problem_index: row for row in rows}
    assert set(by_index) == {"B", "C"}, [row.problem_index for row in rows]

    b_row = by_index["B"]
    assert b_row.solved is True
    assert b_row.actual_mins == 13.0
    assert b_row.was_count == 1
    assert b_row.rating == 1400
    assert b_row.occurred_at == START + 18 * 60
    assert b_row.contest_start_ts == START

    c_row = by_index["C"]
    assert c_row.solved is False
    assert c_row.actual_mins == 102.0
    assert c_row.was_count == 0
    assert c_row.rating == 1700
    assert c_row.occurred_at == START + 120 * 60


# ---------------------------------------------------------------------------
# 12. Problem selection
# ---------------------------------------------------------------------------


def test_select_best_problem() -> None:
    model = predictor.CFPredictor()
    model.s30 = 1800.0
    model.gamma = 0.0
    model.delta = [0.0] * predictor.N_ARCHETYPES

    solved_by_key = cf_data.CFProblem(1900, "A", "Already Solved", 1800, ("dp",))
    mirror_by_name = cf_data.CFProblem(1901, "A", "Shared Mirror", 1800, ("dp",))
    special = cf_data.CFProblem(1902, "A", "Special", 1800, ("*special",))
    interactive = cf_data.CFProblem(1903, "A", "Interactive", 1800, ("interactive",))

    excluded = [solved_by_key, mirror_by_name, special, interactive]
    result = cf_data.select_best_problem(
        model,
        excluded,
        solved_keys={(1900, "A")},
        solved_names={"Shared Mirror"},
        target_mins=25.0,
        blind_jitter=False,
        rng=random.Random(0),
    )
    assert result is None

    normal = cf_data.CFProblem(1200, "B", "Normal", 1800, ("math",))
    result = cf_data.select_best_problem(
        model,
        excluded + [normal],
        solved_keys={(1900, "A")},
        solved_names={"Shared Mirror"},
        target_mins=25.0,
        blind_jitter=False,
        rng=random.Random(0),
    )
    assert isinstance(result, cf_data.CFProblem)
    assert result is normal

    # Interactive is selectable only when explicitly allowed.
    result = cf_data.select_best_problem(
        model,
        [interactive, normal],
        solved_keys=set(),
        solved_names=set(),
        target_mins=25.0,
        blind_jitter=False,
        rng=random.Random(0),
        allow_interactive=True,
    )
    assert isinstance(result, cf_data.CFProblem)
    assert result in (interactive, normal)

    # Closest-pool behaviour with jitter disabled.
    spread = [
        cf_data.CFProblem(1300 + index * 10, "A", f"P{index}", rating, ("dp",))
        for index, rating in enumerate(range(1400, 2600, 100))
    ]
    picked = cf_data.select_best_problem(
        model,
        spread,
        solved_keys=set(),
        solved_names=set(),
        target_mins=25.0,
        blind_jitter=False,
        rng=random.Random(7),
    )
    assert isinstance(picked, cf_data.CFProblem)
    assert abs(float(picked.rating) - 1800.0) <= 300.0


# ---------------------------------------------------------------------------
# 17. Incremental sync with the 15-minute overlap window
# ---------------------------------------------------------------------------


def test_sync_recent_status_overlap(tmp_path: Path) -> None:
    database = db.Database(str(tmp_path / "sync.db"))
    user_id = 7
    database.upsert_user(user_id, "someone", 1800.0)

    def submission(
        index: str,
        timestamp: int,
        verdict: str | None,
        *,
        name: str | None = None,
    ) -> dict:
        return {
            "contestId": 1900,
            "creationTimeSeconds": timestamp,
            "verdict": verdict,
            "problem": {
                "contestId": 1900,
                "index": index,
                "name": name or f"Problem {index}",
                "rating": 1900,
                "tags": ["dp"],
            },
        }

    s0 = submission("C", 8000, "WRONG_ANSWER")
    s1_testing = submission("A", 10000, "TESTING")
    s1_ok = submission("A", 10000, "OK")
    s2 = submission("B", 10050, "WRONG_ANSWER")

    first_calls: list[int] = []

    def first_fetch(handle: str, from_idx: int, count: int) -> list[dict]:
        first_calls.append(from_idx)
        assert from_idx == 1
        return [s2, s1_testing]

    returned = cf_data.sync_recent_status(
        database,
        user_id,
        "someone",
        last_status_ts=0,
        fetch_page=first_fetch,
        page_size=3,
    )
    assert returned == [s2, s1_testing]
    assert first_calls == [1]
    keys, names = database.solved_keys_and_names(user_id)
    assert keys == set()
    assert names == set()
    assert database.get_last_status_ts(user_id) == 10050

    second_calls: list[int] = []

    def second_fetch(handle: str, from_idx: int, count: int) -> list[dict]:
        second_calls.append(from_idx)
        if from_idx > 1:
            raise AssertionError("overlap stop rule should prevent page 2")
        return [s2, s1_ok, s0]

    returned = cf_data.sync_recent_status(
        database,
        user_id,
        "someone",
        last_status_ts=None,
        fetch_page=second_fetch,
        page_size=3,
    )
    assert second_calls == [1]
    assert returned == [s2, s1_ok, s0]

    keys, _ = database.solved_keys_and_names(user_id)
    assert (1900, "A") in keys
    assert (1900, "B") not in keys
    assert (1900, "C") not in keys
    assert database.get_last_status_ts(user_id) == 10050
    database.close()


# ---------------------------------------------------------------------------
# 18. Virtual contest bootstrap extraction
# ---------------------------------------------------------------------------


def test_virtual_contest_bootstrap_extractor() -> None:
    official_start = 1_550_000_000
    virtual_start = 1_700_000_000
    contests_by_id = {
        1200: {
            "startTimeSeconds": official_start,
            "durationSeconds": 7200,
            "name": "Virtual Round",
        }
    }
    rating_changes = [
        {
            "contestId": 1100,
            "oldRating": 1400,
            "newRating": 1600,
            "ratingUpdateTimeSeconds": 1_650_000_000,
        }
    ]
    submissions = [
        # Rating 900 is excluded by the 1300 floor, but defines prev_time = 4.
        _contest_submission(
            "A", 4.0, rating=900, contest_id=1200,
            participant_type="VIRTUAL", part_start=virtual_start,
        ),
        _contest_submission(
            "B", 10.0, verdict="WRONG_ANSWER", rating=1500, contest_id=1200,
            participant_type="VIRTUAL", part_start=virtual_start,
        ),
        _contest_submission(
            "B", 19.0, rating=1500, contest_id=1200,
            participant_type="VIRTUAL", part_start=virtual_start,
        ),
        _contest_submission(
            "C", 45.0, verdict="TIME_LIMIT_EXCEEDED", rating=1800,
            contest_id=1200, participant_type="VIRTUAL", part_start=virtual_start,
        ),
    ]

    rows = cf_data.extract_contest_deltas_from_data(
        rating_changes, contests_by_id, submissions
    )
    by_index = {row.problem_index: row for row in rows}
    assert set(by_index) == {"B", "C"}, [row.problem_index for row in rows]

    b_row = by_index["B"]
    assert b_row.solved is True
    assert b_row.actual_mins == 15.0
    assert b_row.was_count == 1
    assert b_row.rating == 1500
    assert b_row.contest_start_ts == official_start
    assert b_row.occurred_at == virtual_start + 19 * 60

    c_row = by_index["C"]
    assert c_row.solved is False
    assert c_row.actual_mins == 101.0
    assert c_row.was_count == 0
    assert c_row.rating == 1800
    assert c_row.contest_start_ts == official_start
    assert c_row.occurred_at == virtual_start + 7200

    # A practice OK before the virtual started marks B as pre-solved, so B is
    # ignored completely; the timeout now starts from A's +4 (last AC).
    earlier_b_ok = {
        "contestId": 1200,
        "creationTimeSeconds": 1_600_000_000,
        "verdict": "OK",
        "author": {"participantType": "PRACTICE"},
        "problem": {
            "contestId": 1200,
            "index": "B",
            "name": "Problem B",
            "rating": 1500,
            "tags": ["dp"],
        },
    }
    pre_solved_rows = cf_data.extract_contest_deltas_from_data(
        rating_changes, contests_by_id, submissions + [earlier_b_ok]
    )
    pre_solved_by_index = {row.problem_index: row for row in pre_solved_rows}
    assert "B" not in pre_solved_by_index
    assert pre_solved_by_index["C"].actual_mins == 116.0

    explicit_map_rows = cf_data.extract_contest_deltas_from_data(
        rating_changes,
        contests_by_id,
        submissions,
        first_ok_ts={(1200, "B"): 1_600_000_000},
    )
    assert "B" not in {row.problem_index for row in explicit_map_rows}

