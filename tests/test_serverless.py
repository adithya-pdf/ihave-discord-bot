"""Serverless-focused tests: signature verification and memory-backed storage.

These tests do not call Discord or Codeforces. The production backend is Redis,
but the storage domain logic is exercised through the in-process fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from nacl.signing import SigningKey

import main
import storage


def test_signature_verification_accepts_valid_and_rejects_old_or_tampered():
    signing_key = SigningKey.generate()
    original_key = main.DISCORD_PUBLIC_KEY
    main.DISCORD_PUBLIC_KEY = signing_key.verify_key.encode().hex()
    try:
        body = json.dumps({"type": 1}).encode()
        timestamp = str(int(time.time()))
        signature = signing_key.sign(timestamp.encode() + body).signature.hex()

        assert main.verify_signature(body, signature, timestamp)
        assert not main.verify_signature(body, "00" * 64, timestamp)
        assert not main.verify_signature(
            body, signature, str(int(time.time()) - 3600)
        )
    finally:
        main.DISCORD_PUBLIC_KEY = original_key


def test_memory_storage_curve_adjust_and_sessions():
    async def scenario() -> None:
        storage._BACKEND = storage.MemoryBackend()
        try:
            await storage.upsert_user(1, "tester", 1600)
            points = [
                {
                    "rating": rating,
                    "raw_minutes": None,
                    "smoothed_minutes": float(10 + (rating - 800) / 100),
                    "sample_size": 0,
                }
                for rating in storage.estimator.RATING_GRID
            ]
            await storage.rebuild_curve("tester", points)
            assert len(await storage.get_curve("tester")) == len(
                storage.estimator.RATING_GRID
            )

            previous, updated = await storage.adjust_curve("tester", 1600, 25.0)
            assert previous > 0
            assert abs(updated - 25.0) < 0.01
            values = await storage.get_curve_map("tester")
            ratings = sorted(values)
            assert all(values[b] >= values[a] for a, b in zip(ratings, ratings[1:]))

            await storage.add_solved_many("tester", [(100, "A"), (100, "B")])
            assert await storage.get_solved_count("tester") == 2

            session = await storage.create_session(
                1,
                channel_id=2,
                handle="tester",
                expires_at=int(time.time()) + 60,
            )
            assert (await storage.get_session_by_user(1))["id"] == session["id"]
            assert len(await storage.list_active_sessions()) == 1
            assert await storage.claim_feedback(session["id"]) is True
            assert await storage.claim_feedback(session["id"]) is False
            await storage.delete_session(session["id"])
            assert await storage.get_session_by_user(1) is None
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_memory_storage_is_not_used_on_vercel_without_redis():
    async def scenario() -> None:
        storage._BACKEND = None
        os.environ["VERCEL"] = "1"
        try:
            try:
                storage.get_backend()
            except storage.StorageError:
                pass
            else:
                raise AssertionError("Vercel without Redis must fail closed")
        finally:
            os.environ.pop("VERCEL", None)
            storage._BACKEND = None

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit("Use pytest to run these tests.")
