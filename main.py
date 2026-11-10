#!/usr/bin/env python3
"""Vercel-native Discord slash-command bot for Codeforces practice sessions.

This is an **HTTP Interactions** app, not a gateway bot:

* Discord POSTs every slash command / button click to ``/api/interactions``.
* The request is Ed25519-verified with the application public key.
* A deferred response is returned immediately; Vercel's ``wait_until`` keeps the
  invocation alive while Codeforces calls, charts and Discord edits finish.
* State lives in Redis (Upstash / Vercel KV), never on the local filesystem.

Because there is no persistent process, there is no minute-by-minute poller.
``/status`` performs an on-demand Accepted check, and the optional
``/api/cron/poll`` endpoint can drive automatic checks on a schedule (Vercel
Cron Pro, or any external scheduler). See README.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import cf_api
import estimator
import storage

load_dotenv()

logging.basicConfig(
    level=getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("ihave")

DISCORD_API = "https://discord.com/api/v10"
DISCORD_APPLICATION_ID = (os.getenv("DISCORD_APPLICATION_ID") or "").strip()
DISCORD_PUBLIC_KEY = (os.getenv("DISCORD_PUBLIC_KEY") or "").strip()
DISCORD_BOT_TOKEN = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
CRON_SECRET = (os.getenv("CRON_SECRET") or "").strip()

POLL_INTERVAL_SECONDS = 60
MIN_SESSION_MINUTES = 5
MAX_SESSION_MINUTES = 180

EMBED_COLOR = 0x4C72B0
SUCCESS_COLOR = 0x2E8B57
WARNING_COLOR = 0xD9A400
ERROR_COLOR = 0xC0392B
MUTED_COLOR = 0x5A5A6E
FOOTER_TEXT = "Codeforces time-boxed practice · /ihave"

COLD_START_TIP = (
    "Make sure to register for Virtual Contests before starting them! "
    "Unregistered practice submissions are ignored by the time-curve to keep "
    "your data accurate."
)

CF = cf_api.CodeforcesClient()
_DISCORD_HTTP: aiohttp.ClientSession | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def problem_url(contest_id: int, index_name: str) -> str:
    return f"https://codeforces.com/contest/{contest_id}/problem/{index_name}"


def profile_url(handle: str) -> str:
    return f"https://codeforces.com/profile/{handle}"


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def embed(
    title: str,
    description: str | None = None,
    *,
    color: int = EMBED_COLOR,
    fields: list[dict[str, Any]] | None = None,
    url: str | None = None,
    footer: str | None = FOOTER_TEXT,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": title[:256],
        "color": color,
        "fields": fields or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if description:
        payload["description"] = description[:4096]
    if url:
        payload["url"] = url
    if footer:
        payload["footer"] = {"text": footer[:2048]}
    return payload


def error_embed(message: str) -> dict[str, Any]:
    return embed("⚠️ Something went wrong", message, color=ERROR_COLOR)


def field(name: str, value: str, *, inline: bool = False) -> dict[str, Any]:
    return {"name": name[:256], "value": value[:1024], "inline": inline}


def _interaction_user_id(interaction: dict[str, Any]) -> int | None:
    user = interaction.get("user")
    if not isinstance(user, dict):
        member = interaction.get("member")
        user = member.get("user") if isinstance(member, dict) else None
    if not isinstance(user, dict) or user.get("id") is None:
        return None
    try:
        return int(user["id"])
    except (TypeError, ValueError):
        return None


def _options(interaction: dict[str, Any]) -> dict[str, Any]:
    data = interaction.get("data")
    if not isinstance(data, dict):
        return {}
    result: dict[str, Any] = {}
    for option in data.get("options") or []:
        if isinstance(option, dict) and option.get("name") is not None:
            result[str(option["name"])] = option.get("value")
    return result


def _session_remaining(session: dict[str, Any]) -> int:
    try:
        return max(0, int(session["expires_at"]) - int(time.time()))
    except (KeyError, TypeError, ValueError):
        return 0


def _session_limit_minutes(session: dict[str, Any]) -> int:
    try:
        return max(1, (int(session["expires_at"]) - int(session["started_at"])) // 60)
    except (KeyError, TypeError, ValueError):
        return 30


def _curve_table(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        try:
            lines.append(f"{int(row['rating']):>4}  →  {float(row['smoothed_minutes']):>6.1f} min")
        except (KeyError, TypeError, ValueError):
            continue
    return "```\n" + "\n".join(lines) + "\n```"


def _curve_summary(handle: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_samples = sum(int(row.get("sample_size") or 0) for row in rows)
    result = embed(
        f"📈 Solve-time curve — @{handle}",
        "Estimated minutes needed to solve a problem at each rating. "
        "Solid line: smoothed personal estimator. Dashed line: raw averages.",
    )
    result["fields"].append(field("Rating → expected minutes", _curve_table(rows)))
    if total_samples == 0:
        result["fields"].append(
            field(
                "Cold start",
                "No rated contest solves were found, so the fallback curve "
                "anchored around your current rating is shown.",
            )
        )
    else:
        result["fields"].append(
            field(
                "Observations",
                f"{total_samples} timed contest solves feed the raw averages. "
                "The fallback is used when fewer than 5 rated contests exist.",
            )
        )
    return result


def _feedback_components(session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "type": 1,
            "components": [
                {
                    "type": 2,
                    "style": 2,
                    "label": "⚡ Too Easy",
                    "custom_id": f"feedback:easy:{session_id}",
                },
                {
                    "type": 2,
                    "style": 4,
                    "label": "😅 Too Hard",
                    "custom_id": f"feedback:hard:{session_id}",
                },
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Discord HTTP helpers
# ---------------------------------------------------------------------------


async def _discord_http() -> aiohttp.ClientSession:
    global _DISCORD_HTTP
    if _DISCORD_HTTP is None or _DISCORD_HTTP.closed:
        _DISCORD_HTTP = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
    return _DISCORD_HTTP


async def edit_original(
    interaction: dict[str, Any],
    *,
    content: str | None = None,
    embeds: list[dict[str, Any]] | None = None,
    components: list[dict[str, Any]] | None = None,
    file: tuple[str, bytes] | None = None,
) -> None:
    """Edit the deferred ``@original`` interaction response."""
    application_id = str(interaction.get("application_id") or DISCORD_APPLICATION_ID)
    token = str(interaction.get("token") or "")
    if not application_id or not token:
        log.warning("Cannot edit interaction response without application id/token")
        return

    payload: dict[str, Any] = {}
    if content is not None:
        payload["content"] = content[:2000]
    if embeds is not None:
        payload["embeds"] = embeds
    if components is not None:
        payload["components"] = components

    url = f"{DISCORD_API}/webhooks/{application_id}/{token}/messages/@original"
    session = await _discord_http()
    try:
        if file is None:
            async with session.patch(url, json=payload) as response:
                if response.status >= 300:
                    log.warning(
                        "Discord edit failed (%s): %s",
                        response.status,
                        (await response.text())[:300],
                    )
        else:
            filename, data = file
            payload["attachments"] = [{"id": 0, "filename": filename}]
            form = aiohttp.FormData()
            form.add_field(
                "payload_json",
                json.dumps(payload),
                content_type="application/json",
            )
            form.add_field(
                "files[0]",
                data,
                filename=filename,
                content_type="image/png",
            )
            async with session.patch(url, data=form) as response:
                if response.status >= 300:
                    log.warning(
                        "Discord file edit failed (%s): %s",
                        response.status,
                        (await response.text())[:300],
                    )
    except aiohttp.ClientError:
        log.warning("Discord edit request failed", exc_info=True)


async def send_channel_message(channel_id: int, payload: dict[str, Any]) -> bool:
    """Send a normal channel message as the bot (used by the cron poller)."""
    if not DISCORD_BOT_TOKEN:
        log.warning("DISCORD_BOT_TOKEN is not set; cannot send channel message")
        return False
    session = await _discord_http()
    try:
        async with session.post(
            f"{DISCORD_API}/channels/{int(channel_id)}/messages",
            json=payload,
            headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
        ) as response:
            if response.status >= 300:
                log.warning(
                    "Channel message failed (%s): %s",
                    response.status,
                    (await response.text())[:300],
                )
                return False
            return True
    except aiohttp.ClientError:
        log.warning("Channel message request failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Signature verification and deferred work scheduling
# ---------------------------------------------------------------------------


def verify_signature(body: bytes, signature: str, timestamp: str) -> bool:
    public_key = DISCORD_PUBLIC_KEY
    if not public_key:
        return os.getenv("ALLOW_INSECURE_INTERACTIONS") == "1"
    if not signature or not timestamp:
        return False
    try:
        timestamp_value = int(timestamp)
        if abs(int(time.time()) - timestamp_value) > 300:
            return False
    except (TypeError, ValueError):
        return False
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        log.error("PyNaCl is not installed; cannot verify Discord signatures")
        return False

    try:
        verifier = VerifyKey(bytes.fromhex(public_key))
        verifier.verify(
            f"{timestamp}{body.decode('utf-8')}".encode("utf-8"),
            bytes.fromhex(signature),
        )
        return True
    except (BadSignatureError, UnicodeDecodeError, ValueError, TypeError):
        return False


def _schedule(coroutine: Any) -> None:
    """Run post-response work with Vercel's waitUntil when available."""
    if os.getenv("VERCEL"):
        try:
            from vercel.cache.context import get_context

            context = get_context()
            if getattr(context, "wait_until", None) is not None:
                context.wait_until(coroutine)
                return
            log.warning("Vercel context has no wait_until; falling back to a task")
        except Exception:
            log.warning("Could not access Vercel waitUntil", exc_info=True)

    task = asyncio.create_task(coroutine)
    _BACKGROUND_TASKS.add(task)

    def _done(finished: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(finished)
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            log.error("Background interaction task failed", exc_info=error)

    task.add_done_callback(_done)


_BACKGROUND_TASKS: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


async def _sync_handle(
    handle: str,
    *,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = info or await CF.get_user_info(handle)
    canonical = str(info.get("handle") or handle)
    current_rating = estimator.normalize_rating(
        info.get("rating") or info.get("maxRating") or estimator.DEFAULT_RATING
    )

    submissions = await CF.get_user_status(canonical, count=10000)
    solved = estimator.extract_solved_pairs(submissions)
    observations = estimator.extract_contest_deltas(submissions)
    recent = estimator.select_recent_participations(
        observations, estimator.RECENT_PARTICIPATIONS
    )
    rated_contests = len({observation.contest_id for observation in observations})
    points = estimator.build_curve(
        recent,
        current_rating,
        rated_contest_count=rated_contests,
    )

    if solved:
        await storage.add_solved_many(canonical, solved)
    rows = await storage.rebuild_curve(canonical, points)
    await storage.remember_handle(canonical)

    return {
        "handle": canonical,
        "current_rating": current_rating,
        "info": info,
        "submission_count": len(submissions),
        "solved": solved,
        "observations": observations,
        "rated_contests": rated_contests,
        "points": points,
        "rows": rows,
    }


def _find_new_accepted(
    session: dict[str, Any],
    submissions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    baseline = {int(item) for item in session.get("baseline_ids") or []}
    target_contest = int(session.get("contest_id") or 0)
    target_index = str(session.get("index_name") or "")
    for submission in submissions:
        submission_id = submission.get("id")
        if submission_id is None:
            continue
        try:
            if int(submission_id) in baseline:
                continue
        except (TypeError, ValueError):
            continue
        if submission.get("verdict") != "OK":
            continue
        problem = submission.get("problem")
        if not isinstance(problem, dict):
            continue
        contest_id = submission.get("contestId")
        if contest_id is None:
            contest_id = problem.get("contestId")
        try:
            contest_id = int(contest_id)
        except (TypeError, ValueError):
            continue
        if contest_id != target_contest:
            continue
        if str(problem.get("index") or "") != target_index:
            continue
        return submission
    return None


def _accepted_embed(session: dict[str, Any], elapsed: float) -> dict[str, Any]:
    url = problem_url(int(session["contest_id"]), str(session["index_name"]))
    tags = ", ".join(session.get("tags") or []) or "no tags"
    result = embed(
        f"✅ Accepted — {session['problem_name']}",
        f"<@{session['user_id']}> solved **[{session['problem_name']}]({url})** "
        f"in **{format_seconds(elapsed)}** "
        f"(limit: {_session_limit_minutes(session)} min). Well played!",
        color=SUCCESS_COLOR,
        url=url,
    )
    result["fields"].append(field("Rating", f"||{session['rating']}||", inline=True))
    result["fields"].append(field("Tags", f"||{tags}||"))
    return result


def _timeout_embed(session: dict[str, Any]) -> dict[str, Any]:
    url = problem_url(int(session["contest_id"]), str(session["index_name"]))
    result = embed(
        f"⏰ Time's up — {session['problem_name']}",
        f"<@{session['user_id']}> the **{_session_limit_minutes(session)}-minute** "
        "timer ended without an Accepted submission.\n\n"
        "Run `/giveup` to reveal the problem's rating and tags.",
        color=WARNING_COLOR,
        url=url,
    )
    return result


def _accepted_elapsed(session: dict[str, Any], submission: dict[str, Any]) -> float:
    try:
        accepted_at = int(submission.get("creationTimeSeconds") or time.time())
    except (TypeError, ValueError):
        accepted_at = int(time.time())
    elapsed = max(0, accepted_at - int(session["started_at"]))
    return min(elapsed, _session_limit_minutes(session) * 60)


async def _finish_accepted(
    interaction: dict[str, Any],
    session: dict[str, Any],
    submission: dict[str, Any],
) -> None:
    elapsed = _accepted_elapsed(session, submission)
    await storage.mark_solved(
        str(session["handle"]),
        int(session["contest_id"]),
        str(session["index_name"]),
    )
    await storage.delete_session(str(session["id"]))
    await edit_original(
        interaction,
        content=f"<@{session['user_id']}>",
        embeds=[_accepted_embed(session, elapsed)],
        components=[],
    )


async def _mark_timeout(session: dict[str, Any]) -> None:
    session["expired"] = True
    await storage.update_session(session)
    await storage.remove_from_active(str(session["id"]))


# ---------------------------------------------------------------------------
# Slash command handlers
# ---------------------------------------------------------------------------


async def cmd_help(interaction: dict[str, Any]) -> None:
    fields = [
        field("`/link <handle>`", "Link your Codeforces handle and build your curve."),
        field("`/ihave <minutes> [rating]`", "Start a 5–180 minute practice session."),
        field("`/adjust <rating> <minutes>`", 'Recalibrate: "I solve 1700 in 30 minutes".'),
        field("`/curve [handle]`", "Chart plus smoothed minutes-per-rating table."),
        field("`/status`", "Check the active session and its remaining time."),
        field("`/giveup`", "Reveal the rating and tags behind spoilers."),
        field("`/sync`", "Refresh submissions, curve and the problemset cache."),
    ]
    await edit_original(
        interaction,
        embeds=[
            embed(
                "/ihave — Codeforces time-boxed practice",
                "Tell me how many minutes you have and I'll pick a recent, "
                "unsolved Codeforces problem at a rating you can realistically "
                "finish. Tags stay hidden until you solve it or give up.\n\n"
                "Use `/status` to check for an Accepted submission; serverless "
                "mode has no background poller.",
                fields=fields,
            )
        ],
    )


async def cmd_link(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    handle = str(_options(interaction).get("handle") or "").strip().lstrip("@")
    if user_id is None or not handle:
        await edit_original(interaction, embeds=[error_embed("Usage: `/link handle:<handle>`.")])
        return

    info = await CF.get_user_info(handle)
    canonical = str(info.get("handle") or handle)
    synced = await _sync_handle(canonical, info=info)
    await storage.upsert_user(
        user_id,
        synced["handle"],
        synced["current_rating"],
        int(time.time()),
    )

    rated_contests = synced["rated_contests"]
    cold_start = rated_contests < estimator.MIN_RATED_CONTESTS
    result = embed(
        f"🔗 Linked @{canonical}",
        f"Your Codeforces account is connected.\n\n"
        f"I found **{len(synced['solved'])}** solved problems and "
        f"**{rated_contests}** rated contest"
        f"{'' if rated_contests == 1 else 's'} in your history.",
        color=SUCCESS_COLOR,
    )
    result["fields"].extend(
        [
            field("Profile", f"[{canonical}]({profile_url(canonical)})", inline=True),
            field(
                "Current rating",
                f"{synced['current_rating']}"
                + ("" if info.get("rating") else " (unrated default)"),
                inline=True,
            ),
            field(
                "Curve source",
                (
                    "Cold-start fallback (needs 5+ rated contests)"
                    if cold_start
                    else "Personal contest history"
                ),
                inline=True,
            ),
            field("💡 Tip", COLD_START_TIP),
            field("Next step", "Run `/ihave minutes:30` to get a problem that fits."),
        ]
    )
    await edit_original(interaction, embeds=[result])


async def cmd_ihave(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    options = _options(interaction)
    try:
        minutes = int(options.get("minutes"))
    except (TypeError, ValueError):
        await edit_original(interaction, embeds=[error_embed("`minutes` must be a number.")])
        return
    if minutes < MIN_SESSION_MINUTES or minutes > MAX_SESSION_MINUTES:
        await edit_original(
            interaction,
            embeds=[
                error_embed(
                    f"Choose a time between **{MIN_SESSION_MINUTES}** and "
                    f"**{MAX_SESSION_MINUTES}** minutes."
                )
            ],
        )
        return

    user = await storage.get_user(user_id) if user_id is not None else None
    if user is None:
        await edit_original(
            interaction,
            embeds=[error_embed("You haven't linked a handle yet. Run `/link` first.")],
        )
        return

    rating_override = options.get("rating")
    target_rating: int | None = None
    if rating_override is not None:
        try:
            target_rating = estimator.normalize_rating(int(rating_override))
        except (TypeError, ValueError):
            await edit_original(
                interaction,
                embeds=[error_embed("`rating` must be an integer between 800 and 3500.")],
            )
            return

    existing = await storage.get_session_by_user(user_id)
    if existing is not None:
        if not existing.get("expired"):
            await edit_original(
                interaction,
                embeds=[
                    error_embed(
                        f"You already have an active session on "
                        f"**{existing['problem_name']}** with "
                        f"**{format_seconds(_session_remaining(existing))}** left.\n"
                        "Finish it, run `/giveup`, or wait for the timer."
                    )
                ],
            )
            return
        await storage.delete_session(str(existing["id"]))

    handle = str(user["cf_handle"])
    current_rating = estimator.normalize_rating(user.get("current_rating"))

    await storage.ensure_problemset(CF)

    # Quick sync so a problem solved moments ago is not recommended.
    recent_submissions = await CF.get_user_status(handle, count=25)
    recently_solved = estimator.extract_solved_pairs(recent_submissions)
    if recently_solved:
        await storage.add_solved_many(handle, recently_solved)
    baseline_ids = []
    for submission in recent_submissions:
        submission_id = submission.get("id")
        if submission_id is None:
            continue
        try:
            baseline_ids.append(int(submission_id))
        except (TypeError, ValueError):
            continue

    rows = await storage.get_curve(handle)
    if not rows:
        synced = await _sync_handle(handle)
        rows = synced["rows"]
        current_rating = estimator.normalize_rating(synced["current_rating"])
        await storage.upsert_user(user_id, str(synced["handle"]), current_rating)

    curve = {
        int(row["rating"]): float(row["smoothed_minutes"])
        for row in rows
        if row.get("smoothed_minutes") is not None
    }
    if target_rating is None:
        target_rating = estimator.recommend_rating(curve, minutes, current_rating)

    pick = await storage.find_unsolved_problem(handle, target_rating)
    if pick is None:
        await edit_original(
            interaction,
            embeds=[
                error_embed(
                    "I couldn't find an unsolved problem near that rating. "
                    "Run `/sync` and try again."
                )
            ],
        )
        return

    started_at = int(time.time())
    expires_at = started_at + minutes * 60
    pick_rating = int(pick["rating"])
    url = problem_url(int(pick["contest_id"]), str(pick["index_name"]))
    expected = curve.get(pick_rating, float(minutes))

    session = await storage.create_session(
        user_id,
        channel_id=int(interaction.get("channel_id") or 0),
        handle=handle,
        contest_id=int(pick["contest_id"]),
        index_name=str(pick["index_name"]),
        problem_name=str(pick["name"]),
        rating=pick_rating,
        tags=list(pick["tags"]),
        started_at=started_at,
        expires_at=expires_at,
        baseline_ids=baseline_ids,
        expired=False,
    )

    result = embed(
        pick["name"],
        f"<@{user_id}> your **{minutes}-minute** challenge is ready.\n\n"
        f"**Problem:** [{pick['name']}]({url})\n"
        f"**Contest:** {pick['contest_id']} · Problem {pick['index_name']}",
        url=url,
    )
    result["fields"].extend(
        [
            field("⏱ Time limit", f"**{minutes} min** · ends <t:{expires_at}:R>", inline=True),
            field("🎯 Rating", f"||{pick_rating}||", inline=True),
            field("📈 Expected from your curve", f"~{expected:.1f} min", inline=True),
            field("🔒 Tags", "Hidden until you solve it or run `/giveup`."),
            field(
                "📡 Tracking",
                "No background poller in serverless mode. Run `/status` to check "
                "for Accepted; automatic checks need the optional cron endpoint.",
            ),
        ]
    )
    result["footer"] = {
        "text": (
            "Vercel interactions · manual /status checks"
            if not CRON_SECRET
            else "Vercel interactions · cron polling available"
        )
    }
    await edit_original(
        interaction,
        content=f"<@{user_id}>",
        embeds=[result],
        components=_feedback_components(str(session["id"])),
    )


async def cmd_status(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    session = await storage.get_session_by_user(user_id) if user_id is not None else None
    if session is None:
        await edit_original(
            interaction,
            embeds=[
                embed(
                    "No active session",
                    "Start one with `/ihave minutes:30`.\n"
                    "Not linked yet? Run `/link`.",
                    color=MUTED_COLOR,
                )
            ],
        )
        return

    submissions: list[dict[str, Any]] = []
    cf_error: str | None = None
    try:
        submissions = await CF.get_user_status(str(session["handle"]), count=25)
    except cf_api.CFError as exc:
        cf_error = str(exc)

    if submissions:
        solved = estimator.extract_solved_pairs(submissions)
        if solved:
            await storage.add_solved_many(str(session["handle"]), solved)
        accepted = _find_new_accepted(session, submissions)
        if accepted is not None:
            await _finish_accepted(interaction, session, accepted)
            return

    remaining = _session_remaining(session)
    if remaining <= 0:
        await _mark_timeout(session)
        timeout = _timeout_embed(session)
        if cf_error:
            timeout["fields"].append(field("⚠️ Codeforces", cf_error))
        await edit_original(interaction, embeds=[timeout], components=[])
        return

    url = problem_url(int(session["contest_id"]), str(session["index_name"]))
    result = embed(
        f"🎯 {session['problem_name']}",
        f"[{session['problem_name']}]({url}) · "
        f"{session['contest_id']}{session['index_name']}",
        url=url,
    )
    result["fields"].extend(
        [
            field("Rating", f"||{session['rating']}||", inline=True),
            field("Time limit", f"{_session_limit_minutes(session)} min", inline=True),
            field("Remaining", f"**{format_seconds(remaining)}**", inline=True),
            field("Elapsed", format_seconds(int(time.time()) - int(session["started_at"])), inline=True),
        ]
    )
    if cf_error:
        result["fields"].append(field("⚠️ Codeforces check failed", cf_error))
    await edit_original(interaction, embeds=[result])


async def cmd_giveup(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    session = await storage.get_session_by_user(user_id) if user_id is not None else None
    if session is None:
        await edit_original(
            interaction,
            embeds=[error_embed("You don't have an active session to give up on.")],
        )
        return

    url = problem_url(int(session["contest_id"]), str(session["index_name"]))
    tags = ", ".join(session.get("tags") or []) or "no tags"
    elapsed = min(
        max(0, int(time.time()) - int(session["started_at"])),
        _session_limit_minutes(session) * 60,
    )
    result = embed(
        f"🏳️ Revealed — {session['problem_name']}",
        f"You gave up on **[{session['problem_name']}]({url})** after "
        f"**{format_seconds(elapsed)}**.\n\n"
        f"**Problem:** {session['contest_id']} · {session['index_name']}",
        color=WARNING_COLOR,
        url=url,
    )
    result["fields"].extend(
        [
            field("Rating", f"||{session['rating']}||", inline=True),
            field("Time limit", f"{_session_limit_minutes(session)} min", inline=True),
            field("Tags", f"||{tags}||"),
        ]
    )
    await storage.delete_session(str(session["id"]))
    await edit_original(interaction, embeds=[result], components=[])


async def cmd_adjust(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    user = await storage.get_user(user_id) if user_id is not None else None
    if user is None:
        await edit_original(
            interaction,
            embeds=[error_embed("Link a handle first with `/link`.")],
        )
        return

    options = _options(interaction)
    try:
        rating = int(options.get("rating"))
        minutes = float(options.get("minutes"))
    except (TypeError, ValueError):
        await edit_original(
            interaction,
            embeds=[error_embed("`rating` and `minutes` must be numbers.")],
        )
        return
    if not 800 <= rating <= 3500:
        await edit_original(
            interaction,
            embeds=[error_embed("Rating must be between 800 and 3500.")],
        )
        return
    if not 1 <= minutes <= 600:
        await edit_original(
            interaction,
            embeds=[error_embed("Minutes must be between 1 and 600.")],
        )
        return

    previous, updated = await storage.adjust_curve(
        str(user["cf_handle"]),
        rating,
        minutes,
        current_rating=estimator.normalize_rating(user.get("current_rating")),
    )
    result = embed(
        "🛠 Curve recalibrated",
        f"Recorded: you can solve **{estimator.normalize_rating(rating)}** in "
        f"about **{minutes:g} minutes**.\n"
        f"Your smoothed estimate for that rating is now **{updated:.1f} min** "
        f"(was {previous:.1f} min).\n\n"
        "This override persists through `/sync`.",
        color=SUCCESS_COLOR,
    )
    await edit_original(interaction, embeds=[result])


async def cmd_sync(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    user = await storage.get_user(user_id) if user_id is not None else None
    if user is None:
        await edit_original(
            interaction,
            embeds=[error_embed("Link a handle first with `/link`.")],
        )
        return

    info = await CF.get_user_info(str(user["cf_handle"]))
    synced = await _sync_handle(str(user["cf_handle"]), info=info)
    await storage.upsert_user(
        user_id,
        str(synced["handle"]),
        synced["current_rating"],
        int(time.time()),
    )
    problems_refreshed = await storage.ensure_problemset(CF)
    solved_count = await storage.get_solved_count(str(synced["handle"]))

    result = embed(
        f"🔄 Synced @{synced['handle']}",
        f"Scanned **{synced['submission_count']}** submissions and "
        f"**{synced['rated_contests']}** rated contest"
        f"{'' if synced['rated_contests'] == 1 else 's'}.\n"
        f"**{solved_count}** solved problems are now on record.",
        color=SUCCESS_COLOR,
    )
    result["fields"].extend(
        [
            field("Rating", str(synced["current_rating"]), inline=True),
            field(
                "Problemset cache",
                "refreshed just now" if problems_refreshed else "already fresh",
                inline=True,
            ),
        ]
    )
    if synced["rated_contests"] < estimator.MIN_RATED_CONTESTS:
        result["fields"].append(
            field(
                "Cold start",
                "Fewer than 5 rated contests found, so the fallback curve is "
                "still in use.",
            )
        )
    await edit_original(interaction, embeds=[result])


async def cmd_curve(interaction: dict[str, Any]) -> None:
    user_id = _interaction_user_id(interaction)
    requested = _options(interaction).get("handle")
    canonical: str | None = None
    rows: list[dict[str, Any]] = []

    if requested:
        handle = str(requested).strip().lstrip("@")
        canonical = await storage.resolve_handle(handle)
        if canonical:
            rows = await storage.get_curve(canonical)
        if not rows:
            info = await CF.get_user_info(handle)
            synced = await _sync_handle(str(info.get("handle") or handle), info=info)
            canonical = str(synced["handle"])
            rows = synced["rows"]
    else:
        user = await storage.get_user(user_id) if user_id is not None else None
        if user is None:
            await edit_original(
                interaction,
                embeds=[error_embed("Link a handle first, or pass `handle:` to `/curve`.")],
            )
            return
        canonical = str(user["cf_handle"])
        rows = await storage.get_curve(canonical)
        if not rows:
            synced = await _sync_handle(canonical)
            canonical = str(synced["handle"])
            rows = synced["rows"]

    if not canonical or not rows:
        await edit_original(
            interaction,
            embeds=[error_embed("I couldn't build a curve for that handle.")],
        )
        return

    safe_name = "".join(ch for ch in canonical if ch.isalnum() or ch in "-_") or "user"
    filename = f"curve_{safe_name}.png"
    # Imported lazily: matplotlib is heavy and must not delay the 3-second
    # Discord acknowledgement on cold starts.
    import plotter

    buffer = await asyncio.to_thread(plotter.generate_curve_plot, canonical, rows)
    result = _curve_summary(canonical, rows)
    result["image"] = {"url": f"attachment://{filename}"}
    await edit_original(
        interaction,
        embeds=[result],
        file=(filename, buffer.getvalue()),
    )


_HANDLERS = {
    "help": cmd_help,
    "link": cmd_link,
    "ihave": cmd_ihave,
    "status": cmd_status,
    "giveup": cmd_giveup,
    "adjust": cmd_adjust,
    "sync": cmd_sync,
    "curve": cmd_curve,
}

_EPHEMERAL_COMMANDS = {"help", "link", "sync", "adjust"}


async def _handle_command(interaction: dict[str, Any]) -> None:
    data = interaction.get("data")
    name = str(data.get("name") or "") if isinstance(data, dict) else ""
    handler = _HANDLERS.get(name)
    if handler is None:
        await edit_original(interaction, embeds=[error_embed("Unknown command.")])
        return
    try:
        await handler(interaction)
    except cf_api.CFError as exc:
        await edit_original(interaction, embeds=[error_embed(str(exc))])
    except storage.StorageError as exc:
        await edit_original(interaction, embeds=[error_embed(str(exc))])
    except Exception:
        log.exception("Slash command /%s failed", name)
        await edit_original(
            interaction,
            embeds=[error_embed("An unexpected error occurred. Check the Vercel logs.")],
        )


async def _handle_component(interaction: dict[str, Any]) -> dict[str, Any]:
    data = interaction.get("data")
    custom_id = str(data.get("custom_id") or "") if isinstance(data, dict) else ""
    parts = custom_id.split(":")
    if len(parts) != 3 or parts[0] != "feedback" or parts[1] not in {"easy", "hard"}:
        return {
            "type": 4,
            "data": {"flags": 64, "embeds": [error_embed("Unknown component.")]},
        }

    session_id = parts[2]
    session = await storage.get_session(session_id)
    if session is None:
        return {
            "type": 4,
            "data": {
                "flags": 64,
                "embeds": [error_embed("This session is no longer active.")],
            },
        }
    if _interaction_user_id(interaction) != int(session.get("user_id") or 0):
        return {
            "type": 4,
            "data": {
                "flags": 64,
                "embeds": [error_embed("Only the session owner can use these buttons.")],
            },
        }
    if not await storage.claim_feedback(session_id):
        return {
            "type": 4,
            "data": {
                "flags": 64,
                "embeds": [error_embed("Feedback for this session was already recorded.")],
            },
        }

    factor = 0.85 if parts[1] == "easy" else 1.20
    label = "too easy" if parts[1] == "easy" else "too hard"
    try:
        previous, updated = await storage.scale_curve_rating(
            str(session["handle"]),
            int(session["rating"]),
            factor,
            current_rating=estimator.DEFAULT_RATING,
        )
    except Exception:
        await storage.release_feedback(session_id)
        log.exception("Feedback adjustment failed")
        return {
            "type": 4,
            "data": {
                "flags": 64,
                "embeds": [error_embed("I couldn't save that feedback. Try again.")],
            },
        }

    change = "decreased by 15%" if factor < 1 else "increased by 20%"
    return {
        "type": 4,
        "data": {
            "flags": 64,
            "embeds": [
                embed(
                    "📈 Feedback noted",
                    f"**{label}** — your estimated time at **{session['rating']}** "
                    f"went from **{previous:.1f} min** to **{updated:.1f} min** "
                    f"({change}, with the curve kept monotonic).",
                    color=SUCCESS_COLOR,
                )
            ],
        },
    }


# ---------------------------------------------------------------------------
# Optional cron poller
# ---------------------------------------------------------------------------


async def _poll_active_sessions() -> int:
    sessions = await storage.list_active_sessions()
    checked = 0
    for session in sessions:
        if session.get("expired"):
            await storage.remove_from_active(str(session["id"]))
            continue
        try:
            submissions = await CF.get_user_status(str(session["handle"]), count=25)
        except cf_api.CFError as exc:
            log.warning("Cron poll failed for %s: %s", session.get("handle"), exc)
            continue
        checked += 1

        solved = estimator.extract_solved_pairs(submissions)
        if solved:
            await storage.add_solved_many(str(session["handle"]), solved)

        accepted = _find_new_accepted(session, submissions)
        if accepted is not None:
            elapsed = _accepted_elapsed(session, accepted)
            await storage.mark_solved(
                str(session["handle"]),
                int(session["contest_id"]),
                str(session["index_name"]),
            )
            await storage.delete_session(str(session["id"]))
            await send_channel_message(
                int(session["channel_id"]),
                {
                    "content": f"<@{session['user_id']}>",
                    "embeds": [_accepted_embed(session, elapsed)],
                },
            )
            continue

        if _session_remaining(session) <= 0:
            await _mark_timeout(session)
            await send_channel_message(
                int(session["channel_id"]),
                {
                    "content": f"<@{session['user_id']}>",
                    "embeds": [_timeout_embed(session)],
                },
            )
    return checked


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await CF.close()
    await storage.close()
    if _DISCORD_HTTP is not None and not _DISCORD_HTTP.closed:
        await _DISCORD_HTTP.close()


app = FastAPI(title="ihave interactions", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "ihave", "mode": "discord-http-interactions"}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/api/interactions")
async def interactions(request: Request) -> Response:
    body = await request.body()
    signature = request.headers.get("X-Signature-Ed25519", "")
    timestamp = request.headers.get("X-Signature-Timestamp", "")
    if not verify_signature(body, signature, timestamp):
        return Response(status_code=401)

    try:
        interaction = json.loads(body)
    except json.JSONDecodeError:
        return Response(status_code=400)
    if not isinstance(interaction, dict):
        return Response(status_code=400)

    interaction_type = interaction.get("type")
    if interaction_type == 1:  # PING
        return JSONResponse({"type": 1})

    if interaction_type == 2:  # APPLICATION_COMMAND
        data = interaction.get("data")
        name = str(data.get("name") or "") if isinstance(data, dict) else ""
        if name not in _HANDLERS:
            return JSONResponse(
                {
                    "type": 4,
                    "data": {
                        "flags": 64,
                        "embeds": [error_embed(f"Unknown command `/{name}`.")],
                    },
                }
            )
        _schedule(_handle_command(interaction))
        response: dict[str, Any] = {"type": 5}
        if name in _EPHEMERAL_COMMANDS:
            response["data"] = {"flags": 64}
        return JSONResponse(response)

    if interaction_type == 3:  # MESSAGE_COMPONENT
        payload = await _handle_component(interaction)
        return JSONResponse(payload)

    return JSONResponse({"type": 1})


@app.get("/api/cron/poll")
async def cron_poll(request: Request) -> Response:
    if not CRON_SECRET:
        return JSONResponse(
            {"error": "CRON_SECRET is not configured on this deployment."},
            status_code=503,
        )
    authorization = request.headers.get("authorization", "")
    if authorization != f"Bearer {CRON_SECRET}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    checked = await _poll_active_sessions()
    return JSONResponse({"ok": True, "checked": checked})


__all__ = ["app", "verify_signature"]
