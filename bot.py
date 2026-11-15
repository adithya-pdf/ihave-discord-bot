#!/usr/bin/env python3
"""IhHave -- a Codeforces Rating-Space Elo-Survival practice bot for Discord.

The bot links a Codeforces handle, estimates how quickly the user solves
problems as a function of rating, archetype, era and contest pressure, serves a
time-boxed unsolved problem, follows the attempt (early solve, overtime solve,
right-censored give-up) and refits after every completed attempt.

Run with ``python bot.py`` after setting ``DISCORD_TOKEN`` in ``.env``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

import cf_data
import db as db_module
import predictor

load_dotenv()

log = logging.getLogger("ihave")

PREFIX = ";"
DEFAULT_CF_HANDLE = (os.getenv("DEFAULT_CF_HANDLE") or "").strip().lstrip("@") or None
DATA_DIR = (os.getenv("BOT_DATA_DIR") or ".").strip() or "."

POLL_INTERVAL_SECONDS = 30
POLLS_PER_TICK = cf_data.POLLS_PER_TICK
MIN_SESSION_MINUTES = 5
MAX_SESSION_MINUTES = 180
LINK_COOLDOWN_SECONDS = 600.0
CHECK_COOLDOWN_SECONDS = 10.0
EXPIRED_SESSION_WEIGHT = 0.5
SHAPE_REFIT_EVERY = 10
MIN_CONFIDENCE = 0.25
MAX_CONFIDENCE = 0.90
CLASSIC_ERA_Y = 7.0 / 7.5

EMBED_COLOR = 0x4C72B0
SUCCESS_COLOR = 0x2E8B57
WARNING_COLOR = 0xD9A400
ERROR_COLOR = 0xC0392B
MUTED_COLOR = 0x5A5A6E
FOOTER_TEXT = "IhHave · Rating-Space Elo-Survival"

TOPIC_CHOICES: list[str] = ["Ad-Hoc", "Math", "DS", "Graphs", "DP", "Weakest"]
TIME_TOPIC_CHOICES: list[str] = ["Ad-Hoc", "Math", "DS", "Graphs", "DP"]
TOPIC_TO_INDEX: dict[str, int] = {
    "Ad-Hoc": 0,
    "Math": 1,
    "DS": 2,
    "Graphs": 3,
    "DP": 4,
}
ERA_CHOICES: list[str] = ["Modern", "Classic"]
MODE_CHOICES: list[str] = ["Practice", "Contest"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def problem_url(contest_id: int, index: str) -> str:
    return f"https://codeforces.com/contest/{int(contest_id)}/problem/{index}"


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


def signed(value: float, decimals: int = 0) -> str:
    return f"{value:+.{decimals}f}"


def error_embed(message: str) -> discord.Embed:
    return discord.Embed(title="⚠️ Something went wrong", description=message, color=ERROR_COLOR)


def _relative_timestamp(epoch_seconds: float) -> str:
    moment = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    return discord.utils.format_dt(moment, style="R")


def _topic_names(archetypes: list[int]) -> str:
    if not archetypes:
        return "None"
    return ", ".join(predictor.ARCHETYPE_NAMES[index] for index in archetypes)


def _delta_label(value: float) -> str:
    if value > 0:
        return "Slower / Weak"
    if value < 0:
        return "Faster / Strong"
    return "Neutral"


def _normalise_topic(topic: str | None) -> str | None:
    """Accept the slash choices plus common prefix-command spellings."""
    if not topic:
        return None
    text = str(topic).strip()
    if not text:
        return None
    lowered = text.lower()
    for name in TOPIC_CHOICES:
        if lowered == name.lower():
            return name
    aliases = {
        "adhoc": "Ad-Hoc",
        "ad-hoc": "Ad-Hoc",
        "ad hoc": "Ad-Hoc",
        "greedy": "Ad-Hoc",
        "math/nt": "Math",
        "nt": "Math",
        "number theory": "Math",
        "data structures": "DS",
        "search": "DS",
        "graph": "Graphs",
        "graphs": "Graphs",
        "trees": "Graphs",
        "graphs/trees": "Graphs",
        "dp": "DP",
        "bitmasks": "DP",
        "weakest": "Weakest",
        "weak": "Weakest",
    }
    return aliases.get(lowered)


def _normalise_choice(value: str, choices: list[str], default: str) -> str:
    if not value:
        return default
    lowered = str(value).strip().lower()
    for choice in choices:
        if lowered == choice.lower():
            return choice
    return default


# ---------------------------------------------------------------------------
# Prefix-command interaction adapter
# ---------------------------------------------------------------------------
#
# The command implementations are written once against the slash-command
# interaction surface.  These adapters let the legacy prefix commands reuse
# exactly the same code paths (with no defer semantics, because a prefix
# message can be edited at any time).


class _CtxResponse:
    def __init__(self, ctx: commands.Context) -> None:
        self.ctx = ctx

    async def defer(self, *, ephemeral: bool = False, thinking: bool = False) -> None:
        return None

    def is_done(self) -> bool:
        return True

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        view: discord.ui.View | None = None,
        ephemeral: bool = False,
        **_: Any,
    ) -> discord.Message:
        return await self.ctx.send(content=content, embed=embed, view=view)


class _CtxFollowup:
    def __init__(self, ctx: commands.Context) -> None:
        self.ctx = ctx

    async def send(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        view: discord.ui.View | None = None,
        ephemeral: bool = False,
        wait: bool = False,
        **_: Any,
    ) -> discord.Message:
        return await self.ctx.send(content=content, embed=embed, view=view)


class _CtxInteraction:
    """Minimal stand-in for :class:`discord.Interaction` backed by a Context."""

    def __init__(self, ctx: commands.Context) -> None:
        self.ctx = ctx
        self.user = ctx.author
        self.channel = ctx.channel
        self.guild = ctx.guild
        self.message: discord.Message | None = None
        self.response = _CtxResponse(ctx)
        self.followup = _CtxFollowup(ctx)

    async def edit_original_response(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        view: discord.ui.View | None = None,
    ) -> discord.Message:
        return await self.ctx.send(content=content, embed=embed, view=view)


# ---------------------------------------------------------------------------
# UI views
# ---------------------------------------------------------------------------


class PracticeView(discord.ui.View):
    """Persistent per-session controls (timeout=None, fixed custom ids)."""

    def __init__(self, bot: "IhHaveBot") -> None:
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(
        label="✅ Check AC Now",
        style=discord.ButtonStyle.primary,
        custom_id="ihave:check_ac",
    )
    async def check_ac(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.bot.on_check_ac(interaction)

    @discord.ui.button(
        label="🏳️ Give Up",
        style=discord.ButtonStyle.danger,
        custom_id="ihave:give_up",
    )
    async def give_up(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.bot.on_give_up_button(interaction)

    @discord.ui.button(
        label="✖️ Ignore",
        style=discord.ButtonStyle.secondary,
        custom_id="ihave:cancel",
    )
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.bot.on_cancel_button(interaction)


class ConfirmGiveUpView(discord.ui.View):
    """Ephemeral early-give-up confirmation."""

    def __init__(self, bot: "IhHaveBot") -> None:
        super().__init__(timeout=120)
        self.bot = bot

    @discord.ui.button(
        label="Confirm Give Up",
        style=discord.ButtonStyle.danger,
        custom_id="ihave:confirm_give_up",
    )
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.bot.on_confirm_give_up(interaction)

    @discord.ui.button(
        label="Cancel",
        style=discord.ButtonStyle.secondary,
        custom_id="ihave:abort_give_up",
    )
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.edit_message(
            content="Session continues — keep grinding!", view=None
        )


class ActiveSessionCancelView(discord.ui.View):
    """Ephemeral Cancel button shown when /practice finds an active session."""

    def __init__(self, bot: "IhHaveBot") -> None:
        super().__init__(timeout=120)
        self.bot = bot

    @discord.ui.button(
        label="Cancel Active Session",
        style=discord.ButtonStyle.danger,
        custom_id="ihave:cancel_active",
    )
    async def cancel_active(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.bot.on_cancel_active_button(interaction)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class IhHaveBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        # The legacy prefix commands need message content.  Privileged member
        # and presence intents stay off to keep the gateway cache tiny.
        intents.message_content = True
        intents.members = False
        intents.presences = False
        super().__init__(
            command_prefix=commands.when_mentioned_or(PREFIX),
            intents=intents,
            help_command=None,
            case_insensitive=True,
            max_messages=None,
            member_cache_flags=discord.MemberCacheFlags.none(),
            chunk_guilds_at_startup=False,
            allowed_mentions=discord.AllowedMentions(
                everyone=False, roles=False, users=True, replied_user=True
            ),
        )
        os.makedirs(DATA_DIR, exist_ok=True)
        self.db = db_module.Database()
        self.user_locks: dict[int, asyncio.Lock] = {}
        self.last_link_ok: dict[int, float] = {}
        self.check_cooldowns: dict[int, float] = {}
        self._link_semaphore = asyncio.Semaphore(1)
        self._problems_cache: list[cf_data.CFProblem] | None = None
        self._problems_lock = asyncio.Lock()
        self.contest_starts: dict[int, int] | None = None

    # -- lifecycle -----------------------------------------------------------

    async def setup_hook(self) -> None:
        await self.add_cog(IhHaveCog(self))
        self.add_view(PracticeView(self))
        self.poller.start()
        try:
            await self.tree.sync()
            log.info("Slash commands synced")
        except Exception:
            log.warning("Could not sync slash commands", exc_info=True)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")

    async def close(self) -> None:
        try:
            self.poller.cancel()
        except Exception:  # pragma: no cover - defensive
            pass
        self.db.close()
        await super().close()

    def user_lock(self, discord_id: int) -> asyncio.Lock:
        lock = self.user_locks.get(int(discord_id))
        if lock is None:
            lock = asyncio.Lock()
            self.user_locks[int(discord_id)] = lock
        return lock

    # -- error / notification helpers ---------------------------------------

    async def _reply_error(self, interaction: Any, message: str) -> None:
        embed = error_embed(message)
        try:
            if interaction.response.is_done():
                await interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await interaction.response.send_message(embed=embed, ephemeral=True)
        except discord.HTTPException:
            log.debug("Could not deliver an error message", exc_info=True)

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        """Friendly handling for the legacy prefix commands."""
        if isinstance(error, commands.CommandNotFound):
            return
        original = getattr(error, "original", error)
        if isinstance(original, cf_data.CFApiError):
            await ctx.send(embed=error_embed(original.message))
            return
        if isinstance(error, commands.MissingRequiredArgument):
            name = ctx.command.qualified_name if ctx.command else ""
            usage = getattr(ctx.command, "usage", "") or ""
            await ctx.send(
                embed=error_embed(f"Missing argument.\nUsage: `{ctx.prefix}{name} {usage}`")
            )
            return
        if isinstance(error, commands.BadArgument):
            name = ctx.command.qualified_name if ctx.command else ""
            usage = getattr(ctx.command, "usage", "") or ""
            await ctx.send(
                embed=error_embed(
                    f"That argument didn't make sense: {error}\n"
                    f"Usage: `{ctx.prefix}{name} {usage}`"
                )
            )
            return
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(
                embed=error_embed(f"Slow down a little — try again in {error.retry_after:.1f}s.")
            )
            return
        log.error(
            "Unhandled prefix command error in %s",
            ctx.command,
            exc_info=(type(original), original, original.__traceback__),
        )
        try:
            await ctx.send(
                embed=error_embed(
                    "An unexpected error occurred. Check the bot logs for details."
                )
            )
        except discord.HTTPException:
            pass

    async def _notify(
        self,
        discord_id: int,
        channel_id: int | None,
        *,
        content: str | None = None,
        embed: discord.Embed | None = None,
    ) -> None:
        """Send in the original channel, falling back to a DM."""
        channel: Any = None
        if channel_id:
            channel = self.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self.fetch_channel(int(channel_id))
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    channel = None
        if channel is not None:
            try:
                await channel.send(content=content, embed=embed)
                return
            except (discord.Forbidden, discord.HTTPException):
                log.warning("Could not post in channel %s; falling back to DM", channel_id)
        user = self.get_user(int(discord_id))
        if user is None:
            try:
                user = await self.fetch_user(int(discord_id))
            except (discord.NotFound, discord.HTTPException):
                return
        try:
            await user.send(content=content, embed=embed)
        except (discord.Forbidden, discord.HTTPException):
            log.warning("Could not DM user %s", discord_id)

    async def _disable_session_message(self, channel_id: int | None, message_id: int | None) -> None:
        if not channel_id or not message_id:
            return
        channel = self.get_channel(int(channel_id))
        if channel is None:
            return
        try:
            await channel.get_partial_message(int(message_id)).edit(view=None)
        except (discord.Forbidden, discord.HTTPException, AttributeError):
            pass

    # -- Cached resources ----------------------------------------------------

    async def load_problems(self) -> list[cf_data.CFProblem]:
        async with self._problems_lock:
            if self._problems_cache is None:
                self._problems_cache = await asyncio.to_thread(cf_data.fetch_problemset)
            return self._problems_cache

    async def load_contest_starts(self) -> dict[int, int]:
        if self.contest_starts is None:
            try:
                self.contest_starts = await asyncio.to_thread(cf_data.fetch_contest_starts)
            except cf_data.CFApiError as error:
                log.warning("Contest start times unavailable: %s", error)
                self.contest_starts = {}
        return self.contest_starts

    # -- Model loading / refitting (always off the event loop) --------------

    def _load_predictor_blocking(self, discord_id: int) -> predictor.CFPredictor | None:
        user = self.db.get_user(int(discord_id))
        if user is None:
            return None
        base_prior = float(user.get("base_prior") or 1800.0)
        state = self.db.get_model_state(int(discord_id))
        model = predictor.CFPredictor.from_state(state, base_prior=base_prior)
        if state is None:
            history = self.db.load_history(int(discord_id), time.time())
            model.fit(history, base_prior=base_prior, fit_shape=True)
            shape_count = len(history) if model.shape_fit_performed else 0
            self.db.save_model_state(
                int(discord_id), model.to_state(), shape_fit_count=shape_count
            )
        return model

    async def load_predictor(self, discord_id: int) -> predictor.CFPredictor | None:
        return await asyncio.to_thread(self._load_predictor_blocking, int(discord_id))

    def _refit_blocking(self, discord_id: int, force_shape: bool = False) -> dict[str, Any] | None:
        user = self.db.get_user(int(discord_id))
        if user is None:
            return None
        base_prior = float(user.get("base_prior") or 1800.0)
        state = self.db.get_model_state(int(discord_id))
        model = predictor.CFPredictor.from_state(state, base_prior=base_prior)
        old_s30 = float(model.s30)
        history = self.db.load_history(int(discord_id), time.time())
        stored_count = int((state or {}).get("shape_fit_count") or 0)
        fit_shape = bool(force_shape or (len(history) - stored_count >= SHAPE_REFIT_EVERY))
        summary = model.fit(history, base_prior=base_prior, fit_shape=fit_shape)
        new_count = len(history) if model.shape_fit_performed else stored_count
        self.db.save_model_state(
            int(discord_id), model.to_state(), shape_fit_count=new_count
        )
        return {
            "predictor": model,
            "old_s30": old_s30,
            "new_s30": float(model.s30),
            "n_records": len(history),
            "summary": summary,
        }

    async def refit(self, discord_id: int, force_shape: bool = False) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._refit_blocking, int(discord_id), bool(force_shape))

    # -- Attempt polling -----------------------------------------------------

    def _find_ac(
        self,
        session: dict[str, Any],
        submissions: list[dict[str, Any]] | None,
    ) -> tuple[int, int] | None:
        """Return ``(ac_timestamp, wrong_count_before_ac)`` for a session."""
        if not submissions:
            return None
        started_at = int(session.get("started_at") or 0)
        contest_id = int(session.get("contest_id") or 0)
        problem_index = str(session.get("problem_index") or "")
        problem_name = str(session.get("problem_name") or "")
        matching: list[tuple[int, str | None]] = []
        for submission in submissions:
            try:
                timestamp = int(submission.get("creationTimeSeconds") or 0)
            except (TypeError, ValueError):
                continue
            if timestamp < started_at:
                continue
            problem = submission.get("problem")
            if not isinstance(problem, dict):
                problem = {}
            submission_contest = submission.get("contestId")
            if submission_contest is None:
                submission_contest = problem.get("contestId")
            try:
                same_key = int(submission_contest) == contest_id and str(
                    problem.get("index") or ""
                ) == problem_index
            except (TypeError, ValueError):
                same_key = False
            same_name = bool(problem_name) and str(problem.get("name") or "") == problem_name
            if not (same_key or same_name):
                continue
            verdict = submission.get("verdict")
            matching.append((timestamp, str(verdict) if verdict is not None else None))
        matching.sort(key=lambda item: item[0])
        wrong = 0
        for timestamp, verdict in matching:
            if verdict == "OK":
                return timestamp, wrong
            if verdict in cf_data.WRONG_VERDICTS:
                wrong += 1
        return None

    async def _poll_session_submissions(
        self, session: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        discord_id = int(session["discord_id"])
        user = self.db.get_user(discord_id)
        if user is None or not user.get("cf_handle"):
            return None
        try:
            return await asyncio.to_thread(
                cf_data.sync_recent_status,
                self.db,
                discord_id,
                str(user["cf_handle"]),
            )
        except cf_data.CFApiError as error:
            log.warning("Polling %s failed: %s", user.get("cf_handle"), error)
            return None

    async def _record_attempt(
        self,
        session: dict[str, Any],
        *,
        solved: bool,
        actual_mins: float,
        was_count: int,
        weight_mult: float = 1.0,
        occurred_at: int | None = None,
    ) -> dict[str, Any] | None:
        discord_id = int(session["discord_id"])
        payload = {
            "contest_id": int(session["contest_id"]),
            "problem_index": str(session["problem_index"]),
            "problem_name": session.get("problem_name"),
            "rating": session.get("rating"),
            "tags": list(session.get("tags") or []),
            "budget_mins": session.get("budget_mins"),
            "actual_mins": float(actual_mins),
            "was_count": int(was_count),
            "solved": bool(solved),
            "weight_mult": float(weight_mult),
            "contest_start_ts": None,
            "occurred_at": int(occurred_at) if occurred_at is not None else int(time.time()),
        }
        inserted = await asyncio.to_thread(
            self.db.complete_session, discord_id, payload
        )
        if not inserted:
            return None
        refit = await self.refit(discord_id)
        return {"refit": refit, "payload": payload}

    # -- Embeds --------------------------------------------------------------

    def _effective_modern(self, model: predictor.CFPredictor, contest_id: int, rating: float) -> float:
        start = self.contest_starts.get(contest_id) if self.contest_starts else None
        era = predictor.get_era_factor(contest_id, start)
        return float(rating) - model.gamma * era

    def _attempt_embed(
        self,
        session: dict[str, Any],
        result: dict[str, Any],
        *,
        solved: bool,
        actual_mins: float,
        was_count: int,
        overtime: bool = False,
    ) -> discord.Embed:
        discord_id = int(session["discord_id"])
        name = str(session.get("problem_name") or "problem")
        contest_id = int(session.get("contest_id") or 0)
        index = str(session.get("problem_index") or "")
        url = problem_url(contest_id, index)
        tags = list(session.get("tags") or [])
        refit = result.get("refit") or {}
        model = refit.get("predictor")

        if solved:
            title = f"✅ Overtime Solve — {name}" if overtime else f"✅ Solved — {name}"
            color = SUCCESS_COLOR
            intro = (
                f"<@{discord_id}> posted an **overtime AC** at "
                f"**{predictor.format_minutes(actual_mins)}** — it still counts as an "
                f"exact solve."
                if overtime
                else f"<@{discord_id}> solved **{name}** in "
                f"**{predictor.format_minutes(actual_mins)}**."
            )
        else:
            title = f"⏳ Timeout logged — {name}"
            color = WARNING_COLOR
            intro = (
                f"<@{discord_id}> stopped **{name}** after "
                f"**{predictor.format_minutes(actual_mins)}** without an AC. "
                f"Logged as a right-censored observation."
            )
        embed = discord.Embed(title=title, url=url, color=color, description=intro)
        embed.add_field(
            name="Actual Time",
            value=f"{actual_mins:.1f} min ({predictor.format_minutes(actual_mins)})",
            inline=True,
        )
        embed.add_field(name="WAs", value=str(int(was_count)), inline=True)
        embed.add_field(
            name="Contest-equiv. Time",
            value=predictor.format_minutes(
                actual_mins + predictor.CONTEST_WA_PENALTY_MINS * int(was_count)
            ),
            inline=True,
        )
        embed.add_field(name="Nominal Rating", value=str(int(session.get("rating") or 0)), inline=True)
        if model is not None:
            effective = self._effective_modern(model, contest_id, float(session.get("rating") or 0))
            embed.add_field(
                name="Effective Modern Rating", value=f"{effective:.0f}", inline=True
            )
            old_s30 = float((refit or {}).get("old_s30", model.s30))
            new_s30 = float((refit or {}).get("new_s30", model.s30))
            embed.add_field(
                name="S30",
                value=f"{old_s30:.0f} → {new_s30:.0f} [{new_s30 - old_s30:+.0f}]",
                inline=True,
            )
        embed.add_field(name="Tags", value=", ".join(tags) if tags else "none", inline=False)
        embed.set_footer(text=FOOTER_TEXT)
        return embed

    # -- Background poller ---------------------------------------------------

    @tasks.loop(seconds=POLL_INTERVAL_SECONDS)
    async def poller(self) -> None:
        if not self.is_ready():
            return
        try:
            sessions = await asyncio.to_thread(
                self.db.list_sessions_oldest_polled_first, POLLS_PER_TICK
            )
        except Exception:
            log.exception("Could not list active sessions")
            return
        for session in sessions:
            discord_id = int(session["discord_id"])
            try:
                await asyncio.to_thread(self.db.touch_polled, discord_id)
                await self._poll_one_session(session)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Poller failed for user %s", discord_id)

    async def _poll_one_session(self, session: dict[str, Any]) -> None:
        discord_id = int(session["discord_id"])
        channel_id = session.get("channel_id")
        try:
            started_at = int(session.get("started_at") or 0)
        except (TypeError, ValueError):
            started_at = 0
        budget = float(session.get("budget_mins") or 0.0)
        elapsed = max(0.0, (time.time() - started_at) / 60.0)

        submissions = await self._poll_session_submissions(session)
        accepted = self._find_ac(session, submissions)
        if accepted is not None:
            ac_timestamp, wrong = accepted
            actual = max(0.0, (ac_timestamp - started_at) / 60.0)
            result = await self._record_attempt(
                session,
                solved=True,
                actual_mins=actual,
                was_count=wrong,
                occurred_at=ac_timestamp,
            )
            if result is not None:
                embed = self._attempt_embed(
                    session,
                    result,
                    solved=True,
                    actual_mins=actual,
                    was_count=wrong,
                    overtime=actual > budget,
                )
                await self._notify(discord_id, channel_id, content=f"<@{discord_id}>", embed=embed)
                await self._disable_session_message(channel_id, session.get("message_id"))
            return

        if elapsed >= budget and not int(session.get("timer_notified") or 0):
            await asyncio.to_thread(self.db.mark_timer_notified, discord_id)
            embed = discord.Embed(
                title="⏰ Time's up",
                color=WARNING_COLOR,
                description=(
                    f"<@{discord_id}> your **{budget:.0f}m** timer is up! "
                    "Keep grinding if you're close — an overtime AC still counts as an "
                    "exact solve for the model — or click **Give Up** to log a timeout."
                ),
            )
            embed.set_footer(text=FOOTER_TEXT)
            await self._notify(discord_id, channel_id, content=f"<@{discord_id}>", embed=embed)

        expiry_limit = max(budget * 3.0, 180.0)
        if elapsed <= expiry_limit:
            return

        if EXPIRED_SESSION_WEIGHT > 0:
            result = await self._record_attempt(
                session,
                solved=False,
                actual_mins=budget,
                was_count=0,
                weight_mult=EXPIRED_SESSION_WEIGHT,
                occurred_at=started_at + int(budget * 60.0),
            )
            if result is not None:
                embed = self._attempt_embed(
                    session,
                    result,
                    solved=False,
                    actual_mins=budget,
                    was_count=0,
                )
                embed.title = "⌛ Session auto-expired"
                embed.description = (
                    f"<@{discord_id}> this session expired after "
                    f"**{elapsed:.0f}m**. Logged as a right-censored timeout at your "
                    f"**{budget:.0f}m** budget."
                )
                await self._notify(discord_id, channel_id, content=f"<@{discord_id}>", embed=embed)
        else:
            await asyncio.to_thread(self.db.cancel_session, discord_id)
            embed = discord.Embed(
                title="⌛ Session auto-expired",
                color=MUTED_COLOR,
                description=(
                    f"<@{discord_id}> this session expired after **{elapsed:.0f}m** and "
                    "was discarded without logging an attempt."
                ),
            )
            embed.set_footer(text=FOOTER_TEXT)
            await self._notify(discord_id, channel_id, content=f"<@{discord_id}>", embed=embed)
        await self._disable_session_message(channel_id, session.get("message_id"))

    # -- Button handlers -----------------------------------------------------

    async def _session_for_message(self, interaction: discord.Interaction) -> dict[str, Any] | None:
        if interaction.message is None:
            return None
        return self.db.get_session_by_message(int(interaction.message.id))

    async def on_check_ac(self, interaction: discord.Interaction) -> None:
        session = await self._session_for_message(interaction)
        if session is None:
            await interaction.response.send_message(
                "This session has already ended.", ephemeral=True
            )
            return
        if int(session["discord_id"]) != int(interaction.user.id):
            await interaction.response.send_message(
                "Only the user who started this session can use these buttons.",
                ephemeral=True,
            )
            return
        now = time.monotonic()
        last = self.check_cooldowns.get(int(interaction.user.id))
        if last is not None and now - last < CHECK_COOLDOWN_SECONDS:
            remaining = CHECK_COOLDOWN_SECONDS - (now - last)
            await interaction.response.send_message(
                f"Hold on — try again in {remaining:.0f}s.", ephemeral=True
            )
            return
        self.check_cooldowns[int(interaction.user.id)] = now

        await interaction.response.defer(ephemeral=True)
        submissions = await self._poll_session_submissions(session)
        if submissions is None:
            await interaction.followup.send(
                "Could not reach Codeforces just now. Try the button again shortly.",
                ephemeral=True,
            )
            return
        accepted = self._find_ac(session, submissions)
        if accepted is None:
            await interaction.followup.send("No AC detected yet.", ephemeral=True)
            return

        ac_timestamp, wrong = accepted
        started_at = int(session.get("started_at") or 0)
        actual = max(0.0, (ac_timestamp - started_at) / 60.0)
        result = await self._record_attempt(
            session,
            solved=True,
            actual_mins=actual,
            was_count=wrong,
            occurred_at=ac_timestamp,
        )
        if result is None:
            await interaction.followup.send(
                "This session was already completed.", ephemeral=True
            )
            return
        embed = self._attempt_embed(
            session,
            result,
            solved=True,
            actual_mins=actual,
            was_count=wrong,
            overtime=actual > float(session.get("budget_mins") or 0.0),
        )
        await self._notify(
            int(session["discord_id"]),
            session.get("channel_id"),
            content=f"<@{int(session['discord_id'])}>",
            embed=embed,
        )
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))
        await interaction.followup.send("✅ AC detected and logged.", ephemeral=True)

    async def on_give_up_button(self, interaction: discord.Interaction) -> None:
        session = await self._session_for_message(interaction)
        if session is None:
            await interaction.response.send_message(
                "This session has already ended.", ephemeral=True
            )
            return
        if int(session["discord_id"]) != int(interaction.user.id):
            await interaction.response.send_message(
                "Only the user who started this session can use these buttons.",
                ephemeral=True,
            )
            return
        started_at = int(session.get("started_at") or 0)
        budget = float(session.get("budget_mins") or 0.0)
        elapsed = max(0.0, (time.time() - started_at) / 60.0)
        if elapsed < 0.20 * budget:
            await interaction.response.send_message(
                "Giving up this early? You can **Confirm Give Up**, or press "
                "**Cancel** / **Ignore** to keep grinding.",
                view=ConfirmGiveUpView(self),
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        await self._finalize_give_up(interaction, session, None)

    async def on_confirm_give_up(self, interaction: discord.Interaction) -> None:
        session = self.db.get_session(int(interaction.user.id))
        if session is None:
            await interaction.response.send_message(
                "This session has already ended.", ephemeral=True
            )
            return
        if int(session["discord_id"]) != int(interaction.user.id):
            await interaction.response.send_message(
                "Only the user who started this session can use these buttons.",
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        await self._finalize_give_up(interaction, session, None)

    async def on_cancel_button(self, interaction: discord.Interaction) -> None:
        session = await self._session_for_message(interaction)
        if session is None:
            await interaction.response.send_message(
                "This session has already ended.", ephemeral=True
            )
            return
        if int(session["discord_id"]) != int(interaction.user.id):
            await interaction.response.send_message(
                "Only the user who started this session can use these buttons.",
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        cancelled = await asyncio.to_thread(self.db.cancel_session, int(session["discord_id"]))
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))
        if cancelled:
            await interaction.followup.send("Session cancelled — nothing was logged.")
        else:
            await interaction.followup.send("This session had already ended.")

    async def on_cancel_active_button(self, interaction: discord.Interaction) -> None:
        session = self.db.get_session(int(interaction.user.id))
        if session is None:
            await interaction.response.send_message(
                "You do not have an active session.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.db.cancel_session, int(interaction.user.id))
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))
        await interaction.followup.send("Active session cancelled.", ephemeral=True)

    async def _finalize_give_up(
        self,
        interaction: Any,
        session: dict[str, Any],
        requested_mins: float | None,
    ) -> None:
        submissions = await self._poll_session_submissions(session)
        accepted = self._find_ac(session, submissions)
        if accepted is not None:
            ac_timestamp, wrong = accepted
            started_at = int(session.get("started_at") or 0)
            actual = max(0.0, (ac_timestamp - started_at) / 60.0)
            result = await self._record_attempt(
                session,
                solved=True,
                actual_mins=actual,
                was_count=wrong,
                occurred_at=ac_timestamp,
            )
            if result is None:
                await interaction.followup.send("This session was already completed.")
                return
            embed = self._attempt_embed(
                session,
                result,
                solved=True,
                actual_mins=actual,
                was_count=wrong,
                overtime=actual > float(session.get("budget_mins") or 0.0),
            )
            await interaction.followup.send(embed=embed)
            await self._disable_session_message(session.get("channel_id"), session.get("message_id"))
            return

        started_at = int(session.get("started_at") or 0)
        elapsed = max(0.0, (time.time() - started_at) / 60.0)
        actual = float(requested_mins) if requested_mins is not None else max(elapsed, 1.0)
        result = await self._record_attempt(
            session, solved=False, actual_mins=max(actual, 1.0), was_count=0
        )
        if result is None:
            await interaction.followup.send("This session was already completed.")
            return
        embed = self._attempt_embed(
            session,
            result,
            solved=False,
            actual_mins=max(actual, 1.0),
            was_count=0,
        )
        await interaction.followup.send(embed=embed)
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))

    # -- Command implementations (shared by slash + prefix) -----------------

    async def cmd_link(self, interaction: Any, handle: str) -> None:
        discord_id = int(interaction.user.id)
        now = time.monotonic()
        last = self.last_link_ok.get(discord_id)
        if last is not None and now - last < LINK_COOLDOWN_SECONDS:
            remaining = LINK_COOLDOWN_SECONDS - (now - last)
            await interaction.response.send_message(
                f"`/link` is on cooldown. Try again in {int(remaining // 60) + 1} minutes.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()
        if self._link_semaphore.locked():
            try:
                await interaction.edit_original_response(
                    content="Waiting for another /link scan to finish..."
                )
            except discord.HTTPException:
                pass
        async with self._link_semaphore:
            info = await asyncio.to_thread(cf_data.fetch_user_info, handle)
            canonical = str(info.get("handle") or handle)
            base_prior = float(cf_data.fetch_user_rating(canonical, info=info))

            existing = self.db.get_user(discord_id)
            if (
                existing is not None
                and existing.get("cf_handle")
                and str(existing["cf_handle"]).lower() != canonical.lower()
            ):
                await asyncio.to_thread(self.db.clear_user_cf_data, discord_id)
            await asyncio.to_thread(
                self.db.upsert_user, discord_id, canonical, base_prior
            )

            scan = await asyncio.to_thread(
                cf_data.scan_user_history_for_link, self.db, discord_id, canonical
            )
            await asyncio.to_thread(
                self.db.add_bootstrap_rows, discord_id, scan.bootstrap_rows
            )
            self._problems_cache = scan.problems
            self.contest_starts = scan.contest_starts
            refit = await self.refit(discord_id, force_shape=True)
            self.last_link_ok[discord_id] = time.monotonic()

        model = (refit or {}).get("predictor")
        solved_rows = sum(1 for row in scan.bootstrap_rows if row.solved)
        timeout_rows = len(scan.bootstrap_rows) - solved_rows
        solved_total = await asyncio.to_thread(self.db.get_solved_count, discord_id)
        if model is None:
            await interaction.followup.send(embed=error_embed("Could not fit the model."))
            return

        embed = discord.Embed(
            title=f"🔗 Linked @{canonical}",
            url=profile_url(canonical),
            color=SUCCESS_COLOR,
            description=(
                f"Your Codeforces account is connected.\n"
                f"Scanned **{scan.scanned_submissions}** submissions across "
                f"**{scan.rated_contests}** rated contests and "
                f"**{scan.virtual_contests}** virtual contests."
            ),
        )
        embed.add_field(
            name="Contest bootstrap",
            value=(
                f"{solved_rows} timed solves + {timeout_rows} end-of-contest timeouts\n"
                f"{len(scan.bootstrap_rows)} rows derived (idempotent on re-link)"
            ),
            inline=False,
        )
        embed.add_field(name="Solved on record", value=str(solved_total), inline=True)
        embed.add_field(
            name="S30",
            value=f"{model.s30:.0f} ± {float(model.stderr['s30']):.0f}",
            inline=True,
        )
        embed.add_field(
            name="Era discount (γ)",
            value=f"{model.gamma:.0f} pts @ 7.5 yrs",
            inline=True,
        )
        embed.add_field(
            name="Archetype handicaps",
            value="\n".join(
                f"{predictor.ARCHETYPE_NAMES[index]}: {model.delta[index]:+.0f} ± "
                f"{float(model.stderr['delta'][index]):.0f}"
                for index in range(predictor.N_ARCHETYPES)
            ),
            inline=False,
        )
        embed.add_field(
            name="Contest offset",
            value=f"{model.contest_offset:+.0f} pts (± {float(model.stderr['contest_offset']):.0f})",
            inline=True,
        )
        embed.add_field(
            name="Projected 40m live performance",
            value=f"{model.predict_modern_rating(40.0, 0.65, contest=True):.0f}",
            inline=True,
        )
        embed.add_field(
            name="Next step",
            value="Run `/practice 25` or `/times` to explore your curve.",
            inline=False,
        )
        embed.set_footer(text=FOOTER_TEXT)
        await interaction.followup.send(embed=embed)

    async def cmd_practice(
        self,
        interaction: Any,
        minutes: int,
        confidence: float = 0.65,
        topic: str | None = None,
        blind: bool = True,
    ) -> None:
        discord_id = int(interaction.user.id)
        topic = _normalise_topic(topic)
        if not (MIN_SESSION_MINUTES <= int(minutes) <= MAX_SESSION_MINUTES):
            await interaction.response.send_message(
                f"Choose a time between **{MIN_SESSION_MINUTES}** and "
                f"**{MAX_SESSION_MINUTES}** minutes.",
                ephemeral=True,
            )
            return
        if not (MIN_CONFIDENCE <= float(confidence) <= MAX_CONFIDENCE):
            await interaction.response.send_message(
                f"Confidence must be between **{MIN_CONFIDENCE:.2f}** and "
                f"**{MAX_CONFIDENCE:.2f}**.",
                ephemeral=True,
            )
            return

        lock = self.user_lock(discord_id)
        if lock.locked():
            await interaction.response.send_message(
                "A /practice is already in progress.", ephemeral=True
            )
            return
        async with lock:
            user = self.db.get_user(discord_id)
            if user is None:
                await interaction.response.send_message(
                    "Run `/link <handle>` first.", ephemeral=True
                )
                return
            active = self.db.get_session(discord_id)
            if active is not None:
                await interaction.response.send_message(
                    f"You already have an active session on "
                    f"**{active.get('problem_name') or 'a problem'}**. Finish it, or "
                    f"cancel it below.",
                    view=ActiveSessionCancelView(self),
                    ephemeral=True,
                )
                return

            await interaction.response.defer()

            handle = str(user.get("cf_handle") or "")
            try:
                await asyncio.to_thread(
                    cf_data.sync_recent_status, self.db, discord_id, handle
                )
            except cf_data.CFApiError as error:
                log.warning("Incremental sync failed for %s: %s", handle, error)

            model = await self.load_predictor(discord_id)
            if model is None:
                await interaction.followup.send(embed=error_embed("Run `/link <handle>` first."))
                return
            solved_keys, solved_names = await asyncio.to_thread(
                self.db.solved_keys_and_names, discord_id
            )
            problems = await self.load_problems()
            contest_starts = await self.load_contest_starts()

            target_archetype: int | None = None
            topic_note: str | None = None
            if topic == "Weakest":
                counts = await asyncio.to_thread(self.db.archetype_counts, discord_id)
                eligible = [
                    index
                    for index in range(predictor.N_ARCHETYPES)
                    if counts[index] >= 3
                ]
                if eligible:
                    target_archetype = max(eligible, key=lambda index: model.delta[index])
                    topic_note = (
                        f"Focusing on **{predictor.ARCHETYPE_NAMES[target_archetype]}** "
                        f"(your weakest archetype with enough history)."
                    )
                else:
                    topic_note = (
                        "Not enough per-topic history yet, so no topic filter was applied."
                    )
            elif topic in TOPIC_TO_INDEX:
                target_archetype = TOPIC_TO_INDEX[topic]

            pick = await asyncio.to_thread(
                cf_data.select_best_problem,
                model,
                problems,
                solved_keys,
                solved_names,
                float(minutes),
                float(confidence),
                target_archetype,
                None,
                bool(blind),
                contest_starts,
                None,
                random.Random(),
                False,
            )
            if pick is None:
                await interaction.followup.send(
                    embed=error_embed(
                        "I couldn't find an eligible unsolved problem near your target. "
                        "Try a different topic or rerun `/link`."
                    )
                )
                return

            started_at = int(time.time())
            ends_at = started_at + int(minutes) * 60
            url = problem_url(pick.contest_id, pick.index)
            start_ts = contest_starts.get(pick.contest_id)
            era = predictor.get_era_factor(pick.contest_id, start_ts)
            effective = float(pick.rating or 0) - model.gamma * era
            archetype_names = _topic_names(predictor.get_archetypes(pick.tags))

            def maybe_spoiler(value: str) -> str:
                return f"||{value}||" if blind else value

            embed = discord.Embed(
                title=f"🎯 {pick.name}",
                url=url,
                color=EMBED_COLOR,
                description=(
                    f"<@{discord_id}> your **{int(minutes)}-minute** challenge is ready.\n\n"
                    f"**Problem:** [{pick.name}]({url})\n"
                    f"**Contest:** {pick.contest_id} · Problem {pick.index}\n"
                    + (f"\n{topic_note}" if topic_note else "")
                ),
            )
            embed.add_field(
                name="⏱ Time budget",
                value=f"**{int(minutes)} min** · ends {_relative_timestamp(ends_at)}",
                inline=True,
            )
            embed.add_field(
                name="🎲 Confidence",
                value=f"{float(confidence):.0%}",
                inline=True,
            )
            embed.add_field(
                name="Nominal rating",
                value=maybe_spoiler(f"{int(pick.rating or 0)}"),
                inline=True,
            )
            embed.add_field(
                name="Effective modern rating",
                value=maybe_spoiler(f"{effective:.0f}"),
                inline=True,
            )
            embed.add_field(
                name="Archetype",
                value=maybe_spoiler(archetype_names),
                inline=True,
            )
            embed.add_field(
                name="Controls",
                value=(
                    "Use **Check AC Now** after an accepted submission, **Give Up** to "
                    "log a timeout, or **Ignore** to discard the session. Timer expiry "
                    "does not fail the session."
                ),
                inline=False,
            )
            embed.set_footer(text=FOOTER_TEXT)

            view = PracticeView(self)
            message = await interaction.followup.send(embed=embed, view=view, wait=True)
            try:
                await asyncio.to_thread(
                    self.db.create_session,
                    discord_id,
                    channel_id=int(interaction.channel.id),
                    message_id=int(message.id),
                    contest_id=int(pick.contest_id),
                    problem_index=str(pick.index),
                    problem_name=pick.name,
                    rating=int(pick.rating or 0),
                    tags=list(pick.tags),
                    budget_mins=float(minutes),
                    confidence=float(confidence),
                    blind_mode=bool(blind),
                    started_at=started_at,
                )
            except Exception:
                log.exception("Could not create the session row for %s", discord_id)
                try:
                    await message.delete()
                except (discord.HTTPException, AttributeError):
                    pass
                await interaction.followup.send(
                    embed=error_embed("Could not start the session. Please try again.")
                )

    async def cmd_done(
        self,
        interaction: Any,
        actual_mins: float | None = None,
        was: int | None = None,
    ) -> None:
        discord_id = int(interaction.user.id)
        if actual_mins is not None and not (0.0 < float(actual_mins) <= 720.0):
            await interaction.response.send_message(
                "`actual_mins` must be between 0 and 720.", ephemeral=True
            )
            return
        if was is not None and not (0 <= int(was) <= 50):
            await interaction.response.send_message(
                "`was` must be between 0 and 50.", ephemeral=True
            )
            return
        session = self.db.get_session(discord_id)
        if session is None:
            await interaction.response.send_message(
                "You do not have an active session.", ephemeral=True
            )
            return
        await interaction.response.defer()

        submissions = await self._poll_session_submissions(session)
        accepted = self._find_ac(session, submissions)
        started_at = int(session.get("started_at") or 0)
        elapsed = max(0.0, (time.time() - started_at) / 60.0)
        if actual_mins is not None:
            actual = float(actual_mins)
        elif accepted is not None:
            actual = max(0.0, (accepted[0] - started_at) / 60.0)
        else:
            actual = max(elapsed, 1.0)
        if was is not None:
            wrong = int(was)
        elif accepted is not None:
            wrong = int(accepted[1])
        else:
            wrong = 0

        result = await self._record_attempt(
            session,
            solved=True,
            actual_mins=actual,
            was_count=wrong,
            occurred_at=accepted[0] if accepted is not None else None,
        )
        if result is None:
            await interaction.followup.send("This session was already completed.")
            return
        embed = self._attempt_embed(
            session,
            result,
            solved=True,
            actual_mins=actual,
            was_count=wrong,
            overtime=actual > float(session.get("budget_mins") or 0.0),
        )
        await interaction.followup.send(embed=embed)
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))

    async def cmd_give_up(
        self, interaction: Any, actual_mins: float | None = None
    ) -> None:
        discord_id = int(interaction.user.id)
        if actual_mins is not None and not (0.0 < float(actual_mins) <= 720.0):
            await interaction.response.send_message(
                "`actual_mins` must be between 0 and 720.", ephemeral=True
            )
            return
        session = self.db.get_session(discord_id)
        if session is None:
            await interaction.response.send_message(
                "You do not have an active session.", ephemeral=True
            )
            return
        await interaction.response.defer()
        await self._finalize_give_up(interaction, session, actual_mins)

    async def cmd_cancel(self, interaction: Any) -> None:
        discord_id = int(interaction.user.id)
        session = self.db.get_session(discord_id)
        if session is None:
            await interaction.response.send_message(
                "You do not have an active session.", ephemeral=True
            )
            return
        await interaction.response.defer()
        cancelled = await asyncio.to_thread(self.db.cancel_session, discord_id)
        await self._disable_session_message(session.get("channel_id"), session.get("message_id"))
        if cancelled:
            await interaction.followup.send("Session cancelled — nothing was logged.")
        else:
            await interaction.followup.send("This session had already ended.")

    async def cmd_perf(self, interaction: Any) -> None:
        discord_id = int(interaction.user.id)
        user = self.db.get_user(discord_id)
        if user is None:
            await interaction.response.send_message(
                "Run `/link <handle>` first.", ephemeral=True
            )
            return
        await interaction.response.defer()
        model = await self.load_predictor(discord_id)
        if model is None:
            await interaction.followup.send(embed=error_embed("Run `/link <handle>` first."))
            return
        handle = str(user.get("cf_handle") or "")
        current_rating = int(user.get("base_prior") or 0)
        try:
            info = await asyncio.to_thread(cf_data.fetch_user_info, handle)
            current_rating = int(cf_data.fetch_user_rating(handle, info=info))
        except cf_data.CFApiError as error:
            log.warning("Could not refresh current rating for %s: %s", handle, error)

        counts = await asyncio.to_thread(self.db.archetype_counts, discord_id)
        readiness = model.predict_modern_rating(40.0, 0.65, contest=True)
        embed = discord.Embed(
            title=f"📊 Performance — @{handle}",
            url=profile_url(handle),
            color=EMBED_COLOR,
            description=(
                f"**Current CF rating:** {current_rating}\n"
                f"**Live Contest Readiness (40m @ 65%):** {readiness:.0f}\n"
                f"**S30:** {model.s30:.0f} ± {float(model.stderr['s30']):.0f}"
            ),
        )
        embed.add_field(
            name="Era discount (γ)",
            value=f"-{model.gamma:.0f} pts for problems >= {predictor.ERA_SPAN_YEARS:g} years old",
            inline=False,
        )
        lines = ["Budget          Modern65  Classic65  Modern50  Classic50"]
        for budget in (15, 25, 40, 60):
            modern_65 = model.predict_modern_rating(float(budget), 0.65)
            modern_50 = model.predict_modern_rating(float(budget), 0.50)
            classic_65 = modern_65 + model.gamma * CLASSIC_ERA_Y
            classic_50 = modern_50 + model.gamma * CLASSIC_ERA_Y
            label = f"{budget}m" + (" (Live)" if budget == 40 else "")
            lines.append(
                f"{label:<14}{modern_65:>9.0f}{classic_65:>11.0f}"
                f"{modern_50:>10.0f}{classic_50:>10.0f}"
            )
        embed.add_field(
            name="Time-budget table (modern vs Classic-era ~7 yrs old)",
            value="```\n" + "\n".join(lines) + "\n```",
            inline=False,
        )
        archetype_lines = []
        for index in range(predictor.N_ARCHETYPES):
            delta = model.delta[index]
            stderr = float(model.stderr["delta"][index])
            archetype_lines.append(
                f"{predictor.ARCHETYPE_NAMES[index]}: {delta:+.0f} ± {stderr:.0f} pts "
                f"[{_delta_label(delta)}] — {counts[index]} attempts"
            )
        embed.add_field(
            name="Archetype breakdown",
            value="\n".join(archetype_lines),
            inline=False,
        )
        embed.add_field(
            name="Learned contest offset",
            value=(
                f"{model.contest_offset:+.0f} pts (± "
                f"{float(model.stderr['contest_offset']):.0f})"
            ),
            inline=True,
        )
        embed.set_footer(text=FOOTER_TEXT)
        await interaction.followup.send(embed=embed)

    async def cmd_times(
        self,
        interaction: Any,
        rating: int | None = None,
        topic: str | None = None,
        era: str = "Modern",
        mode: str = "Practice",
    ) -> None:
        discord_id = int(interaction.user.id)
        topic = _normalise_topic(topic)
        era = _normalise_choice(era, ERA_CHOICES, "Modern")
        mode = _normalise_choice(mode, MODE_CHOICES, "Practice")
        user = self.db.get_user(discord_id)
        if user is None:
            await interaction.response.send_message(
                "Run `/link <handle>` first.", ephemeral=True
            )
            return
        if rating is not None and not (800 <= int(rating) <= 3500):
            await interaction.response.send_message(
                "`rating` must be between 800 and 3500.", ephemeral=True
            )
            return
        await interaction.response.defer()
        model = await self.load_predictor(discord_id)
        if model is None:
            await interaction.followup.send(embed=error_embed("Run `/link <handle>` first."))
            return
        attempts = await asyncio.to_thread(self.db.count_attempts, discord_id)
        era_y = CLASSIC_ERA_Y if era == "Classic" else 0.0
        contest = mode == "Contest"
        topic_index = TOPIC_TO_INDEX.get(topic) if topic else None
        stderr_s30 = float(model.stderr["s30"])
        footer = (
            f"Based on {attempts} attempts (S30 = {model.s30:.0f} ± {stderr_s30:.0f}); "
            f"rows far from S30 are extrapolations."
        )
        setting = f"{era.lower()} problems, {mode.lower()} conditions"
        if topic:
            setting += f", topic {topic}"

        if rating is None:
            center = int(round(model.s30 / 100.0) * 100)
            low = max(800, min(center - 600, 2300))
            ratings = [low + 100 * offset for offset in range(13)]
            lines = [
                f"{'Rating':<8}{'50% (Med)':>11}{'65% (Rel)':>11}"
                f"{'80% (Safe)':>12}{'P(<=25m)':>10}{'P(<=40m)':>10}"
            ]
            for row_rating in ratings:
                median = predictor.format_minutes(
                    model.predict_solve_time(
                        float(row_rating), 0.50, arch_idx=topic_index,
                        era_y=era_y, contest=contest
                    )
                )
                reliable = predictor.format_minutes(
                    model.predict_solve_time(
                        float(row_rating), 0.65, arch_idx=topic_index,
                        era_y=era_y, contest=contest
                    )
                )
                safe = predictor.format_minutes(
                    model.predict_solve_time(
                        float(row_rating), 0.80, arch_idx=topic_index,
                        era_y=era_y, contest=contest
                    )
                )
                p25 = model.predict_solve_probability(
                    float(row_rating), target_mins=25.0, arch_idx=topic_index,
                    era_y=era_y, contest=contest
                )
                p40 = model.predict_solve_probability(
                    float(row_rating), target_mins=40.0, arch_idx=topic_index,
                    era_y=era_y, contest=contest
                )
                lines.append(
                    f"{row_rating:<8}{median:>11}{reliable:>11}{safe:>12}"
                    f"{p25 * 100:>9.0f}%{p40 * 100:>9.0f}%"
                )
            embed = discord.Embed(
                title=f"⏱ Time needed by rating — @{user.get('cf_handle')}",
                color=EMBED_COLOR,
                description=(
                    f"13-row ladder centred on your S30 ({setting}).\n"
                    "```\n" + "\n".join(lines) + "\n```"
                ),
            )
            embed.set_footer(text=footer)
            await interaction.followup.send(embed=embed)
            return

        target = int(rating)
        archetypes: list[int | None] = [None, 0, 1, 2, 3, 4]
        row_names = ["Overall Baseline", *predictor.ARCHETYPE_NAMES]
        time_lines = [
            f"{'Row':<18}{'Handicap':>10}{'50% (Med)':>11}{'65% (Rel)':>11}{'80% (Safe)':>12}"
        ]
        probability_lines = [
            f"{'Row':<18}{'P(<=15m)':>10}{'P(<=25m)':>10}{'P(<=40m)':>10}{'P(<=60m)':>10}"
        ]
        for row_name, archetype in zip(row_names, archetypes):
            handicap = "—" if archetype is None else f"{model.delta[archetype]:+.0f}"
            median = predictor.format_minutes(
                model.predict_solve_time(
                    float(target), 0.50, arch_idx=archetype, era_y=era_y, contest=contest
                )
            )
            reliable = predictor.format_minutes(
                model.predict_solve_time(
                    float(target), 0.65, arch_idx=archetype, era_y=era_y, contest=contest
                )
            )
            safe = predictor.format_minutes(
                model.predict_solve_time(
                    float(target), 0.80, arch_idx=archetype, era_y=era_y, contest=contest
                )
            )
            time_lines.append(
                f"{row_name:<18}{handicap:>10}{median:>11}{reliable:>11}{safe:>12}"
            )
            probabilities = [
                model.predict_solve_probability(
                    float(target),
                    target_mins=float(minutes),
                    arch_idx=archetype,
                    era_y=era_y,
                    contest=contest,
                )
                for minutes in (15, 25, 40, 60)
            ]
            probability_lines.append(
                f"{row_name:<18}"
                + "".join(f"{value * 100:>9.0f}%" for value in probabilities)
            )
        embed = discord.Embed(
            title=f"⏱ Rating {target} — @{user.get('cf_handle')}",
            color=EMBED_COLOR,
            description=(
                f"**Time needed at confidence p** ({setting})\n"
                "```\n" + "\n".join(time_lines) + "\n```\n"
                f"**Solve probabilities** ({setting})\n"
                "```\n" + "\n".join(probability_lines) + "\n```"
            ),
        )
        embed.set_footer(text=footer)
        await interaction.followup.send(embed=embed)

    async def cmd_calibration(self, interaction: Any) -> None:
        discord_id = int(interaction.user.id)
        user = self.db.get_user(discord_id)
        if user is None:
            await interaction.response.send_message(
                "Run `/link <handle>` first.", ephemeral=True
            )
            return
        await interaction.response.defer()
        history = await asyncio.to_thread(
            self.db.load_history, discord_id, time.time()
        )
        min_train = 10
        if len(history) < min_train:
            await interaction.followup.send(
                f"Not enough data yet — need at least {min_train} logged attempts, "
                f"have {len(history)}."
            )
            return
        result = await asyncio.to_thread(predictor.walk_forward_backtest, history)
        if result.n == 0:
            await interaction.followup.send(
                "Not enough data yet — no completed practice sessions matched the "
                "backtest's eligibility rules."
            )
            return

        lines = [f"{'Bin':<12}{'Predicted':>10}{'Actual':>9}{'N':>6}"]
        for index, (predicted, actual, count) in enumerate(result.bins):
            if count == 0:
                continue
            label = f"{index / 10:.1f}-{(index + 1) / 10:.1f}"
            lines.append(
                f"{label:<12}{predicted:>10.2f}{actual:>9.2f}{count:>6}"
            )
        embed = discord.Embed(
            title=f"📈 Calibration — @{user.get('cf_handle')}",
            color=EMBED_COLOR,
            description=(
                f"**Log-loss:** {result.log_loss:.4f} "
                f"(coin-flip baseline 0.6931)\n"
                f"**Brier score:** {result.brier_score:.4f}\n"
                f"Evaluated on your last **{result.n}** practice attempts.\n"
                "```\n" + "\n".join(lines) + "\n```"
            ),
        )
        embed.set_footer(text=FOOTER_TEXT)
        await interaction.followup.send(embed=embed)

    async def cmd_help(self, interaction: Any) -> None:
        embed = discord.Embed(
            title="IhHave — Rating-Space Elo-Survival",
            color=EMBED_COLOR,
            description=(
                "I estimate the rating you can solve in a given time window, "
                "serve an unsolved problem, and learn from every attempt."
            ),
        )
        commands_list = [
            ("`/link <handle>`", "Link a Codeforces handle and bootstrap from contest history."),
            ("`/practice <minutes> [confidence] [topic] [blind]`", "Start a time-boxed session."),
            ("`/times [rating] [topic] [era] [mode]`", "Rating -> time quantile tables."),
            ("`/perf` (alias `/stats`)", "Readiness, era discount and archetype handicaps."),
            ("`/calibration`", "Walk-forward reliability on your own attempts."),
            ("`/done [actual_mins] [was]`", "Log the active session as solved."),
            ("`/giveup [actual_mins]`", "Log a right-censored give-up."),
            ("`/cancel`", "Discard the active session without logging."),
        ]
        for name, description in commands_list:
            embed.add_field(name=name, value=description, inline=False)
        embed.set_footer(text=FOOTER_TEXT)
        await interaction.response.send_message(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Commands (slash + legacy prefix)
# ---------------------------------------------------------------------------


def _app_guard(func: Any) -> Any:
    @functools.wraps(func)
    async def wrapper(self: "IhHaveCog", interaction: Any, *args: Any, **kwargs: Any) -> None:
        try:
            await func(self, interaction, *args, **kwargs)
        except cf_data.CFApiError as error:
            await self.bot._reply_error(interaction, error.message)
        except Exception:
            log.exception("Unhandled slash command error in %s", func.__name__)
            await self.bot._reply_error(
                interaction, "An unexpected error occurred. Check the bot logs."
            )

    return wrapper


class IhHaveCog(commands.Cog):
    """User-facing commands; the model/network work lives on the bot."""

    def __init__(self, bot: IhHaveBot) -> None:
        self.bot = bot

    # -- slash commands ------------------------------------------------------

    @app_commands.command(name="link", description="Link a Codeforces handle and bootstrap your model.")
    @app_commands.describe(handle="Codeforces handle, e.g. tourist")
    @_app_guard
    async def link(self, interaction: discord.Interaction, handle: str) -> None:
        await self.bot.cmd_link(interaction, handle)

    @app_commands.command(name="practice", description="Start a time-boxed practice session.")
    @app_commands.describe(
        minutes="Time budget in minutes (5-180)",
        confidence="Target solve probability (0.25-0.90)",
        topic="Optional topic filter; Weakest picks your weakest archetype",
        blind="Hide rating/archetype until the attempt ends",
    )
    @app_commands.choices(
        topic=[app_commands.Choice(name=name, value=name) for name in TOPIC_CHOICES]
    )
    @_app_guard
    async def practice(
        self,
        interaction: discord.Interaction,
        minutes: app_commands.Range[int, 5, 180],
        confidence: app_commands.Range[float, 0.25, 0.90] = 0.65,
        topic: str | None = None,
        blind: bool = True,
    ) -> None:
        await self.bot.cmd_practice(interaction, int(minutes), float(confidence), topic, blind)

    @app_commands.command(name="done", description="Log the active session as solved.")
    @app_commands.describe(actual_mins="Actual solve time in minutes", was="Wrong submissions before AC")
    @_app_guard
    async def done(
        self,
        interaction: discord.Interaction,
        actual_mins: float | None = None,
        was: int | None = None,
    ) -> None:
        await self.bot.cmd_done(interaction, actual_mins, was)

    @app_commands.command(name="giveup", description="Log the active session as a right-censored timeout.")
    @app_commands.describe(actual_mins="Time spent before giving up (defaults to elapsed)")
    @_app_guard
    async def giveup(
        self, interaction: discord.Interaction, actual_mins: float | None = None
    ) -> None:
        await self.bot.cmd_give_up(interaction, actual_mins)

    @app_commands.command(name="cancel", description="Discard the active session without logging it.")
    @_app_guard
    async def cancel(self, interaction: discord.Interaction) -> None:
        await self.bot.cmd_cancel(interaction)

    @app_commands.command(name="perf", description="Show readiness, handicaps and time-budget predictions.")
    @_app_guard
    async def perf(self, interaction: discord.Interaction) -> None:
        await self.bot.cmd_perf(interaction)

    @app_commands.command(name="stats", description="Alias for /perf.")
    @_app_guard
    async def stats(self, interaction: discord.Interaction) -> None:
        await self.bot.cmd_perf(interaction)

    @app_commands.command(name="times", description="Predict time needed at 50%, 65% and 80% confidence.")
    @app_commands.describe(
        rating="Target rating (omit for a 13-row ladder around your S30)",
        topic="Optional topic filter for the ladder view",
        era="Modern or Classic (~7 years old)",
        mode="Practice or live Contest conditions",
    )
    @app_commands.choices(
        topic=[app_commands.Choice(name=name, value=name) for name in TIME_TOPIC_CHOICES],
        era=[app_commands.Choice(name=name, value=name) for name in ERA_CHOICES],
        mode=[app_commands.Choice(name=name, value=name) for name in MODE_CHOICES],
    )
    @_app_guard
    async def times(
        self,
        interaction: discord.Interaction,
        rating: int | None = None,
        topic: str | None = None,
        era: str = "Modern",
        mode: str = "Practice",
    ) -> None:
        await self.bot.cmd_times(interaction, rating, topic, era, mode)

    @app_commands.command(name="calibration", description="Walk-forward reliability of your model.")
    @_app_guard
    async def calibration(self, interaction: discord.Interaction) -> None:
        await self.bot.cmd_calibration(interaction)

    @app_commands.command(name="help", description="Show the command reference.")
    @_app_guard
    async def help_command(self, interaction: discord.Interaction) -> None:
        await self.bot.cmd_help(interaction)

    # -- legacy prefix commands ---------------------------------------------

    @commands.command(name="link", usage="<handle>")
    async def link_prefix(self, ctx: commands.Context, handle: str | None = None) -> None:
        """Link a Codeforces handle."""
        target = (handle or DEFAULT_CF_HANDLE or "").strip().lstrip("@")
        if not target:
            await ctx.send(
                embed=error_embed(
                    "Usage: `;link <handle>` (or set `DEFAULT_CF_HANDLE` in `.env`)."
                )
            )
            return
        await self.bot.cmd_link(_CtxInteraction(ctx), target)

    @commands.command(name="practice", aliases=["ihave"], usage="<minutes> [confidence] [topic] [blind]")
    async def practice_prefix(
        self,
        ctx: commands.Context,
        minutes: int,
        confidence: float = 0.65,
        topic: str | None = None,
        blind: bool = True,
    ) -> None:
        """Start a time-boxed session."""
        await self.bot.cmd_practice(
            _CtxInteraction(ctx), minutes, confidence, topic, blind
        )

    @commands.command(name="done", usage="[actual_mins] [was]")
    async def done_prefix(
        self,
        ctx: commands.Context,
        actual_mins: float | None = None,
        was: int | None = None,
    ) -> None:
        """Log the active session as solved."""
        await self.bot.cmd_done(_CtxInteraction(ctx), actual_mins, was)

    @commands.command(name="giveup", usage="[actual_mins]")
    async def giveup_prefix(
        self, ctx: commands.Context, actual_mins: float | None = None
    ) -> None:
        """Log the active session as a timeout."""
        await self.bot.cmd_give_up(_CtxInteraction(ctx), actual_mins)

    @commands.command(name="cancel")
    async def cancel_prefix(self, ctx: commands.Context) -> None:
        """Discard the active session without logging it."""
        await self.bot.cmd_cancel(_CtxInteraction(ctx))

    @commands.command(name="perf", aliases=["stats"])
    async def perf_prefix(self, ctx: commands.Context) -> None:
        """Show readiness and handicaps."""
        await self.bot.cmd_perf(_CtxInteraction(ctx))

    @commands.command(name="times", usage="[rating] [topic] [era] [mode]")
    async def times_prefix(
        self,
        ctx: commands.Context,
        rating: int | None = None,
        topic: str | None = None,
        era: str = "Modern",
        mode: str = "Practice",
    ) -> None:
        """Rating -> time quantile tables."""
        await self.bot.cmd_times(_CtxInteraction(ctx), rating, topic, era, mode)

    @commands.command(name="calibration")
    async def calibration_prefix(self, ctx: commands.Context) -> None:
        """Walk-forward reliability."""
        await self.bot.cmd_calibration(_CtxInteraction(ctx))

    @commands.command(name="help")
    async def help_prefix(self, ctx: commands.Context) -> None:
        """Show the command reference."""
        await self.bot.cmd_help(_CtxInteraction(ctx))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    token = (os.getenv("DISCORD_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "DISCORD_TOKEN is not set. Copy .env.example to .env and paste your "
            "bot token (and enable the Message Content Intent in the Discord "
            "Developer Portal)."
        )
    if token == "your_discord_bot_token_here":
        raise SystemExit(
            "DISCORD_TOKEN is still the placeholder from .env.example. Open .env "
            "and replace `your_discord_bot_token_here` with the real bot token."
        )

    bot = IhHaveBot()
    try:
        bot.run(token, log_handler=None)
    except discord.LoginFailure as exc:
        raise SystemExit(
            "Discord rejected the bot token. Double-check DISCORD_TOKEN in .env."
        ) from exc
    except discord.PrivilegedIntentsRequired as exc:
        raise SystemExit(
            "The Message Content Intent is disabled for this bot. Open the Discord "
            "Developer Portal -> your application -> Bot -> Privileged Gateway "
            "Intents and enable Message Content Intent, then restart."
        ) from exc


if __name__ == "__main__":
    main()
