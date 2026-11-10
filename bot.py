#!/usr/bin/env python3
"""``;ihave`` -- a Codeforces time-boxed practice bot for Discord.

A single-process, asynchronous bot. It links a Codeforces handle to a Discord
user, learns how long that user needs per problem rating from their live and
virtual contest history, and then recommends a random unsolved problem that
fits a requested time budget.

Run with ``python bot.py`` after setting ``DISCORD_TOKEN`` in ``.env``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import discord
from discord.ext import commands
from dotenv import load_dotenv

import cf_api
import estimator
import plotter
import storage

load_dotenv()

log = logging.getLogger("ihave")

PREFIX = ";"
DEFAULT_CF_HANDLE = (os.getenv("DEFAULT_CF_HANDLE") or "").strip().lstrip("@") or None
POLL_INTERVAL_SECONDS = 60
MIN_SESSION_MINUTES = 5
MAX_SESSION_MINUTES = 180

EMBED_COLOR = 0x4C72B0
SUCCESS_COLOR = 0x2E8B57
WARNING_COLOR = 0xD9A400
ERROR_COLOR = 0xC0392B
MUTED_COLOR = 0x5A5A6E
FOOTER_TEXT = "Codeforces time-boxed practice"

COLD_START_TIP = (
    "Make sure to register for Virtual Contests before starting them! "
    "Unregistered practice submissions are ignored by the time-curve to keep "
    "your data accurate."
)


# ---------------------------------------------------------------------------
# Small formatting helpers
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


def error_embed(message: str) -> discord.Embed:
    return discord.Embed(title="⚠️ Something went wrong", description=message, color=ERROR_COLOR)


def _relative_timestamp(epoch_seconds: int) -> str:
    moment = datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc)
    return discord.utils.format_dt(moment, style="R")


# ---------------------------------------------------------------------------
# Active session state
# ---------------------------------------------------------------------------


@dataclass
class PracticeSession:
    user_id: int
    channel_id: int
    handle: str
    contest_id: int
    index_name: str
    problem_name: str
    rating: int
    tags: tuple[str, ...]
    started_at: int
    expires_at: int
    baseline_ids: set[int]
    expired: bool = False
    finished: bool = False
    message: discord.Message | None = None
    view: "PracticeView | None" = None
    task: asyncio.Task | None = None

    @property
    def limit_minutes(self) -> int:
        return max(1, (self.expires_at - self.started_at) // 60)

    @property
    def remaining_seconds(self) -> int:
        return max(0, self.expires_at - int(time.time()))

    @property
    def elapsed_seconds(self) -> int:
        return max(0, int(time.time()) - self.started_at)


# ---------------------------------------------------------------------------
# Feedback buttons
# ---------------------------------------------------------------------------


class PracticeView(discord.ui.View):
    """Per-session feedback buttons.

    The view keeps working after the timer expires (and after the session is
    retired) so the user can still tell the bot that the problem was too easy or
    too hard. Only one adjustment is accepted per session to avoid stacking
    accidental clicks.
    """

    def __init__(self, client: "IhHaveBot", *, user_id: int, handle: str, rating: int):
        super().__init__(timeout=None)
        self.client = client
        self.user_id = int(user_id)
        self.handle = handle
        self.rating = int(rating)
        self.recorded = False
        self.message: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this session can use these buttons.",
                ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(label="⚡ Too Easy", style=discord.ButtonStyle.secondary)
    async def too_easy(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._apply_feedback(interaction, factor=0.85, label="too easy")

    @discord.ui.button(label="😅 Too Hard", style=discord.ButtonStyle.danger)
    async def too_hard(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._apply_feedback(interaction, factor=1.20, label="too hard")

    async def _apply_feedback(
        self,
        interaction: discord.Interaction,
        *,
        factor: float,
        label: str,
    ) -> None:
        if self.recorded:
            await interaction.response.send_message(
                "Feedback for this session has already been recorded.",
                ephemeral=True,
            )
            return

        self.recorded = True
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

        try:
            previous, updated = await storage.scale_curve_rating(self.handle, self.rating, factor)
        except Exception:
            self.recorded = False
            for child in self.children:
                child.disabled = False  # type: ignore[attr-defined]
            log.exception("Could not store feedback for %s at %s", self.handle, self.rating)
            await interaction.response.send_message(
                "I couldn't save that feedback. Please try again in a moment.",
                ephemeral=True,
            )
            return

        if factor < 1:
            change = "decreased by 15%"
        else:
            change = "increased by 20%"
        await interaction.response.send_message(
            f"📈 Noted — **{label}**. Your estimated time at **{self.rating}** "
            f"went from **{previous:.1f} min** to **{updated:.1f} min** "
            f"({change}, with the rest of the curve kept monotonic)."
        )

        if interaction.message is not None:
            try:
                await interaction.message.edit(view=self)
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Commands (real Cog so discord.py checks/cooldowns work correctly)
# ---------------------------------------------------------------------------


class PracticeCog(commands.Cog):
    """Thin user-facing prefix commands.

    Networking, persistence and session tasks live on :class:`IhHaveBot`; the
    Cog exists so discord.py's check, cooldown and help machinery always sees a
    genuine :class:`commands.Cog` attached to each command.
    """

    def __init__(self, bot: "IhHaveBot") -> None:
        self.bot = bot

    @commands.command(name="link", usage="<handle>")
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def link(self, ctx: commands.Context, handle: str | None = None) -> None:
        """Link a Codeforces handle and build the initial time curve."""
        await self.bot._cmd_link(ctx, handle)

    @commands.command(name="ihave", usage="<minutes> [rating]")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def ihave(
        self,
        ctx: commands.Context,
        minutes: int,
        rating_override: int | None = None,
    ) -> None:
        """Start a time-boxed practice session."""
        await self.bot._cmd_ihave(ctx, minutes, rating_override)

    @commands.command(name="adjust", usage="<rating> <minutes>")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def adjust(self, ctx: commands.Context, rating: int, minutes: float) -> None:
        """Manually recalibrate the solve-time curve."""
        await self.bot._cmd_adjust(ctx, rating, minutes)

    @commands.command(name="curve", usage="[handle]")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def curve(self, ctx: commands.Context, handle: str | None = None) -> None:
        """Render the personal solve-time curve."""
        await self.bot._cmd_curve(ctx, handle)

    @commands.command(name="status")
    async def status(self, ctx: commands.Context) -> None:
        """Show the active session and remaining time."""
        await self.bot._cmd_status(ctx)

    @commands.command(name="giveup")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def giveup(self, ctx: commands.Context) -> None:
        """Cancel the session and reveal the hidden details."""
        await self.bot._cmd_giveup(ctx)

    @commands.command(name="sync", usage="")
    @commands.cooldown(1, 20, commands.BucketType.user)
    async def sync(self, ctx: commands.Context) -> None:
        """Refresh submissions, curve and the problemset cache."""
        await self.bot._cmd_sync(ctx)

    @commands.command(name="help")
    async def help_command(self, ctx: commands.Context) -> None:
        """Show the command reference."""
        await self.bot._cmd_help(ctx)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class IhHaveBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required for prefix commands
        super().__init__(
            command_prefix=commands.when_mentioned_or(PREFIX),
            intents=intents,
            help_command=None,
            case_insensitive=True,
            allowed_mentions=discord.AllowedMentions(
                everyone=False, roles=False, users=True, replied_user=True
            ),
        )
        self.cf = cf_api.CodeforcesClient()
        self.sessions: dict[int, PracticeSession] = {}
        self._background: set[asyncio.Task] = set()
        self._plot_lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    async def setup_hook(self) -> None:
        if self.get_cog("PracticeCog") is None:
            await self.add_cog(PracticeCog(self))
        await storage.init_db()
        await self.cf.start()
        log.info("SQLite and aiohttp are ready")
        self._spawn(self._warm_problemset_cache())

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")

    async def close(self) -> None:
        pending: list[asyncio.Task] = []
        current = asyncio.current_task()
        for session in list(self.sessions.values()):
            task = session.task
            if task is not None and task is not current and not task.done():
                task.cancel()
                pending.append(task)
        for task in list(self._background):
            if task is not current and not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.sessions.clear()
        await self.cf.close()
        await storage.close_db()
        await super().close()

    def _spawn(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def _warm_problemset_cache(self) -> None:
        try:
            refreshed = await storage.ensure_problemset(self.cf)
            log.info(
                "Problemset cache %s", "refreshed" if refreshed else "already fresh"
            )
        except Exception:
            log.warning("Could not warm the problemset cache", exc_info=True)

    # -- shared helpers ------------------------------------------------------

    async def _get_channel(self, channel_id: int) -> discord.abc.Messageable | None:
        channel = self.get_channel(channel_id)
        if channel is not None:
            return channel
        try:
            return await self.fetch_channel(channel_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return None

    async def _sync_handle(
        self,
        handle: str,
        *,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch a handle's full history and rebuild their curve."""
        info = info or await self.cf.get_user_info(handle)
        canonical = str(info.get("handle") or handle)
        raw_rating = info.get("rating") or info.get("maxRating")
        current_rating = estimator.normalize_rating(raw_rating or estimator.DEFAULT_RATING)

        submissions = await self.cf.get_user_status(canonical, count=10000)
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

    def _curve_embed(self, handle: str, rows: list[dict[str, Any]]) -> discord.Embed:
        embed = discord.Embed(
            title=f"📈 Solve-time curve — @{handle}",
            color=EMBED_COLOR,
            description=(
                "Estimated minutes needed to solve a problem at each rating.\n"
                "Solid line: smoothed personal estimator. "
                "Dashed line: raw per-rating averages."
            ),
        )
        lines: list[str] = []
        total_samples = 0
        for row in rows:
            try:
                rating = int(row["rating"])
                minutes = float(row["smoothed_minutes"])
            except (KeyError, TypeError, ValueError):
                continue
            total_samples += int(row.get("sample_size") or 0)
            lines.append(f"{rating:>4}  →  {minutes:>6.1f} min")

        table = "```\n" + "\n".join(lines) + "\n```"
        embed.add_field(name="Rating → expected minutes", value=table[:1024], inline=False)

        if total_samples == 0:
            embed.add_field(
                name="Cold start",
                value=(
                    "No rated contest solves were found, so the fallback curve "
                    "anchored around your current rating is shown. Register for "
                    "Virtual Contests to build a personal curve."
                ),
                inline=False,
            )
        else:
            embed.add_field(
                name="Observations",
                value=(
                    f"{total_samples} timed contest solves feed the raw averages. "
                    "If you have fewer than 5 rated contests, the rating-anchored "
                    "fallback is used instead."
                ),
                inline=False,
            )
        embed.set_footer(text=FOOTER_TEXT)
        return embed

    async def _announce_solved(
        self,
        session: PracticeSession,
        submission: dict[str, Any],
    ) -> None:
        session.finished = True
        await storage.mark_solved(session.handle, session.contest_id, session.index_name)

        try:
            accepted_at = int(submission.get("creationTimeSeconds") or time.time())
        except (TypeError, ValueError):
            accepted_at = int(time.time())
        elapsed = max(0, accepted_at - session.started_at)
        if elapsed > session.limit_minutes * 60:
            elapsed = session.limit_minutes * 60

        url = problem_url(session.contest_id, session.index_name)
        tags = ", ".join(session.tags) if session.tags else "no tags"
        embed = discord.Embed(
            title=f"✅ Accepted — {session.problem_name}",
            url=url,
            color=SUCCESS_COLOR,
            description=(
                f"<@{session.user_id}> solved **[{session.problem_name}]({url})** "
                f"in **{format_seconds(elapsed)}** "
                f"(limit: {session.limit_minutes} min). Well played!"
            ),
        )
        embed.add_field(name="Rating", value=f"||{session.rating}||", inline=True)
        embed.add_field(name="Tags", value=f"||{tags}||", inline=False)
        embed.set_footer(text=FOOTER_TEXT)

        channel = await self._get_channel(session.channel_id)
        if channel is not None:
            try:
                await channel.send(content=f"<@{session.user_id}>", embed=embed)
            except discord.HTTPException:
                log.warning("Could not announce accepted session for %s", session.user_id)

        await self._disable_view(session)
        self.sessions.pop(session.user_id, None)
        session.task = None

    async def _announce_timeout(self, session: PracticeSession) -> None:
        session.expired = True
        session.task = None

        url = problem_url(session.contest_id, session.index_name)
        embed = discord.Embed(
            title=f"⏰ Time's up — {session.problem_name}",
            url=url,
            color=WARNING_COLOR,
            description=(
                f"<@{session.user_id}> the **{session.limit_minutes}-minute** timer "
                f"ended without an Accepted submission.\n\n"
                f"Run `;giveup` to reveal the problem's rating and tags."
            ),
        )
        embed.set_footer(text=FOOTER_TEXT)

        channel = await self._get_channel(session.channel_id)
        if channel is not None:
            try:
                await channel.send(content=f"<@{session.user_id}>", embed=embed)
            except discord.HTTPException:
                log.warning("Could not announce timeout for %s", session.user_id)

    async def _disable_view(self, session: PracticeSession) -> None:
        view = session.view
        if view is None:
            return
        view.recorded = True
        for child in view.children:
            child.disabled = True  # type: ignore[attr-defined]
        if session.message is not None:
            try:
                await session.message.edit(view=view)
            except discord.HTTPException:
                pass

    async def _retire_session(self, session: PracticeSession) -> None:
        session.finished = True
        task, session.task = session.task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._disable_view(session)
        self.sessions.pop(session.user_id, None)

    # -- session polling -----------------------------------------------------

    def _find_new_accepted(
        self,
        session: PracticeSession,
        submissions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for submission in submissions:
            submission_id = submission.get("id")
            if submission_id is None:
                continue
            try:
                if int(submission_id) in session.baseline_ids:
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
            if contest_id != session.contest_id:
                continue
            if str(problem.get("index") or "") != session.index_name:
                continue
            return submission
        return None

    async def _monitor_session(self, session: PracticeSession) -> None:
        try:
            while not session.finished:
                remaining = session.expires_at - int(time.time())
                if remaining <= 0:
                    break
                await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
                if session.finished:
                    return

                try:
                    submissions = await self.cf.get_user_status(session.handle, count=25)
                except cf_api.CFError as exc:
                    log.warning("Session poll for %s failed: %s", session.handle, exc)
                    if int(time.time()) >= session.expires_at:
                        break
                    continue

                solved = estimator.extract_solved_pairs(submissions)
                if solved:
                    await storage.add_solved_many(session.handle, solved)

                accepted = self._find_new_accepted(session, submissions)
                if accepted is not None:
                    await self._announce_solved(session, accepted)
                    return

                if int(time.time()) >= session.expires_at:
                    break

            if not session.finished:
                await self._announce_timeout(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Session monitor crashed for %s", session.handle)
            session.finished = True
            self.sessions.pop(session.user_id, None)

    # -- commands ------------------------------------------------------------

    async def _cmd_link(self, ctx: commands.Context, handle: str | None = None) -> None:
        """Implementation for ``;link`` (see :class:`PracticeCog`)."""
        raw_handle = (handle or DEFAULT_CF_HANDLE or "").strip().lstrip("@")
        if not raw_handle:
            await ctx.send(
                embed=error_embed(
                    "Usage: `;link <handle>`\n"
                    "You can also set `DEFAULT_CF_HANDLE` in `.env` to make the "
                    "argument optional."
                )
            )
            return

        async with ctx.typing():
            info = await self.cf.get_user_info(raw_handle)
            canonical = str(info.get("handle") or raw_handle)
            synced = await self._sync_handle(canonical, info=info)
            await storage.upsert_user(
                ctx.author.id,
                canonical,
                synced["current_rating"],
                int(time.time()),
            )

        rated_contests = synced["rated_contests"]
        cold_start = rated_contests < estimator.MIN_RATED_CONTESTS
        embed = discord.Embed(
            title=f"🔗 Linked @{canonical}",
            color=SUCCESS_COLOR,
            description=(
                f"Your Codeforces account is connected.\n\n"
                f"I found **{len(synced['solved'])}** solved problems and "
                f"**{rated_contests}** rated contest"
                f"{'' if rated_contests == 1 else 's'} in your history."
            ),
        )
        embed.add_field(
            name="Profile",
            value=f"[{canonical}]({profile_url(canonical)})",
            inline=True,
        )
        embed.add_field(
            name="Current rating",
            value=(
                f"{synced['current_rating']}"
                + ("" if info.get("rating") else " (unrated default)")
            ),
            inline=True,
        )
        embed.add_field(
            name="Curve source",
            value=(
                "Cold-start fallback (needs 5+ rated contests)"
                if cold_start
                else "Personal contest history"
            ),
            inline=True,
        )
        embed.add_field(name="💡 Tip", value=COLD_START_TIP, inline=False)
        embed.add_field(
            name="Next step",
            value="Run `;ihave 30` to get a problem that fits a 30-minute session.",
            inline=False,
        )
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)

    async def _cmd_ihave(
        self,
        ctx: commands.Context,
        minutes: int,
        rating_override: int | None = None,
    ) -> None:
        """Implementation for ``;ihave`` (see :class:`PracticeCog`)."""
        if minutes < MIN_SESSION_MINUTES or minutes > MAX_SESSION_MINUTES:
            await ctx.send(
                embed=error_embed(
                    f"Choose a time between **{MIN_SESSION_MINUTES}** and "
                    f"**{MAX_SESSION_MINUTES}** minutes, e.g. `;ihave 30`."
                )
            )
            return

        user = await storage.get_user(ctx.author.id)
        if user is None:
            await ctx.send(
                embed=error_embed(
                    "You haven't linked a Codeforces handle yet. "
                    "Run `;link <handle>` first."
                )
            )
            return

        existing = self.sessions.get(ctx.author.id)
        if existing is not None:
            if not existing.expired:
                await ctx.send(
                    embed=error_embed(
                        f"You already have an active session on "
                        f"**{existing.problem_name}** with "
                        f"**{format_seconds(existing.remaining_seconds)}** left.\n"
                        "Finish it, run `;giveup`, or wait for the timer."
                    )
                )
                return
            await self._retire_session(existing)

        handle = str(user["cf_handle"])
        current_rating = estimator.normalize_rating(user["current_rating"])
        target_rating: int | None
        if rating_override is not None:
            if not (estimator.RATING_MIN <= rating_override <= estimator.RATING_MAX):
                await ctx.send(
                    embed=error_embed(
                        f"Rating override must be between "
                        f"**{estimator.RATING_MIN}** and **{estimator.RATING_MAX}**."
                    )
                )
                return
            target_rating = estimator.normalize_rating(rating_override)
        else:
            target_rating = None

        async with ctx.typing():
            await storage.ensure_problemset(self.cf)

            # Quick sync: make sure a problem solved moments ago is not picked.
            recent_submissions = await self.cf.get_user_status(handle, count=25)
            recently_solved = estimator.extract_solved_pairs(recent_submissions)
            if recently_solved:
                await storage.add_solved_many(handle, recently_solved)
            baseline_ids: set[int] = set()
            for submission in recent_submissions:
                submission_id = submission.get("id")
                if submission_id is None:
                    continue
                try:
                    baseline_ids.add(int(submission_id))
                except (TypeError, ValueError):
                    continue

            rows = await storage.get_curve(handle)
            if not rows:
                synced = await self._sync_handle(handle)
                rows = synced["rows"]
                current_rating = estimator.normalize_rating(synced["current_rating"])
                await storage.upsert_user(
                    ctx.author.id,
                    str(synced["handle"]),
                    current_rating,
                    int(time.time()),
                )

            curve = {
                int(row["rating"]): float(row["smoothed_minutes"])
                for row in rows
                if row["smoothed_minutes"] is not None
            }
            if target_rating is None:
                target_rating = estimator.recommend_rating(
                    curve, minutes, current_rating
                )

            pick = await storage.find_unsolved_problem(handle, target_rating)
            if pick is None:
                await ctx.send(
                    embed=error_embed(
                        "I couldn't find an unsolved problem near that rating. "
                        "Try `;sync` and run `;ihave` again."
                    )
                )
                return

        started_at = int(time.time())
        expires_at = started_at + minutes * 60
        pick_rating = int(pick["rating"])
        url = problem_url(pick["contest_id"], pick["index_name"])
        expected = curve.get(pick_rating, float(minutes))

        embed = discord.Embed(
            title=pick["name"],
            url=url,
            color=EMBED_COLOR,
            description=(
                f"<@{ctx.author.id}> your **{minutes}-minute** challenge is ready.\n\n"
                f"**Problem:** [{pick['name']}]({url})\n"
                f"**Contest:** {pick['contest_id']} · Problem {pick['index_name']}"
            ),
        )
        embed.add_field(
            name="⏱ Time limit",
            value=f"**{minutes} min** · ends {_relative_timestamp(expires_at)}",
            inline=True,
        )
        embed.add_field(
            name="🎯 Rating",
            value=f"||{pick_rating}||",
            inline=True,
        )
        embed.add_field(
            name="📈 Expected from your curve",
            value=f"~{expected:.1f} min",
            inline=True,
        )
        embed.add_field(
            name="🔒 Tags",
            value="Hidden until you solve it or run `;giveup`.",
            inline=False,
        )
        footer_note = (
            "manual rating override" if rating_override is not None else "personal curve"
        )
        embed.set_footer(text=f"{FOOTER_TEXT} · polling every {POLL_INTERVAL_SECONDS}s · {footer_note}")
        embed.timestamp = datetime.now(timezone.utc)

        view = PracticeView(
            self,
            user_id=ctx.author.id,
            handle=handle,
            rating=pick_rating,
        )
        message = await ctx.send(content=f"<@{ctx.author.id}>", embed=embed, view=view)
        view.message = message

        session = PracticeSession(
            user_id=ctx.author.id,
            channel_id=ctx.channel.id,
            handle=handle,
            contest_id=int(pick["contest_id"]),
            index_name=str(pick["index_name"]),
            problem_name=str(pick["name"]),
            rating=pick_rating,
            tags=tuple(pick["tags"]),
            started_at=started_at,
            expires_at=expires_at,
            baseline_ids=baseline_ids,
            message=message,
            view=view,
        )
        self.sessions[ctx.author.id] = session
        session.task = asyncio.create_task(
            self._monitor_session(session),
            name=f"ihave-session-{ctx.author.id}",
        )

    async def _cmd_adjust(self, ctx: commands.Context, rating: int, minutes: float) -> None:
        """Implementation for ``;adjust`` (see :class:`PracticeCog`)."""
        user = await storage.get_user(ctx.author.id)
        if user is None:
            await ctx.send(
                embed=error_embed("Link a handle first with `;link <handle>`.")
            )
            return
        if rating < estimator.RATING_MIN or rating > estimator.RATING_MAX:
            await ctx.send(
                embed=error_embed(
                    f"Rating must be between {estimator.RATING_MIN} and "
                    f"{estimator.RATING_MAX}."
                )
            )
            return
        if minutes < 1 or minutes > 600:
            await ctx.send(
                embed=error_embed("Minutes must be between 1 and 600.")
            )
            return

        handle = str(user["cf_handle"])
        previous, updated = await storage.adjust_curve(handle, rating, minutes)
        embed = discord.Embed(
            title="🛠 Curve recalibrated",
            color=SUCCESS_COLOR,
            description=(
                f"Recorded: you can solve **{estimator.normalize_rating(rating)}** "
                f"in about **{minutes:g} minutes**.\n"
                f"Your smoothed estimate for that rating is now "
                f"**{updated:.1f} min** (was {previous:.1f} min).\n\n"
                "This override persists through `;sync`."
            ),
        )
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)

    async def _cmd_curve(self, ctx: commands.Context, handle: str | None = None) -> None:
        """Implementation for ``;curve`` (see :class:`PracticeCog`)."""
        canonical: str | None = None
        rows: list[dict[str, Any]] = []

        async with ctx.typing():
            if handle:
                requested = handle.strip().lstrip("@")
                if not requested:
                    await ctx.send(embed=error_embed("Usage: `;curve [handle]`."))
                    return
                canonical = await storage.resolve_handle(requested)
                if canonical:
                    rows = await storage.get_curve(canonical)
                if not rows:
                    info = await self.cf.get_user_info(requested)
                    synced = await self._sync_handle(str(info.get("handle") or requested), info=info)
                    canonical = str(synced["handle"])
                    rows = synced["rows"]
            else:
                user = await storage.get_user(ctx.author.id)
                if user is not None:
                    canonical = str(user["cf_handle"])
                elif DEFAULT_CF_HANDLE:
                    canonical = await storage.resolve_handle(DEFAULT_CF_HANDLE)
                    canonical = canonical or DEFAULT_CF_HANDLE
                else:
                    await ctx.send(
                        embed=error_embed(
                            "Link a handle first with `;link <handle>`, or pass "
                            "one explicitly: `;curve tourist`."
                        )
                    )
                    return

                assert canonical is not None
                rows = await storage.get_curve(canonical)
                if not rows:
                    synced = await self._sync_handle(canonical)
                    canonical = str(synced["handle"])
                    rows = synced["rows"]

            if not canonical or not rows:
                await ctx.send(
                    embed=error_embed("I couldn't build a curve for that handle.")
                )
                return

        safe_name = "".join(
            character for character in canonical if character.isalnum() or character in "-_"
        ) or "user"
        filename = f"curve_{safe_name}.png"
        # Matplotlib state is process-global; serialize rendering across users.
        async with self._plot_lock:
            buffer = await asyncio.to_thread(plotter.generate_curve_plot, canonical, rows)
        file = discord.File(buffer, filename=filename)
        embed = self._curve_embed(canonical, rows)
        embed.set_image(url=f"attachment://{filename}")
        await ctx.send(embed=embed, file=file)

    async def _cmd_status(self, ctx: commands.Context) -> None:
        """Implementation for ``;status`` (see :class:`PracticeCog`)."""
        session = self.sessions.get(ctx.author.id)
        if session is None:
            embed = discord.Embed(
                title="No active session",
                color=MUTED_COLOR,
                description=(
                    "Start one with `;ihave 30`.\n"
                    "Not linked yet? Run `;link <handle>`."
                ),
            )
            embed.set_footer(text=FOOTER_TEXT)
            await ctx.send(embed=embed)
            return

        url = problem_url(session.contest_id, session.index_name)
        embed = discord.Embed(
            title=f"🎯 {session.problem_name}",
            url=url,
            color=WARNING_COLOR if session.expired else EMBED_COLOR,
        )
        embed.add_field(
            name="Problem",
            value=f"[{session.problem_name}]({url}) · {session.contest_id}{session.index_name}",
            inline=False,
        )
        embed.add_field(name="Rating", value=f"||{session.rating}||", inline=True)
        embed.add_field(
            name="Time limit", value=f"{session.limit_minutes} min", inline=True
        )
        if session.expired:
            embed.add_field(
                name="Status",
                value=(
                    "⏰ The timer ended without an Accepted submission. "
                    "Run `;giveup` to reveal the rating and tags."
                ),
                inline=False,
            )
            embed.color = WARNING_COLOR
        else:
            embed.add_field(
                name="Remaining",
                value=f"**{format_seconds(session.remaining_seconds)}**",
                inline=True,
            )
            embed.add_field(
                name="Elapsed",
                value=format_seconds(session.elapsed_seconds),
                inline=True,
            )
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)

    async def _cmd_giveup(self, ctx: commands.Context) -> None:
        """Implementation for ``;giveup`` (see :class:`PracticeCog`)."""
        session = self.sessions.get(ctx.author.id)
        if session is None:
            await ctx.send(
                embed=error_embed("You don't have an active session to give up on.")
            )
            return

        url = problem_url(session.contest_id, session.index_name)
        tags = ", ".join(session.tags) if session.tags else "no tags"
        elapsed = min(session.elapsed_seconds, session.limit_minutes * 60)
        embed = discord.Embed(
            title=f"🏳️ Revealed — {session.problem_name}",
            url=url,
            color=WARNING_COLOR,
            description=(
                f"You gave up on **[{session.problem_name}]({url})** after "
                f"**{format_seconds(elapsed)}**.\n\n"
                f"**Problem:** {session.contest_id} · {session.index_name}"
            ),
        )
        embed.add_field(name="Rating", value=f"||{session.rating}||", inline=True)
        embed.add_field(
            name="Time limit", value=f"{session.limit_minutes} min", inline=True
        )
        embed.add_field(name="Tags", value=f"||{tags}||", inline=False)
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)
        await self._retire_session(session)

    async def _cmd_sync(self, ctx: commands.Context) -> None:
        """Implementation for ``;sync`` (see :class:`PracticeCog`)."""
        user = await storage.get_user(ctx.author.id)
        if user is None:
            await ctx.send(
                embed=error_embed("Link a handle first with `;link <handle>`.")
            )
            return

        async with ctx.typing():
            info = await self.cf.get_user_info(str(user["cf_handle"]))
            synced = await self._sync_handle(str(user["cf_handle"]), info=info)
            await storage.upsert_user(
                ctx.author.id,
                str(synced["handle"]),
                synced["current_rating"],
                int(time.time()),
            )
            problems_refreshed = await storage.ensure_problemset(self.cf)
            solved_count = await storage.get_solved_count(str(synced["handle"]))

        embed = discord.Embed(
            title=f"🔄 Synced @{synced['handle']}",
            color=SUCCESS_COLOR,
            description=(
                f"Scanned **{synced['submission_count']}** submissions and "
                f"**{synced['rated_contests']}** rated contest"
                f"{'' if synced['rated_contests'] == 1 else 's'}.\n"
                f"**{solved_count}** solved problems are now on record."
            ),
        )
        embed.add_field(
            name="Rating",
            value=str(synced["current_rating"]),
            inline=True,
        )
        embed.add_field(
            name="Problemset cache",
            value="refreshed just now" if problems_refreshed else "already fresh",
            inline=True,
        )
        if synced["rated_contests"] < estimator.MIN_RATED_CONTESTS:
            embed.add_field(
                name="Cold start",
                value=(
                    "Fewer than 5 rated contests found, so the fallback curve is "
                    "still in use."
                ),
                inline=False,
            )
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)

    async def _cmd_help(self, ctx: commands.Context) -> None:
        """Implementation for ``;help`` (see :class:`PracticeCog`)."""
        commands_list = [
            ("`;link <handle>`", "Link your Codeforces handle and build your curve."),
            ("`;ihave <minutes> [rating]`", "Start a 5–180 minute session (optional rating override)."),
            ("`;adjust <rating> <minutes>`", "Recalibrate: \"I solve 1700 in 30 minutes\"."),
            ("`;curve [handle]`", "PNG chart plus a smoothed minutes-per-rating table."),
            ("`;status`", "Show the active session and remaining time."),
            ("`;giveup`", "Cancel and reveal the rating and tags in spoilers."),
            ("`;sync`", "Refresh submissions, curve and the problemset cache."),
        ]
        embed = discord.Embed(
            title=";ihave — Codeforces time-boxed practice",
            color=EMBED_COLOR,
            description=(
                "Tell me how many minutes you have and I'll pick a recent, "
                "unsolved Codeforces problem at the rating you can realistically "
                "finish in that time. Tags stay hidden until you solve it or give up."
            ),
        )
        for name, description in commands_list:
            embed.add_field(name=name, value=description, inline=False)
        embed.set_footer(text=FOOTER_TEXT)
        await ctx.send(embed=embed)

    # -- error handling ------------------------------------------------------

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return

        original = getattr(error, "original", error)

        if isinstance(original, cf_api.CFError):
            await ctx.send(embed=error_embed(original.message))
            return

        if isinstance(error, commands.MissingRequiredArgument):
            usage = ctx.command.usage if ctx.command and ctx.command.usage else ""
            name = ctx.command.qualified_name if ctx.command else ""
            await ctx.send(
                embed=error_embed(
                    f"Missing argument.\nUsage: `{ctx.prefix}{name} {usage}`"
                )
            )
            return

        if isinstance(error, commands.BadArgument):
            usage = ctx.command.usage if ctx.command and ctx.command.usage else ""
            name = ctx.command.qualified_name if ctx.command else ""
            await ctx.send(
                embed=error_embed(
                    f"That argument didn't make sense: {error}\n"
                    f"Usage: `{ctx.prefix}{name} {usage}`"
                )
            )
            return

        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(
                embed=error_embed(
                    f"Slow down a little — try again in {error.retry_after:.1f}s."
                )
            )
            return

        log.error(
            "Unhandled command error in %s",
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
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
            "and replace `your_discord_bot_token_here` with the real bot token, "
            "save the file, then run `python bot.py` again."
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
