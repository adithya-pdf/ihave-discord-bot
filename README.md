# `;ihave` — Codeforces time-boxed practice bot

A single-process, asynchronous Discord bot that answers one question:

> *"I have 30 minutes. Which Codeforces problem should I solve right now?"*

You link your Codeforces handle, the bot learns how many minutes you typically
spend on each problem rating from your **live and virtual contest history**, and
then serves a recent, unsolved problem at the rating your personal speed curve
says fits your time budget. Problem tags stay hidden so the rating label does
not spoil the approach.

No web backend, no external database server, no Docker, no ML libraries — just
`discord.py`, `aiohttp`, `matplotlib`, `numpy`, and a local SQLite file
accessed through `aiosqlite`.

---

## Features

- **`;link <handle>`** — validates the handle with the public Codeforces API,
  syncs submissions, and builds the initial time-per-rating curve.
- **`;ihave <minutes> [rating]`** — 5 to 180 minute practice session. Picks a
  random recent unsolved problem, hides the rating in a Discord spoiler and
  hides tags entirely, then polls Codeforces every 60 seconds for an Accepted
  submission.
- **`;adjust <rating> <minutes>`** — manual calibration ("I can consistently
  solve 1700s in 30 minutes"), persisted as an override and re-applied after
  every `;sync`.
- **`;curve [handle]`** — renders a clean Matplotlib chart (minutes vs rating)
  and posts a smoothed per-rating table next to it.
- **`;status`** — active session, remaining time, and target problem.
- **`;giveup`** — ends the session and reveals the rating and tags inside
  Discord spoiler tags (`||graphs, dp||`).
- **`;sync`** — refreshes submissions, rebuilds the curve, and refreshes the
  local problemset cache if it is older than 24 hours.
- **Interactive feedback** — `⚡ Too Easy` / `😅 Too Hard` buttons adjust the
  expected time for that rating by −15% / +20% while keeping the curve
  non-decreasing.

---

## Requirements

- Python **3.12**
- A Discord application/bot token
- Internet access to `codeforces.com` (no API key or OAuth needed)

Install dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Discord bot setup

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create a new application.
2. Go to **Bot** → **Add Bot**.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**.
   Prefix commands cannot read `;ihave 30` without it.
4. Copy the bot token.
5. Invite the bot with the `bot` scope and these permissions:
   **View Channels**, **Send Messages**, **Embed Links**, **Attach Files**,
   **Read Message History**.

   Replace `YOUR_APP_ID` in this URL:

   ```
   https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=117760&scope=bot
   ```

6. Configure the environment:

   ```bash
   cp .env.example .env
   ```

   `.env`:

   ```env
   DISCORD_TOKEN=your_real_token_here
   DEFAULT_CF_HANDLE=
   ```

   `DEFAULT_CF_HANDLE` is optional. When set, `;link` can be used without an
   argument and `;curve` falls back to that handle.

7. Run the bot:

   ```bash
   python bot.py
   ```

   The bot creates `data/bot.db` on first start and warms the problemset cache
   in the background (a few seconds; the public API is intentionally paced).

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | — | Bot token (required). |
| `DEFAULT_CF_HANDLE` | empty | Fallback handle for `;link` / `;curve`. |
| `BOT_DB_PATH` | `data/bot.db` | Alternative SQLite path (directory is created). |
| `LOG_LEVEL` | `INFO` | Python log level. |

---

## Commands

| Command | Example | Description |
| --- | --- | --- |
| `;link` | `;link tourist` | Link a handle, sync submissions, build the curve. |
| `;ihave` | `;ihave 30` or `;ihave 45 1700` | Start a session for 5–180 minutes, optionally forcing a rating. |
| `;adjust` | `;adjust 1700 30` | Record "1700 in 30 minutes" and re-smooth the curve. |
| `;curve` | `;curve` or `;curve tourist` | Post the chart and the smoothed rating table. |
| `;status` | `;status` | Show the active session and countdown. |
| `;giveup` | `;giveup` | Cancel and reveal rating + tags in spoilers. |
| `;sync` | `;sync` | Refresh submissions, curve, and problemset cache. |
| `;help` | `;help` | Command reference. |

`;link` also sends the one-time virtual-contest tip:

> Make sure to register for Virtual Contests before starting them! Unregistered
> practice submissions are ignored by the time-curve to keep your data accurate.

---

## How the estimation algorithm works

Everything below lives in [`estimator.py`](estimator.py) and is pure logic
(numpy only), which is why it can be unit tested without Discord, SQLite, or
network access.

### 1. Contest submissions only

From `user.status` we keep submissions where:

- `verdict == "OK"`
- `author.participantType` is `CONTESTANT` or `VIRTUAL`
- the problem has a `rating`
- `0 < relativeTimeSeconds <= 18000` (five hours)

Practice submissions are deliberately ignored: unregistered practice says
nothing reliable about contest speed.

### 2. Per-contest time deltas

Accepted submissions are grouped by
`(contestId, participantType, author.startTimeSeconds)` — so a virtual restart
is its own participation. Within a participation:

- keep only the **earliest** AC per `problem.index`;
- sort the solved problems by `relativeTimeSeconds`:
  `t1 <= t2 <= ... <= tk`;
- the time spent on each problem is

  ```
  Δ1 = t1 / 60
  Δi = (ti - t(i-1)) / 60
  ```

`relativeTimeSeconds` is already relative to contest start, so no external start
timestamp is subtracted.

### 3. Recency filter

Only the **35 most recent distinct contest participations** feed the curve.
Recency is ordered by contest id (which grows over time) and participation start
time, so a user's current speed dominates the estimate.

### 4. Robust aggregation

Δ values are bucketed by rating on the 800…3500 / step-100 grid. A bucket needs
at least **2** solves to produce a raw average. Buckets with 4+ solves use a 10%
trimmed mean (at least one value trimmed from each end); tiny buckets use the
median. This keeps a single 3-minute fluke from defining a rating.

### 5. Monotonic smoothing (numpy only, no scikit-learn)

Higher ratings can show *lower* raw averages because of survivorship bias: in a
two-hour contest only the easy problems of the highest ratings get solved at
all. To remove those impossible dips:

1. `numpy.interp` fills missing rating buckets by linear interpolation between
   observed buckets (and clamps beyond the observed range).
2. `numpy.maximum.accumulate` performs a forward cumulative maximum, making the
   final curve non-decreasing in rating.

Raw averages are retained separately for the dashed line on the chart.

### 6. Cold start

If the user has **fewer than 5 rated contests** (or no usable buckets), the
curve is anchored around their current rating `R` (1500 if unrated):

| Rating | Minutes |
| --- | --- |
| `R - 300` | 12 |
| `R - 200` | 18 |
| `R - 100` | 28 |
| `R` | 40 |
| `R + 100` | 55 |
| `R + 200` | 75 |

The anchors are interpolated onto the grid, extrapolated for very low/high
ratings, clamped to 3–240 minutes, and made non-decreasing.

### 7. Reverse lookup for `;ihave`

Given `target_minutes`:

1. Pick the **highest** rating whose smoothed time is
   `<= target_minutes * 1.08`.
2. If every rating is too slow, pick the bucket closest to the target by
   absolute difference.
3. Clamp to `[max(800, current_rating - 500), current_rating + 300]`, snap to
   the 100-point grid.

The 8% tolerance prevents a 29.9-minute curve value from being rejected for a
30-minute request.

### 8. Problem selection

From the local `problems` cache:

- only the **400 highest contest ids** (recent problems);
- only the requested rating (adjacent tiers are tried if the exact tier has no
  eligible problem);
- excluding every problem present in `user_solved` — including practice solves;
- prefer the **top 70% by `solved_count`** within the tier, then pick uniformly
  at random from that pool.

### 9. Session tracking

`;ihave` performs a quick 25-submission sync first, records the ids of those
submissions, then polls `user.status` every 60 seconds. A submission counts as
the session win when it is a new id, has verdict `OK`, and matches the target
`contestId` + `index`. Elapsed time comes from the submission's
`creationTimeSeconds`.

Sessions live in memory only; a bot restart forgets active timers (documented
limitation).

### 10. Feedback buttons

- `⚡ Too Easy` targets `old_minutes * 0.85` (−15%).
- `😅 Too Hard` targets `old_minutes * 1.20` (+20%).

The bot then **pins** the smoothed curve through that exact point: ratings at or
below it are capped at the new value, ratings above it are floored at the new
value. For a non-decreasing curve this is the minimal change that keeps the
curve monotonic, so the requested adjustment is actually visible at that rating
instead of being swallowed by a forward-maximum pass. Both buttons are disabled
after one adjustment per session to prevent accidental stacking.

Button tweaks are tuning signals, not permanent calibration: they update
`smoothed_minutes` but not `curve_overrides`, so a later `;sync` rebuilds from
fresh contest history. Use `;adjust` for a durable override.

### 11. Manual calibration and monotonic projection

`;adjust 1700 30` records the point in `curve_overrides` and pins the curve
through `(1700, 30)`. On every subsequent `;sync`, the fresh contest-derived
curve is recomputed and all stored overrides are pinned into it in ascending
rating order. This makes manual calibration survive data refreshes while the
result remains non-decreasing: if you say you solve 1700s in 30 minutes, slower
neighbouring buckets are pulled down (or faster neighbours pushed up) just
enough to keep the curve consistent.

---

## Codeforces API, caching, and rate limiting

`cf_api.py` uses only public endpoints:

| Endpoint | Use |
| --- | --- |
| `problemset.problems` | Problem metadata + solved counts, cached for 24h. |
| `user.info` | Handle validation, canonical spelling, current rating. |
| `user.status` | Submission history, quick sync, session polling. |
| `contest.list` | Only to identify April Fools / Kotlin Heroes contests by name. Falls back to a curated denylist if unavailable. |

**Rate limiting:** a module-level singleton `asyncio.Lock` serializes every
request from the whole process — command handlers, cache refresh, and all
background pollers. Inside the lock the client sleeps until at least **2.05
seconds** have passed since the previous request started, then timestamps the
new request. This keeps the bot far below Codeforces' public call limit.

**Retries:** network timeouts/connection errors, HTTP 429, and HTTP 5xx are
retried up to 4 times with exponential backoff (3s, 6s, 12s, longer on 429).
Every attempt has a 30-second timeout. API-level failures (unknown handle, bad
request) raise `CFError` with a human-friendly message that the bot posts
directly.

**Special contests:** problems without a rating are dropped, as are all
problems from contests whose name contains "April Fools", "Kotlin Heroes", or
"Kotlin", plus a known-id denylist as a fallback.

---

## Database

SQLite lives at `data/bot.db` (auto-created with `os.makedirs("data",
exist_ok=True)`). All access uses `aiosqlite`; WAL mode is enabled.

```
problems(contest_id, index_name, name, rating, solved_count, tags)
    PRIMARY KEY (contest_id, index_name)

users(discord_id, cf_handle, current_rating, last_synced)
    PRIMARY KEY (discord_id)

user_solved(cf_handle, contest_id, index_name)
    PRIMARY KEY (cf_handle, contest_id, index_name)

user_curve(cf_handle, rating, raw_minutes, smoothed_minutes, sample_size)
    PRIMARY KEY (cf_handle, rating)

curve_overrides(cf_handle, rating, minutes, updated_at)
    PRIMARY KEY (cf_handle, rating)     -- durable ;adjust calibrations

meta(key, value)                        -- problemset cache timestamp etc.
```

`raw_minutes` is `NULL` for ratings with no observations; `;curve` only draws
the dashed raw line for buckets with `sample_size > 0`.

---

## Project layout

```
bot.py                 Discord bot: commands, session loop, embeds, buttons
cf_api.py              Public Codeforces API client + global rate limiter
estimator.py           Pure numpy estimation engine (no discord / no db)
storage.py             aiosqlite persistence and problem selection
plotter.py             Matplotlib PNG chart generator
requirements.txt       Pinned minimum dependencies
.env.example           Environment template
.gitignore             Ignores .env, data/, caches, venv
tests/test_estimator.py Synthetic unit tests for the estimation engine
README.md              This file
```

---

## Running the tests

```bash
pytest -q
```

The tests use only hardcoded synthetic submissions and observations; they make
no network calls and do not import `storage.py`.

Covered:

1. earliest-AC deduplication, contest deltas (10/20/45 minutes), practice
   submissions ignored, five-hour cutoff;
2. monotonic smoothing of a 1600:35 → 1700:55 → 1800:44 dip;
3. `recommend_rating` for 15, 30, and 60 minutes plus clamp behaviour;
4. cold-start fallback for 0, 2, and extreme ratings.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `DISCORD_TOKEN is not set` | Copy `.env.example` to `.env` and paste the token. |
| `DISCORD_TOKEN is still the placeholder` | Open `.env`, replace `your_discord_bot_token_here` with the real bot token, and save. |
| Bot ignores `;ihave` | Enable **Message Content Intent** in the Developer Portal; restart the bot. |
| `RuntimeWarning: coroutine ... was never awaited` | Make sure `bot.run(...)` is the last call and no code after it. |
| `Codeforces rate limit reached` | Wait a minute. The 2.05s global pacing normally prevents this. |
| `Codeforces handle ... was not found` | Handles are case-insensitive; check spelling and that the profile exists. |
| Empty problemset / no problems found | Run `;sync`; the cache may have failed to warm at startup. |
| Chart missing / font warnings | Matplotlib falls back to DejaVu Sans; no action needed. |
| First `;ihave` is slow | The problemset cache refresh makes several paced API calls; later runs use SQLite. |
| Bot restarted mid-session | Sessions are in memory only; run `;ihave` again. |

---

## Limitations

- Active sessions are in memory, so restarting the bot cancels timers.
- Only the most recent 10,000 submissions are scanned (`user.status` cap).
- Contest solve deltas approximate "time spent" by the gap between ACs; a user
  who thinks for 20 minutes before their first accepted submission is measured
  exactly that way (the first problem includes all pre-solve thinking).
- `contest.list` is fetched once per cache refresh purely to filter special
  contests; the bot still works with the built-in denylist if that call fails.
- Ratings and tags are hidden in embeds as spoilers; anyone can still click the
  problem link and see the rating on Codeforces.
