# `ihave` — Vercel slash-command bot for Codeforces practice

A Discord **HTTP Interactions** app that recommends an unsolved Codeforces
problem for a time budget (`/ihave minutes:30`), hides the tags, keeps a personal
minutes-per-rating curve, and tracks an active session.

This version is built for **Vercel Functions**, not the Discord gateway. Discord
POSTs slash commands and button clicks to one HTTPS endpoint; the app verifies
the Ed25519 signature, returns a deferred response, and uses Vercel's
`wait_until` to finish Codeforces calls and edit the message.

> **Important:** serverless mode has no persistent process, so there is no
> automatic 60-second AC poller by default. `/status` performs an on-demand
> check, and an optional `/api/cron/poll` endpoint can drive automatic checks on
> a schedule. See [Session tracking](#session-tracking) and
> [Optional cron polling](#optional-cron-polling).

---

## What runs where

```
Discord  --POST /api/interactions-->  main.py (FastAPI on Vercel)
                                        |
                                        +-- Ed25519 signature check
                                        +-- deferred response (3s limit)
                                        +-- wait_until(...)
                                              |
                                              +-- Codeforces API (aiohttp)
                                              +-- Upstash Redis (state)
                                              +-- Discord webhook (edit response)
                                              +-- matplotlib (optional chart)
```

- **No gateway WebSocket**, no `on_message`, no `on_ready`, no member events.
- **No local SQLite**: Vercel Functions have an ephemeral filesystem, so state
  lives in Redis (Upstash or Vercel KV).
- **No long-running background loop** inside the request. The optional cron
  endpoint is the only scheduled work.

| Capability | Status |
|---|---|
| Slash commands | Yes |
| Buttons (`Too Easy` / `Too Hard`) | Yes |
| Personal curve / chart | Yes |
| Link + sync from Codeforces | Yes |
| Manual `/status` AC check | Yes |
| Automatic 60s polling | Only with a cron schedule (Pro/external) |
| Read normal chat messages | **No** — HTTP interactions only |
| Join/leave events | **No** |
| Local SQLite | Replaced by Redis |

---

## Setup

### 1. Create the Discord application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create an application.
2. On **General Information**, copy:
   - **Application ID** → `DISCORD_APPLICATION_ID`
   - **Public Key** → `DISCORD_PUBLIC_KEY`
3. On **Bot**, reset/copy the token → `DISCORD_BOT_TOKEN`.
4. **You do not need the Message Content Intent.** This bot never reads messages.
5. Invite the app with both scopes `bot` and `applications.commands`:

   ```
   https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=117760&scope=bot%20applications.commands
   ```

   `117760` covers View Channels, Send Messages, Embed Links, Attach Files and
   Read Message History. Adjust if you want fewer permissions.

### 2. Create Redis

On Vercel, use one of:

- **Upstash Redis** from the Vercel Marketplace, or an Upstash database directly.
- **Vercel KV** (Upstash-backed), if your project uses it.

Set either pair of environment variables:

```env
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
```

or:

```env
KV_REST_API_URL=https://...
KV_REST_API_TOKEN=...
```

On Vercel the app **requires** Redis. Local development falls back to an
in-process memory store (data disappears when the process exits).

### 3. Deploy to Vercel

Using the Vercel CLI:

```bash
npm i -g vercel
vercel
vercel --prod
```

Or connect the GitHub repository in the Vercel dashboard. No build command is
needed; Vercel detects `main.py` as the Python entrypoint.

Set these environment variables in **Project → Settings → Environment
Variables** (Production and Preview as needed):

| Variable | Required | Purpose |
|---|---|---|
| `DISCORD_PUBLIC_KEY` | Yes | Verifies every Discord interaction |
| `DISCORD_APPLICATION_ID` | Yes | Builds webhook edit URLs |
| `DISCORD_BOT_TOKEN` | Yes | `/api/cron/poll` sends channel messages; command registration |
| `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` | Yes on Vercel | Durable state |
| `KV_REST_API_URL` / `KV_REST_API_TOKEN` | Alternative | Vercel KV names |
| `CRON_SECRET` | Optional | Protects `/api/cron/poll`; enables the cron footer |
| `GUILD_ID` | Optional | Register commands in one guild for instant updates |
| `ALLOW_INSECURE_INTERACTIONS` | Local only | `1` skips signature verification |
| `LOG_LEVEL` | Optional | Defaults to `INFO` |

### 4. Set the Interactions Endpoint URL

After the production deployment exists, copy its URL and set:

```
https://YOUR-PROJECT.vercel.app/api/interactions
```

in **General Information → Interactions Endpoint URL**. Discord immediately
sends a signed `PING`; the app must answer with `{"type": 1}`. Save only after it
verifies.

### 5. Register the slash commands

From the project root, with the same environment variables available:

```bash
python register_commands.py
```

Global commands can take a few minutes to appear. Set `GUILD_ID` for instant
guild-scoped registration during testing.

### 6. Test

```
/help
/link handle:tourist
/ihave minutes:30
/status
```

---

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# .env with the same keys as above
uvicorn main:app --reload --port 8000
```

Without Redis env vars, `storage.py` uses an in-memory backend: useful for
clicking through `/help` locally, but it is not shared between processes and is
not suitable for Vercel. To send unsigned local test requests, set:

```env
ALLOW_INSECURE_INTERACTIONS=1
```

Never set that in production.

To test real Discord interactions locally, expose port 8000 with a tunnel
(`ngrok http 8000`) and use the tunnel URL as the Interactions Endpoint URL.
Discord will verify the signature against your public key.

---

## Commands

| Command | Description |
|---|---|
| `/link handle:<handle>` | Validate the handle, sync submissions and build the curve. |
| `/ihave minutes:<5-180> [rating:<800-3500>]` | Start a time-boxed session with an optional rating override. |
| `/adjust rating:<800-3500> minutes:<1-600>` | Record "I solve this rating in this time" and re-smooth. |
| `/curve [handle:<handle>]` | Post the PNG chart and the smoothed minutes-per-rating table. |
| `/status` | Show the session and check Codeforces for a new AC. |
| `/giveup` | End the session and reveal rating/tags in `||spoilers||`. |
| `/sync` | Refresh submissions, curve and the problemset cache if stale. |
| `/help` | Command reference. |

`/link` also sends the tip:

> Make sure to register for Virtual Contests before starting them! Unregistered
> practice submissions are ignored by the time-curve to keep your data accurate.

---

## How the estimation works

The math is unchanged; it lives in [`estimator.py`](estimator.py) and is pure
numpy, with no Discord or storage dependencies.

### 1. Contest submissions only

From `user.status`, keep submissions where:

- `verdict == "OK"`
- `author.participantType` is `CONTESTANT` or `VIRTUAL`
- the problem has a `rating`
- `0 < relativeTimeSeconds <= 18000` (five hours)

Practice submissions are ignored.

### 2. Per-contest deltas

Group by `(contestId, participantType, author.startTimeSeconds)`. Within one
participation:

- keep the **earliest AC per problem index**;
- sort by `relativeTimeSeconds`;
- compute:

  ```
  Δ1 = t1 / 60
  Δi = (ti - t(i-1)) / 60
  ```

`relativeTimeSeconds` is already contest-relative.

### 3. Recency

Only the **35 most recent distinct contest participations** are used.

### 4. Robust buckets

Bucket Δ by rating (800…3500, step 100). A bucket needs at least **2 solves**.
Four or more solves use a 10% trimmed mean; smaller buckets use the median.

### 5. Monotonic smoothing

Higher ratings can look faster in raw data because only their easy problems were
solved (survivorship bias). The estimator:

1. interpolates missing buckets with `numpy.interp`;
2. applies `numpy.maximum.accumulate` to force a non-decreasing curve.

No scikit-learn.

### 6. Cold start

With fewer than 5 rated contests, anchor around the current Codeforces rating
`R` (1500 if unrated):

| Rating | Minutes |
|---|---|
| `R - 300` | 12 |
| `R - 200` | 18 |
| `R - 100` | 28 |
| `R` | 40 |
| `R + 100` | 55 |
| `R + 200` | 75 |

### 7. Time → rating

For `/ihave minutes:Y`:

1. `limit = Y * 1.08`;
2. choose the **highest** rating whose smoothed time is `<= limit`;
3. if none qualify, choose the closest by absolute difference;
4. clamp to `[max(800, current_rating - 500), current_rating + 300]`, snapped to
   the 100-point grid.

The 8% tolerance keeps a 30-minute request from rejecting a 31-minute estimate.

### 8. Problem selection

From the cached problemset:

- one of the **400 highest contest ids**;
- exact rating first, then nearby tiers (`±100`, `±200`, …);
- exclude every problem in the user's solved set, including practice solves;
- prefer the **top 70% by `solved_count`** within the tier;
- pick uniformly at random from that pool.

---

## Session tracking

`/ihave` stores a session in Redis with:

- the problem, rating and tags;
- `started_at`, `expires_at`;
- the ids of the latest submissions at session start (`baseline_ids`);
- the Discord user and channel.

There is no always-on poller. `/status`:

1. fetches the user's latest 25 submissions;
2. looks for an `OK` submission with a **new id** matching the target
   `contestId` + `index`;
3. on success, marks it solved and reveals rating/tags;
4. if the timer expired, reports time-up and tells the user to run `/giveup`.

The `Too Easy` / `Too Hard` buttons update the curve by `-15%` / `+20%` with a
monotonic pin; one feedback click per session is enforced with a Redis `SET NX`.

---

## Optional cron polling

`/api/cron/poll` iterates active sessions, checks Codeforces, announces ACs or
timeouts via `DISCORD_BOT_TOKEN`, and clears finished sessions. It requires:

```env
CRON_SECRET=some-long-random-value
```

and an `Authorization: Bearer <CRON_SECRET>` request header.

Vercel Cron:

```json
{
  "crons": [
    { "path": "/api/cron/poll", "schedule": "* * * * *" }
  ]
}
```

**Plan limits matter:** Vercel Cron on **Hobby runs once per day** and
per-minute expressions fail deployment; minute-level polling requires **Pro**
or an external scheduler (cron-job.org, GitHub Actions, etc.). See
[Vercel Cron usage and pricing](https://vercel.com/docs/cron-jobs/usage-and-pricing).

Without a scheduler, the bot still works: `/status` performs the same check on
demand. The function runs one poll at a time and is bounded by Vercel's maximum
duration (300 seconds on Hobby).

---

## Redis layout

All keys are prefixed with `ihave:`.

| Key | Type | Contents |
|---|---|---|
| `ihave:user:<discord_id>` | JSON string | handle, current rating, last sync |
| `ihave:handle:<handle>` | string | canonical handle spelling |
| `ihave:curve:<handle>` | JSON string | 28 rating points |
| `ihave:overrides:<handle>` | JSON string | durable `/adjust` values |
| `ihave:solved:<handle>` | set | `contestId:index` pairs |
| `ihave:problems:<rating>` | JSON string | cached tier |
| `ihave:problems:recent` | JSON string | top 400 contest ids |
| `ihave:problems:updated` | string | cache timestamp |
| `ihave:session:<id>` | JSON string | active session |
| `ihave:session:user:<id>` | string | current session id |
| `ihave:sessions:active` | set | active session ids |
| `ihave:feedback:<id>` | string | one-feedback NX lock |

The problemset is split by rating to keep individual Redis values small.

---

## Vercel specifics and limits

- A deferred Discord response is sent immediately; actual work runs inside
  `wait_until`. Vercel keeps the invocation alive up to the configured maximum
  duration (`maxDuration: 300` in [`vercel.json`](vercel.json)). Ensure **Fluid
  compute** is enabled if post-response work is being cut off early.
- `/link` and `/sync` fetch up to 10,000 submissions; they can take several
  seconds because the Codeforces client spaces requests by 2.05 seconds.
- Vercel Functions have an ephemeral filesystem. Nothing durable is written
  locally; Redis is required.
- Vercel Hobby functions have a **300 second** maximum duration; Pro supports
  longer durations. A full history sync plus chart rendering is normally well
  inside that.
- `/curve` renders in a worker thread and uploads the PNG as an attachment edit.

---

## Project structure

```
main.py                 FastAPI interactions endpoint + slash handlers
storage.py              Redis/memory storage, curves, sessions, cache
cf_api.py               Public Codeforces API client + rate limiter
estimator.py            Pure numpy solve-time estimation
plotter.py              Matplotlib PNG rendering
register_commands.py    One-time Discord slash command registration
vercel.json             Function duration config
requirements.txt        Python dependencies
.env.example            Environment template
tests/test_estimator.py Synthetic estimator tests
tests/test_serverless.py Signature + memory-storage tests
```

---

## Tests

```bash
pytest -q
```

The estimator tests use hardcoded synthetic submissions and never touch Redis,
Discord or the network.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Endpoint URL won't save / `401` | `DISCORD_PUBLIC_KEY` must be the application's Public Key, not the bot token. |
| Slash commands missing | Run `register_commands.py`; global commands can take time. Set `GUILD_ID` for instant guild sync. |
| `Redis is not configured` | Set `UPSTASH_REDIS_REST_URL` + token or `KV_REST_API_URL` + token in Vercel. |
| `Discord edit failed (404)` | The interaction token expired or the application id is wrong; check logs. |
| `/ihave` says no problem | The problemset cache may be missing; run `/sync` or retry. |
| `/status` never sees an AC | Codeforces can lag a few seconds; run `/status` again. Automatic checks require cron. |
| Function timeout | Hobby is capped at 300s; retry, or upgrade Pro. Check Vercel logs. |
| Button feedback says already recorded | One adjustment per session is intentional. |
| Cron endpoint 503 | `CRON_SECRET` is not set. |
| Cron endpoint 401 | Send `Authorization: Bearer <CRON_SECRET>`. |

---

## Security notes

- Every interaction is Ed25519-verified against `DISCORD_PUBLIC_KEY`.
- `/link` proves only that a Codeforces handle **exists**, not that the Discord
  user owns it. The Codeforces API has no OAuth flow in this project; data is
  public and read-only.
- Keep `.env` local. It is ignored by Git; never commit bot tokens or Redis
  credentials.
