# IhHave — Codeforces Rating-Space Elo-Survival practice bot

A single-process Discord bot that answers one question:

> *"I have 25 minutes. Which Codeforces problem should I solve right now?"*

You link a Codeforces handle. The bot bootstraps a survival model from your rated
contest history, then predicts the problem rating you can solve in a given time
window. It serves a matching unsolved problem, follows the attempt (early solve,
overtime solve, or right-censored give-up), and refits the model after every
attempt. It can also run the model backwards (`/times`) to show the minutes you
need at 50%, 65% and 80% confidence for any rating.

No web backend, no external database server, no ML libraries and no numerical
stack: the model and data layer are pure standard-library Python plus SQLite.

---

## Features

- **`/link <handle>`** — validates the handle, bootstraps from your most recent
  40 live and virtual participation sessions (clean sequential solve times and
  end-of-contest timeouts), caches every accepted problem, and fits the initial
  model.
- **`/practice <minutes> [confidence] [topic] [blind]`** — 5–180 minute session.
  Picks a random unsolved problem near your predicted rating. Blind mode adds a
  `[-100, 0, +100]` rating jitter and spoiler-tags the rating, effective rating
  and archetype. Timer expiry does **not** fail the session.
- **Interactive controls** — **Check AC Now**, **Give Up** (with an early
  confirmation prompt), and **Ignore**. An overtime AC still counts as an exact
  solve; a give-up is logged as a right-censored observation.
- **`/times`** — rating → time quantiles. A 13-row rating ladder around your S30
  at 50% / 65% / 80% confidence plus `P(<=25m)` and `P(<=40m)`, or a single
  rating deep dive across all five archetypes.
- **`/perf` (alias `/stats`)** — live-contest readiness, era discount, learned
  contest offset, timer-budget table and archetype handicaps.
- **`/calibration`** — walk-forward log-loss, Brier score and a 10-bin
  reliability table on your own attempts.
- **`/done`, `/giveup`, `/cancel`** — manual control over the active session.
- Legacy prefix commands (`;link`, `;practice`/`;ihave`, `;done`, `;giveup`,
  `;cancel`, `;perf`/`;stats`, `;times`, `;calibration`) remain available.

---

## Quick start (local)

Requirements: Python **3.10+** (tested on 3.12), a Discord bot token, and
outbound access to `codeforces.com`.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements-dev.txt   # runtime deps + pytest
cp .env.example .env                  # then paste your token
python bot.py
```

The bot creates the SQLite database on first start and opens the problemset
cache under `{BOT_DATA_DIR}/.cache/`.

### Discord application setup

1. Create an application at the
   [Discord Developer Portal](https://discord.com/developers/applications).
2. **Bot → Add Bot**, then copy the token.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**.
   It is required because this bot still ships prefix commands. (Slash commands
   do not need it; the legacy `;` commands do.)
4. Invite the bot with the `bot` and `applications.commands` scopes and the
   permissions **View Channels**, **Send Messages**, **Embed Links**, and
   **Read Message History**.

Slash commands are synced automatically on startup (`setup_hook`).

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | — | Bot token (required). |
| `DEFAULT_CF_HANDLE` | empty | Optional fallback handle for the prefix `;link` command. |
| `BOT_DATA_DIR` | `.` | Directory for the SQLite database and `.cache/`. Docker sets `/app/data`. |
| `BOT_DB_PATH` | unset | Legacy override for the exact SQLite file path. Takes precedence over `BOT_DATA_DIR`. |
| `LOG_LEVEL` | `INFO` | Python log level. |

The bot never reads a token from a command-line argument and never bakes one
into the image. Docker deployment uses `env_file: .env`.

---

## Commands

| Command | Example | Description |
| --- | --- | --- |
| `/link` | `/link tourist` | Link a handle and bootstrap the model from contest history. Re-linking a different handle clears the old contest cache first. |
| `/practice` | `/practice 25 confidence:0.65 topic:DP blind:true` | Start a time-boxed session. `topic` accepts Ad-Hoc, Math, DS, Graphs, DP or Weakest. |
| `/done` | `/done 21.5 was:2` | Log the active session as solved. Defaults come from one final poll. |
| `/giveup` | `/giveup 18` | Final poll, then log a right-censored timeout (defaults to elapsed time). |
| `/cancel` | `/cancel` | Discard the session and log nothing. |
| `/perf` / `/stats` | `/perf` | Readiness, era discount, time-budget table, archetype handicaps, contest offset. |
| `/times` | `/times 1900 topic:DP era:Classic mode:Contest` | Time needed at 50/65/80% confidence; omit `rating` for the 13-row ladder. |
| `/calibration` | `/calibration` | Walk-forward log-loss/Brier/reliability on your practice attempts. |
| `/help` | `/help` | Command reference. |

Buttons on a practice message:

- **Check AC Now** — one incremental poll (10 s per-user cooldown); if an AC is
  found the attempt is finalized publicly in the channel.
- **Give Up** — asks for confirmation if less than 20% of the budget has
  elapsed; otherwise does one final poll before logging the timeout.
- **Ignore** — deletes the session without logging an attempt.

---

## How the model works (plain English)

Each user is summarised by eight interpretable parameters (all in Codeforces
rating points unless noted):

| Parameter | Meaning |
| --- | --- |
| `s30` | The rating you solve with 50% probability in 30 minutes on a **modern** problem. |
| `gamma` | Era discount: how many points easier a problem older than 7.5 years feels than a modern problem of the same nominal rating. |
| `delta[0..4]` | Per-archetype handicap. Positive means you are slower/weaker than your rating suggests on that topic; negative means faster/stronger. |
| `contest_offset` | Extra effective difficulty of a live-contest solve versus relaxed practice. Positive means you are slower in contests. |
| `kappa` | Time → rating exchange rate (rating points per unit of log-time; 180 ≈ 125 points per doubling). |
| `k` | Log-logistic consistency shape (default 2.5). |

For an attempt at nominal rating `R` with elapsed time `t`, the model computes

```text
r_eff = R - gamma * era_factor + mean(delta[archetype]) + contest_offset * is_contest
z     = k * ln(t / 30) + (k / kappa) * (s30 - r_eff)
p     = sigmoid(z)
```

A solve contributes a log-logistic density and a give-up contributes a
right-censored survival term. Elapsed time already includes time lost to wrong
answers, so fitting adds **zero** minutes per WA (`FIT_WA_PENALTY_MINS = 0`);
the `+5m per WA` "contest-equivalent time" is display-only.

The eight rating-space parameters enter linearly, so the weighted likelihood is
concave. Gaussian priors (`s30 ±150`, `gamma ~ N(200, 75)`, `delta ~ N(0, 60)`,
`contest_offset ~ N(0, 50)`) make the log-posterior strictly concave. The bot
finds the unique MAP estimate with Newton's method (Gaussian elimination,
partial pivoting, step clipping and a backtracking line search) and reports
standard errors from the inverse Hessian. Observations decay with a 75-day
half-life (floor 5%), and contest-derived rows carry a 0.7 bootstrap weight.

**Cold start.** `/link` reads `user.rating`, `contest.list` and the problemset
first, then makes one paged pass over `user.status` (1000 rows per page, no date
cutoff, capped at 20,000 submissions). Inside that pass it caches every AC in
`cf_solved` and keeps live (`CONTESTANT`) and virtual (`VIRTUAL`) participation
submissions from the 40 most recent participation sessions. The pure
bootstrapper reconstructs clean sequential solve times
(`actual = this AC - previous AC`) and one end-of-contest right-censored timeout
when all post-last-AC attempts target a single unsolved problem. Virtual
sessions ignore problems already solved before the virtual started, use the
virtual play time for `occurred_at` (recency decay), and keep the official
contest date for `contest_start_ts` (era discounting). Incremental polls use a
15-minute overlap so a `TESTING` submission that turns `OK` after the cursor
advanced is not missed.

**Backtest.** `/calibration` refits on each attempt's preceding history (most
recent 300 records), predicts `P(solve <= budget)` out of sample, and reports
log-loss (vs. the 0.6931 coin-flip baseline), Brier score and 10 reliability
bins.

---

## Data, memory and threading

- **SQLite** lives at `{BOT_DATA_DIR}/bot.db` (Docker: `/app/data/bot.db`), in
  WAL mode, with one connection guarded by a `threading.Lock`. `PRAGMA
  user_version` tracks migrations; a pre-survival `data/bot.db` schema is
  detected and migrated by dropping the old tables. `bot.close()` closes the
  connection cleanly.
- **Problemset cache** — `{BOT_DATA_DIR}/.cache/problemset_compact.json` is a
  compact `[contestId, index, name, rating, tags]` list written immediately
  after parsing. A 24-hour TTL applies; corrupt files are refetched; a stale
  file is used with a warning if the API is down. Restarts never parse the
  bulky raw Codeforces response.
- **Submission paging** — `user.status` is consumed one page at a time and
  reduced to compact tuples before the next page is requested, so at most one
  raw JSON page is alive at once.
- **`/link` concurrency** — a process-wide `asyncio.Semaphore(1)` serialises
  `/link` scans so two concurrent calls cannot multiply peak memory.
- **Codeforces rate limit** — one process-wide `threading.Lock` +
  `time.monotonic` limiter enforces at least 2.05 s between *any* two requests,
  shared by `/link`, the poller and button handlers. Transient HTTP 429/5xx,
  network errors and JSON `{"status":"FAILED", "comment":"... limit exceeded"}`
  responses are retried with exponential backoff and jitter (max 5 tries).
- **Discord memory** — `Intents.default()` with `members=False`,
  `presences=False` and `message_content=True` (prefix commands),
  `MemberCacheFlags.none()`, `chunk_guilds_at_startup=False` and
  `max_messages=None`.
- **Event loop** — every Codeforces call and every `CFPredictor.fit()` /
  `walk_forward_backtest()` invocation from `bot.py` runs inside
  `asyncio.to_thread`. SQLite reads are local and sub-millisecond.

---

## Docker deployment (Hetzner VPS)

### 1. Create the server

A small Hetzner Cloud CX22 (2 vCPU / 4 GB) running Ubuntu 24.04 is plenty. Only
SSH needs to be reachable; the bot makes outbound HTTPS connections to Discord
and Codeforces.

```bash
ssh root@YOUR_SERVER_IP
curl -fsSL https://get.docker.com | sh
```

### 2. Get the project and configure it

```bash
git clone <your-repository-url> ihave
cd ihave
cp .env.example .env
nano .env             # DISCORD_TOKEN=..., optionally DEFAULT_CF_HANDLE=...
```

Never commit `.env`; it is excluded by `.gitignore` and `.dockerignore`.

### 3. Start it

```bash
docker compose up -d --build
docker compose logs -f
```

The container runs as non-root `appuser` (UID 1000). `BOT_DATA_DIR=/app/data` is
backed by the named volume `bot_data`, so the SQLite database, its `-wal`/`-shm`
files and `.cache/` survive restarts and upgrades.

Common operations:

```bash
docker compose restart          # or: docker compose up -d --build after a pull
docker compose down             # keeps the bot_data volume
docker compose ps
docker compose logs --tail=200
```

### 4. Backups

Stop the bot briefly or use a volume snapshot, then archive the volume:

```bash
docker compose stop
docker run --rm -v ihave_bot_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/ihave-backup.tgz -C /data .
docker compose start
```

To use a host bind mount instead of the named volume, uncomment the
`user: "${UID:-1000}:${GID:-1000}"` and `./data:/app/data` lines in
`docker-compose.yml`, create the directory, and make it writable by UID 1000:

```bash
mkdir -p ./data && sudo chown -R 1000:1000 ./data
```

Both options keep SQLite WAL files writable for the non-root container user.

---

## Systemd user service (no Docker / no root)

When Docker is unavailable but the account can run a user-level systemd service,
this is the smallest production-grade setup. It also survives logout and reboots
once lingering is enabled.

```bash
cd ~/ihave-bot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p data

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ihave.service <<'UNIT'
[Unit]
Description=IhHave Codeforces Discord Bot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=%h/ihave-bot
Environment=BOT_DATA_DIR=%h/ihave-bot/data
Environment=LOG_LEVEL=INFO
Environment=PYTHONUNBUFFERED=1
ExecStart=%h/ihave-bot/.venv/bin/python %h/ihave-bot/bot.py
Restart=always
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGTERM
SyslogIdentifier=ihave-bot

[Install]
WantedBy=default.target
UNIT

loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now ihave.service
```

Manage and inspect it without root:

```bash
systemctl --user status ihave.service
systemctl --user restart ihave.service
journalctl --user -u ihave.service -f
```

`loginctl enable-linger "$USER"` is what makes the user manager (and the bot)
start after a reboot. If that command is denied on a hardened host, the same unit
can be installed system-wide as `/etc/systemd/system/ihave.service` with
`User=deploy` and `WorkingDirectory=/home/deploy/ihave-bot`.

---

## Database location

| Context | Path |
| --- | --- |
| Local default | `./bot.db` (because `BOT_DATA_DIR` defaults to `.`) |
| Docker | `/app/data/bot.db` (named volume `bot_data`) |
| Legacy override | `$BOT_DB_PATH` if set |

The compact caches live next to it in `{BOT_DATA_DIR}/.cache/`:
`problemset_compact.json` (24 h TTL) and `contests_compact.json` (24 h TTL).

---

## Testing

The unit tests are fully offline: synthetic data only, no Codeforces calls and
no Discord connection.

```bash
# Windows
.venv\Scripts\python -m pytest -q
# Linux/macOS
.venv/bin/python -m pytest -q
```

Eighteen tests across three files:

- `tests/test_predictor.py` — tag/era mapping, overtime vs. fast solves,
  right-censored timeouts, era discount, archetype divergence, decay weights,
  analytic gradient/Hessian vs. finite differences, determinism and standard
  errors, WA non-double-counting, contest-offset learning, walk-forward
  backtest, exact rating↔time inverses, duration formatting.
- `tests/test_db.py` — practice re-attempts, bootstrap idempotency, atomic
  `complete_session`, `occurred_at` → `days_ago`, incremental `cf_solved` and
  solved-key union.
- `tests/test_cf_data.py` — live-contest and virtual-contest bootstrap
  extraction, pre-solved virtual problem filtering, problem selector, and the
  15-minute incremental-sync overlap rule.

A production image intentionally installs only `requirements.txt`; `pytest`
lives in `requirements-dev.txt`.

---

## Project layout

| File | Responsibility |
| --- | --- |
| `bot.py` | Discord layer only: slash + prefix commands, buttons, poller, refits via `asyncio.to_thread`. |
| `predictor.py` | Pure standard-library survival model, tag/era mapping, Newton MAP fit, backtest. |
| `db.py` | SQLite persistence, migrations, atomic session completion, clean shutdown. |
| `cf_data.py` | Codeforces HTTP client, compact cache, problem selector, cold-start bootstrapper. |
| `tests/` | Offline unit tests. |
| `requirements.txt` / `requirements-dev.txt` | Runtime pins / runtime + pytest. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Non-root container and persistent volume. |

---

## Known limitations

1. **Silent give-ups are invisible.** If a user opens a problem, reads it, and
   never submits, Codeforces records nothing. Contest-derived data therefore
   omits the hardest non-attempts and biases `s30` upward.
2. **Bootstrap solve times include reading time on other problems.** A contest
   solve measured as "time since the previous AC" also contains reading/thinking
   on other problems, so contest rows overstate pure solve time. Right-censored
   end-of-contest timeouts include the same noise.
3. **Give-ups are informative, not missing-at-random.** People tend to quit when
   they feel stuck, so right-censoring is correlated with difficulty; the model
   treats a timeout as "solve time > elapsed", which is the best it can do
   without observing the counterfactual.
4. **Blind mode is a Discord-side courtesy.** Rating, effective rating and
   archetype are spoiler-tagged inside the embed and the bot links to
   `/contest/<id>/problem/<index>`. If Codeforces shows tags on the problem page
   under your account settings, they remain visible in the browser.
5. **`/link` trusts the handle it is given.** There is no profile-token or OAuth
   ownership verification; the design target is a personal bot or a trusted
   private Discord server. Anyone who can run `/link` can claim a handle.
