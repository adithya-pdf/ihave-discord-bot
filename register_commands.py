#!/usr/bin/env python3
"""Register the bot's slash commands with Discord.

Run this once after deploying (or when command definitions change):

    python register_commands.py

Environment:
    DISCORD_BOT_TOKEN          bot token from the Developer Portal
    DISCORD_APPLICATION_ID     application id
    GUILD_ID                   optional: register in one guild for instant updates
"""

from __future__ import annotations

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()

API = "https://discord.com/api/v10"

COMMANDS = [
    {
        "name": "link",
        "description": "Link a Codeforces handle and build your solve-time curve.",
        "options": [
            {
                "name": "handle",
                "description": "Codeforces handle (case-insensitive)",
                "type": 3,
                "required": True,
                "min_length": 1,
                "max_length": 64,
            }
        ],
    },
    {
        "name": "ihave",
        "description": "Pick a recent unsolved problem for a time budget.",
        "options": [
            {
                "name": "minutes",
                "description": "Minutes available (5-180)",
                "type": 4,
                "required": True,
                "min_value": 5,
                "max_value": 180,
            },
            {
                "name": "rating",
                "description": "Optional rating override (800-3500)",
                "type": 4,
                "required": False,
                "min_value": 800,
                "max_value": 3500,
            },
        ],
    },
    {
        "name": "adjust",
        "description": 'Recalibrate: "I can solve this rating in this many minutes."',
        "options": [
            {
                "name": "rating",
                "description": "Problem rating (800-3500)",
                "type": 4,
                "required": True,
                "min_value": 800,
                "max_value": 3500,
            },
            {
                "name": "minutes",
                "description": "Minutes you consistently need at that rating",
                "type": 10,
                "required": True,
                "min_value": 1,
                "max_value": 600,
            },
        ],
    },
    {
        "name": "curve",
        "description": "Show your smoothed minutes-per-rating curve.",
        "options": [
            {
                "name": "handle",
                "description": "Optional Codeforces handle (defaults to your linked one)",
                "type": 3,
                "required": False,
                "min_length": 1,
                "max_length": 64,
            }
        ],
    },
    {
        "name": "status",
        "description": "Show the active session and check for an Accepted submission.",
    },
    {
        "name": "giveup",
        "description": "End the active session and reveal the problem rating and tags.",
    },
    {
        "name": "sync",
        "description": "Refresh submissions, curve and the problemset cache.",
    },
    {
        "name": "help",
        "description": "Show the command reference.",
    },
]


async def main() -> int:
    token = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    application_id = (os.getenv("DISCORD_APPLICATION_ID") or "").strip()
    guild_id = (os.getenv("GUILD_ID") or "").strip()

    if not token or not application_id:
        print(
            "Set DISCORD_BOT_TOKEN and DISCORD_APPLICATION_ID in .env first.",
            file=sys.stderr,
        )
        return 1

    if guild_id:
        url = f"{API}/applications/{application_id}/guilds/{guild_id}/commands"
        scope = f"guild {guild_id}"
    else:
        url = f"{API}/applications/{application_id}/commands"
        scope = "global"

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.put(url, json=COMMANDS) as response:
            text = await response.text()
            if response.status >= 300:
                print(f"Discord rejected the command sync ({response.status}):", file=sys.stderr)
                print(text[:4000], file=sys.stderr)
                return 1
            print(f"Registered {len(COMMANDS)} commands ({scope}).")
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
