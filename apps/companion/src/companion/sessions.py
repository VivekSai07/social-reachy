"""Session-level memory: when conversations happened and a short recap of
each, so the assistant can be temporally aware ("it's been 2 days") across
process restarts.

Sibling to memory.py, mirroring its on-disk-location convention and
locking pattern, but deliberately not unified into it: the two stores have
genuinely different shapes (deduplicated fact bullets vs. an append-only
timestamped session log), each with exactly one reader/writer call site
(prompts.py / main.py), so a shared interface would be speculative.

Timestamps are stored in milliseconds, matching memory.py's MemoryFact.created_at,
so callers can compare the two without a unit-conversion bug.
"""

from __future__ import annotations

import os
import random
import string
import sqlite3
import logging
import threading
import time
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

SESSIONS_DB_FILENAME = "sessions.v1.db"

_STORE_LOCK = threading.Lock()


def sessions_db_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the sessions database path for this app instance.

    Mirrors memory.memory_path_for_instance()'s resolution exactly, so both
    stores live side by side.
    """
    if instance_path is not None:
        return Path(instance_path).expanduser() / SESSIONS_DB_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "companion" / SESSIONS_DB_FILENAME


def _make_session_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"s_{int(time.time() * 1000)}_{suffix}"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER,
            summary TEXT
        )
        """
    )
    return conn


def start_session(instance_path: str | Path | None = None) -> tuple[str, int]:
    """Record a new session start; returns (session_id, start_ts_ms) to close out later.

    Degrades gracefully on storage errors (mirrors memory.py's contract):
    a broken/locked sessions.v1.db must not prevent the app from starting,
    so on failure this still returns a usable (session_id, start_ts) pair,
    just one that won't be persisted this run.
    """
    session_id = _make_session_id()
    start_ts = int(time.time() * 1000)
    path = sessions_db_path_for_instance(instance_path)
    try:
        with _STORE_LOCK, closing(_connect(path)) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, start_ts, end_ts, summary) VALUES (?, ?, NULL, NULL)",
                (session_id, start_ts),
            )
            conn.commit()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Failed to record session start at %s: %s", path, exc)
    return session_id, start_ts


def end_session(session_id: str, summary: str, instance_path: str | Path | None = None) -> None:
    """Close out a session with its summary. Idempotent: a no-op if already closed."""
    end_ts = int(time.time() * 1000)
    path = sessions_db_path_for_instance(instance_path)
    try:
        with _STORE_LOCK, closing(_connect(path)) as conn:
            conn.execute(
                "UPDATE sessions SET end_ts = ?, summary = ? WHERE session_id = ? AND end_ts IS NULL",
                (end_ts, summary, session_id),
            )
            conn.commit()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Failed to record session end at %s: %s", path, exc)


def _format_elapsed(delta_seconds: float) -> str:
    """Return a short human phrase for a time delta, for direct use in the prompt."""
    minutes = delta_seconds / 60
    hours = minutes / 60
    days = hours / 24
    if minutes < 2:
        return "moments ago"
    if minutes < 60:
        return f"about {int(minutes)} minutes ago"
    if hours < 36:
        rounded_hours = round(hours)
        return f"about {rounded_hours} hour{'s' if rounded_hours != 1 else ''} ago"
    rounded_days = round(days)
    return f"about {rounded_days} day{'s' if rounded_days != 1 else ''} ago"


def format_session_context_for_prompt(instance_path: str | Path | None = None) -> str:
    """Return the prompt fragment describing the most recently completed session.

    Returns "" if no session has ever completed (mirrors
    memory.format_memory_for_prompt()'s empty-string-on-nothing convention;
    also the fallback on any storage error, same degrade-gracefully contract
    as memory.py's _read_memory_file()).
    The `end_ts IS NOT NULL` filter excludes the just-opened current session,
    so this is safe to call right after start_session() in the same run.
    """
    path = sessions_db_path_for_instance(instance_path)
    try:
        with _STORE_LOCK, closing(_connect(path)) as conn:
            row = conn.execute(
                "SELECT end_ts, summary FROM sessions WHERE end_ts IS NOT NULL ORDER BY end_ts DESC LIMIT 1"
            ).fetchone()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Failed to read session context at %s: %s", path, exc)
        return ""

    if row is None:
        return ""

    end_ts, summary = row
    elapsed = _format_elapsed(time.time() - end_ts / 1000)
    lines = [f"You last talked with the user {elapsed}."]
    if summary:
        lines.append(f"What came up then: {summary}")
    return "\n".join(lines)
