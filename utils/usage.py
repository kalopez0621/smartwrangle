"""
utils/usage.py
==============
Monthly AI question limits for SmartWrangle (Version 2, Stage 2).

What this file does
-------------------
Each signed-in person gets a set number of free AI Analyst questions
per calendar month (20 by default). This file keeps the count in a
small hosted Postgres database (Neon), because Streamlit Community
Cloud erases its own files whenever the app restarts.

How the counting works
----------------------
- One row per person per month: (email, "2026-11", questions used).
- A question is counted only AFTER the AI answers it. If the AI call
  fails, the person keeps that question.
- A new month means a new row, so counts reset on their own on the
  1st of each month (UTC). No clean-up job is needed.
- Emails on the "unlimited" list (Karla's, for testing) are never
  limited and never counted.

Fail closed
-----------
If the database settings are missing, or the database cannot be
reached, the AI Analyst stays locked with a "try again" message. A
database problem can never turn into unlimited free AI.

Where the settings live
-----------------------
Streamlit secrets (laptop: .streamlit/secrets.toml, cloud: Settings >
Secrets). See .streamlit/secrets.toml.example for the layout:

    [database]
    url = "postgresql://..."        # Neon connection string

    [usage]
    free_questions_per_month = 20
    unlimited_emails = ["you@example.com"]
"""

from datetime import datetime, timezone

import streamlit as st

# Used only when [usage] free_questions_per_month is missing from secrets
DEFAULT_FREE_QUESTIONS = 20

# The table that holds the counts. Created automatically the first time
# the app connects, so Karla never has to write any SQL herself.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ai_usage (
    user_email  TEXT        NOT NULL,          -- who asked (lowercased Google email)
    month       TEXT        NOT NULL,          -- calendar month in UTC, like '2026-11'
    questions   INTEGER     NOT NULL DEFAULT 0,-- answered questions this month
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_email, month)            -- one row per person per month
)
"""


class UsageUnavailable(Exception):
    """Raised when the usage database cannot be reached or is not set up."""


# ── Settings ─────────────────────────────────────────────────────────────────

def _secrets_section(name: str) -> dict:
    """
    Return one section of Streamlit secrets as a plain dict, or {}.

    Streamlit raises an error when there is no secrets file at all,
    so we catch that and treat it as "nothing set up".
    """
    try:
        section = st.secrets.get(name)
    except Exception:
        return {}
    return dict(section) if section else {}


def _database_url():
    """The Neon connection string from [database] url, or None."""
    return _secrets_section("database").get("url") or None


def usage_is_configured() -> bool:
    """True when a database connection string is set in secrets."""
    return _database_url() is not None


def get_monthly_limit() -> int:
    """Free questions per person per month, from [usage] in secrets."""
    value = _secrets_section("usage").get("free_questions_per_month", DEFAULT_FREE_QUESTIONS)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        # A typo in secrets (like "twenty") falls back to the default
        return DEFAULT_FREE_QUESTIONS


def is_unlimited(email: str) -> bool:
    """True if this email is on the [usage] unlimited_emails list."""
    if not email:
        return False
    listed = _secrets_section("usage").get("unlimited_emails", [])
    # Compare in lowercase so "Karla@Gmail.com" matches "karla@gmail.com"
    return email.strip().lower() in {str(e).strip().lower() for e in listed}


# ── Dates ────────────────────────────────────────────────────────────────────

def current_month(now=None) -> str:
    """The current calendar month in UTC, like '2026-11'."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


def next_reset_label(now=None) -> str:
    """The date the counts reset, in plain words, like 'December 1'."""
    now = now or datetime.now(timezone.utc)
    if now.month == 12:
        first_of_next = datetime(now.year + 1, 1, 1)
    else:
        first_of_next = datetime(now.year, now.month + 1, 1)
    return f"{first_of_next.strftime('%B')} {first_of_next.day}"


# ── Database ─────────────────────────────────────────────────────────────────

def _connect():
    """
    Open a connection to the usage database.

    Each call opens a fresh, short connection. Neon's free plan puts the
    database to sleep when idle and wakes it on the next connection, so
    a long-lived connection would often be broken anyway.
    """
    url = _database_url()
    if not url:
        raise UsageUnavailable("No [database] url in secrets.")
    try:
        import psycopg  # imported here so the other tabs never need it
        conn = psycopg.connect(url, connect_timeout=15, autocommit=True)
    except Exception as e:
        raise UsageUnavailable(f"Could not connect: {e}") from e

    # Make sure the table exists. CREATE TABLE IF NOT EXISTS is safe to
    # run every time, but we skip it after the first success per server.
    if not _TABLE_READY:
        try:
            conn.execute(CREATE_TABLE_SQL)
            _mark_table_ready()
        except Exception as e:
            conn.close()
            raise UsageUnavailable(f"Could not create table: {e}") from e
    return conn


# Remembered for the life of the server process
_TABLE_READY = False


def _mark_table_ready():
    """Remember that the table exists, so we skip the check next time."""
    global _TABLE_READY
    _TABLE_READY = True


def get_questions_used(email: str, month=None) -> int:
    """
    How many questions this person has used this month.

    Raises UsageUnavailable if the database cannot be reached.
    """
    month = month or current_month()
    with _connect() as conn:
        try:
            row = conn.execute(
                "SELECT questions FROM ai_usage WHERE user_email = %s AND month = %s",
                (email.strip().lower(), month),
            ).fetchone()
        except Exception as e:
            raise UsageUnavailable(f"Could not read usage: {e}") from e
    return row[0] if row else 0


def _cache_key(email: str, month=None) -> str:
    """Session-state name for this person's cached count this month."""
    return f"_usage_used_{month or current_month()}_{email.strip().lower()}"


def record_question(email: str, month=None) -> int:
    """
    Add 1 to this person's count for the month and return the new total.

    Uses one INSERT ... ON CONFLICT statement, so it creates the row the
    first time and adds 1 every time after, safely even if two questions
    arrive at once. Raises UsageUnavailable on database errors.
    """
    month = month or current_month()
    with _connect() as conn:
        try:
            row = conn.execute(
                """
                INSERT INTO ai_usage (user_email, month, questions)
                VALUES (%s, %s, 1)
                ON CONFLICT (user_email, month)
                DO UPDATE SET questions  = ai_usage.questions + 1,
                              updated_at = now()
                RETURNING questions
                """,
                (email.strip().lower(), month),
            ).fetchone()
        except Exception as e:
            raise UsageUnavailable(f"Could not save usage: {e}") from e

    # Keep the cached count in step with the database
    st.session_state[_cache_key(email, month)] = row[0]
    return row[0]


# ── What the AI tab calls ────────────────────────────────────────────────────

def check_allowance(email: str) -> dict:
    """
    Decide whether this person may ask another question right now.

    Returns a dict:
        allowed    True or False
        unlimited  True for emails on the unlimited list
        used       questions used this month (None if unlimited)
        limit      the monthly limit
        remaining  questions left (None if unlimited)
        reason     "ok", "limit_reached", "not_configured", or "unavailable"
    Never raises: database problems come back as reason="unavailable".
    """
    limit = get_monthly_limit()

    if not usage_is_configured():
        return {"allowed": False, "unlimited": False, "used": None,
                "limit": limit, "remaining": None, "reason": "not_configured"}

    if is_unlimited(email):
        return {"allowed": True, "unlimited": True, "used": None,
                "limit": limit, "remaining": None, "reason": "ok"}

    # Read the count from the database once per visit, then keep it in
    # session state. Streamlit redraws every tab on every click, so without
    # this the app would query the database on each click anywhere in the
    # app, keeping Neon awake and using up its free compute hours.
    cache_key = _cache_key(email)
    try:
        if cache_key in st.session_state:
            used = st.session_state[cache_key]
        else:
            used = get_questions_used(email)
            st.session_state[cache_key] = used
    except UsageUnavailable:
        return {"allowed": False, "unlimited": False, "used": None,
                "limit": limit, "remaining": None, "reason": "unavailable"}

    remaining = max(0, limit - used)
    return {"allowed": remaining > 0, "unlimited": False, "used": used,
            "limit": limit, "remaining": remaining,
            "reason": "ok" if remaining > 0 else "limit_reached"}
