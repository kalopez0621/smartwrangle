"""
utils/usage.py
==============
Monthly AI question limits for SmartWrangle (Version 2, Stage 2).

What this file does
-------------------
Each signed-in person gets a set number of free AI Analyst questions
per month (20 by default). This file keeps the count in a
small hosted Postgres database (Neon), because Streamlit Community
Cloud erases its own files whenever the app restarts.

How the counting works (personal monthly reset)
-----------------------------------------------
- Each person's monthly cycle starts the day of their FIRST AI question
  (not the day they sign in), the way Stripe renews a subscription.
- It renews on that same day every month. If the month is too short
  (first question on the 31st, then February), it renews on the last
  day of that month instead, then goes back to the 31st when it can.
- One row per person: (email, first question date, current cycle start,
  questions used in the current cycle).
- A question is counted only AFTER the AI answers it. If the AI call
  fails, the person keeps that question.
- When a new cycle begins, the count starts again from 0. This is
  worked out from the dates, so no clean-up job is needed.
- Dates are in UTC, like before.
- The old table (ai_usage, counted by calendar month) is no longer
  used. It is left in place untouched; everyone starts a fresh cycle.
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

import calendar
from datetime import date, datetime, timezone

import streamlit as st

# Used only when [usage] free_questions_per_month is missing from secrets
DEFAULT_FREE_QUESTIONS = 20

# The table that holds the counts. Created automatically the first time
# the app connects, so Karla never has to write any SQL herself.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ai_usage_cycles (
    user_email   TEXT        PRIMARY KEY,         -- lowercased Google email
    anchor_date  DATE        NOT NULL,            -- day of their first AI question (UTC)
    cycle_start  DATE        NOT NULL,            -- first day of the cycle being counted
    questions    INTEGER     NOT NULL DEFAULT 0,  -- answered questions in that cycle
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
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

def today_utc(now=None) -> date:
    """Today's date in UTC."""
    return (now or datetime.now(timezone.utc)).date()


def _renewal_day(year: int, month: int, anchor_day: int) -> date:
    """
    The renewal date in a given month for someone whose cycle started on
    anchor_day. Uses the last day of the month when the month is shorter
    (anchor 31 in April gives April 30; anchor 30 in February gives Feb 28
    or 29).
    """
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(anchor_day, last_day))


def _shift_month(year: int, month: int, step: int):
    """Move (year, month) forward or back by step months."""
    index = year * 12 + (month - 1) + step
    return index // 12, index % 12 + 1


def cycle_start_for(anchor: date, today: date) -> date:
    """
    The first day of the cycle that contains today.

    That is this month's renewal date if it has already come, otherwise
    last month's. It is never earlier than the anchor itself.
    """
    this_month = _renewal_day(today.year, today.month, anchor.day)
    if this_month <= today:
        start = this_month
    else:
        y, m = _shift_month(today.year, today.month, -1)
        start = _renewal_day(y, m, anchor.day)
    return max(start, anchor)


def next_reset_date(anchor: date, today: date) -> date:
    """The date the next cycle begins (the count goes back to 0)."""
    start = cycle_start_for(anchor, today)
    y, m = _shift_month(start.year, start.month, 1)
    return _renewal_day(y, m, anchor.day)


def date_label(d: date) -> str:
    """A date in plain words, like 'December 20'."""
    return f"{d.strftime('%B')} {d.day}"


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


def _read_row(email: str):
    """
    This person's saved row as (anchor_date, cycle_start, questions),
    or None if they have never had a question answered.

    Raises UsageUnavailable if the database cannot be reached.
    """
    with _connect() as conn:
        try:
            return conn.execute(
                "SELECT anchor_date, cycle_start, questions "
                "FROM ai_usage_cycles WHERE user_email = %s",
                (email.strip().lower(),),
            ).fetchone()
        except Exception as e:
            raise UsageUnavailable(f"Could not read usage: {e}") from e


def _status_from_row(row, today: date) -> dict:
    """
    Turn a saved row into this cycle's numbers.

    Returns {"used": int, "reset": date or None}. reset is None for
    someone who has not asked a question yet, since their cycle has not
    started.
    """
    if row is None:
        return {"used": 0, "reset": None}
    anchor, saved_start, questions = row
    current_start = cycle_start_for(anchor, today)
    # A row from an earlier cycle means a new cycle has begun: count from 0
    used = questions if saved_start == current_start else 0
    return {"used": used, "reset": next_reset_date(anchor, today)}


def get_status(email: str, today=None) -> dict:
    """
    Questions used in this person's current cycle and their next reset date.
    Raises UsageUnavailable if the database cannot be reached.
    """
    return _status_from_row(_read_row(email), today or today_utc())


def _cache_key(email: str) -> str:
    """Session-state name for this person's cached row."""
    return f"_usage_row_{email.strip().lower()}"


def record_question(email: str, today=None) -> int:
    """
    Add 1 to this person's count for the current cycle and return the
    new total. Raises UsageUnavailable on database errors.

    Runs inside one transaction with a row lock (SELECT ... FOR UPDATE),
    so two questions arriving at once are both counted correctly:
      - first question ever: creates the row; today becomes the anchor
      - a new cycle has begun: moves cycle_start forward, count = 1
      - same cycle: count + 1
    """
    today = today or today_utc()
    email_key = email.strip().lower()

    with _connect() as conn:
        try:
            with conn.transaction():
                # Create the row on the first question. If it already
                # exists, DO NOTHING keeps the original anchor date.
                conn.execute(
                    """
                    INSERT INTO ai_usage_cycles (user_email, anchor_date, cycle_start, questions)
                    VALUES (%s, %s, %s, 0)
                    ON CONFLICT (user_email) DO NOTHING
                    """,
                    (email_key, today, today),
                )
                anchor, saved_start = conn.execute(
                    "SELECT anchor_date, cycle_start FROM ai_usage_cycles "
                    "WHERE user_email = %s FOR UPDATE",
                    (email_key,),
                ).fetchone()

                current_start = cycle_start_for(anchor, today)
                if saved_start == current_start:
                    sql = "questions = questions + 1"
                else:
                    sql = "questions = 1"      # first question of a new cycle

                row = conn.execute(
                    f"""
                    UPDATE ai_usage_cycles
                    SET {sql}, cycle_start = %s, updated_at = now()
                    WHERE user_email = %s
                    RETURNING anchor_date, cycle_start, questions
                    """,
                    (current_start, email_key),
                ).fetchone()
        except Exception as e:
            raise UsageUnavailable(f"Could not save usage: {e}") from e

    # Keep the cached row in step with the database
    st.session_state[_cache_key(email)] = tuple(row)
    return row[2]


# ── What the AI tab calls ────────────────────────────────────────────────────

def check_allowance(email: str, today=None) -> dict:
    """
    Decide whether this person may ask another question right now.

    Returns a dict:
        allowed    True or False
        unlimited  True for emails on the unlimited list
        used       questions used this cycle (None if unlimited)
        limit      the monthly limit
        remaining  questions left (None if unlimited)
        reset      date the count next resets (None before the first
                   question, or if unlimited or the database is down)
        reason     "ok", "limit_reached", "not_configured", or "unavailable"
    Never raises: database problems come back as reason="unavailable".
    """
    limit = get_monthly_limit()

    if not usage_is_configured():
        return {"allowed": False, "unlimited": False, "used": None,
                "limit": limit, "remaining": None, "reset": None,
                "reason": "not_configured"}

    if is_unlimited(email):
        return {"allowed": True, "unlimited": True, "used": None,
                "limit": limit, "remaining": None, "reset": None,
                "reason": "ok"}

    # Read the row from the database once per visit, then keep it in
    # session state. Streamlit redraws every tab on every click, so without
    # this the app would query the database on each click anywhere in the
    # app, keeping Neon awake and using up its free compute hours.
    # The cycle math runs on every redraw, so a visit that crosses a
    # renewal date still sees the fresh count.
    cache_key = _cache_key(email)
    try:
        if cache_key in st.session_state:
            row = st.session_state[cache_key]
        else:
            row = _read_row(email)
            st.session_state[cache_key] = tuple(row) if row else None
    except UsageUnavailable:
        return {"allowed": False, "unlimited": False, "used": None,
                "limit": limit, "remaining": None, "reset": None,
                "reason": "unavailable"}

    status = _status_from_row(row, today or today_utc())
    used = status["used"]
    remaining = max(0, limit - used)
    return {"allowed": remaining > 0, "unlimited": False, "used": used,
            "limit": limit, "remaining": remaining, "reset": status["reset"],
            "reason": "ok" if remaining > 0 else "limit_reached"}
