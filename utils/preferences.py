"""
utils/preferences.py
====================
The "How should the AI explain things?" setting (polish Part 4).

What this file does
-------------------
Each signed-in person can pick how the AI Analyst explains things:

    beginner  "Beginner"         plain words, step by step
    some      "Some experience"  a balance (the default)
    pro       "Data pro"         short and technical

The choice is saved per person (by Google email) in one small table,
user_prefs, in the same Neon database that counts AI questions. It
stays saved between visits and can be changed anytime on the AI tab.

Only this one choice is stored. No job titles or personal background.

Fail open (on purpose)
----------------------
Unlike the question counter, this setting is not about money. If the
database cannot be reached, the AI simply uses "Some experience" and
the person can keep working. Nothing is blocked.
"""

import streamlit as st

# Shares the connection settings and the "database is down" error with
# the question counter, so there is still only one [database] secret.
from utils.usage import _connect, UsageUnavailable

# ── The three levels ─────────────────────────────────────────────────────────
# Key saved in the database -> label shown on screen
LEVELS = {
    "beginner": "Beginner",
    "some":     "Some experience",
    "pro":      "Data pro",
}
DEFAULT_LEVEL = "some"

# Extra instructions added to the AI's system prompt for each level.
# Kept short so they shape the answer without crowding out the data.
LEVEL_INSTRUCTIONS = {
    "beginner": (
        "The user is new to data work. Use plain everyday words. When you "
        "must use a data term (like outlier, median, or category column), "
        "explain it in one short sentence the first time. Give fixes as "
        "numbered steps that point to SmartWrangle's tabs and buttons. "
        "Skip formulas and code."
    ),
    "some": (
        "The user has some data experience. Use common data terms without "
        "defining basic ones, and briefly explain advanced ideas. Balance "
        "the why with the how."
    ),
    "pro": (
        "The user is an experienced data professional. Be concise and "
        "technical. Use precise statistical and modeling terms without "
        "defining them. Lead with the key numbers and trade-offs, and skip "
        "beginner explanations."
    ),
}

# Created automatically the first time it is needed
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_prefs (
    user_email     TEXT        PRIMARY KEY,   -- lowercased Google email
    explain_level  TEXT        NOT NULL,      -- 'beginner', 'some', or 'pro'
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# Remembered for the life of the server process
_TABLE_READY = False


def _connect_prefs():
    """Open a database connection and make sure user_prefs exists."""
    global _TABLE_READY
    conn = _connect()          # raises UsageUnavailable if the database is down
    if not _TABLE_READY:
        try:
            conn.execute(CREATE_TABLE_SQL)
            _TABLE_READY = True
        except Exception as e:
            conn.close()
            raise UsageUnavailable(f"Could not create user_prefs: {e}") from e
    return conn


def _cache_key(email: str) -> str:
    """Session-state name for this person's saved level."""
    return f"_explain_level_{email.strip().lower()}"


def get_saved_level(email: str):
    """
    This person's saved level ('beginner', 'some', 'pro'), or None if
    they have never picked one.

    Read from the database once per visit, then kept in session state,
    so clicks elsewhere in the app do not wake the database.
    Raises UsageUnavailable if the database cannot be reached.
    """
    key = _cache_key(email)
    if key in st.session_state:
        return st.session_state[key]

    with _connect_prefs() as conn:
        try:
            row = conn.execute(
                "SELECT explain_level FROM user_prefs WHERE user_email = %s",
                (email.strip().lower(),),
            ).fetchone()
        except Exception as e:
            raise UsageUnavailable(f"Could not read user_prefs: {e}") from e

    level = row[0] if row and row[0] in LEVELS else None
    st.session_state[key] = level
    return level


def get_level(email: str):
    """
    The level the AI should use right now, and whether it was saved.

    Returns (level, status):
        status "saved"    the person picked this level before
        status "not_set"  never picked; using the default
        status "offline"  database down; using the default (fail open)
    Never raises.
    """
    try:
        saved = get_saved_level(email)
    except UsageUnavailable:
        return DEFAULT_LEVEL, "offline"
    if saved is None:
        return DEFAULT_LEVEL, "not_set"
    return saved, "saved"


def save_level(email: str, level: str) -> bool:
    """
    Save this person's choice. Returns True if it was saved.

    One INSERT ... ON CONFLICT statement creates the row the first time
    and updates it after that. Never raises: if the database is down the
    choice still applies for this visit, it just is not remembered.
    """
    if level not in LEVELS:
        return False

    # Apply it for this visit right away, saved or not
    st.session_state[_cache_key(email)] = level

    try:
        with _connect_prefs() as conn:
            conn.execute(
                """
                INSERT INTO user_prefs (user_email, explain_level)
                VALUES (%s, %s)
                ON CONFLICT (user_email)
                DO UPDATE SET explain_level = EXCLUDED.explain_level,
                              updated_at    = now()
                """,
                (email.strip().lower(), level),
            )
        return True
    except Exception:
        return False


def level_instructions(level: str) -> str:
    """The extra system-prompt text for a level (default if unknown)."""
    return LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS[DEFAULT_LEVEL])
