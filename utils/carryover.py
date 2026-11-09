"""
utils/carryover.py
==================
Keeps a person's file and cleaning steps when they sign in with Google.

The problem
-----------
Google sign-in (st.login) leaves the app, goes to Google, and comes back
to the app's home address as a brand-new visit. A new visit starts with
empty session state, so the uploaded file and every cleaning step were
lost, and the person had to start over.

How this fixes it
-----------------
1. Every visit gets a random "ticket" (like a coat-check ticket). A tiny
   script saves it in a browser cookie named sw_carry.
2. When the person clicks "Sign in with Google", save_and_sign_in() puts
   their data (the file, the cleaned version, the action history, the
   undo steps, the AI conversation) in a server-side holding area under
   that ticket, then starts the sign-in.
3. When Google sends them back, restore_after_sign_in() reads the ticket
   from the cookie, takes the data back out, and puts it into the new
   visit's session state.

Privacy and memory
------------------
- The ticket is a long random string, so nobody can guess another
  person's ticket.
- Each ticket works once: restoring removes the data from the holding area.
- Unclaimed data is thrown away after 15 minutes, and at most
  MAX_WAITING visits are held at once (the oldest go first).
- Nothing is written to disk or to the database. Signing OUT still
  clears everything, on purpose.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import secrets     # makes the long random ticket
import threading   # a lock, because several visitors can sign in at once
import time        # for the 15-minute expiry

import streamlit as st

# ── Settings ───────────────────────────────────────────────────────────────────
COOKIE_NAME   = "sw_carry"   # browser cookie that holds the ticket
WAIT_SECONDS  = 15 * 60      # how long data waits for its owner to come back
MAX_WAITING   = 20           # most visits held at once (protects server memory)

# The session state keys that make up "your work". Results that the app
# can recalculate (quality score, charts, PDF) are left out on purpose.
KEYS_TO_KEEP = [
    "original_df",       # the file as uploaded
    "working_df",        # the cleaned version
    "col_types",         # detected column types
    "cleaning_log",      # Action History
    "version_history",   # undo steps
    "dataset_name",      # file name shown in the sidebar
    "date_warnings",     # unreadable-date notes for the Understand tab
    "loaded_file_id",    # which upload this was
    "analyst_history",   # the AI conversation so far
    "reset_round",       # keeps the "I want to start over" box in step
]


@st.cache_resource
def _holding_area() -> dict:
    """
    One shared holding area for the whole app (st.cache_resource makes
    every visit see the same dict). Each entry is:
        ticket -> (time it expires, {session key: value})
    """
    return {"lock": threading.Lock(), "items": {}}


def _drop_old(items: dict, now: float):
    """Remove expired entries, then the oldest ones if there are too many."""
    for ticket in [t for t, (expires, _) in items.items() if expires < now]:
        del items[ticket]
    while len(items) >= MAX_WAITING:
        oldest = min(items, key=lambda t: items[t][0])
        del items[oldest]


def give_browser_a_ticket():
    """
    Make sure this visit has a ticket saved in the browser's cookie.

    Call this on every run while the person is signed out. The cookie
    script is only sent once per visit. st.html with
    unsafe_allow_javascript=True runs the one-line script in the page,
    which writes the cookie (path=/ so the whole app can read it,
    max-age=3600 so it disappears after an hour).
    """
    if "carry_ticket" not in st.session_state:
        st.session_state.carry_ticket = secrets.token_urlsafe(24)
    if not st.session_state.get("carry_cookie_sent"):
        ticket = st.session_state.carry_ticket
        st.html(
            f"<script>document.cookie = '{COOKIE_NAME}={ticket}; path=/; "
            f"max-age=3600; SameSite=Lax';</script>",
            unsafe_allow_javascript=True,
            width=1,
        )
        st.session_state.carry_cookie_sent = True


def save_and_sign_in():
    """
    on_click for every "Sign in with Google" button.

    Puts the current file and cleaning steps in the holding area under
    this visit's ticket, then starts Google sign-in. If no file is loaded,
    it simply starts the sign-in.
    """
    ticket = st.session_state.get("carry_ticket")
    if ticket and "working_df" in st.session_state:
        work = {k: st.session_state[k] for k in KEYS_TO_KEEP if k in st.session_state}
        area = _holding_area()
        with area["lock"]:
            now = time.time()
            _drop_old(area["items"], now)
            area["items"][ticket] = (now + WAIT_SECONDS, work)
    st.login()


def restore_after_sign_in() -> bool:
    """
    Put the file and cleaning steps back after Google sends the person
    back to the app. Call this near the top of app.py, before the sidebar.

    Only looks once per visit, and only when no file is loaded yet.
    Returns True if something was put back.
    """
    if "working_df" in st.session_state or st.session_state.get("carry_checked"):
        return False
    st.session_state.carry_checked = True

    # st.context.cookies holds the cookies the browser sent with this visit
    ticket = st.context.cookies.get(COOKIE_NAME)
    if not ticket:
        return False

    area = _holding_area()
    with area["lock"]:
        entry = area["items"].pop(ticket, None)   # pop: each ticket works once
    if entry is None:
        return False
    expires, work = entry
    if expires < time.time():
        return False

    for key, value in work.items():
        st.session_state[key] = value
    return True
