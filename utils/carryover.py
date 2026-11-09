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

Why not a cookie read by the server
-----------------------------------
The first version kept a ticket in a browser cookie and read it with
st.context.cookies. That works on a laptop, but Streamlit Community
Cloud's proxy removes most cookies before they reach the app, so the live
app never saw the ticket. (Streamlit staff confirmed this on the forum:
discuss.streamlit.io, "st.context.cookies.get() works perfectly in local
but collects an empty dict once deployed in Cloud Community".)

How this version works
----------------------
1. Every visit gets a random "ticket" (like a coat-check ticket).
2. A small script in the page watches the two "Sign in with Google"
   buttons. The moment one is clicked, it saves the ticket in the
   browser's own storage (localStorage), which stays on the computer.
3. The button's on_click, save_and_sign_in(), puts the person's work
   (the file, the cleaned version, the action history, the undo steps,
   the AI conversation) in a server-side holding area under that ticket,
   then starts the sign-in.
4. When Google sends them back, the same script finds the saved ticket,
   deletes it from storage, and reloads the app once with the ticket in
   the address (?carry=...). The app reads it with st.query_params, which
   works the same on the laptop and on Streamlit Cloud.
5. restore_after_sign_in() takes the work back out of the holding area,
   puts it into the new visit, and removes ?carry from the address.

Privacy and memory
------------------
- The ticket is a long random string, so nobody can guess another
  person's ticket. It holds no personal information.
- Each ticket works once: restoring removes the work from the holding area.
- Unclaimed work is thrown away after 15 minutes, and at most
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
STORAGE_KEY   = "sw_carry_ticket"   # name used in the browser's localStorage
URL_PARAM     = "carry"             # ?carry=<ticket> on the way back
WAIT_SECONDS  = 15 * 60             # how long work waits for its owner
MAX_WAITING   = 20                  # most visits held at once (server memory)

# The "Sign in with Google" buttons the page script watches. Streamlit gives
# a widget with key="X" the CSS class "st-key-X" (the keys are in utils/auth.py).
SIGN_IN_BUTTON_KEYS = ["auth_sidebar_login", "auth_tab_login"]

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


def _ticket() -> str:
    """This visit's random ticket (made on first use, then kept)."""
    if "carry_ticket" not in st.session_state:
        st.session_state.carry_ticket = secrets.token_urlsafe(24)
    return st.session_state.carry_ticket


def add_page_script():
    """
    Add the small page script that carries the ticket through sign-in.

    Call this on every run when sign-in is set up (app.py does it in the
    sidebar). st.html with unsafe_allow_javascript=True runs the script in
    the page. It does two jobs:

    1. Watches the sign-in buttons. On a click, it saves this visit's
       ticket in localStorage, just before the app leaves for Google.
    2. Only when no file is loaded: if a saved ticket is waiting (the
       person just came back from Google), it deletes it from storage and
       reloads the app once with ?carry=<ticket> in the address.

    Every browser step is wrapped in try/catch, so a browser that blocks
    storage simply skips this feature instead of breaking the page.
    """
    ticket = _ticket()
    no_file_loaded = "working_df" not in st.session_state

    # CSS selectors for the watched buttons, e.g. ".st-key-auth_tab_login"
    selectors = ", ".join(f".st-key-{key}" for key in SIGN_IN_BUTTON_KEYS)

    # Part 2 of the script, only sent while no file is loaded
    return_check = f"""
        var waiting = null;
        try {{ waiting = localStorage.getItem('{STORAGE_KEY}'); }} catch (e) {{}}
        if (waiting) {{
            try {{ localStorage.removeItem('{STORAGE_KEY}'); }} catch (e) {{}}
            var url = new URL(window.location.href);
            url.searchParams.set('{URL_PARAM}', waiting);
            window.location.replace(url.toString());
        }}
    """ if no_file_loaded else ""

    st.html(
        f"""<script>
        (function () {{
            // The current visit's ticket (updated on every run)
            window.swCarryTicket = '{ticket}';

            // Part 1: watch the sign-in buttons (added only once per page)
            if (!window.swCarryWatching) {{
                window.swCarryWatching = true;
                document.addEventListener('click', function (event) {{
                    var button = event.target.closest && event.target.closest('{selectors}');
                    if (button) {{
                        try {{ localStorage.setItem('{STORAGE_KEY}', window.swCarryTicket); }} catch (e) {{}}
                    }}
                }}, true);   // true = see the click before Streamlit handles it
            }}

            // Part 2: back from Google? Hand the ticket to the app.
            {return_check}
        }})();
        </script>""",
        unsafe_allow_javascript=True,
        width=1,
    )


def save_and_sign_in():
    """
    on_click for every "Sign in with Google" button.

    Puts the current file and cleaning steps in the holding area under
    this visit's ticket, then starts Google sign-in. If no file is loaded,
    it simply starts the sign-in.
    """
    ticket = _ticket()
    if "working_df" in st.session_state:
        work = {k: st.session_state[k] for k in KEYS_TO_KEEP if k in st.session_state}
        area = _holding_area()
        with area["lock"]:
            now = time.time()
            _drop_old(area["items"], now)
            area["items"][ticket] = (now + WAIT_SECONDS, work)
    st.login()


def restore_after_sign_in() -> bool:
    """
    Put the file and cleaning steps back when the page script reloads the
    app with ?carry=<ticket>. Call this near the top of app.py, before the
    sidebar. Returns True if something was put back.
    """
    ticket = st.query_params.get(URL_PARAM)
    if not ticket:
        return False

    # Take the ticket out of the address right away, so a refresh or a
    # shared link never carries it.
    del st.query_params[URL_PARAM]

    if "working_df" in st.session_state:
        return False          # a file is already loaded; leave it alone

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
