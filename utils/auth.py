"""
utils/auth.py
=============
Google sign-in for SmartWrangle (Version 2, Stage 1).

What this file does
-------------------
Only the AI Analyst tab needs a login, because the AI is the only
part of the app that costs money. The Understand, Discover, and
Clean & Export tabs stay open to everyone, with no login.

The sign-in itself is Streamlit's built-in st.login(), which sends
the visitor to Google and back. Streamlit then fills st.user with
the person's name and email from Google.

Where the settings live
-----------------------
st.login() reads ONLY from Streamlit secrets, never from .env:
    - Laptop:          .streamlit/secrets.toml  (listed in .gitignore)
    - Streamlit Cloud: the app's Settings > Secrets box

Both need an [auth] section with these five values:
    redirect_uri, cookie_secret, client_id, client_secret,
    server_metadata_url
See .streamlit/secrets.toml.example for the layout.

Important behavior to know
--------------------------
Signing in and signing out both reload the whole page. Streamlit
starts a fresh session when that happens, so an uploaded file and
any cleaning steps are cleared. That is why the sign-in button
also sits in the sidebar, so people can sign in BEFORE uploading.
"""

import streamlit as st

# Keeps the file and cleaning steps through the Google sign-in round trip
from utils.carryover import save_and_sign_in

# ── Settings the [auth] section must contain ─────────────────────
# If any of these is missing, sign-in cannot work, so the app treats
# login as "not set up" instead of crashing.
REQUIRED_AUTH_KEYS = (
    "redirect_uri",
    "cookie_secret",
    "client_id",
    "client_secret",
    "server_metadata_url",
)


def auth_is_configured() -> bool:
    """
    Check whether Google sign-in has been set up in Streamlit secrets.

    Returns True only when an [auth] section exists and has all five
    required values. Returns False if there is no secrets file at all
    (Streamlit raises an error in that case, so we catch it).
    """
    try:
        auth_settings = st.secrets.get("auth")
    except Exception:
        # No secrets.toml on the laptop, or no Secrets on the cloud app.
        return False

    if not auth_settings:
        return False

    # Every required value must be present and not blank.
    return all(auth_settings.get(key) for key in REQUIRED_AUTH_KEYS)


def is_logged_in() -> bool:
    """
    Return True if the visitor is signed in with Google.

    st.user.is_logged_in only exists when sign-in is set up, so we
    check auth_is_configured() first to avoid an error.
    """
    if not auth_is_configured():
        return False
    return bool(getattr(st.user, "is_logged_in", False))


def get_user_email():
    """
    Return the signed-in person's email address, or None.

    Stage 2 (usage limits) will use this email to count each
    person's AI questions per month.
    """
    if not is_logged_in():
        return None
    return st.user.get("email")


def _display_name() -> str:
    """Pick the friendliest label for the signed-in person."""
    return st.user.get("name") or st.user.get("email") or "your Google account"


def render_sidebar_account():
    """
    Draw the sign-in / sign-out area in the sidebar.

    - Sign-in not set up: show nothing, so the sidebar stays clean.
    - Signed out: a "Sign in with Google" button and a short note.
    - Signed in:  who is signed in and a "Sign out" button.
    """
    if not auth_is_configured():
        return

    st.markdown("**Account**")

    if is_logged_in():
        st.caption(f"Signed in as {_display_name()}")
        # on_click runs st.logout before the page reruns.
        # key= keeps this button separate from any other button.
        st.button(
            "Sign out",
            key="auth_sidebar_logout",
            on_click=st.logout,
            width="stretch",
        )
        st.caption("Signing out reloads the page and clears the uploaded file.")
    else:
        # save_and_sign_in sets the work aside, then starts st.login()
        # (utils/carryover.py explains how the work comes back afterward)
        st.button(
            "Sign in with Google",
            key="auth_sidebar_login",
            on_click=save_and_sign_in,
            width="stretch",
        )
        st.caption(
            "Only needed for the AI Analyst. Your file and cleaning steps "
            "stay when you sign in."
        )

    st.markdown("---")


def require_login_for_ai() -> bool:
    """
    Gate for the AI Analyst tab.

    Returns True when the visitor may use the AI. Otherwise it draws
    a message explaining why not and returns False, and the AI tab
    stops there. It never stops the other tabs.

    If sign-in is not set up, the AI stays locked ("fail closed"),
    so a missing setting can never open the paid AI to everyone.
    """
    if not auth_is_configured():
        st.info(
            "**The AI Analyst requires sign-in, which is not set up yet.**\n\n"
            "Add an `[auth]` section to `.streamlit/secrets.toml` on your "
            "laptop, or to Settings > Secrets on Streamlit Cloud. "
            "See `.streamlit/secrets.toml.example` for the layout.\n\n"
            "The Understand, Discover, and Clean & Export tabs work without sign-in."
        )
        return False

    if not is_logged_in():
        st.markdown("### :material/auto_awesome: AI Analyst")
        st.info(
            "**Sign in with Google to use the AI Analyst.**\n\n"
            "Your file and cleaning steps will still be here after you sign in.\n\n"
            "The other three tabs work without signing in."
        )
        st.button("Sign in with Google", key="auth_tab_login", on_click=save_and_sign_in)
        return False

    return True
