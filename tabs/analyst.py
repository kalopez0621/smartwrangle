"""
tabs/analyst.py
===============
The "AI Analyst" tab — Tab 4 of SmartWrangle 2.0.

What this tab does
------------------
Lets the user ask plain-English questions about their dataset
and get structured, data-aware answers powered by Claude.

The AI never sees the raw data rows. Instead, we build a context
prompt from everything the app has already computed:
    - Dataset name, row count, column count
    - Quality score and grade
    - Column type inventory (col_types dict)
    - Top findings from the quality engine
    - The person's chosen explanation level (Beginner, Some
      experience, or Data pro)
    - Conversation history (so follow-up questions work)

This keeps API costs low, protects data privacy, and produces
focused answers because the model only gets relevant context.
"""

import os
from pathlib import Path

import streamlit as st
import anthropic

# Login gate: only signed-in visitors can use the AI Analyst
from utils.auth import require_login_for_ai, get_user_email

# Monthly question limits (Version 2, Stage 2)
from utils.usage import check_allowance, record_question, date_label, UsageUnavailable

# "How should the AI explain things?" setting (polish Part 4)
from utils.preferences import LEVELS, get_level, save_level, level_instructions

# ── Local .env file location (laptop only) ───────────────────────
# On your laptop the API key lives in a file named .env in the
# project folder, one level up from this file (tabs/ -> project root).
# On Streamlit Cloud there is no .env file, so it is simply skipped.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# ── AI model setting ─────────────────────────────────────────────
# The Claude model the AI Analyst uses. Keep it in ONE place so the
# next model change is a one-line edit.
# claude-haiku-4-5-20251001 may be retired as early as Oct 15, 2026.
# claude-haiku-5-5 replaces it and costs about 10x less per question.
CLAUDE_MODEL = "claude-haiku-5-5"

# ── API key lookup ───────────────────────────────────────────────
def _get_api_key():
    """
    Find the Anthropic API key without ever storing it in the code.

    Where it looks, in order:
        1. Streamlit secrets  (Streamlit Cloud: Settings > Secrets,
                               or a local .streamlit/secrets.toml)
        2. Environment / .env (your laptop's .env file)

    Returns the key as a string, or None if it was not found.
    """
    # Step 1: Streamlit secrets.
    # If no secrets file exists at all, Streamlit raises an error
    # instead of returning None, so we catch it and move on.
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None

    if key:
        return key

    # Step 2: read the local .env file into environment variables.
    # This runs on every check, so a key added while the app is
    # running is picked up without a restart.
    # The try/except keeps the app running if python-dotenv is missing.
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=True)
    except ImportError:
        pass

    return os.environ.get("ANTHROPIC_API_KEY")


# ── Context builder ────────────────────────────────────────────────────────────

def _build_system_prompt(df, col_types: dict, quality_result: dict,
                         level: str = "some") -> str:
    """
    Build the system prompt from dataset context already computed by the app.

    This is the most important function in this file. The quality of the
    AI's answers depends entirely on the quality of this context.

    We never send raw data rows — only metadata and statistics.
    This keeps the prompt small, fast, and privacy-safe.

    level is the person's explanation level ('beginner', 'some', 'pro');
    it adds a short "how to explain" paragraph to the instructions.
    """

    # ── Column inventory by type ───────────────────────────────────────────────
    from engine.detector import get_columns_of_type

    date_cols        = get_columns_of_type(col_types, "date_column")
    numeric_cols     = get_columns_of_type(col_types, "metric", "financial")
    categorical_cols = get_columns_of_type(col_types, "categorical")
    id_cols          = get_columns_of_type(col_types, "id_column")
    text_cols        = get_columns_of_type(col_types, "text", "high_cardinality")

    # ── Basic numeric stats for numeric columns ────────────────────────────────
    numeric_stats = ""
    if numeric_cols:
        stats_lines = []
        for col in numeric_cols[:8]:   # cap at 8 to keep prompt size reasonable
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_lines.append(
                    f"  - {col}: min={col_data.min():.2f}, "
                    f"max={col_data.max():.2f}, "
                    f"mean={col_data.mean():.2f}, "
                    f"median={col_data.median():.2f}"
                )
        if stats_lines:
            numeric_stats = "Numeric column statistics:\n" + "\n".join(stats_lines)

    # ── Missing value summary ──────────────────────────────────────────────────
    missing_cols = [
        f"{col} ({df[col].isnull().mean()*100:.1f}% missing)"
        for col in df.columns
        if df[col].isnull().sum() > 0
    ]
    missing_summary = (
        ", ".join(missing_cols) if missing_cols
        else "No missing values."
    )

    # ── Quality findings ───────────────────────────────────────────────────────
    # score_dataset() does not store sentences under a "findings" key, so the
    # old code always told the AI "No issues found." get_quality_report() turns
    # the score result into the same plain-English findings shown on the
    # Understand tab, so the AI now sees the real problems.
    from engine.quality import get_quality_report
    findings = [f["message"] for f in get_quality_report(quality_result)]
    findings_text = (
        "\n".join(f"  - {f}" for f in findings[:8])   # cap at 8 to keep the prompt short
        if findings else "  - No issues found."
    )

    # ── Assemble the full system prompt ───────────────────────────────────────
    system_prompt = f"""You are an AI Data Analyst embedded inside SmartWrangle, \
a professional data wrangling application.

Your job is to answer questions about the dataset the user has uploaded. \
You have the dataset's metadata and statistics, not the raw rows.

Always respond in plain English. Be specific and reference actual column names, \
numbers, and findings from the context below. Never fabricate statistics not \
provided here. If you cannot answer from the context provided, say so clearly \
and explain what additional information would be needed.

Keep answers concise and structured. Use bullet points when listing multiple \
findings. Avoid generic advice. Every answer should reference this specific dataset.

Length: keep each answer under about 400 words. If a full answer would be \
longer, cover the most important points and offer to go deeper on the rest.

Writing style: never use em dashes (the long dash). Use a period, comma, colon, \
or parentheses instead. Write in plain, simple sentences.

How to explain: {level_instructions(level)}

App names: SmartWrangle's tabs are exactly "Understand", "Discover", \
"Clean & Export", and "AI Analyst". When you point the user to a tab, use \
these exact names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset name   : {st.session_state.get('dataset_name', 'Unknown')}
Rows           : {len(df):,}
Columns        : {len(df.columns)}
Quality score  : {quality_result.get('score', 'N/A')} / 100 \
({quality_result.get('grade', 'N/A')})

Column inventory:
  - Date columns       : {', '.join(date_cols) if date_cols else 'None'}
  - Numeric columns    : {', '.join(numeric_cols) if numeric_cols else 'None'}
  - Category columns   : {', '.join(categorical_cols) if categorical_cols else 'None'}
  - Identifier columns : {', '.join(id_cols) if id_cols else 'None'}
  - Text/high-card cols: {', '.join(text_cols) if text_cols else 'None'}

Missing values : {missing_summary}

{numeric_stats}

Quality findings:
{findings_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    return system_prompt


# ── Small display helpers ─────────────────────────────────────────────────────

# Chat avatars: line icons in the app's blue instead of Streamlit's default
# red and orange. The colors themselves are set in _AVATAR_CSS below.
USER_AVATAR = ":material/person:"
AI_AVATAR   = ":material/auto_awesome:"

# Streamlit draws an icon avatar inside an element whose test id starts with
# "stChatMessageAvatar". This paints that square blue with a white icon.
_AVATAR_CSS = """
<style>
    [data-testid^="stChatMessageAvatar"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    [data-testid^="stChatMessageAvatar"] span,
    [data-testid^="stChatMessageAvatar"] svg {
        color: white !important;
        fill: white !important;
    }
</style>
"""


def _avatar(role: str) -> str:
    """Pick the avatar icon for a chat role ("user" or "assistant")."""
    return USER_AVATAR if role == "user" else AI_AVATAR


# Invisible marker placed just above the newest question, so the page can
# scroll straight to it after an answer arrives. scroll-margin-top leaves
# room for Streamlit's top bar so the question is not hidden under it.
_LATEST_ANCHOR = ('<div id="sw-latest-question" '
                  'style="scroll-margin-top: 70px; height: 0"></div>')


def _show_history(history):
    """
    Draw every earlier question and answer in the conversation, oldest
    first, with an invisible marker just above the newest question.
    """
    # Position of the newest question (None if there are no questions yet)
    last_question = max(
        (i for i, m in enumerate(history) if m["role"] == "user"),
        default=None,
    )
    for i, message in enumerate(history):
        if i == last_question:
            st.markdown(_LATEST_ANCHOR, unsafe_allow_html=True)
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role, avatar=_avatar(role)):
            st.markdown(message["content"])
            if role == "assistant" and message.get("level") in LEVELS:
                st.caption(_level_label(message["level"]))


def _level_label(level: str) -> str:
    """Small note under an answer, like 'Answered for: Beginner'."""
    return f"Answered for: {LEVELS[level]}"


def _for_api(history):
    """
    The conversation in the shape the Claude API expects: only "role" and
    "content". The app also keeps a "level" on each answer for its label,
    and the API would reject that extra field.
    """
    return [{"role": m["role"], "content": m["content"]} for m in history]


def _scroll_to_latest_question():
    """
    Scroll the page so the newest question sits at the top of the screen,
    with its answer right below it. Saves scrolling down to find it.

    The script jumps several times over about 2 seconds, because Streamlit
    may still be drawing the conversation, and stops if the person scrolls.
    """
    st.html(
        """<script>
        (function () {
            // Jump instantly (no animation). A smooth scroll can be stopped
            // partway by the browser while Streamlit is still drawing.
            // Re-check a few times over about 2 seconds in case the page
            // grows or shifts, but stop as soon as the person scrolls
            // or touches the page themselves.
            var userMoved = false;
            function stop() { userMoved = true; }
            window.addEventListener('wheel', stop, {once: true, passive: true});
            window.addEventListener('touchstart', stop, {once: true, passive: true});
            window.addEventListener('keydown', stop, {once: true});

            function jump() {
                if (userMoved) { return; }
                var marker = document.getElementById('sw-latest-question');
                if (marker) { marker.scrollIntoView({behavior: 'auto', block: 'start'}); }
            }
            [50, 300, 700, 1200, 2000].forEach(function (ms) { setTimeout(jump, ms); });
        })();
        </script>""",
        unsafe_allow_javascript=True,
        width=1,
    )


def _show_allowance(allowance: dict):
    """
    Tell the person where they stand with their monthly questions.

    allowance comes from utils.usage.check_allowance().
    """
    reason = allowance["reason"]
    limit = allowance["limit"]

    if reason == "not_configured":
        st.info(
            "**The AI Analyst needs its usage database, which is not set up yet.**\n\n"
            "Add a `[database]` section with the Neon connection string to "
            "`.streamlit/secrets.toml` on your laptop, or to Settings > Secrets "
            "on Streamlit Cloud. See `.streamlit/secrets.toml.example`."
        )
    elif reason == "unavailable":
        st.warning(
            "The AI Analyst can't check your question count right now. "
            "Please try again in a minute. The other tabs still work."
        )
    elif reason == "limit_reached":
        st.info(
            f"**You've used all {limit} free AI questions for this month.**\n\n"
            f"Your questions reset on {date_label(allowance['reset'])}. "
            "The Understand, Discover, and Clean & Export tabs stay free."
        )
    elif allowance["unlimited"]:
        st.caption("Unlimited questions on this account.")
    else:
        _show_allowance_card(allowance["remaining"], limit, allowance["reset"])


def _show_allowance_card(remaining: int, limit: int, reset=None):
    """
    Draw the "questions left" card: a big number, a row of blocks
    (blue = left, grey = used), and the reset date underneath.

    reset is the date the count goes back to the full limit. It is None
    for someone who has not asked yet, because their monthly cycle starts
    on the day of their first question.

    With more than 30 questions the blocks would get too small, so the
    card switches to one continuous bar instead. When 3 or fewer are
    left, the blue turns amber as a gentle heads-up.
    """
    used = limit - remaining
    # "1 of 20 free questions left" reads right, so the word follows the limit
    word = "question" if limit == 1 else "questions"
    fill = "#d97706" if remaining <= 3 else "#2563eb"   # amber when running low
    reset_text = (f"Resets on {date_label(reset)}" if reset
                  else "Your monthly count starts with your first question")

    if limit <= 30:
        # One small block per question: left ones first, then used ones
        blocks = (f'<span class="sw-q-block" style="background:{fill}"></span>' * remaining
                  + '<span class="sw-q-block sw-q-used"></span>' * used)
        meter = f'<div class="sw-q-blocks">{blocks}</div>'
    else:
        pct = 0 if limit == 0 else round(remaining / limit * 100)
        meter = (f'<div class="sw-q-bar"><div style="width:{pct}%; background:{fill}; '
                 f'height:100%; border-radius:4px"></div></div>')

    st.markdown(f"""
    <style>
        /* Card around the counter so it stands out from the page */
        .sw-q-card {{
            display: inline-block;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 18px;
            margin: 4px 0 8px 0;
        }}
        .sw-q-top {{ display: flex; align-items: baseline; gap: 8px; }}
        .sw-q-num {{ font-size: 30px; font-weight: 700; line-height: 1; }}
        .sw-q-label {{ font-size: 14px; color: #475569; font-weight: 500; }}
        .sw-q-blocks {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0 8px 0; max-width: 340px; }}
        .sw-q-block {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
        .sw-q-used {{ background: #e2e8f0; }}
        .sw-q-bar {{ width: 300px; max-width: 100%; height: 10px; background: #e2e8f0;
                     border-radius: 4px; margin: 10px 0 8px 0; }}
        .sw-q-reset {{ font-size: 12.5px; color: #64748b; }}
    </style>
    <div class="sw-q-card" role="status"
         aria-label="{remaining} of {limit} free {word} left this month">
        <div class="sw-q-top">
            <span class="sw-q-num" style="color:{fill}">{remaining}</span>
            <span class="sw-q-label">of {limit} free {word} left this month</span>
        </div>
        {meter}
        <div class="sw-q-reset">{reset_text}</div>
    </div>
    """, unsafe_allow_html=True)


def _show_level_picker(user_email: str) -> str:
    """
    Draw the "How should the AI explain things?" choice and return the
    level to use ('beginner', 'some', or 'pro').

    The choice is saved per person in the database the moment it changes
    and applies to the very next answer. If it was never picked, the AI
    uses "Some experience" and a one-line hint invites the person to pick.
    If the database is down, the AI still works with the default.
    """
    level, status = get_level(user_email)
    widget_key = "explain_level_choice"

    def _on_change():
        # Runs when the person clicks a different level
        choice = st.session_state.get(widget_key)
        if choice:
            saved = save_level(user_email, choice)
            st.session_state["_explain_level_saved"] = saved

    st.segmented_control(
        "How should the AI explain things?",
        options=list(LEVELS),
        format_func=LEVELS.get,          # show "Beginner" instead of "beginner"
        default=level,
        key=widget_key,
        on_change=_on_change,
        help="Beginner: plain words and step-by-step help. "
             "Some experience: a balance. Data pro: short and technical. "
             "You can change this anytime.",
    )

    # Clicking the selected choice again un-selects it. Keep the last level.
    choice = st.session_state.get(widget_key) or level

    if status == "not_set" and "_explain_level_saved" not in st.session_state:
        st.caption("Optional. Pick the level that fits you. "
                   "We'll remember it the next time you sign in.")
    elif st.session_state.get("_explain_level_saved") is False or status == "offline":
        st.caption("Your choice applies to this visit but could not be saved just now.")
    return choice


# ── Main tab renderer ──────────────────────────────────────────────────────────

def render_analyst_tab():
    """
    Render the AI Analyst tab.

    Session state keys this function reads and writes:
        st.session_state.working_df       current dataset
        st.session_state.col_types        column type dict
        st.session_state.quality_result   quality score dict
        st.session_state.analyst_history  conversation history list
    """

    # ── Guard: Google sign-in required (Version 2, Stage 1) ────────────────────
    # The AI is the only part of the app that costs money, so it is the only
    # tab that needs a login. require_login_for_ai() shows a sign-in message
    # and returns False when the visitor is not signed in. It also returns
    # False if sign-in is not set up, so the AI is never left open by mistake.
    if not require_login_for_ai():
        return

    # ── Guard — no dataset loaded ──────────────────────────────────────────────
    if "working_df" not in st.session_state:
        st.info("Upload a dataset using the sidebar to get started.")
        return

    df        = st.session_state.working_df
    col_types = st.session_state.col_types

    # ── Compute quality result if not cached ───────────────────────────────────
    if "quality_result" not in st.session_state:
        from engine.quality import score_dataset
        st.session_state.quality_result = score_dataset(df, col_types)

    quality_result = st.session_state.quality_result

    # ── Initialize conversation history ───────────────────────────────────────
    # analyst_history stores the full conversation as a list of dicts:
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # This is sent to Claude on every call so follow-up questions work correctly.
    if "analyst_history" not in st.session_state:
        st.session_state.analyst_history = []

    # ── Check API key ──────────────────────────────────────────────────────────
    # _get_api_key() checks Streamlit secrets first, then the local .env.
    # If neither has a key, show setup steps instead of crashing.
    # The other three tabs keep working without a key.
    api_key = _get_api_key()
    if not api_key:
        st.info(
            "**The AI Analyst needs an Anthropic API key.**\n\n"
            "- On your laptop: copy `.env.example`, name the copy `.env`, "
            "and paste your key into it. Then refresh this page.\n"
            "- On Streamlit Cloud: open the app's Settings > Secrets and add "
            "`ANTHROPIC_API_KEY = \"your-key\"`.\n\n"
            "The Understand, Discover, and Clean & Export tabs work without a key."
        )
        return

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("### :material/auto_awesome: AI Analyst")
    st.markdown(_AVATAR_CSS, unsafe_allow_html=True)    # blue chat avatars
    st.caption(
        "Ask any question about your dataset in plain English. "
        "The AI has full context about your data's structure, quality, "
        "and statistics, but it never sees your raw data rows."
    )

    # ── Monthly question allowance (Version 2, Stage 2) ────────────────────────
    # Each signed-in person gets a set number of free questions per month.
    # check_allowance() never raises; database problems come back as
    # allowed=False so the AI stays locked instead of becoming unlimited.
    user_email = get_user_email()
    allowance = check_allowance(user_email)
    _show_allowance(allowance)

    if not allowance["allowed"]:
        # Still show the earlier conversation so nothing is lost on screen,
        # but hide the suggestions and the question box.
        _show_history(st.session_state.analyst_history)
        return

    # ── Explanation level (polish Part 4) ──────────────────────────────────────
    # One optional question. The answer shapes how the AI explains things.
    explain_level = _show_level_picker(user_email)

    # ── Suggested questions ────────────────────────────────────────────────────
    # These give new users a starting point and demonstrate the tab's value.
    st.markdown("**Try asking:**")

    suggested = [
        "What are the biggest data quality issues I should fix first?",
        "Which columns are most useful for analysis and why?",
        "What type of analysis or model would work best for this dataset?",
        "Summarize what this dataset contains in plain English.",
    ]

    # Let the button text wrap onto a second line instead of being cut off
    # with "...". The CSS only targets buttons inside the container below
    # (Streamlit gives a container with key="X" the CSS class "st-key-X"),
    # so no other button in the app changes.
    st.markdown("""
    <style>
        .st-key-analyst_suggestions button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .st-key-analyst_suggestions button {
            height: auto !important;
            min-height: 2.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Render suggestions as clickable buttons in 2 rows of 2.
    # Four in one row made each button too narrow to show the full question.
    with st.container(key="analyst_suggestions"):
        cols = st.columns(2) + st.columns(2)       # row 1: 2 columns, row 2: 2 columns
        for i, (col, question) in enumerate(zip(cols, suggested)):
            with col:
                if st.button(
                    question,
                    key=f"suggestion_{i}",
                    width="stretch"
                ):
                    # Clicking a suggestion populates it as the user's question
                    st.session_state.analyst_prefill = question
                    st.rerun()

    st.markdown("---")

    # ── Conversation history display ───────────────────────────────────────────
    # Show all previous messages in the conversation so the user can
    # scroll back and see earlier questions and answers.
    history = st.session_state.analyst_history
    _show_history(history)

    # Just got a new answer? Bring the newest question into view.
    if st.session_state.pop("_analyst_scroll_to_latest", False):
        _scroll_to_latest_question()

    # ── Question input ─────────────────────────────────────────────────────────
    # Check if a suggestion was clicked — if so, prefill the input
    prefill = st.session_state.pop("analyst_prefill", "")

    user_question = st.chat_input(
        placeholder="Ask anything about your dataset...",
    )

    # Handle both typed input and suggestion clicks
    question_to_ask = user_question or prefill

    # ── Send question to Claude ────────────────────────────────────────────────
    # answered is set to True only when the AI replies, so the page can
    # refresh afterward and show the new "questions left" number.
    answered = False
    if question_to_ask:

        # Add user message to history and display it
        st.session_state.analyst_history.append({
            "role": "user",
            "content": question_to_ask
        })

        with st.chat_message("user", avatar=_avatar("user")):
            st.markdown(question_to_ask)

        # Build the system prompt from current dataset context
        system_prompt = _build_system_prompt(df, col_types, quality_result, explain_level)

        # Call the Anthropic API
        with st.chat_message("assistant", avatar=_avatar("assistant")):
            with st.spinner("Analyzing your dataset..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)

                    response = client.messages.create(
                        model=CLAUDE_MODEL,   # set at the top of this file
                        # Room for a full structured answer. 1,024 cut long
                        # answers off mid-sentence. At Haiku's output price this
                        # is still about a tenth of a cent per answer.
                        max_tokens=2048,
                        system=system_prompt,
                        messages=_for_api(st.session_state.analyst_history)
                    )

                    # Extract the text response
                    # Newer models can return several content blocks
                    # (for example a "thinking" block before the text).
                    # Keep only the text blocks and join them together,
                    # instead of assuming the first block is the answer.
                    answer = "".join(
                        block.text
                        for block in response.content
                        if block.type == "text"
                    )

                    # stop_reason "max_tokens" means the answer ran out of
                    # room before it finished. Say so instead of leaving a
                    # sentence hanging.
                    if response.stop_reason == "max_tokens":
                        answer += (
                            "\n\n*This answer reached its length limit and was "
                            "cut short. Ask a narrower question, or ask me to "
                            "continue.*"
                        )

                    # Display the answer, with a note of the level it was written for
                    st.markdown(answer)
                    st.caption(_level_label(explain_level))

                    # Add assistant response to history for follow-up questions.
                    # "level" is only for the label; _for_api() leaves it out.
                    st.session_state.analyst_history.append({
                        "role": "assistant",
                        "content": answer,
                        "level": explain_level,
                    })
                    answered = True

                except Exception as e:
                    # The question got no answer, so take it back out of the
                    # history. Otherwise the next call would send two questions
                    # in a row with no answer between them.
                    st.session_state.analyst_history.pop()

                    # Show a plain-English message for the most common errors
                    if isinstance(e, anthropic.AuthenticationError):
                        st.error(
                            "Invalid API key. Check ANTHROPIC_API_KEY in your "
                            ".env file or Streamlit secrets."
                        )
                    elif isinstance(e, anthropic.RateLimitError):
                        st.error("Rate limit reached. Wait a moment and try again.")
                    else:
                        st.error(f"API error: {str(e)}")

    # ── Count the answered question ────────────────────────────────────────────
    # Counted only after a real answer, so a failed AI call costs nothing.
    # This sits outside the try/except above on purpose: st.rerun() works by
    # raising a special signal, which "except Exception" would swallow.
    if answered:
        if not allowance["unlimited"]:
            try:
                record_question(user_email)
            except UsageUnavailable:
                # The answer was already shown; only the count failed to save.
                st.caption("Note: your question count could not be updated just now.")
                return
        # After the redraw, scroll to the new question and its answer
        st.session_state["_analyst_scroll_to_latest"] = True
        st.rerun()   # redraw so the "questions left" line is current

    # ── Clear conversation button ──────────────────────────────────────────────
    if history:
        st.markdown("---")
        if st.button("Clear conversation", key="clear_analyst"):
            st.session_state.analyst_history = []
            st.rerun()


# ── END OF FILE ────────────────────────────────────────────────────────────────