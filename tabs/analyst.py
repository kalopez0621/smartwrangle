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
    - Conversation history (so follow-up questions work)

This keeps API costs low, protects data privacy, and produces
focused answers because the model only gets relevant context.
"""

import streamlit as st
import anthropic


# ── Context builder ────────────────────────────────────────────────────────────

def _build_system_prompt(df, col_types: dict, quality_result: dict) -> str:
    """
    Build the system prompt from dataset context already computed by the app.

    This is the most important function in this file. The quality of the
    AI's answers depends entirely on the quality of this context.

    We never send raw data rows — only metadata and statistics.
    This keeps the prompt small, fast, and privacy-safe.
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
    findings = quality_result.get("findings", [])
    findings_text = (
        "\n".join(f"  - {f}" for f in findings[:6])
        if findings else "  - No issues found."
    )

    # ── Assemble the full system prompt ───────────────────────────────────────
    system_prompt = f"""You are an AI Data Analyst embedded inside SmartWrangle, \
a professional data wrangling application.

Your job is to answer questions about the dataset the user has uploaded. \
You have access to the dataset's metadata and statistics — not the raw rows.

Always respond in plain English. Be specific and reference actual column names, \
numbers, and findings from the context below. Never fabricate statistics not \
provided here. If you cannot answer from the context provided, say so clearly \
and explain what additional information would be needed.

Keep answers concise and structured. Use bullet points when listing multiple \
findings. Avoid generic advice — every answer should reference this specific dataset.

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
    # st.secrets reads from .streamlit/secrets.toml locally
    # and from Streamlit Cloud secrets in production.
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        st.error(
            "Anthropic API key not found. "
            "Add ANTHROPIC_API_KEY to your Streamlit secrets."
        )
        return

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Analyst")
    st.caption(
        "Ask any question about your dataset in plain English. "
        "The AI has full context about your data's structure, quality, "
        "and statistics — but never sees your raw data rows."
    )

    # ── Suggested questions ────────────────────────────────────────────────────
    # These give new users a starting point and demonstrate the tab's value.
    st.markdown("**Try asking:**")

    suggested = [
        "What are the biggest data quality issues I should fix first?",
        "Which columns are most useful for analysis and why?",
        "What type of analysis or model would work best for this dataset?",
        "Summarize what this dataset contains in plain English.",
    ]

    # Render suggestions as clickable pills in a single row
    cols = st.columns(len(suggested))
    for i, (col, question) in enumerate(zip(cols, suggested)):
        with col:
            if st.button(
                question,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                # Clicking a suggestion populates it as the user's question
                st.session_state.analyst_prefill = question
                st.rerun()

    st.markdown("---")

    # ── Conversation history display ───────────────────────────────────────────
    # Show all previous messages in the conversation so the user can
    # scroll back and see earlier questions and answers.
    history = st.session_state.analyst_history

    if history:
        for message in history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    # ── Question input ─────────────────────────────────────────────────────────
    # Check if a suggestion was clicked — if so, prefill the input
    prefill = st.session_state.pop("analyst_prefill", "")

    user_question = st.chat_input(
        placeholder="Ask anything about your dataset...",
    )

    # Handle both typed input and suggestion clicks
    question_to_ask = user_question or prefill

    # ── Send question to Claude ────────────────────────────────────────────────
    if question_to_ask:

        # Add user message to history and display it
        st.session_state.analyst_history.append({
            "role": "user",
            "content": question_to_ask
        })

        with st.chat_message("user"):
            st.markdown(question_to_ask)

        # Build the system prompt from current dataset context
        system_prompt = _build_system_prompt(df, col_types, quality_result)

        # Call the Anthropic API
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your dataset..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)

                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        system=system_prompt,
                        messages=st.session_state.analyst_history
                    )

                    # Extract the text response
                    answer = response.content[0].text

                    # Display the answer
                    st.markdown(answer)

                    # Add assistant response to history for follow-up questions
                    st.session_state.analyst_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except anthropic.AuthenticationError:
                    st.error(
                        "Invalid API key. Check your ANTHROPIC_API_KEY in secrets."
                    )
                except anthropic.RateLimitError:
                    st.error(
                        "Rate limit reached. Wait a moment and try again."
                    )
                except Exception as e:
                    st.error(f"API error: {str(e)}")

    # ── Clear conversation button ──────────────────────────────────────────────
    if history:
        st.markdown("---")
        if st.button("Clear conversation", key="clear_analyst"):
            st.session_state.analyst_history = []
            st.rerun()


# ── END OF FILE ────────────────────────────────────────────────────────────────