"""
tabs/clean_export.py
====================
The "Clean & Export" tab — Tab 3 of SmartWrangle 2.0.

What this tab does
------------------
This tab answers the third question: "What do I do with my data?"

It is the action layer — where the user fixes problems and takes
the data with them. Everything is guided and in plain English.

A user who has never cleaned data before should be able to look at
this tab and know exactly what to click and why.

Structure of this tab
----------------------
    1. Suggested Actions     Auto-detected problems with one-click fixes.
                             SmartWrangle tells the user what to do —
                             they don't have to figure it out themselves.
    2. All Cleaning Tools    All available operations listed clearly with
                             plain-English descriptions and expandable forms.
    3. Transformation Tools  Log scale, scaling, encoding — explained simply.
    4. Action History        A log of everything that has been done so far.
    5. Export                Download cleaned CSV, a text summary report,
                             and a PDF report with charts.

The undo system
---------------
Every action stores a snapshot before making changes.
The Undo button (always visible at the top) restores the last snapshot.

How to add this tab to app.py
------------------------------
    from tabs.clean_export import render_clean_export_tab
    with tab3:
        render_clean_export_tab()
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np

# Engine modules
from engine.cleaner  import (
    save_snapshot, undo_last_action,
    remove_duplicates, drop_column, fill_missing,
    drop_missing_rows, trim_whitespace, standardize_text_case,
    rename_column, log_transform, standard_scale, minmax_scale,
    one_hot_encode, extract_date_part, get_cleaning_suggestions,
    numeric_conversion_preview, convert_to_numeric,
    find_placeholders, placeholder_selected_by_default, blank_placeholders,
    EXTRACT_METHODS, extract_number_preview, extract_number,
    DATE_ORDERS, date_conversion_preview, convert_to_dates,
)
from engine.detector import (
    get_columns_of_type, detect_column_types, count_unreadable_dates,
    infer_expected_type,
)
from engine.quality  import score_dataset

# Utility modules
from utils.sanitizer import sanitize_for_display
from utils.exporter  import df_to_csv_bytes, build_summary_report, report_to_bytes, get_export_filename
from utils.pdf_report import build_pdf_report
from utils.wording   import plural          # "1 value" / "3 values"


def _show_undo_bar():
    """
    Render the undo button and current dataset size at the top of the tab.

    This is always visible so the user always knows they can undo.
    It also shows a quick summary of the current dataset state.
    """
    df      = st.session_state.working_df
    history = st.session_state.get("version_history", [])

    # ── Current state of the data, on its own line above the buttons ─────────
    orig_rows = len(st.session_state.original_df)
    curr_rows = len(df)
    removed   = orig_rows - curr_rows
    missing   = int(df.isnull().sum().sum())

    parts = [
        f"**{curr_rows:,}** rows",
        f"**{len(df.columns)}** columns",
    ]
    if removed > 0:
        parts.append(f"**{removed:,}** rows removed from original")
    if missing > 0:
        parts.append(f"**{missing:,}** missing values remaining")
    else:
        parts.append(":green[:material/check_circle:] No missing values")

    st.markdown("  ·  ".join(parts))

    # ── One tidy row: Undo | start-over tick | Reset ─────────────────────────
    # vertical_alignment="center" lines the checkbox up with the two buttons.
    col_undo, col_confirm, col_reset = st.columns([1, 1, 1], vertical_alignment="center")

    with col_undo:
        # Disable the button if there's nothing to undo
        undo_disabled = len(history) == 0
        if st.button(
            "Undo Last Action",
            icon=":material/undo:",
            disabled=undo_disabled,
            width="stretch",
            type="secondary",
            key="undo_btn"
        ):
            restored, new_history, msg = undo_last_action(history)
            if restored is not None:
                # Replace working_df with the restored version
                st.session_state.working_df    = restored
                st.session_state.version_history = new_history

                # Re-detect column types for the restored data. Without this,
                # undoing (for example) a one-hot encode left the old column
                # list in place and the dropdowns no longer matched the data.
                st.session_state.col_types = detect_column_types(restored)

                # Clear cached results so they recalculate with new data
                for key in ["quality_result", "insights_cache"]:
                    if key in st.session_state:
                        del st.session_state[key]

                # Add to the action log
                st.session_state.cleaning_log.append("Undid the last action")
                st.rerun()   # refresh the page to show updated data

    # ── Reset to the original upload ───────────────────────────────────────────
    # Needs a tick first so it cannot be clicked by accident.
    # It is saved like any other action, so Undo brings the cleaned
    # version back.
    # reset_round is added to the checkbox's key. Adding 1 after a reset
    # draws a fresh, unticked checkbox (Streamlit does not allow changing a
    # checkbox's value after it is drawn).
    reset_round = st.session_state.get("reset_round", 0)
    with col_confirm:
        confirm = st.checkbox("I want to start over", key=f"reset_confirm_{reset_round}")

    with col_reset:
        if st.button(
            "Reset to Original",
            icon=":material/restart_alt:",
            disabled=not confirm,
            width="stretch",
            key="reset_btn",
        ):
            original = st.session_state.original_df.copy()
            restored = original.copy()
            # Re-read the date columns exactly the way the first upload did.
            # detect_column_types() converts dates inside `restored`, and
            # count_unreadable_dates() compares it with `original` to recount
            # the bad-date warning for the Understand tab.
            col_types = detect_column_types(restored)
            st.session_state.date_warnings = count_unreadable_dates(original, restored, col_types)
            # Fresh, unticked confirmation box next time
            st.session_state.reset_round = reset_round + 1
            # Save a snapshot, swap in the original, log it, refresh
            _apply_action(
                restored,
                "Reset to the original upload. Use Undo to bring back your cleaned version."
            )


def _apply_action(new_df: pd.DataFrame, log_message: str):
    """
    Apply a cleaning action — save snapshot, update working_df, log, refresh.

    This helper is called after every cleaning or transformation operation.
    It centralizes the steps that must happen after every change so we
    don't repeat the same code in every button handler.

    Parameters
    ----------
    new_df      : pd.DataFrame   The new DataFrame after the operation.
    log_message : str            Plain-English description of what was done.
    """
    # Save snapshot of the current state BEFORE overwriting it
    # This is what allows undo to work
    st.session_state.version_history = save_snapshot(
        st.session_state.working_df,
        st.session_state.get("version_history", [])
    )

    # Update the working dataset
    st.session_state.working_df = new_df

    # 🔥 Re-run detection so column types update correctly
    st.session_state.col_types = detect_column_types(new_df)

    # Add to the action log
    st.session_state.cleaning_log.append(log_message)

    # Clear cached analysis results so they recalculate with the new data
    for key in ["quality_result", "insights_cache"]:
        if key in st.session_state:
            del st.session_state[key]

    # Show a success message briefly at the top of the page
    st.toast(log_message, icon=":material/check_circle:")

    # Rerun the page to show the updated dataset and recalculated stats
    st.rerun()


def _render_suggested_actions(df: pd.DataFrame, col_types: dict):
    """
    Show auto-detected problems with one-click fix buttons.

    This is the most important part of the tab for non-technical users.
    Instead of making them figure out what to do, SmartWrangle shows
    them exactly what needs fixing and gives them a button to fix it.
    """
    suggestions = get_cleaning_suggestions(df, col_types)

    if not suggestions:
        st.success(
            "No issues detected. Your data looks clean and ready to use.",
            icon=":material/check_circle:",
        )
        return

    st.markdown(f"**{plural(len(suggestions), 'suggestion')} found:**")

    # Let the button text wrap onto more lines instead of being cut off
    # with "...". The CSS only targets buttons inside the container below
    # (Streamlit gives a container with key="X" the CSS class "st-key-X"),
    # the same way the AI Analyst tab does it.
    st.markdown("""
    <style>
        .st-key-suggested_actions button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .st-key-suggested_actions button {
            height: auto !important;
            min-height: 2.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="suggested_actions"):
        _render_suggestion_rows(df, suggestions)


def _render_suggestion_rows(df: pd.DataFrame, suggestions: list):
    """Draw one row per suggestion: the issue on the left, its fix button on the right."""
    for i, s in enumerate(suggestions):
        # Color-code by priority
        if s["priority"] == 1:
            icon = ":red[:material/error:]"       # urgent
        elif s["priority"] == 2:
            icon = ":orange[:material/warning:]"  # moderate
        else:
            icon = ":blue[:material/info:]"       # optional

        # Show the issue and a button to fix it
        col_issue, col_action = st.columns([3, 1])

        with col_issue:
            st.markdown(f"{icon} {s['issue']}")

        with col_action:
            # Each button needs a unique key — we use the operation + index
            btn_key = f"suggest_{s['operation']}_{i}"

            if st.button(s["action"], key=btn_key, width="stretch"):
                # Execute the suggested operation
                op  = s["operation"]
                col = s.get("column")

                if op == "remove_duplicates":
                    new_df, msg = remove_duplicates(df)
                elif op == "fill_missing_median":
                    new_df, msg = fill_missing(df, col, "median")
                elif op == "fill_missing_mode":
                    new_df, msg = fill_missing(df, col, "mode")
                elif op == "drop_missing_rows":
                    new_df, msg = drop_missing_rows(df, col)
                elif op == "log_transform":
                    new_df, msg = log_transform(df, col)
                elif op == "convert_to_numeric":
                    new_df, msg = convert_to_numeric(df, col)
                elif op == "blank_placeholders":
                    # Same default choices the tool below starts with
                    chosen = [(p["column"], p["key"]) for p in find_placeholders(df)
                              if placeholder_selected_by_default(p)]
                    new_df, msg = blank_placeholders(df, chosen)
                else:
                    st.warning(f"Operation '{op}' not yet implemented.")
                    return

                _apply_action(new_df, msg)


def _render_cleaning_tools(df: pd.DataFrame, col_types: dict):
    """
    Render all available cleaning operations with expandable forms.

    Each operation is in its own expander so the list doesn't overwhelm
    users who just want to use the suggested actions.
    """
    all_cols     = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    string_cols  = [
        c for c in all_cols
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
    ]
    date_cols    = get_columns_of_type(col_types, "date_column")

    # ── Remove duplicates ──────────────────────────────────────────────────────
    with st.expander("Remove Duplicate Rows", icon=":material/content_copy:"):
        st.caption(
            "Preview the duplicate rows first, then decide whether to remove them. "
            "Keeps the first occurrence of each duplicate."
        )

        # Count exact duplicates — rows where every column value is identical
        n_dups = int(df.duplicated().sum())

        if n_dups == 0:
            st.success("No duplicate rows found.")

        else:
            # ── Summary line ───────────────────────────────────────────────────
            st.warning(
                f"{plural(n_dups, 'duplicate row')} found out of {len(df):,} total rows "
                f"({n_dups / len(df) * 100:.1f}%)."
            )

            # ── Preview section ────────────────────────────────────────────────
            # df.duplicated(keep=False) marks EVERY copy as True — both the
            # original row AND the duplicate. This means the preview shows
            # pairs of identical rows side-by-side, so you can visually
            # confirm they are genuinely the same record before removing anything.
            #
            # We then sort by all columns so identical rows appear adjacent —
            # Alice row 0 and Alice row 2 will be next to each other, not
            # scattered through the table.
            #
            # The 'Original Row #' column shows the row's position in the
            # full dataset, so you can trace any row back to its source.

            MAX_PREVIEW = 50   # cap preview to avoid flooding the UI

            dup_all = df[df.duplicated(keep=False)].copy()

            # Sort by all columns → identical rows become adjacent
            dup_preview = (
                dup_all
                .sort_values(by=df.columns.tolist())
                .reset_index()
                .rename(columns={"index": "Original Row #"})
            )

            # If there are more than MAX_PREVIEW rows, only show the first 50
            showing_all = len(dup_preview) <= MAX_PREVIEW
            preview_df  = dup_preview.head(MAX_PREVIEW)

            st.markdown("**Preview: identical rows are grouped together**")
            st.caption(
                f"Showing {len(preview_df):,} of {len(dup_preview):,} rows "
                f"involved in duplicates. "
                f"{'All duplicates shown.' if showing_all else f'First {MAX_PREVIEW} shown. Remove them to see the rest.'}"
            )

            # Display the preview table
            # width makes it fill the available space
            st.dataframe(preview_df, width="stretch", hide_index=True)

            st.markdown(
                "_The rows above are grouped in pairs (or larger groups). "
                "Each group contains identical records. "
                "The first row in each group will be **kept**, and "
                "the others will be **removed**._"
            )

            # ── Remove button ──────────────────────────────────────────────────
            st.markdown("---")
            if st.button(
                f"Remove {plural(n_dups, 'duplicate row')}",
                key="clean_dups",
                type="primary"
            ):
                new_df, msg = remove_duplicates(df)
                _apply_action(new_df, msg)

    # ── Mark placeholder values as blank ───────────────────────────────────────
    with st.expander("Mark Placeholder Values as Blank", icon=":material/block:"):
        st.caption(
            "Some files type a word like ERROR, UNKNOWN or N/A (or the code -1) "
            "into a cell instead of leaving it empty. Turning these into real blanks "
            "makes the missing-value counts honest, and lets number columns convert."
        )
        found = find_placeholders(df)
        if not found:
            st.success("No placeholder values found.")
        else:
            # One option per (column, placeholder) pair. The option itself is the
            # pair, which does not change; the count is only in the display text.
            # (If the count were part of the option, any other cleaning step that
            # changed a count would make Streamlit forget what was ticked.)
            by_pair = {(p["column"], p["key"]): p for p in found}
            defaults = [pair for pair, p in by_pair.items() if placeholder_selected_by_default(p)]
            picked = st.multiselect(
                "Which values should become blank?",
                options=list(by_pair),
                default=defaults,
                format_func=lambda pair: (
                    f"'{by_pair[pair]['shown_as']}' in {pair[0]} "
                    f"({by_pair[pair]['count']:,} cells)"
                ),
                key="ph_pick",
            )
            # Explain why some options start unticked
            if any(p["is_number"] for p in found):
                st.caption(
                    "**-1 is not selected by default.** Some systems use it for "
                    "\"missing\", but in other data it is a real number. Select it "
                    "only if -1 cannot be a real value in that column."
                )
            if any(not p["is_number"] and p["count"] == 1 for p in found):
                st.caption(
                    "Words that appear only once are not selected by default, "
                    "because they may be real data (for example, a movie titled "
                    "\"Unknown\")."
                )
            n_cells = sum(by_pair[pair]["count"] for pair in picked)
            if st.button(f"Mark {plural(n_cells, 'value')} as blank", key="clean_ph",
                         disabled=not picked):
                new_df, msg = blank_placeholders(df, list(picked))
                _apply_action(new_df, msg)

    # ── Fill missing values ────────────────────────────────────────────────────
    with st.expander("Fill Missing Values", icon=":material/format_color_fill:"):
        st.caption(
            "Replace empty cells with an estimated value. "
            "For numbers, use the median (middle value), since extreme values affect it less. "
            "For categories: use the most common value."
        )
        cols_with_missing = [c for c in all_cols if df[c].isnull().sum() > 0]
        if not cols_with_missing:
            st.success("No missing values found.")
        else:
            col_fm   = st.selectbox("Column", cols_with_missing, key="fm_col")

            # ── Date columns need special handling ─────────────────────────────
            # Filling a date column with a text constant like "None" or "0"
            # corrupts the column — it stores the text string in date cells,
            # which breaks any date-based analysis, charts, and the time trend
            # insight. We restrict options for date columns and explain why.
            col_fm_type = col_types.get(col_fm, "categorical")
            is_date_col = col_fm_type == "date_column"

            if is_date_col:
                # For date columns, only offer mode (most common date) and drop
                st.warning(
                    f"**'{col_fm}' is a date column.** "
                    f"Filling date columns with text values (like '0' or 'None') "
                    f"corrupts the column and breaks all date-based charts. "
                    f"Use **mode** (fills with the most common date) or "
                    f"**remove the affected rows** instead."
                )
                # Each choice is a short key the code uses, shown with a
                # plain label through format_func.
                date_labels = {
                    "mode": "Most common date (recommended)",
                    "drop_rows": "Remove rows with missing dates",
                }
                strategy_key = st.selectbox(
                    "Fill method",
                    list(date_labels),
                    format_func=lambda k: date_labels[k],
                    key="fm_strategy_date"
                )
                const_val = None
            else:
                # Median and mean only make sense for number columns.
                # Offering them for a text column crashed the app, so text
                # columns only get mode (most common value) and constant.
                # Each choice is a short key the code uses, shown with a
                # plain label through format_func.
                fill_labels = {
                    "median":   "Median: the middle value (best for numbers with outliers)",
                    "mean":     "Mean: the average value",
                    "mode":     "Mode: the most common value (best for categories)",
                    "constant": "A value you choose",
                }
                if pd.api.types.is_numeric_dtype(df[col_fm]):
                    fill_options = ["median", "mean", "mode", "constant"]
                else:
                    fill_options = ["mode", "constant"]
                    st.caption(
                        "This is a text column, so median and mean are not offered. "
                        "If it holds numbers stored as text, use "
                        "**Convert Text or Currency to Numbers** first."
                    )
                strategy_key = st.selectbox(
                    "Fill method",
                    fill_options,
                    format_func=lambda k: fill_labels[k],
                    key="fm_strategy"
                )

                if strategy_key == "constant":
                    const_val = st.text_input("Fill value", "0", key="fm_const")
                    # Try to convert to number if it looks like one
                    try:
                        const_val = float(const_val)
                        if const_val == int(const_val):
                            const_val = int(const_val)
                    except ValueError:
                        pass   # keep as string
                else:
                    const_val = None

            if st.button("Fill missing values", key="clean_fill"):
                # Date columns with "drop" strategy use drop_missing_rows, not fill_missing
                if strategy_key == "drop_rows":
                    new_df, msg = drop_missing_rows(df, col_fm)
                else:
                    new_df, msg = fill_missing(df, col_fm, strategy_key, const_val)
                _apply_action(new_df, msg)

    # ── Drop rows with missing values ──────────────────────────────────────────
    with st.expander("Remove Rows With Missing Values", icon=":material/playlist_remove:"):
        st.caption(
            "Remove rows that have empty cells. "
            "Use this when you want to keep only complete records."
        )
        col_dr = st.selectbox(
            "Which column?",
            ["Any column (remove row if any cell is empty)"] + all_cols,
            key="dr_col"
        )
        col_arg = None if col_dr.startswith("Any") else col_dr

        n_affected = (
            df.isnull().any(axis=1).sum()
            if col_arg is None
            else df[col_arg].isnull().sum()
        )

        if n_affected == 0:
            st.success("No rows with missing values found.")
        else:
            st.warning(f"This will remove {n_affected:,} rows.")
            if st.button("Remove rows", key="clean_dr"):
                new_df, msg = drop_missing_rows(df, col_arg)
                _apply_action(new_df, msg)

    # ── Drop column ────────────────────────────────────────────────────────────
    with st.expander("Remove a Column", icon=":material/view_column:"):
        st.caption(
            "Permanently remove a column you don't need. "
            "Common use: remove ID columns or columns with too many missing values."
        )
        col_dc = st.selectbox("Column to remove", all_cols, key="dc_col")
        st.warning(f"This will permanently remove '{col_dc}' from the dataset.")
        if st.button("Remove column", key="clean_dc"):
            new_df, msg = drop_column(df, col_dc)
            _apply_action(new_df, msg)

    # ── Trim whitespace ────────────────────────────────────────────────────────
    with st.expander("Fix Extra Spaces in Text", icon=":material/space_bar:"):
        st.caption(
            "Remove invisible spaces from the start and end of text values. "
            "These can cause the same category to appear as two different ones."
        )
        if not string_cols:
            st.info("No text columns found.")
        else:
            col_tw = st.selectbox("Column", string_cols, key="tw_col")
            if st.button("Trim whitespace", key="clean_tw"):
                new_df, msg = trim_whitespace(df, col_tw)
                _apply_action(new_df, msg)

    # ── Standardize text case ──────────────────────────────────────────────────
    with st.expander("Standardize Text Capitalization", icon=":material/text_fields:"):
        st.caption(
            "Make text consistent. For example, 'american airlines', "
            "'AMERICAN AIRLINES', and 'American Airlines' all become the same."
        )
        if not string_cols:
            st.info("No text columns found.")
        else:
            col_tc  = st.selectbox("Column", string_cols, key="tc_col")
            # Short keys for the code, plain labels on screen
            case_labels = {"title": "Title Case", "lower": "all lowercase",
                           "upper": "ALL UPPERCASE"}
            case_key = st.selectbox(
                "Format",
                list(case_labels),
                format_func=lambda k: case_labels[k],
                key="tc_case"
            )
            if st.button("Standardize capitalization", key="clean_tc"):
                new_df, msg = standardize_text_case(df, col_tc, case_key)
                _apply_action(new_df, msg)

    # ── Rename column ──────────────────────────────────────────────────────────
    with st.expander("Rename a Column", icon=":material/edit:"):
        st.caption("Give a column a cleaner, more readable name.")
        col_rn  = st.selectbox("Column to rename", all_cols, key="rn_col")
        new_rn  = st.text_input("New name", value=col_rn, key="rn_new")
        if st.button("Rename column", key="clean_rn"):
            new_df, msg = rename_column(df, col_rn, new_rn)
            _apply_action(new_df, msg)

    # ── Convert Text or Currency to Numbers ───────────────────────────────────
    with st.expander("Convert Text or Currency to Numbers", icon=":material/attach_money:"):
        st.caption(
            "Turn numbers stored as text (like 7, $2,350.00, or (75.50)) into real "
            "numbers so charts, scaling, and modeling tools can use them. "
            "You will see how many values convert before anything changes."
        )

        # Only show object/text columns
        currency_candidates = [
            c for c in all_cols
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
        ]

        if not currency_candidates:
            st.info("No text columns available for conversion.")
        else:
            col_cc = st.selectbox(
                "Column to convert",
                currency_candidates,
                key="cc_col"
            )

            preview_values = df[col_cc].dropna().astype(str).head(5).tolist()
            # Escape $ so currency samples are not read as math formulas
            st.caption(f"Sample values: {preview_values}".replace("$", "\\$"))

            # ── Preview: what WOULD happen (nothing is changed yet) ───────────
            # Before this check existed, converting a column like "93 min"
            # turned every value into a blank with no warning.
            p = numeric_conversion_preview(df, col_cc)
            # A backslash before $ stops Streamlit from reading "$...$" as a
            # math formula, so values like '$137K-$171K' display correctly.
            examples = ", ".join(f"'{e}'" for e in p["examples"]).replace("$", "\\$")
            can_convert = False

            if p["n_ok"] == 0:
                # Nothing can be read as a number: block the conversion
                st.error(
                    f"None of the {p['n_filled']:,} values in '{col_cc}' can be read "
                    f"as numbers (for example: {examples}), so converting would "
                    f"blank out the whole column. Nothing will be changed."
                )
            elif p["n_lost"] > 0:
                # Some values would be lost: make the user confirm first
                st.warning(
                    f"{p['n_ok']:,} of {p['n_filled']:,} values will become numbers. "
                    f"{p['n_lost']:,} cannot be read as numbers and will become blank "
                    f"(for example: {examples})."
                )
                can_convert = st.checkbox(
                    f"I understand {plural(p['n_lost'], 'value')} will become blank",
                    key="cc_confirm",
                )
            else:
                st.success(f"All {p['n_filled']:,} filled values can be converted.")
                can_convert = True

            # The button stays greyed out until it is safe (or confirmed)
            if st.button("Convert to numbers", key="clean_currency",
                         disabled=not can_convert):
                new_df, msg = convert_to_numeric(df, col_cc)
                _apply_action(new_df, msg)

    # ── Convert text to dates ──────────────────────────────────────────────────
    # Date columns with date-like names are converted automatically on upload.
    # This tool covers the rest, and files that write the day before the month.
    with st.expander("Convert Text to Dates", icon=":material/calendar_month:"):
        st.caption(
            "Turn dates stored as text (like 03/05/2024 or March 5, 2024) into real "
            "dates, so time charts and Extract Year or Month can use them. "
            "You will see how each value is read before anything changes."
        )
        text_cols_dt = [
            c for c in all_cols
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
        ]
        if not text_cols_dt:
            st.info("No text columns found. Columns that are already dates do not need this.")
        else:
            # Columns whose values look like dates go to the top of the list
            looks_like_dates = [c for c in text_cols_dt
                                if infer_expected_type(df[c]) == "date_column"]
            text_cols_dt = looks_like_dates + [c for c in text_cols_dt
                                               if c not in looks_like_dates]

            col_dt = st.selectbox("Column to convert", text_cols_dt, key="dt_col")
            order_dt = st.radio(
                "How are dates like 03/05/2024 written in this file?",
                list(DATE_ORDERS.keys()),
                format_func=lambda k: DATE_ORDERS[k],
                key="dt_order",
            )

            # ── Preview: what WOULD happen (nothing is changed yet) ───────────
            p = date_conversion_preview(df, col_dt, order_dt)

            # Warn when the values themselves show the other order is right
            if p["order_hint"]:
                st.warning(p["order_hint"].replace("$", "\\$"))
            elif p["n_ambiguous"]:
                st.info(
                    f"{plural(p['n_ambiguous'], 'value')} give a different date depending on "
                    f"the order (like 03/05/2024). Make sure the choice above matches "
                    f"how this file writes dates."
                )

            st.dataframe(p["examples"], hide_index=True, width="stretch")

            lost = ", ".join(f"'{e}'" for e in p["lost_examples"]).replace("$", "\\$")
            can_convert_dt = False
            if p["n_ok"] == 0:
                # Nothing can be read as a date: block the conversion
                st.error(
                    f"None of the {p['n_filled']:,} values in '{col_dt}' can be read "
                    f"as dates, so converting would blank out the whole column. "
                    f"Nothing will be changed."
                )
            elif p["n_lost"] > 0:
                # Some values would be lost: make the user confirm first
                st.warning(
                    f"{p['n_ok']:,} of {p['n_filled']:,} values will become dates. "
                    f"{p['n_lost']:,} cannot be read as dates and will become blank "
                    f"(for example: {lost})."
                )
                can_convert_dt = st.checkbox(
                    f"I understand {plural(p['n_lost'], 'value')} will become blank",
                    key="dt_confirm",
                )
            else:
                st.success(f"All {p['n_filled']:,} filled values can be read as dates.")
                can_convert_dt = True

            # The button stays greyed out until it is safe (or confirmed)
            if st.button("Convert to dates", key="clean_dates",
                         disabled=not can_convert_dt):
                new_df, msg = convert_to_dates(df, col_dt, order_dt)
                _apply_action(new_df, msg)

    # ── Pull numbers out of text ───────────────────────────────────────────────
    with st.expander("Pull Numbers Out of Text", icon=":material/pin:"):
        st.caption(
            "For values with words around the number, like '93 min', '4 Seasons' or "
            "'\\$137K-\\$171K (Glassdoor est.)'. The numbers go into a NEW column, so "
            "the original stays as it is."
        )
        text_cols = [
            c for c in all_cols
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
        ]
        if not text_cols:
            st.info("No text columns found.")
        else:
            col_xn = st.selectbox("Column", text_cols, key="xn_col")
            method = st.radio(
                "Which number?",
                options=list(EXTRACT_METHODS),
                # Show the explanation. The backslash before $ stops Streamlit from
                # reading "$137K-$171K" as a math formula.
                format_func=lambda m: EXTRACT_METHODS[m].replace("$", "\\$"),
                key="xn_method",
            )
            use_suffix = st.checkbox(
                "Read K as thousand and M as million (137K -> 137,000)",
                value=True, key="xn_suffix",
            )

            # ── Preview: what WOULD happen (nothing is changed yet) ───────────
            p = extract_number_preview(df, col_xn, method, use_suffix)
            st.dataframe(sanitize_for_display(p["examples"]), hide_index=True, width="stretch")
            if p["n_found"] == 0:
                st.error(f"No numbers were found in '{col_xn}'. Nothing will be changed.")
            elif p["n_none"]:
                st.warning(
                    f"{p['n_found']:,} of {p['n_filled']:,} values contain a number. "
                    f"{p['n_none']:,} have none and will be blank in the new column."
                )
            else:
                st.success(f"All {p['n_filled']:,} filled values contain a number.")

            if st.button("Pull numbers into a new column", key="clean_xn",
                         disabled=p["n_found"] == 0):
                new_df, msg = extract_number(df, col_xn, method, use_suffix)
                _apply_action(new_df, msg)


def _render_transformation_tools(df: pd.DataFrame, col_types: dict):
    """
    Render transformation operations with plain-English explanations.

    Transformations change the values in a column (as opposed to cleaning
    which changes the structure). Each one is explained simply.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = (
        get_columns_of_type(col_types, "categorical") +
        get_columns_of_type(col_types, "high_cardinality")
    )
    date_cols    = get_columns_of_type(col_types, "date_column")

    # ── Log transform ──────────────────────────────────────────────────────────
    with st.expander("Compress Extreme Values (Log Scale)", icon=":material/compress:"):
        st.caption(
            "When a column has a few very large values (like claim amounts in the millions), "
            "charts and models get distorted. Log scale compresses the range so everything "
            "is more readable. A new column is added, and your original stays as it is."
        )
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col_lt = st.selectbox("Column", numeric_cols, key="lt_col")
            if st.button("Apply log scale", key="transform_lt"):
                new_df, msg = log_transform(df, col_lt)
                _apply_action(new_df, msg)

    # ── Standard scale ─────────────────────────────────────────────────────────
    with st.expander("Normalize for Modeling (Standard Scale)", icon=":material/balance:"):
        st.caption(
            "Rescales a column so it has an average of 0. "
            "Use this when preparing data for a machine learning model "
            "so all columns are on equal footing. A new column is added."
        )
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col_ss = st.selectbox("Column", numeric_cols, key="ss_col")
            if st.button("Apply standard scale", key="transform_ss"):
                new_df, msg = standard_scale(df, col_ss)
                _apply_action(new_df, msg)

    # ── Min-max scale ──────────────────────────────────────────────────────────
    with st.expander("Scale to 0–1 Range (Min-Max Scale)", icon=":material/straighten:"):
        st.caption(
            "Compresses all values into the range 0 to 1. "
            "The smallest value becomes 0, the largest becomes 1. "
            "Useful for comparing columns measured in very different units."
        )
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col_mm = st.selectbox("Column", numeric_cols, key="mm_col")
            if st.button("Apply 0–1 scale", key="transform_mm"):
                new_df, msg = minmax_scale(df, col_mm)
                _apply_action(new_df, msg)

    # ── One-hot encode ─────────────────────────────────────────────────────────
    with st.expander("Convert Categories to Yes/No Columns (One-Hot Encode)", icon=":material/toggle_on:"):
        st.caption(
            "Turns a category column into several 0/1 columns, one per category. "
            "Required for most machine learning models, which can't use text directly. "
            "Example: 'Claim Type' becomes separate Yes/No columns for each claim type."
        )
        if not cat_cols:
            st.info("No category columns found.")
        else:
            col_ohe = st.selectbox("Column", cat_cols, key="ohe_col")
            n_unique = df[col_ohe].nunique()
            if n_unique > 30:
                st.warning(
                    f"'{col_ohe}' has {n_unique} categories, so this would add "
                    f"{n_unique} new columns. Consider dropping this column instead."
                )
            if st.button("Encode categories", key="transform_ohe"):
                new_df, msg = one_hot_encode(df, col_ohe)
                _apply_action(new_df, msg)

    # ── Extract date part ──────────────────────────────────────────────────────
    with st.expander("Extract Year or Month From a Date", icon=":material/event:"):
        st.caption(
            "Pull out just the year or month from a date column as a new number column. "
            "Useful for modeling. Machine learning models can't use raw dates, "
            "but they can use the year (2005, 2006...) as a numeric value."
        )
        if not date_cols:
            st.info("No date columns found.")
        else:
            col_dp   = st.selectbox("Date column", date_cols, key="dp_col")
            part_dp  = st.selectbox("What to extract", ["year", "month", "day"], key="dp_part")
            if st.button(f"Extract {part_dp}", key="transform_dp"):
                new_df, msg = extract_date_part(df, col_dp, part_dp)
                _apply_action(new_df, msg)


def render_clean_export_tab():
    """
    Render the entire Clean & Export tab.

    Called once by app.py whenever the user is on this tab.

    Session state keys this function reads and writes:
        st.session_state.working_df       current dataset
        st.session_state.original_df      never-modified original
        st.session_state.col_types        column type dict
        st.session_state.cleaning_log     list of completed action strings
        st.session_state.version_history  list of undo snapshots
    """

    # ── Guard ──────────────────────────────────────────────────────────────────
    if "working_df" not in st.session_state:
        st.info("Upload a dataset using the sidebar to get started.")
        return

    df        = st.session_state.working_df
    col_types = st.session_state.col_types

    # ── Always-visible undo bar ────────────────────────────────────────────────
    _show_undo_bar()
    st.markdown("---")

    # ── SECTION 1: Suggested Actions ──────────────────────────────────────────
    st.markdown("### :material/lightbulb: Suggested Actions")
    st.caption(
        "SmartWrangle found these issues in your data. "
        "Click a button to fix each one automatically."
    )
    _render_suggested_actions(df, col_types)

    st.markdown("---")

    # ── SECTION 2: All Cleaning Tools ─────────────────────────────────────────
    st.markdown("### :material/cleaning_services: Cleaning Tools")
    st.caption(
        "Click any section below to expand it and use that tool. "
        "Every action can be undone with the Undo button above."
    )
    _render_cleaning_tools(df, col_types)

    st.markdown("---")

    # ── SECTION 3: Transformation Tools ───────────────────────────────────────
    st.markdown("### :material/tune: Preparation for Analysis")
    st.caption(
        "These tools change the values in your columns to prepare them "
        "for charts and models. They always create a new column, so "
        "your original column is never overwritten."
    )
    _render_transformation_tools(df, col_types)

    st.markdown("---")

    # ── SECTION 4: Action History ──────────────────────────────────────────────
    st.markdown("### :material/history: Action History")

    cleaning_log = st.session_state.get("cleaning_log", [])

    if not cleaning_log:
        st.caption("No actions taken yet. Your data is in its original state.")
    else:
        st.caption(f"{plural(len(cleaning_log), 'action')} completed:")
        for i, action in enumerate(cleaning_log, 1):
            st.markdown(f"  {i}. {action}")

    st.markdown("---")

    # ── SECTION 5: Export ──────────────────────────────────────────────────────
    st.markdown("### :material/download: Export Your Data")
    st.caption(
        "Download your cleaned dataset as a CSV file (opens in Excel or Google Sheets), "
        "or download a plain-English summary report of what was found and what was done."
    )

    # Show a quick pre-export summary
    missing_remaining = int(df.isnull().sum().sum())
    if missing_remaining > 0:
        st.warning(
            f"Your dataset still has {missing_remaining:,} missing values. "
            f"You may want to fix these before exporting. "
            f"Use the Suggested Actions or Cleaning Tools above."
        )
    else:
        st.success("No missing values. Your data is ready to export.",
                   icon=":material/check_circle:")

    # ── Two download buttons side by side ─────────────────────────────────────
    col_csv, col_report = st.columns(2)

    with col_csv:
        st.markdown("**Cleaned Dataset**")
        st.caption("The dataset as it looks now, with all your changes included.")

        csv_filename = get_export_filename("smartwrangle_data", "csv")
        csv_bytes    = df_to_csv_bytes(df)

        st.download_button(
            label=f"Download CSV ({len(df):,} rows)",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            width="stretch",
            type="primary",
        )

    with col_report:
        st.markdown("**Summary Report**")
        st.caption(
            "A plain-English report: dataset overview, quality findings, "
            "key patterns, and every action taken."
        )

        # Build the quality result for the report
        if "quality_result" not in st.session_state:
            st.session_state.quality_result = score_dataset(df, col_types)

        report_text = build_summary_report(
            df,
            col_types,
            st.session_state.quality_result,
            cleaning_log,
            dataset_name=st.session_state.get("dataset_name", "Dataset")
        )
        report_bytes    = report_to_bytes(report_text)
        report_filename = get_export_filename("smartwrangle_report", "txt")

        st.download_button(
            label="Download Report (.txt)",
            data=report_bytes,
            file_name=report_filename,
            mime="text/plain",
            width="stretch",
        )

    # ── Report preview ─────────────────────────────────────────────────────────
    with st.expander("Preview Report", expanded=False):
        st.text(report_text)

    # ── PDF report ─────────────────────────────────────────────────────────────
    # Built only when the user clicks, because drawing the charts takes a
    # second or two. The finished file is kept in session state together with
    # a "fingerprint" of the data, so the app can tell when the data has
    # changed since the PDF was made.
    st.markdown("**PDF Report**")
    st.caption(
        "A printable report to share: before and after numbers, findings, the "
        "automatic charts with their headlines, key numbers, and every cleaning step."
    )

    try:
        # One number that changes whenever any value in the table changes
        data_print = int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        data_print = id(df)        # rare data types cannot be hashed; fall back
    fingerprint = (st.session_state.get("loaded_file_id"), data_print, len(cleaning_log))

    # The Create button gets the same blue look as the download buttons.
    # Streamlit gives a button with key="make_pdf" the CSS class
    # "st-key-make_pdf", so this style only touches this one button.
    st.markdown("""
    <style>
        .st-key-make_pdf button {
            background: #3b82f6; color: white; border-color: #3b82f6;
            border-radius: 8px; font-weight: 600;
        }
        .st-key-make_pdf button:hover, .st-key-make_pdf button:focus,
        .st-key-make_pdf button:focus-visible, .st-key-make_pdf button:active {
            background: #2563eb; color: white; border-color: #2563eb;
        }
    </style>
    """, unsafe_allow_html=True)

    # Two columns, like the CSV and summary report buttons above:
    # Create on the left, Download on the right once the PDF is ready.
    col_make, col_get = st.columns(2)

    if col_make.button("Create PDF report", key="make_pdf", width="stretch",
                       icon=":material/picture_as_pdf:"):
        with st.spinner("Building your PDF report..."):
            try:
                pdf_bytes = build_pdf_report(
                    st.session_state.original_df,
                    df,
                    col_types,
                    cleaning_log,
                    dataset_name=st.session_state.get("dataset_name", "Dataset"),
                    date_warnings=st.session_state.get("date_warnings", {}),
                )
                st.session_state.pdf_report = {"fingerprint": fingerprint, "bytes": pdf_bytes}
            except Exception as error:
                st.error(f"The PDF report could not be created: {error}")

    saved_pdf = st.session_state.get("pdf_report")
    if saved_pdf and saved_pdf["fingerprint"] == fingerprint:
        col_get.download_button(
            label="Download PDF Report",
            data=saved_pdf["bytes"],
            file_name=get_export_filename("smartwrangle_report", "pdf"),
            mime="application/pdf",
            key="download_pdf",
            width="stretch",
        )
    elif saved_pdf:
        st.caption(
            "Your data has changed since the last PDF was made. "
            "Click Create PDF report again for an up-to-date copy."
        )


# ── END OF FILE ────────────────────────────────────────────────────────────────