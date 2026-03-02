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
    5. Export                Download cleaned CSV + text summary report.

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
)
from engine.detector import get_columns_of_type, detect_column_types
from engine.quality  import score_dataset

# Utility modules
from utils.sanitizer import sanitize_for_display
from utils.exporter  import df_to_csv_bytes, build_summary_report, report_to_bytes, get_export_filename


def _show_undo_bar():
    """
    Render the undo button and current dataset size at the top of the tab.

    This is always visible so the user always knows they can undo.
    It also shows a quick summary of the current dataset state.
    """
    df      = st.session_state.working_df
    history = st.session_state.get("version_history", [])

    # Two columns: left for undo, right for dataset summary
    col_undo, col_status = st.columns([1, 3])

    with col_undo:
        # Disable the button if there's nothing to undo
        undo_disabled = len(history) == 0
        if st.button(
            "↩ Undo Last Action",
            disabled=undo_disabled,
            use_container_width=True,
            type="secondary",
            key="undo_btn"
        ):
            restored, new_history, msg = undo_last_action(history)
            if restored is not None:
                # Replace working_df with the restored version
                st.session_state.working_df    = restored
                st.session_state.version_history = new_history

                # Clear cached results so they recalculate with new data
                for key in ["quality_result", "insights_cache"]:
                    if key in st.session_state:
                        del st.session_state[key]

                # Add to the action log
                st.session_state.cleaning_log.append(f"↩ Undid last action")
                st.rerun()   # refresh the page to show updated data

    with col_status:
        # Show a simple "current state" summary
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
            parts.append("✅ No missing values")

        st.markdown("  ·  ".join(parts))


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
    st.toast(log_message, icon="✅")

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
            "✅ No issues detected. Your data looks clean and ready to use."
        )
        return

    st.markdown(f"**{len(suggestions)} suggestion(s) found:**")

    for i, s in enumerate(suggestions):
        # Color-code by priority
        if s["priority"] == 1:
            icon = "🔴"   # urgent
        elif s["priority"] == 2:
            icon = "🟡"   # moderate
        else:
            icon = "🔵"   # optional

        # Show the issue and a button to fix it
        col_issue, col_action = st.columns([3, 1])

        with col_issue:
            st.markdown(f"{icon} {s['issue']}")

        with col_action:
            # Each button needs a unique key — we use the operation + index
            btn_key = f"suggest_{s['operation']}_{i}"

            if st.button(s["action"], key=btn_key, use_container_width=True):
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
    with st.expander("🗑 Remove Duplicate Rows"):
        st.caption(
            "Removes rows that are exact copies of another row. "
            "Keeps the first occurrence of each duplicate."
        )
        n_dups = df.duplicated().sum()
        if n_dups == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{n_dups:,} duplicate rows found.")
            if st.button("Remove duplicates", key="clean_dups"):
                new_df, msg = remove_duplicates(df)
                _apply_action(new_df, msg)

    # ── Fill missing values ────────────────────────────────────────────────────
    with st.expander("🩹 Fill Missing Values"):
        st.caption(
            "Replace empty cells with an estimated value. "
            "For numbers: use median (middle value) — it's less affected by extremes. "
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
                strategy = st.selectbox(
                    "Fill method",
                    ["mode — most common date (recommended)",
                     "drop — remove rows with missing dates"],
                    key="fm_strategy"
                )
                strategy_key = "mode" if strategy.startswith("mode") else "drop_rows"
                const_val = None
            else:
                # Non-date columns get the full set of options
                strategy = st.selectbox(
                    "Fill method",
                    ["median — middle value (best for numbers with outliers)",
                     "mean — average value",
                     "mode — most common value (best for categories)",
                     "constant — a value you choose"],
                    key="fm_strategy"
                )
                # Extract just the strategy keyword before the " —" description
                strategy_key = strategy.split(" —")[0].strip()

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
    with st.expander("✂ Remove Rows With Missing Values"):
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
    with st.expander("🗂 Remove a Column"):
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
    with st.expander("✏ Fix Extra Spaces in Text"):
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
    with st.expander("🔡 Standardize Text Capitalization"):
        st.caption(
            "Make text consistent — e.g. 'american airlines', "
            "'AMERICAN AIRLINES', and 'American Airlines' all become the same."
        )
        if not string_cols:
            st.info("No text columns found.")
        else:
            col_tc  = st.selectbox("Column", string_cols, key="tc_col")
            case_tc = st.selectbox(
                "Format",
                ["title — Title Case", "lower — all lowercase", "upper — ALL UPPERCASE"],
                key="tc_case"
            )
            case_key = case_tc.split(" —")[0].strip()
            if st.button("Standardize capitalization", key="clean_tc"):
                new_df, msg = standardize_text_case(df, col_tc, case_key)
                _apply_action(new_df, msg)

    # ── Rename column ──────────────────────────────────────────────────────────
    with st.expander("🏷 Rename a Column"):
        st.caption("Give a column a cleaner, more readable name.")
        col_rn  = st.selectbox("Column to rename", all_cols, key="rn_col")
        new_rn  = st.text_input("New name", value=col_rn, key="rn_new")
        if st.button("Rename column", key="clean_rn"):
            new_df, msg = rename_column(df, col_rn, new_rn)
            _apply_action(new_df, msg)

    # ── Convert Currency / Text to Numeric ────────────────────────────────────
    with st.expander("💰 Convert Currency to Numeric"):
        st.caption(
            "Convert currency-formatted text (like $2,350.00) into real numeric values "
            "so charts, scaling, and modeling tools can use them."
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
            st.caption(f"Sample values: {preview_values}")

            if st.button("Convert to numeric", key="clean_currency"):

                # Step 1 — convert to plain strings and strip whitespace
                cleaned_series = df[col_cc].astype(str).str.strip()

                # Step 2 — detect accounting-format negatives BEFORE stripping
                # ($2,350.00) means NEGATIVE $2,350 in standard accounting notation
                is_negative = (
                    cleaned_series.str.startswith("(") &
                    cleaned_series.str.endswith(")")
                )

                # Step 3 — strip currency symbols, commas, spaces, parentheses
                cleaned_series = cleaned_series.str.replace(
                    r"[\$,€£¥\s()]", "", regex=True
                )

                # Step 4 — restore the minus sign on rows that had parentheses
                cleaned_series = cleaned_series.where(
                    ~is_negative,
                    "-" + cleaned_series
                )

                # Step 5 — convert to float; non-parseable values become NaN
                new_df = df.copy()
                new_df[col_cc] = pd.to_numeric(cleaned_series, errors="coerce")

                _apply_action(
                    new_df,
                    f"Converted '{col_cc}' from currency/text to numeric."
                )


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
    with st.expander("📉 Compress Extreme Values (Log Scale)"):
        st.caption(
            "When a column has a few very large values (like claim amounts in the millions), "
            "charts and models get distorted. Log scale compresses the range so everything "
            "is more readable. A new column is added — your original is preserved."
        )
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col_lt = st.selectbox("Column", numeric_cols, key="lt_col")
            if st.button("Apply log scale", key="transform_lt"):
                new_df, msg = log_transform(df, col_lt)
                _apply_action(new_df, msg)

    # ── Standard scale ─────────────────────────────────────────────────────────
    with st.expander("⚖ Normalize for Modeling (Standard Scale)"):
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
    with st.expander("📏 Scale to 0–1 Range (Min-Max Scale)"):
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
    with st.expander("🔢 Convert Categories to Yes/No Columns (One-Hot Encode)"):
        st.caption(
            "Turns a category column into multiple 0/1 columns — one per category. "
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
                    f"'{col_ohe}' has {n_unique} categories — this would add "
                    f"{n_unique} new columns. Consider dropping this column instead."
                )
            if st.button("Encode categories", key="transform_ohe"):
                new_df, msg = one_hot_encode(df, col_ohe)
                _apply_action(new_df, msg)

    # ── Extract date part ──────────────────────────────────────────────────────
    with st.expander("📅 Extract Year or Month From a Date"):
        st.caption(
            "Pull out just the year or month from a date column as a new number column. "
            "Useful for modeling — machine learning models can't use raw dates, "
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
    st.markdown("### 💡 Suggested Actions")
    st.caption(
        "SmartWrangle found these issues in your data. "
        "Click a button to fix each one automatically."
    )
    _render_suggested_actions(df, col_types)

    st.markdown("---")

    # ── SECTION 2: All Cleaning Tools ─────────────────────────────────────────
    st.markdown("### 🧹 Cleaning Tools")
    st.caption(
        "Click any section below to expand it and use that tool. "
        "Every action can be undone with the Undo button above."
    )
    _render_cleaning_tools(df, col_types)

    st.markdown("---")

    # ── SECTION 3: Transformation Tools ───────────────────────────────────────
    st.markdown("### ⚙ Preparation for Analysis")
    st.caption(
        "These tools change the values in your columns to prepare them "
        "for charts and models. They always create a new column — "
        "your original column is never overwritten."
    )
    _render_transformation_tools(df, col_types)

    st.markdown("---")

    # ── SECTION 4: Action History ──────────────────────────────────────────────
    st.markdown("### 📋 Action History")

    cleaning_log = st.session_state.get("cleaning_log", [])

    if not cleaning_log:
        st.caption("No actions taken yet. Your data is in its original state.")
    else:
        st.caption(f"{len(cleaning_log)} action(s) completed:")
        for i, action in enumerate(reversed(cleaning_log), 1):
            st.markdown(f"  {len(cleaning_log) - i + 1}. {action}")

    st.markdown("---")

    # ── SECTION 5: Export ──────────────────────────────────────────────────────
    st.markdown("### ⬇ Export Your Data")
    st.caption(
        "Download your cleaned dataset as a CSV file (opens in Excel or Google Sheets), "
        "or download a plain-English summary report of what was found and what was done."
    )

    # Show a quick pre-export summary
    missing_remaining = int(df.isnull().sum().sum())
    if missing_remaining > 0:
        st.warning(
            f"Your dataset still has {missing_remaining:,} missing values. "
            f"You may want to fix these before exporting — "
            f"use the Suggested Actions or Cleaning Tools above."
        )
    else:
        st.success("✅ No missing values — your data is ready to export.")

    # ── Two download buttons side by side ─────────────────────────────────────
    col_csv, col_report = st.columns(2)

    with col_csv:
        st.markdown("**Cleaned Dataset**")
        st.caption("The dataset as it currently looks — all your changes included.")

        csv_filename = get_export_filename("smartwrangle_data", "csv")
        csv_bytes    = df_to_csv_bytes(df)

        st.download_button(
            label=f"Download CSV ({len(df):,} rows)",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
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
            use_container_width=True,
        )

    # ── Report preview ─────────────────────────────────────────────────────────
    with st.expander("Preview Report", expanded=False):
        st.text(report_text)


# ── END OF FILE ────────────────────────────────────────────────────────────────