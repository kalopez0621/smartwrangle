"""
utils/sanitizer.py
==================
Arrow-safe DataFrame sanitizer for SmartWrangle 2.0.

What this file does
-------------------
Streamlit uses a technology called Apache Arrow to display DataFrames
in the browser quickly. Arrow is fast, but it is strict — it cannot
display certain column types that pandas uses internally.

If you try to display a DataFrame with one of these problematic types,
Streamlit throws a cryptic error like:
    "ArrowInvalid: Could not convert value..."
    "ArrowTypeError: Expected bytes, got a str object..."

This file provides one function — sanitize_for_display() — that you
call before passing any DataFrame to st.dataframe() or st.table().
It converts every problematic column type into something Arrow can
handle, without changing the actual data values.

Column types that need fixing
------------------------------
    datetime64[us]  →  string  (Arrow can't serialize all datetime resolutions)
    datetime64[ns]  →  string  (same reason)
    StringDtype     →  object  (pandas 2.x new string type, Arrow struggles with it)
    Interval        →  string  (used by pd.cut() and pd.qcut() for binning)
    Categorical     →  object  (pandas category dtype with mixed underlying types)
    Period          →  string  (pandas Period type for time spans)

Important
---------
This sanitizer is ONLY for display purposes.
It returns a NEW DataFrame — it never modifies the original.
The working_df in session state is always kept in its original form.

How to use this file
--------------------
    from utils.sanitizer import sanitize_for_display

    # Before passing to any Streamlit display function:
    st.dataframe(sanitize_for_display(working_df))
    st.table(sanitize_for_display(summary_df))
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# pandas : for DataFrame operations and dtype checking
import pandas as pd


def sanitize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all Arrow-incompatible column types to safe equivalents.

    This is the only function in this file. Call it every time you
    pass a DataFrame to st.dataframe(), st.table(), or any other
    Streamlit function that renders a table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sanitize. Can be any DataFrame in the app —
        the working dataset, a summary table, a grouped result, etc.

    Returns
    -------
    pd.DataFrame
        A new DataFrame that is safe to display in Streamlit.
        The original DataFrame is never modified.

    Example
    -------
    >>> st.dataframe(sanitize_for_display(st.session_state.working_df))
    """
    # Always work on a copy — never modify the original DataFrame
    # .copy() creates a completely independent duplicate
    safe_df = df.copy()

    # Loop through every column in the DataFrame
    for col in safe_df.columns:

        # Get the dtype (data type) of this column
        col_dtype = safe_df[col].dtype

        # ── Fix 1: Datetime columns ────────────────────────────────────────
        # pandas uses datetime64[ns] or datetime64[us] for date/time columns.
        # Arrow can sometimes struggle with these, especially datetime64[us]
        # which is the default in pandas 2.x when reading CSVs.
        # Converting to string avoids the issue entirely.
        # The dates still display correctly — they just look like "2024-01-15"
        # instead of being stored as a timestamp internally.
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            # Format as YYYY-MM-DD for readability
            # .dt.strftime() applies a date format pattern to each value
            # %Y = 4-digit year, %m = 2-digit month, %d = 2-digit day
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%d")
            continue   # Move to the next column — no more checks needed

        # ── Fix 2: pandas StringDtype ──────────────────────────────────────
        # pandas 2.x introduced a new "StringDtype" that is different from
        # the classic "object" dtype. StringDtype stores text more efficiently,
        # but some versions of the Arrow serializer don't support it.
        # Converting to "object" (the classic string dtype) fixes this.
        if isinstance(col_dtype, pd.StringDtype):
            safe_df[col] = safe_df[col].astype(object)
            continue

        # ── Fix 3: Interval dtype ──────────────────────────────────────────
        # pd.cut() and pd.qcut() create columns with an Interval dtype.
        # For example, cutting ages into bins produces values like (18, 35].
        # Arrow cannot serialize Interval values — convert to readable strings.
        if isinstance(col_dtype, pd.IntervalDtype):
            safe_df[col] = safe_df[col].astype(str)
            continue

        # ── Fix 4: Categorical dtype ───────────────────────────────────────
        # pd.Categorical stores a limited set of text values efficiently.
        # If the underlying values are mixed types (e.g. strings AND None),
        # Arrow may fail. Converting to object is safest.
        if isinstance(col_dtype, pd.CategoricalDtype):
            safe_df[col] = safe_df[col].astype(object)
            continue

        # ── Fix 5: Period dtype ────────────────────────────────────────────
        # pd.Period represents a span of time (e.g. "2024-Q1" for a quarter).
        # This rarely appears in raw data but can appear after time-based
        # grouping operations. Convert to string for safe display.
        if isinstance(col_dtype, pd.PeriodDtype):
            safe_df[col] = safe_df[col].astype(str)
            continue

        # ── Fix 6: Object columns with mixed types ─────────────────────────
        # Sometimes a column typed as "object" contains a mix of strings,
        # integers, and None values. Arrow can fail on these.
        # We standardize everything to strings as a safety measure.
        if col_dtype == object:
            # .where() keeps values that satisfy the condition,
            # replaces others with the second argument (here: None stays None)
            # We convert to str only for non-null values
            safe_df[col] = safe_df[col].apply(
                lambda x: str(x) if x is not None and not pd.isna(x) else x
            )
            continue

        # All other dtypes (float64, int64, bool) are Arrow-compatible
        # and don't need any conversion.

    return safe_df


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up column names so they are safe to use everywhere in the app.

    Column names in real-world datasets can cause problems:
    - Spaces in names: "Claim Amount" needs quotes in some contexts
    - Special characters: "Amount ($)" breaks certain operations
    - Leading/trailing spaces: " Column " doesn't match "Column"

    This function standardizes column names by:
    - Stripping leading and trailing spaces
    - Replacing internal spaces with underscores (optional — see below)

    Note: We strip spaces but do NOT replace spaces with underscores,
    because the app displays column names to users and "Claim Amount"
    is more readable than "Claim_Amount".

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose column names to clean.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with cleaned column names.
    """
    # Build a mapping from old column names to new (cleaned) ones
    # Strip is the only change — preserves spaces within the name
    new_columns = {col: str(col).strip() for col in df.columns}

    # .rename() applies the mapping — only columns in the dict are changed
    return df.rename(columns=new_columns)


# ── END OF FILE ────────────────────────────────────────────────────────────────
# Nothing should be placed below this line.
# sanitize_for_display() is called by all three tab files before
# passing any DataFrame to a Streamlit display function.