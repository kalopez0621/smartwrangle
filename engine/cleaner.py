"""
engine/cleaner.py
=================
All data cleaning and transformation logic for SmartWrangle 2.0.

What this file does
-------------------
This module handles every operation that changes the dataset —
from fixing simple problems (removing duplicates, filling empty cells)
to preparing data for analysis (rescaling numbers, encoding categories).

In v1.0, these operations were split across two separate tabs:
Cleaning and Transformations. In v2.0, they live here together
because from a user's perspective it's all the same goal:
"Fix my data so I can use it."

Every operation in this file follows the same three rules:

    Rule 1 — Never modify the original.
        We always work on a copy of the data. The original upload
        is preserved in session state and can always be restored.

    Rule 2 — Return a plain-English log entry.
        Every function returns a string describing what it did in
        plain language. This gets displayed in the Clean & Export tab
        so the user always knows what has been done to their data.

    Rule 3 — Be safe.
        Every function handles edge cases — what if the column is
        already empty? What if there are no duplicates to remove?
        We never let a cleaning operation crash the app.

The undo system
---------------
Before every operation, we save a snapshot of the current dataset.
If the user clicks "Undo", we restore the last snapshot.
We keep a maximum of 10 snapshots to avoid using too much memory.

How to use this file
--------------------
    from engine.cleaner import (
        remove_duplicates,
        drop_column,
        fill_missing,
        drop_missing_rows,
        trim_whitespace,
        log_transform,
        standard_scale,
        minmax_scale,
        one_hot_encode,
        extract_date_part,
        get_cleaning_suggestions,
    )

    # Each function takes a DataFrame and returns:
    #   (new_df, log_message)
    #
    # Example:
    new_df, message = remove_duplicates(working_df)
    # message = "Removed 42 duplicate rows."
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# re          : finds numbers inside text (Pull Numbers Out of Text)
# pandas      : for working with DataFrames (tables of data)
# numpy       : for math operations like log transformation
# StandardScaler, MinMaxScaler : from scikit-learn — used to rescale numbers
#   StandardScaler : makes a column have mean=0 and standard deviation=1
#   MinMaxScaler   : compresses a column into the range [0, 1]
# warnings    : to hide pandas' "guessing the date format" notices
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum number of undo snapshots we keep in memory.
# After this limit, the oldest snapshot is discarded.
# Keeping too many snapshots of a 94,000-row dataset would use a lot of RAM.
MAX_UNDO_HISTORY = 10


# ── Snapshot / undo helpers ────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, version_history: list) -> list:
    """
    Save a copy of the current DataFrame to the undo history.

    Call this BEFORE making any change to the data. That way, if the
    user clicks "Undo", we can restore the state from before the change.

    Parameters
    ----------
    df              : pd.DataFrame   The current working dataset.
    version_history : list           The existing list of snapshots
                                     (stored in Streamlit session state).

    Returns
    -------
    list
        The updated version_history with the new snapshot appended.
        If the history exceeds MAX_UNDO_HISTORY, the oldest entry
        is removed first (like a queue — first in, first out).

    Example
    -------
    # Before doing anything to the data:
    st.session_state.version_history = save_snapshot(
        st.session_state.working_df,
        st.session_state.version_history
    )
    """
    # .copy() creates a fully independent copy of the DataFrame.
    # Without .copy(), we'd just be saving a reference to the same
    # data, and subsequent changes would overwrite the "snapshot".
    snapshot = df.copy()

    # Add the snapshot to the end of the history list
    version_history.append(snapshot)

    # If we've exceeded the limit, remove the oldest snapshot (index 0)
    if len(version_history) > MAX_UNDO_HISTORY:
        version_history.pop(0)

    return version_history


def undo_last_action(version_history: list) -> tuple:
    """
    Restore the dataset to the state before the last operation.

    Parameters
    ----------
    version_history : list   The list of saved snapshots.

    Returns
    -------
    tuple of (restored_df, updated_history, success_message)
        restored_df      : pd.DataFrame   The previous version of the data.
        updated_history  : list           The history with the last entry removed.
        success_message  : str            A message to show the user.

    If there is nothing to undo, returns (None, version_history, error_message).
    """
    # Check if there is anything to undo
    if not version_history:
        return None, version_history, "Nothing to undo — this is the original dataset."

    # Remove and return the last snapshot from the history
    # .pop() removes the last item and returns it
    restored_df     = version_history.pop()
    success_message = "Last action undone successfully."

    return restored_df, version_history, success_message


# ── Text helper ────────────────────────────────────────────────────────────────

def _text_only(series: pd.Series, text_func) -> pd.Series:
    """
    Apply a text operation (strip, title, lower, upper) to the filled cells
    of a column and leave blank cells blank.

    Why this exists: series.astype(str) turns a blank cell (NaN) into the
    text "nan" on some column types. That hides the blank from the missing
    value count and creates a fake "nan" category in charts.

    Parameters
    ----------
    series    : pd.Series   The column to change.
    text_func : function    Takes a text Series and returns the changed Series,
                            for example  lambda s: s.str.strip()

    Returns
    -------
    pd.Series with the change applied and the blank cells untouched.
    """
    is_blank = series.isna()                       # True where the cell is empty
    changed  = text_func(series.astype(str))       # change every cell as text
    return changed.where(~is_blank, series)        # put the original blanks back


# ── Cleaning operations ────────────────────────────────────────────────────────
# Each function below:
#   - Takes the current working DataFrame as input
#   - Makes the requested change on a COPY (never the original)
#   - Returns (new_df, log_message)
#
# The log_message is always plain English — no jargon.


def remove_duplicates(df: pd.DataFrame) -> tuple:
    """
    Remove rows that are exact copies of another row.

    A duplicate row is one where every single column value is identical
    to another row. For example, if the same claim was entered twice
    with all the same details, that's a duplicate.

    Why remove them?
    ----------------
    Duplicate rows make some groups look bigger than they are.
    For example, if American Airlines has 50 real claims but
    10 are duplicated, charts would show 60 — misleading analysis.

    Parameters
    ----------
    df : pd.DataFrame   The current working dataset.

    Returns
    -------
    tuple of (new_df, log_message)
        new_df      : The dataset with duplicates removed.
        log_message : Plain-English description of what was done.
    """
    # Count duplicates BEFORE removing them so we can report the number
    n_duplicates = int(df.duplicated().sum())

    # If there are no duplicates, tell the user and return the data unchanged
    if n_duplicates == 0:
        return df.copy(), "No duplicate rows found — nothing to remove."

    # drop_duplicates() removes all rows that are copies of an earlier row.
    # keep='first' means we keep the first occurrence and remove the copies.
    new_df = df.drop_duplicates(keep="first").reset_index(drop=True)
    # reset_index() renumbers the rows from 0 after removal
    # drop=True means we don't keep the old row numbers as a column

    rows_removed = len(df) - len(new_df)

    log_message = (
        f"Removed {rows_removed:,} duplicate row(s). "
        f"Dataset now has {len(new_df):,} rows."
    )

    return new_df, log_message


def drop_column(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Remove a column from the dataset entirely.

    Use this to remove columns that are not useful for analysis,
    like ID numbers, internal codes, or free-text notes that
    cannot be charted or modeled.

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The name of the column to remove.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    # Check that the column actually exists before trying to remove it
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found — nothing was changed."

    # Drop the column — axis=1 means "remove a column" (axis=0 means "remove a row")
    new_df = df.drop(columns=[column_name])

    log_message = (
        f"Removed column '{column_name}'. "
        f"Dataset now has {len(new_df.columns)} columns."
    )

    return new_df, log_message


def fill_missing(
    df: pd.DataFrame,
    column_name: str,
    strategy: str,
    constant_value=None
) -> tuple:
    """
    Fill empty (missing) cells in a column with a chosen value.

    Empty cells are a problem because many calculations and charts
    cannot handle them. This function lets the user choose how to
    fill them in.

    Strategies
    ----------
    'mean'     Fill with the average value of the column.
               Best for: numeric columns with a roughly even spread.
               Example: fill missing ages with the average age.

    'median'   Fill with the middle value of the column.
               Best for: numeric columns with outliers (like income or
               claim amounts) where the average is pulled way up by a
               few extreme values.
               Example: fill missing claim amounts with the median claim.

    'mode'     Fill with the most common value in the column.
               Best for: text/category columns.
               Example: fill missing airline names with the most common airline.

    'constant' Fill with a specific value you choose.
               Best for: when you know what the missing value should be.
               Example: fill missing payouts with $0 (meaning nothing was paid).

    Parameters
    ----------
    df             : pd.DataFrame   The current working dataset.
    column_name    : str            The column to fill.
    strategy       : str            One of: 'mean', 'median', 'mode', 'constant'
    constant_value : any            Only used when strategy='constant'.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    # Check that the column exists
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found — nothing was changed."

    # Count missing values before filling
    n_missing = int(df[column_name].isnull().sum())

    # If there are no missing values, nothing to do
    if n_missing == 0:
        return df.copy(), f"'{column_name}' has no missing values — nothing to fill."

    # Safety check: mean and median need numbers. On a text column they
    # raise an error, so return a friendly message instead of crashing.
    if strategy in ("mean", "median") and not pd.api.types.is_numeric_dtype(df[column_name]):
        return df.copy(), (
            f"'{column_name}' is a text column, so it cannot be filled with the "
            f"{strategy}. Use the most common value or a constant instead."
        )

    # Work on a copy so we never modify the original
    new_df = df.copy()

    # ── Apply the chosen strategy ──────────────────────────────────────────────
    if strategy == "mean":
        # .mean() calculates the average of all non-missing values
        fill_value   = new_df[column_name].mean()
        strategy_desc = f"the column average ({fill_value:,.2f})"

    elif strategy == "median":
        # .median() finds the middle value when all values are sorted
        fill_value   = new_df[column_name].median()
        strategy_desc = f"the median value ({fill_value:,.2f})"

    elif strategy == "mode":
        # .mode() returns the most frequently occurring value(s)
        # [0] picks the first one in case there's a tie
        mode_values  = new_df[column_name].mode()
        if len(mode_values) == 0:
            return df.copy(), f"Could not determine a mode for '{column_name}'."
        fill_value   = mode_values[0]
        strategy_desc = f"the most common value ('{fill_value}')"

    elif strategy == "constant":
        # Use whatever value the user provided
        if constant_value is None:
            return df.copy(), "No fill value provided for constant strategy."
        fill_value    = constant_value
        strategy_desc = f"the value '{constant_value}'"

    else:
        return df.copy(), f"Unknown strategy '{strategy}'. Nothing was changed."

    # .fillna() replaces all NaN (empty) values with fill_value
    new_df[column_name] = new_df[column_name].fillna(fill_value)

    log_message = (
        f"Filled {n_missing:,} missing value(s) in '{column_name}' "
        f"with {strategy_desc}."
    )

    return new_df, log_message


def drop_missing_rows(df: pd.DataFrame, column_name: str = None) -> tuple:
    """
    Remove rows that have missing (empty) values.

    Two modes:
    - If column_name is given: only remove rows where THAT column is empty.
    - If column_name is None: remove any row that has ANY empty cell.

    When to use this vs fill_missing
    ---------------------------------
    Use drop_missing_rows when:
    - The row without the value is not useful at all
    - Missing data is rare enough that removing rows won't hurt the analysis

    Use fill_missing when:
    - You want to keep the row and just estimate the missing value
    - Missing data is common (removing rows would shrink the dataset too much)

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str or None    Specific column, or None to drop all rows
                                 with any missing value.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    rows_before = len(df)

    if column_name:
        # Remove rows where only this specific column is empty
        if column_name not in df.columns:
            return df.copy(), f"Column '{column_name}' not found."

        new_df = df.dropna(subset=[column_name]).reset_index(drop=True)
        scope_desc = f"in '{column_name}'"

    else:
        # Remove rows where ANY column is empty
        new_df = df.dropna().reset_index(drop=True)
        scope_desc = "in any column"

    rows_removed = rows_before - len(new_df)

    if rows_removed == 0:
        return df.copy(), f"No rows with missing values found {scope_desc}."

    log_message = (
        f"Removed {rows_removed:,} row(s) with missing values {scope_desc}. "
        f"Dataset now has {len(new_df):,} rows."
    )

    return new_df, log_message


def trim_whitespace(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Remove extra spaces from the beginning and end of text values.

    This is a common data quality issue in datasets exported from
    forms or legacy systems. For example, ' American Airlines '
    (with spaces) would not match 'American Airlines' (without),
    creating false duplicates in grouped analysis.

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The text column to clean.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    new_df = df.copy()

    # Convert to string first (in case any values are not already strings)
    # .str.strip() removes leading and trailing whitespace.
    # _text_only() keeps blank cells blank. Without it, astype(str) can
    # turn a blank cell into the 4-letter text "nan".
    new_df[column_name] = _text_only(new_df[column_name], lambda s: s.str.strip())

    log_message = (
        f"Trimmed extra whitespace from all values in '{column_name}'."
    )

    return new_df, log_message


def standardize_text_case(
    df: pd.DataFrame,
    column_name: str,
    case: str = "title"
) -> tuple:
    """
    Standardize the capitalization of text values in a column.

    Inconsistent capitalization causes the same category to be counted
    separately. For example, 'american airlines', 'American Airlines',
    and 'AMERICAN AIRLINES' would appear as three different categories
    in a chart, when they should all be the same.

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The text column to standardize.
    case        : str            'title'  → Title Case (first letter of each word)
                                 'lower'  → all lowercase
                                 'upper'  → ALL UPPERCASE

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    if case not in ("title", "lower", "upper"):
        return df.copy(), f"Unknown case '{case}'. Choose 'title', 'lower', or 'upper'."

    new_df = df.copy()

    # Apply the requested case transformation using pandas string methods.
    # _text_only() keeps blank cells blank (see trim_whitespace above).
    if case == "title":
        new_df[column_name] = _text_only(new_df[column_name], lambda s: s.str.title())
        case_desc = "Title Case"
    elif case == "lower":
        new_df[column_name] = _text_only(new_df[column_name], lambda s: s.str.lower())
        case_desc = "lowercase"
    else:
        new_df[column_name] = _text_only(new_df[column_name], lambda s: s.str.upper())
        case_desc = "UPPERCASE"

    log_message = (
        f"Standardized '{column_name}' text to {case_desc}."
    )

    return new_df, log_message


def rename_column(
    df: pd.DataFrame,
    old_name: str,
    new_name: str
) -> tuple:
    """
    Give a column a new name.

    Useful when column names are unclear, abbreviated, or come from
    a system that uses codes instead of readable names.

    Parameters
    ----------
    df       : pd.DataFrame   The current working dataset.
    old_name : str            The current column name.
    new_name : str            The desired new name.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if old_name not in df.columns:
        return df.copy(), f"Column '{old_name}' not found."

    # Check if the new name would create a duplicate column
    if new_name in df.columns and new_name != old_name:
        return df.copy(), (
            f"A column named '{new_name}' already exists. "
            f"Please choose a different name."
        )

    # df.rename() with columns={old: new} renames just that one column
    new_df = df.rename(columns={old_name: new_name})

    log_message = f"Renamed column '{old_name}' to '{new_name}'."

    return new_df, log_message


# ── Placeholder values ─────────────────────────────────────────────────────────
# Real files often type a word or code into a cell instead of leaving it empty.
# These look like real values, so they hide from the missing-value counts and
# stop number columns from converting. The functions below find them and can
# turn them into true blanks.

# Words that usually mean "no value here". Matching ignores capital letters
# and outer spaces, so "Error", " ERROR " and "error" all match "error".
PLACEHOLDER_WORDS = [
    "error", "unknown", "n/a", "na", "#n/a", "null", "none", "nan",
    "-", "--", "?", "missing", "not available",
]

# A number code that some systems use for "missing". It is NOT blanked by
# default, because in other data -1 is a real value (a score change, a temperature).
PLACEHOLDER_NUMBER = "-1"


def _placeholder_key(series: pd.Series) -> pd.Series:
    """
    Turn every value into a lowercase, trimmed text "key" for matching.
    Blank cells stay blank so they are never counted as placeholders.
    Number columns become text too, so -1 and -1.0 both turn into "-1".
    """
    if pd.api.types.is_numeric_dtype(series):
        # -1.0 becomes "-1"; other numbers become text that will not match
        key = series.map(lambda v: "-1" if v == -1 else str(v))
    else:
        key = series.astype(str).str.strip().str.lower()
    return key.where(series.notna())


def find_placeholders(df: pd.DataFrame) -> list:
    """
    List every placeholder value found in the dataset.

    Returns
    -------
    list of dicts, one per (column, placeholder) pair, largest count first:
        column      str    The column name
        key         str    The matched placeholder in lowercase ("error", "-1")
        shown_as    str    How it most often appears in the file ("ERROR")
        count       int    How many cells hold it
        is_number   bool   True for -1 (not selected by default in the app)
    """
    found = []
    targets = set(PLACEHOLDER_WORDS) | {PLACEHOLDER_NUMBER}

    for col in df.columns:
        series = df[col]
        # Dates and True/False columns cannot hold placeholder text
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue

        key = _placeholder_key(series)
        hits = key[key.isin(targets)]
        if hits.empty:
            continue

        for k, count in hits.value_counts().items():
            # The spelling used most often in the file, for display
            shown = series[key == k].astype(str).str.strip().mode()
            found.append({
                "column":    col,
                "key":       k,
                "shown_as":  shown.iloc[0] if len(shown) else k,
                "count":     int(count),
                "is_number": k == PLACEHOLDER_NUMBER,
            })

    found.sort(key=lambda d: d["count"], reverse=True)
    return found


def placeholder_selected_by_default(item: dict) -> bool:
    """
    Decide whether a found placeholder starts out ticked in the app.

    Ticked:     words (ERROR, UNKNOWN, N/A ...) that appear 2 or more times
                in the column. A real placeholder usually repeats.
    Not ticked: -1, because it is often a real number, and any word that
                appears only once, because it may be real data. (The
                Netflix file has a movie actually titled "Unknown".)
    """
    return (not item["is_number"]) and item["count"] >= 2


def blank_placeholders(df: pd.DataFrame, selections: list) -> tuple:
    """
    Turn the chosen placeholder values into true blank cells.

    Parameters
    ----------
    df         : pd.DataFrame   The current working dataset.
    selections : list           (column, key) pairs from find_placeholders(),
                                for example [("Item", "error"), ("Rating", "-1")]

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if not selections:
        return df.copy(), "No placeholder values were selected, so nothing was changed."

    new_df = df.copy()
    total = 0
    columns_changed = set()

    for col, k in selections:
        if col not in new_df.columns:
            continue
        mask = _placeholder_key(new_df[col]) == k     # cells holding this placeholder
        n = int(mask.sum())
        if n:
            # Whole-number columns cannot hold blanks in pandas, so switch
            # them to decimal numbers first (5 becomes 5.0; values unchanged).
            if pd.api.types.is_integer_dtype(new_df[col]):
                new_df[col] = new_df[col].astype(float)
            new_df.loc[mask, col] = np.nan            # make them truly blank
            total += n
            columns_changed.add(col)

    if total == 0:
        return df.copy(), "No matching placeholder values were found, so nothing was changed."

    return new_df, (
        f"Marked {total:,} placeholder value(s) as blank in "
        f"{len(columns_changed)} column(s). They now count as missing values."
    )


# ── Text or currency to numbers ────────────────────────────────────────────────

def _parse_numbers(series: pd.Series) -> pd.Series:
    """
    Read text like "$2,350.00", "(75.50)" or " 7 " as numbers.

    Steps:
        1. Treat every value as text and remove outer spaces.
        2. Note accounting negatives: (2,350.00) means -2,350.00.
        3. Remove currency symbols, commas, spaces and parentheses.
        4. Put the minus sign back on the accounting negatives.
        5. Convert to numbers. Anything still not a number becomes blank.

    Cells that were already blank stay blank.
    """
    text = series.astype(str).str.strip()                                  # step 1
    is_negative = text.str.startswith("(") & text.str.endswith(")")        # step 2
    text = text.str.replace(r"[\$,€£¥\s()]", "", regex=True)               # step 3
    text = text.where(~is_negative, "-" + text)                            # step 4
    numbers = pd.to_numeric(text, errors="coerce")                         # step 5
    return numbers.where(series.notna())                                   # keep blanks blank


def numeric_conversion_preview(df: pd.DataFrame, column_name: str) -> dict:
    """
    Check what converting a column to numbers WOULD do, without changing it.

    The Clean & Export tab shows this before the user clicks Convert, so a
    column like "93 min" or "$137K-$171K" is not wiped out by surprise.

    Returns
    -------
    dict with keys:
        n_filled   int    Cells that have a value now
        n_ok       int    Of those, how many can be read as numbers
        n_lost     int    Of those, how many would become blank
        examples   list   Up to 3 values that cannot be read as numbers
    """
    original = df[column_name]
    numbers  = _parse_numbers(original)
    filled   = original.notna()
    lost     = filled & numbers.isna()          # had a value, would become blank
    return {
        "n_filled": int(filled.sum()),
        "n_ok":     int((filled & numbers.notna()).sum()),
        "n_lost":   int(lost.sum()),
        "examples": original[lost].astype(str).unique()[:3].tolist(),
    }


def convert_to_numeric(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Convert a text or currency column into real numbers.

    Use numeric_conversion_preview() first to see how many values would
    become blank. This function does the conversion and reports the same
    numbers in its log message.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    preview = numeric_conversion_preview(df, column_name)

    # Safety net: never wipe out a whole column
    if preview["n_ok"] == 0:
        return df.copy(), (
            f"'{column_name}' was not converted: none of its values can be "
            f"read as numbers."
        )

    new_df = df.copy()
    new_df[column_name] = _parse_numbers(df[column_name])

    log_message = f"Converted '{column_name}' to numbers ({preview['n_ok']:,} values)."
    if preview["n_lost"]:
        log_message += (
            f" {preview['n_lost']:,} value(s) could not be read and are now blank."
        )
    return new_df, log_message


# ── Convert text to dates ──────────────────────────────────────────────────────
# SmartWrangle already turns date text into real dates when the column NAME
# sounds like a date ("Order Date", "created_at"). This tool is for every
# other column, such as "Shipped" or "Start", and for files that write the
# day before the month (European style).

# The two ways to read a value like 03/05/2024, shown in the app
DATE_ORDERS = {
    "month_first": "Month first (US style): 03/05/2024 is March 5, 2024",
    "day_first":   "Day first (UK and most other countries): 03/05/2024 is May 3, 2024",
}


def _read_dates(text: pd.Series, day_first: bool) -> pd.Series:
    """Read text values one by one as dates, in whatever format each uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # pandas warns about guessing formats
        try:
            out = pd.to_datetime(text, format="mixed", dayfirst=day_first,
                                 errors="coerce")
        except (ValueError, TypeError):
            # Rare: a column that mixes time zones cannot be read in one go,
            # so read each value on its own and drop the time zone.
            def one(value):
                """Read a single value; blank if it is not a date."""
                d = pd.to_datetime(value, dayfirst=day_first, errors="coerce")
                if pd.isna(d):
                    return pd.NaT
                return d.tz_localize(None) if d.tzinfo else d
            out = pd.to_datetime(text.map(one), errors="coerce")
    # Dates with a time zone are kept as local clock time without the zone,
    # so the whole column has one consistent type.
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _parse_dates(series: pd.Series, day_first: bool) -> pd.Series:
    """
    Read every value in a column as a date. Unreadable values become NaT (blank).

    Values that start with a 4-digit year (2024-03-05, 2024/03/05 14:30) are
    always read as year, month, day. This matters: with day-first reading
    on, the general reader would otherwise turn 2024-03-05 into May 3.
    Every other value ("03/05/2024", "March 5, 2024") is read in the order
    the user chose.
    """
    # Work on clean text: stray spaces are common in real files
    text = series.astype("string").str.strip()

    year_first = text.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", na=False)
    other = text.notna() & ~year_first

    # Start with all blanks, then fill in each group
    dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if year_first.any():
        dates.loc[year_first] = _read_dates(text[year_first], day_first=False)
    if other.any():
        dates.loc[other] = _read_dates(text[other], day_first=day_first)
    return dates


def date_conversion_preview(df: pd.DataFrame, column_name: str,
                            order: str = "month_first") -> dict:
    """
    Check what converting a column to dates WOULD do, without changing it.

    Returns
    -------
    dict with keys:
        n_filled     int        Cells that have a value now
        n_ok         int        Of those, how many can be read as dates
        n_lost       int        Of those, how many would become blank
        lost_examples list      Up to 3 values that cannot be read
        examples     DataFrame  Up to 5 different values and the date each becomes
        n_ambiguous  int        Values that give a DIFFERENT date in the other order
                                (like 03/05/2024), so the order choice matters
        order_hint   str|None   A warning when the values show the file uses
                                the OTHER order: '25/12/2024' only makes sense
                                day first, '12/25/2024' only month first
    """
    original = df[column_name]
    day_first = order == "day_first"
    dates = _parse_dates(original, day_first)
    other = _parse_dates(original, not day_first)     # the other reading

    filled = original.notna()
    lost = filled & dates.isna()

    # Values both readings understand but disagree on (03/05 vs 05/03)
    both = dates.notna() & other.notna()
    ambiguous = both & (dates != other)

    # Five different values, to show the user what will happen
    sample = original[filled].drop_duplicates().head(5)
    examples = pd.DataFrame({
        "Value now": sample.astype(str).values,
        "Becomes": [
            f"{d:%B} {d.day}, {d.year}" if pd.notna(d) else "(blank: not a date)"
            for d in dates[sample.index]
        ],
    })

    return {
        "n_filled": int(filled.sum()),
        "n_ok": int((filled & dates.notna()).sum()),
        "n_lost": int(lost.sum()),
        "lost_examples": original[lost].astype(str).unique()[:3].tolist(),
        "examples": examples,
        "n_ambiguous": int(ambiguous.sum()),
        "order_hint": _order_hint(original, order),
    }


def _order_hint(series: pd.Series, order: str):
    """
    Look for values that can only be read one way, like 25/12/2024 (there is
    no month 25, so the day must come first). If they point to the order the
    user did NOT pick, return a short warning; otherwise return None.

    This is needed because pandas quietly reads 25/12/2024 as December 25
    even when month first is chosen, while 05/03/2024 in the same file is
    read as March 5. The result would be a column of mixed-up dates.
    """
    parts = series.astype("string").str.strip().str.extract(
        r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}"
    )
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    day_first_only = int(((first > 12) & (second <= 12)).sum())
    month_first_only = int(((second > 12) & (first <= 12)).sum())

    if order == "month_first" and day_first_only and not month_first_only:
        example = series[(first > 12) & (second <= 12)].astype(str).iloc[0].strip()
        return (f"{day_first_only:,} value(s) like '{example}' only make sense with "
                f"the day first. This file probably writes the day first.")
    if order == "day_first" and month_first_only and not day_first_only:
        example = series[(second > 12) & (first <= 12)].astype(str).iloc[0].strip()
        return (f"{month_first_only:,} value(s) like '{example}' only make sense with "
                f"the month first. This file probably writes the month first.")
    return None


def convert_to_dates(df: pd.DataFrame, column_name: str,
                     order: str = "month_first") -> tuple:
    """
    Convert a text column into real dates, in place.

    Use date_conversion_preview() first. Like convert_to_numeric(), this
    refuses to run when no value can be read, so a column is never wiped out.

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    preview = date_conversion_preview(df, column_name, order)

    # Safety net: never wipe out a whole column
    if preview["n_ok"] == 0:
        return df.copy(), (
            f"'{column_name}' was not converted: none of its values can be "
            f"read as dates."
        )

    new_df = df.copy()
    new_df[column_name] = _parse_dates(df[column_name], order == "day_first")

    order_words = "day first" if order == "day_first" else "month first"
    log_message = (
        f"Converted '{column_name}' to dates ({preview['n_ok']:,} values, "
        f"read {order_words})."
    )
    if preview["n_lost"]:
        log_message += (
            f" {preview['n_lost']:,} value(s) could not be read and are now blank."
        )
    return new_df, log_message


# ── Pull numbers out of text ───────────────────────────────────────────────────
# For values like "93 min", "4 Seasons" or "$137K-$171K (Glassdoor est.)",
# where the converter above correctly refuses because there is extra text.

# One number inside text, with three parts:
#   (-?)            an optional minus sign, only when it is not glued to a word
#                   or digit before it, so the "-" in "137-171" is a range
#                   dash, not a negative sign
#   (\d[\d,]*(?:\.\d+)?)   the digits, allowing commas and a decimal part
#   ([kK]|M)(?![A-Za-z])   an optional K (thousand) or capital M (million) right
#                   after the number, only when no other letter follows, so
#                   "93 min" is not 93 million. Lowercase m is left alone because
#                   it usually means minutes or meters ("10m 30s").
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(-?)(\d[\d,]*(?:\.\d+)?)((?:[kK]|M)(?![A-Za-z]))?")

# What each method does, shown in the app
EXTRACT_METHODS = {
    "first":   "First number   ('93 min' -> 93)",
    "average": "Average of a range   ('$137K-$171K' -> 154,000)",
    "last":    "Last number   ('Season 1 of 4' -> 4)",
}


def _numbers_in(text, use_suffix: bool) -> list:
    """Return every number found in one value, as floats."""
    numbers = []
    for minus, digits, suffix in _NUMBER_PATTERN.findall(str(text)):
        value = float(digits.replace(",", ""))           # "1,250" -> 1250.0
        if use_suffix and suffix:
            value *= 1_000 if suffix.lower() == "k" else 1_000_000
        numbers.append(-value if minus else value)
    return numbers


def _extract(series: pd.Series, method: str, use_suffix: bool) -> pd.Series:
    """Apply the chosen method to every cell. No number found -> blank."""
    def pick(value):
        """Return the one number this cell should become, or blank."""
        if pd.isna(value):
            return np.nan                                  # blank stays blank
        nums = _numbers_in(value, use_suffix)
        if not nums:
            return np.nan                                  # text with no number
        if method == "first":
            return nums[0]
        if method == "last":
            return nums[-1]
        return sum(nums[:2]) / len(nums[:2])               # average of a range
    return series.map(pick).astype(float)


def extract_number_preview(df: pd.DataFrame, column_name: str,
                           method: str, use_suffix: bool = True) -> dict:
    """
    Show what pulling numbers out of a column WOULD give, without changing it.

    Returns
    -------
    dict with keys:
        examples   pd.DataFrame   Up to 5 different values: before and after
        n_filled   int            Cells that have a value now
        n_found    int            Of those, how many contain a number
        n_none     int            Of those, how many have no number (will be blank)
    """
    original = df[column_name]
    result   = _extract(original, method, use_suffix)
    filled   = original.notna()

    # 5 different example values, so the user sees the variety in the column
    sample = original.dropna().drop_duplicates().head(5)
    examples = pd.DataFrame({
        "Original value":  sample.astype(str).values,
        "Number":          _extract(sample, method, use_suffix).values,
    })
    return {
        "examples": examples,
        "n_filled": int(filled.sum()),
        "n_found":  int((filled & result.notna()).sum()),
        "n_none":   int((filled & result.isna()).sum()),
    }


def extract_number(df: pd.DataFrame, column_name: str,
                   method: str, use_suffix: bool = True) -> tuple:
    """
    Pull a number out of each text value into a NEW column.

    The original column is kept unchanged, like the other preparation tools.
    The new column is named '<column>_number' ('<column>_number_2' if taken).

    Parameters
    ----------
    method     : "first", "average" or "last" (see EXTRACT_METHODS)
    use_suffix : True reads K as thousand and M as million

    Returns
    -------
    tuple of (new_df, log_message)
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."
    if method not in EXTRACT_METHODS:
        return df.copy(), f"Unknown method '{method}'."

    preview = extract_number_preview(df, column_name, method, use_suffix)
    if preview["n_found"] == 0:
        return df.copy(), f"No numbers were found in '{column_name}', so nothing was changed."

    # Pick a column name that is not already used
    new_col, n = f"{column_name}_number", 2
    while new_col in df.columns:
        new_col, n = f"{column_name}_number_{n}", n + 1

    new_df = df.copy()
    new_df[new_col] = _extract(df[column_name], method, use_suffix)

    log_message = (
        f"Pulled numbers out of '{column_name}' into new column '{new_col}' "
        f"({preview['n_found']:,} values, method: {method})."
    )
    if preview["n_none"]:
        log_message += f" {preview['n_none']:,} value(s) had no number and are blank."
    return new_df, log_message


# ── Transformation operations ──────────────────────────────────────────────────
# Transformations change the VALUES in a column, not the structure of the data.
# They are mainly used to prepare data for statistical modeling.
#
# Important: transformations add a new column rather than replacing the original.
# The new column gets the suffix '_log', '_scaled', '_encoded', etc.
# This way the user can compare before and after, and the original is preserved.


def log_transform(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Apply a logarithmic transformation to compress extreme values.

    What is a log transform?
    ------------------------
    Imagine a column where most values are between $100 and $1,000
    but a few are $10,000,000. A chart of this data is unreadable —
    everything looks flat at the bottom because the scale is dominated
    by the giant outliers.

    A log transform compresses the scale so that:
        $1       → 0.0
        $10      → 2.3
        $100     → 4.6
        $1,000   → 6.9
        $10,000  → 9.2
        $1,000,000 → 13.8

    The data is still in the same order (larger values are still larger),
    but everything fits on a readable scale.

    We use log1p (log of 1+x) instead of plain log(x) because:
    - It handles zero values safely (log(0) is undefined, log(1+0) = 0)
    - Most datasets have some zero values

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The numeric column to transform.

    Returns
    -------
    tuple of (new_df, log_message)
        The new column is named '{column_name}_log'.
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    # Check that the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df.copy(), (
            f"'{column_name}' is not a numeric column — "
            f"log transform only works on numbers."
        )

    # Check for negative values (log of a negative number is undefined)
    if (df[column_name].dropna() < 0).any():
        return df.copy(), (
            f"'{column_name}' contains negative values. "
            f"Log transform requires all values to be 0 or greater."
        )

    new_df = df.copy()

    # np.log1p(x) = log(1 + x)
    # This is applied element-by-element to every value in the column
    new_col_name = f"{column_name}_log"
    new_df[new_col_name] = np.log1p(new_df[column_name])

    # Report the improvement in skewness
    original_skew = round(df[column_name].skew(), 1)
    new_skew      = round(new_df[new_col_name].skew(), 1)

    log_message = (
        f"Applied log scale to '{column_name}' → new column '{new_col_name}'. "
        f"Value spread improved from {original_skew} to {new_skew} "
        f"(closer to 0 is better)."
    )

    return new_df, log_message


def standard_scale(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Rescale a numeric column to have a mean of 0 and standard deviation of 1.

    What does this mean?
    --------------------
    After scaling:
    - Values near the average become close to 0
    - Values 1 standard deviation above average become ~1.0
    - Values 1 standard deviation below average become ~-1.0

    Why do this?
    ------------
    Many machine learning models compare columns to each other.
    If one column is measured in dollars (0–1,000,000) and another
    in days (0–365), the model would incorrectly treat the dollar
    column as more important just because its numbers are bigger.

    Standard scaling puts all columns on the same footing so the
    model can compare them fairly.

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The numeric column to scale.

    Returns
    -------
    tuple of (new_df, log_message)
        The new column is named '{column_name}_scaled'.
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df.copy(), (
            f"'{column_name}' is not a numeric column — "
            f"scaling only works on numbers."
        )

    new_df = df.copy()

    # StandardScaler expects a 2D array (matrix), not a 1D column.
    # .values reshapes the column into an array.
    # .reshape(-1, 1) means "one column, as many rows as needed".
    col_values = new_df[column_name].fillna(0).values.reshape(-1, 1)

    # StandardScaler from scikit-learn:
    #   fit()       calculates the mean and std from the data
    #   transform() applies the scaling formula: (value - mean) / std
    # fit_transform() does both in one step
    scaler     = StandardScaler()
    scaled     = scaler.fit_transform(col_values)

    new_col_name = f"{column_name}_scaled"

    # .flatten() converts the 2D array back to a 1D series
    new_df[new_col_name] = scaled.flatten()

    log_message = (
        f"Scaled '{column_name}' → new column '{new_col_name}'. "
        f"Values now have mean ≈ 0 and standard deviation ≈ 1. "
        f"Original column preserved."
    )

    return new_df, log_message


def minmax_scale(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Rescale a numeric column so all values fall between 0 and 1.

    What does this mean?
    --------------------
    The minimum value becomes 0.0, the maximum becomes 1.0, and
    everything in between is scaled proportionally.

    Example:
        Original: $0, $500, $1,000
        Scaled:    0.0,  0.5,   1.0

    When to use this vs standard_scale
    -----------------------------------
    Use minmax_scale when:
    - You want values to stay positive (0 to 1)
    - You're working with neural networks or distance-based models
    - The column has a clear minimum and maximum boundary

    Use standard_scale when:
    - The column has outliers that would compress everything else
    - You're working with linear models or algorithms that assume
      values can be negative

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The numeric column to scale.

    Returns
    -------
    tuple of (new_df, log_message)
        The new column is named '{column_name}_minmax'.
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return df.copy(), (
            f"'{column_name}' is not a numeric column."
        )

    new_df = df.copy()

    col_values = new_df[column_name].fillna(0).values.reshape(-1, 1)

    # MinMaxScaler formula: (value - min) / (max - min)
    scaler       = MinMaxScaler()
    scaled       = scaler.fit_transform(col_values)
    new_col_name = f"{column_name}_minmax"
    new_df[new_col_name] = scaled.flatten()

    log_message = (
        f"Scaled '{column_name}' to 0–1 range → new column '{new_col_name}'. "
        f"Original column preserved."
    )

    return new_df, log_message


def one_hot_encode(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Convert a category column into multiple yes/no (0/1) columns.

    What does this mean?
    --------------------
    Machine learning models cannot use text directly. They need numbers.
    One-hot encoding converts a category column like 'Claim Type' into
    separate columns, one per category, each containing 0 or 1:

    Before:
        Claim Type
        Property Damage
        Passenger Theft
        Property Damage

    After:
        Claim Type_Property Damage  Claim Type_Passenger Theft
                1                              0
                0                              1
                1                              0

    Why is this useful?
    -------------------
    This lets a model understand that 'Property Damage' and
    'Passenger Theft' are different things, without assuming one
    is "greater than" the other (which would happen if we used numbers
    like 1 and 2).

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The category column to encode.

    Returns
    -------
    tuple of (new_df, log_message)
        The original column is removed and replaced by new 0/1 columns.
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    # Count unique values — warn if there are too many
    n_unique = df[column_name].nunique()
    if n_unique > 30:
        return df.copy(), (
            f"'{column_name}' has {n_unique} unique values — "
            f"one-hot encoding would add {n_unique} new columns, "
            f"which is too many. Consider dropping this column instead, "
            f"or using it only for grouping in charts."
        )

    new_df = df.copy()

    # pd.get_dummies() is pandas' built-in one-hot encoder.
    # prefix=column_name    → new columns named 'Claim Type_Property Damage' etc.
    # drop_first=False      → keep all categories (don't drop one for math reasons)
    # dtype=int             → values are 0 and 1, not True and False
    dummies = pd.get_dummies(
        new_df[column_name],
        prefix=column_name,
        drop_first=False,
        dtype=int
    )

    # Remove the original text column and add the new 0/1 columns
    new_df = new_df.drop(columns=[column_name])
    new_df = pd.concat([new_df, dummies], axis=1)
    # axis=1 means "add as new columns" (axis=0 would mean "add as new rows")

    log_message = (
        f"Encoded '{column_name}' into {n_unique} yes/no columns "
        f"for modeling. Original column removed."
    )

    return new_df, log_message


def extract_date_part(
    df: pd.DataFrame,
    column_name: str,
    part: str
) -> tuple:
    """
    Extract the year, month, or day from a date column as a new column.

    Why is this useful?
    -------------------
    Machine learning models cannot use raw dates. But the YEAR or MONTH
    extracted from a date can be very useful. For example, knowing a
    claim was filed in 2005 vs 2009 might be predictive of its outcome.

    Parameters
    ----------
    df          : pd.DataFrame   The current working dataset.
    column_name : str            The date column to extract from.
    part        : str            What to extract: 'year', 'month', or 'day'

    Returns
    -------
    tuple of (new_df, log_message)
        A new column named '{column_name}_{part}' is added.
    """
    if column_name not in df.columns:
        return df.copy(), f"Column '{column_name}' not found."

    # Check that the column is actually a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        return df.copy(), (
            f"'{column_name}' is not a date column. "
            f"Make sure the column has been converted to a date format first."
        )

    if part not in ("year", "month", "day"):
        return df.copy(), f"Unknown part '{part}'. Choose 'year', 'month', or 'day'."

    new_df = df.copy()

    new_col_name = f"{column_name}_{part}"

    # .dt is the pandas accessor for datetime operations
    # .dt.year, .dt.month, .dt.day each extract just that part
    if part == "year":
        new_df[new_col_name] = new_df[column_name].dt.year
    elif part == "month":
        new_df[new_col_name] = new_df[column_name].dt.month
    else:
        new_df[new_col_name] = new_df[column_name].dt.day

    log_message = (
        f"Extracted {part} from '{column_name}' → new column '{new_col_name}'."
    )

    return new_df, log_message


# ── Smart suggestions ──────────────────────────────────────────────────────────

def get_cleaning_suggestions(df: pd.DataFrame, col_types: dict) -> list:
    """
    Automatically detect problems and suggest the right fix for each one.

    This function powers the "Suggested Actions" section at the top
    of the Clean & Export tab. Instead of making the user figure out
    what their data needs, SmartWrangle tells them.

    Each suggestion is a dictionary with:
        priority    int     1 = fix this first, 2 = fix next, 3 = optional
        issue       str     What the problem is (plain English)
        action      str     What to do about it (plain English)
        operation   str     The internal operation name (used by the tab
                            to know which button to show)
        column      str     Which column the suggestion applies to
                            (or None if it applies to the whole dataset)

    Parameters
    ----------
    df        : pd.DataFrame   The current working dataset.
    col_types : dict           Output from detector.detect_column_types().

    Returns
    -------
    list of dicts, sorted by priority (highest priority first)
    """
    # We import get_columns_of_type here to avoid a circular import
    from engine.detector import get_columns_of_type

    suggestions = []

    # ── Check for duplicates ───────────────────────────────────────────────────
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        dup_pct = round(n_dups / len(df) * 100, 1)
        suggestions.append({
            "priority":  1,
            "issue":     f"{n_dups:,} duplicate rows found ({dup_pct}% of your data).",
            "action":    "Remove duplicate rows",
            "operation": "remove_duplicates",
            "column":    None,
        })

    # ── Check for placeholder values (ERROR, UNKNOWN, N/A ...) ────────────────
    # Listed before missing values because blanking placeholders changes the
    # missing-value counts. Only the default-ticked ones are counted here, so
    # the one-click fix never touches -1 or one-off words.
    default_placeholders = [p for p in find_placeholders(df) if placeholder_selected_by_default(p)]
    if default_placeholders:
        n_ph   = sum(p["count"] for p in default_placeholders)
        n_cols = len({p["column"] for p in default_placeholders})
        words  = sorted({p["shown_as"] for p in default_placeholders})[:3]
        suggestions.append({
            "priority":  1,
            "issue": (
                f"{n_ph:,} placeholder values like {', '.join(repr(w) for w in words)} "
                f"in {n_cols} column(s) are hiding missing data."
            ),
            "action":    "Mark placeholder values as blank",
            "operation": "blank_placeholders",
            "column":    None,
        })

    # ── Check for numbers or money stored as text ─────────────────────────────
    # A column like Amount ("$1,094.88") cannot be added up or charted until
    # it is converted. The one-click fix is only offered when every filled
    # value can be read as a number, so the button never blanks out data.
    # Columns where some values would be lost (like "seven" in Quantity) are
    # left to the Convert tool, which shows those values and asks first.
    from engine.detector import infer_expected_type
    for col in df.columns:
        if col_types.get(col) not in ("categorical", "high_cardinality"):
            continue
        expected = infer_expected_type(df[col])
        if expected not in ("metric", "financial"):
            continue
        preview = numeric_conversion_preview(df, col)
        if preview["n_ok"] == 0 or preview["n_lost"] > 0:
            continue
        kind_text = "money amounts" if expected == "financial" else "numbers"
        suggestions.append({
            "priority":  1,
            "issue": (
                f"'{col}' holds {kind_text} stored as text, so it cannot be "
                f"added up or charted."
            ),
            "action":    f"Convert '{col}' to numbers",
            "operation": "convert_to_numeric",
            "column":    col,
        })

    # ── Check for missing values per column ───────────────────────────────────
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue

        missing_pct = round(n_missing / len(df) * 100, 1)
        col_type    = col_types.get(col, "categorical")

        # Suggest the best fill strategy based on column type
        if col_type == "financial" or col_type == "metric":
            # For numeric columns, median is safer than mean (outliers)
            suggested_op  = "fill_missing_median"
            suggested_action = (
                f"Fill {n_missing:,} empty values in '{col}' with the median value"
            )
        elif col_type == "date_column":
            # For date columns, just drop the rows (can't estimate dates)
            suggested_op  = "drop_missing_rows"
            suggested_action = (
                f"Remove {n_missing:,} rows with a missing date in '{col}'"
            )
        else:
            # For category columns, use the most common value
            suggested_op  = "fill_missing_mode"
            suggested_action = (
                f"Fill {n_missing:,} empty values in '{col}' "
                f"with the most common value"
            )

        # Priority: high if > 5% missing, medium otherwise
        priority = 1 if missing_pct > 5 else 2

        suggestions.append({
            "priority":  priority,
            "issue":     f"'{col}' is {missing_pct}% empty ({n_missing:,} missing values).",
            "action":    suggested_action,
            "operation": suggested_op,
            "column":    col,
        })

    # ── Check for highly skewed numeric columns ────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        # Only suggest for non-negative columns (log transform requires >= 0)
        if abs(col_data.skew()) >= 1 and (col_data >= 0).all():
            suggestions.append({
                "priority":  3,
                "issue": (
                    f"'{col}' has a very uneven value distribution — "
                    f"a few extreme values are much larger than the rest."
                ),
                "action":    f"Apply log scale to '{col}' to compress extreme values",
                "operation": "log_transform",
                "column":    col,
            })

    # ── Sort by priority — 1 (most urgent) first ──────────────────────────────
    suggestions.sort(key=lambda x: x["priority"])

    return suggestions


# ── END OF FILE ────────────────────────────────────────────────────────────────
# Nothing should be placed below this line.
# All functions in this file are imported and called by tabs/clean_export.py.