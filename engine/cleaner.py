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
# pandas      : for working with DataFrames (tables of data)
# numpy       : for math operations like log transformation
# StandardScaler, MinMaxScaler : from scikit-learn — used to rescale numbers
#   StandardScaler : makes a column have mean=0 and standard deviation=1
#   MinMaxScaler   : compresses a column into the range [0, 1]
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
    # .str.strip() removes leading and trailing whitespace
    new_df[column_name] = new_df[column_name].astype(str).str.strip()

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

    # Apply the requested case transformation using pandas string methods
    if case == "title":
        new_df[column_name] = new_df[column_name].astype(str).str.title()
        case_desc = "Title Case"
    elif case == "lower":
        new_df[column_name] = new_df[column_name].astype(str).str.lower()
        case_desc = "lowercase"
    else:
        new_df[column_name] = new_df[column_name].astype(str).str.upper()
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