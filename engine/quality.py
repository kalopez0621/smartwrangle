"""
engine/quality.py
=================
Dataset quality scoring engine for SmartWrangle 2.0.

What this file does
-------------------
When someone uploads a dataset, this module examines it and produces
two things:

    1. A quality SCORE from 0 to 100 (like a grade on a report card)
    2. A plain-English REPORT explaining what is good, what needs
       attention, and what to do about it

The score and report are shown on the "Understand" tab. They give
any user — technical or not — an immediate sense of whether their
data is ready to work with.

How the score works
-------------------
Every dataset starts at 100 points. Points are deducted for problems:

    Missing values      Up to -30 points  (empty cells hurt analysis)
    Duplicate rows      Up to -20 points  (copies skew results)
    Extreme outliers    Up to -10 points  (one huge value breaks charts)
    Skewed numbers      Up to  -5 points per column  (hard to model)

The final score maps to a plain-English grade:
    90-100  →  Excellent
    75-89   →  Good
    60-74   →  Fair
    0-59    →  Needs Work

How to use this file
--------------------
    from engine.quality import score_dataset, get_quality_report

    result  = score_dataset(df, col_types)
    report  = get_quality_report(result)

    # result is a dictionary with the score and all the details
    # report is a list of plain-English strings ready to display
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# pandas  : the main library for working with tabular data (DataFrames)
# numpy   : math and statistics functions used for skewness and outlier detection
import pandas as pd
import numpy as np


# ── Score thresholds ───────────────────────────────────────────────────────────
# These numbers define the boundaries between grade levels.
# Changing these numbers here changes them everywhere in the app.
SCORE_EXCELLENT = 90   # 90 and above → Excellent
SCORE_GOOD      = 75   # 75 to 89     → Good
SCORE_FAIR      = 60   # 60 to 74     → Fair
                       # Below 60     → Needs Work

# How many points to deduct for each type of problem.
# These are caps — the penalty never exceeds these values no matter
# how bad the problem is.
MAX_MISSING_PENALTY   = 30   # worst-case penalty for missing values
MAX_DUPLICATE_PENALTY = 20   # worst-case penalty for duplicate rows
MAX_OUTLIER_PENALTY   = 10   # worst-case penalty for extreme outliers
SKEW_PENALTY_PER_COL  =  5   # penalty per highly skewed numeric column


def _grade_from_score(score: float) -> str:
    """
    Convert a numeric score into a plain-English grade word.

    This is a helper function (the underscore prefix is a Python
    convention meaning 'intended for internal use in this file').

    Parameters
    ----------
    score : float
        A number between 0 and 100.

    Returns
    -------
    str
        One of: 'Excellent', 'Good', 'Fair', 'Needs Work'

    Example
    -------
    >>> _grade_from_score(92)
    'Excellent'
    >>> _grade_from_score(55)
    'Needs Work'
    """
    if score >= SCORE_EXCELLENT:
        return "Excellent"
    elif score >= SCORE_GOOD:
        return "Good"
    elif score >= SCORE_FAIR:
        return "Fair"
    else:
        return "Needs Work"


def _grade_color(grade: str) -> str:
    """
    Return a hex color code for each grade level.

    These colors are used in the Understand tab to visually signal
    whether the dataset is in good shape. Green = good, Red = bad.

    Parameters
    ----------
    grade : str
        Output from _grade_from_score().

    Returns
    -------
    str
        A CSS hex color string like '#2ecc71'.
    """
    # Each grade gets a color that intuitively signals its severity.
    # These are the same colors used in traffic lights and report cards.
    colors = {
        "Excellent":  "#2ecc71",   # green  — ready to go
        "Good":       "#27ae60",   # darker green — mostly ready
        "Fair":       "#f39c12",   # amber  — some work needed
        "Needs Work": "#e74c3c",   # red    — significant issues
    }
    # .get() returns the color if found, or gray if an unknown grade is passed
    return colors.get(grade, "#95a5a6")


def _analyze_missing(df: pd.DataFrame) -> dict:
    """
    Analyze missing values across the entire dataset.

    A missing value is a cell with no data — shown as NaN (Not a Number)
    in pandas, or blank in Excel/CSV. Missing values are a problem
    because many calculations and charts cannot handle them.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.

    Returns
    -------
    dict with keys:
        total_missing   int     Total number of empty cells
        missing_pct     float   Percentage of ALL cells that are empty
        cols_affected   list    Column names that have at least one missing value
        worst_col       str     The column with the highest missing percentage
        worst_col_pct   float   How empty that worst column is (as a percentage)
        penalty         float   Points deducted from the quality score
    """
    # Count how many cells are missing in each column
    # isnull() returns True/False for each cell, sum() counts the Trues
    missing_per_col = df.isnull().sum()

    # Total cells in the entire dataset = rows × columns
    total_cells   = df.shape[0] * df.shape[1]

    # Total missing across all columns
    total_missing = int(missing_per_col.sum())

    # What percentage of ALL cells are missing
    # We avoid dividing by zero in case the dataset is somehow empty
    missing_pct = (
        round(total_missing / total_cells * 100, 4)
        if total_cells > 0
        else 0.0
    )

    # Which columns have at least one missing value
    cols_affected = missing_per_col[missing_per_col > 0].index.tolist()

    # Find the single worst column (most missing)
    worst_col     = None
    worst_col_pct = 0.0

    if cols_affected:
        # idxmax() finds the column name with the highest value
        worst_col = missing_per_col.idxmax()

        # Calculate what percentage of that column is empty
        worst_col_pct = round(
            df[worst_col].isnull().mean() * 100, 1
        )

    # Calculate the penalty score
    # The more missing values, the bigger the penalty —
    # but it never exceeds MAX_MISSING_PENALTY (30 points)
    penalty = min(MAX_MISSING_PENALTY, missing_pct * 10)

    return {
        "total_missing": total_missing,
        "missing_pct":   missing_pct,
        "cols_affected": cols_affected,
        "worst_col":     worst_col,
        "worst_col_pct": worst_col_pct,
        "penalty":       round(penalty, 2),
    }


def _analyze_duplicates(df: pd.DataFrame) -> dict:
    """
    Analyze duplicate rows in the dataset.

    A duplicate row is a row that is identical to another row in every
    column. Duplicates are a problem because they make certain groups
    look larger than they really are, skewing charts and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.

    Returns
    -------
    dict with keys:
        n_duplicates    int     Number of duplicate rows found
        duplicate_pct   float   Percentage of rows that are duplicates
        penalty         float   Points deducted from the quality score
    """
    # duplicated() returns True for every row that is a copy of an earlier row
    # .sum() counts how many Trues there are
    n_duplicates = int(df.duplicated().sum())

    # What percentage of the dataset are duplicates
    duplicate_pct = (
        round(n_duplicates / len(df) * 100, 2)
        if len(df) > 0
        else 0.0
    )

    # Penalty scales with how many duplicates there are,
    # capped at MAX_DUPLICATE_PENALTY (20 points)
    penalty = min(MAX_DUPLICATE_PENALTY, duplicate_pct * 4)

    return {
        "n_duplicates":  n_duplicates,
        "duplicate_pct": duplicate_pct,
        "penalty":       round(penalty, 2),
    }


def _analyze_numeric_columns(df: pd.DataFrame, col_types: dict) -> dict:
    """
    Analyze numeric columns for skewness and extreme outliers.

    What is skewness?
    -----------------
    Skewness measures how lopsided a column's values are.
    A symmetric column (like height) has skewness near 0.
    A column like income or claim amounts is right-skewed —
    most values are small but a few are very large. Skewed
    columns can distort charts and are harder to model.

    What are outliers?
    ------------------
    An outlier is a value that is far outside the normal range.
    For example, if most claims are under $1,000 but one is
    $125,000,000 — that's an extreme outlier. It doesn't mean
    the value is wrong, but it affects how the data looks in charts.

    We use the IQR method to detect outliers:
        Lower bound = Q1 - 1.5 × IQR
        Upper bound = Q3 + 1.5 × IQR
    Any value outside these bounds is flagged as an outlier.
    (Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1)

    Parameters
    ----------
    df       : pd.DataFrame   The dataset.
    col_types: dict           Output from detector.detect_column_types().
                              Used to identify financial vs metric columns.

    Returns
    -------
    dict with keys:
        columns         list    All numeric column names analyzed
        skewed_cols     list    Columns with |skewness| >= 1
        outlier_cols    list    Columns where > 5% of values are outliers
        details         dict    Per-column breakdown of skew and outlier stats
        skew_penalty    float   Total penalty from skewed columns
        outlier_penalty float   Total penalty from extreme outliers
    """
    # Select all numeric columns (int and float)
    # We use select_dtypes to let pandas find them automatically
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # We will fill these lists as we analyze each column
    skewed_cols    = []   # columns that are highly skewed
    outlier_cols   = []   # columns with many outliers

    # Per-column details — used in the advanced section of the Understand tab
    details        = {}

    # Running totals for penalties
    skew_penalty_total    = 0.0
    outlier_penalty_total = 0.0

    for col in numeric_cols:

        # Drop missing values before calculating stats
        # (skew and percentile calculations break on NaN values)
        col_data = df[col].dropna()

        # Skip this column if it's empty after dropping NaN
        if len(col_data) == 0:
            continue

        # ── Skewness ──────────────────────────────────────────────────────
        # .skew() returns a positive number for right-skewed data
        # (long tail to the right) and negative for left-skewed.
        # We use abs() because we care about the magnitude, not direction.
        skew_value = col_data.skew()

        # A column is "highly skewed" if |skewness| >= 1
        # This is a widely accepted threshold in data science
        is_highly_skewed = abs(skew_value) >= 1

        if is_highly_skewed:
            skewed_cols.append(col)
            # Each highly skewed column costs SKEW_PENALTY_PER_COL points (5)
            skew_penalty_total += SKEW_PENALTY_PER_COL

        # ── Outlier detection using IQR method ────────────────────────────
        # Step 1: Find Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)

        # Step 2: Calculate IQR (Interquartile Range = Q3 minus Q1)
        # IQR represents the middle 50% of the data
        IQR = Q3 - Q1

        # Step 3: Calculate the "fence" boundaries
        # Values outside these fences are considered outliers
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        # Step 4: Count how many values fall outside the fences
        outlier_mask  = (col_data < lower_fence) | (col_data > upper_fence)
        outlier_count = int(outlier_mask.sum())
        outlier_pct   = round(outlier_count / len(col_data) * 100, 1)

        # Flag columns where more than 5% of values are outliers
        has_many_outliers = outlier_pct > 5
        if has_many_outliers:
            outlier_cols.append(col)

        # ── Extreme outlier check (financial columns specifically) ─────────
        # For financial columns, we also check if the maximum value is
        # wildly higher than the 99th percentile. This catches cases like
        # Claim Amount where the max ($125M) is 18,000x the 99th percentile.
        p99     = col_data.quantile(0.99)
        max_val = col_data.max()

        # "Extreme" = max is more than 10x the 99th percentile
        is_extreme = (p99 > 0) and (max_val > p99 * 10)

        # Penalty for outliers: scaled by percentage, capped at MAX_OUTLIER_PENALTY
        outlier_penalty = min(MAX_OUTLIER_PENALTY, outlier_pct * 0.5)
        outlier_penalty_total += outlier_penalty

        # ── Store per-column details ────────────────────────────────────────
        details[col] = {
            "skewness":        round(skew_value, 2),
            "is_highly_skewed": is_highly_skewed,
            "outlier_count":   outlier_count,
            "outlier_pct":     outlier_pct,
            "has_many_outliers": has_many_outliers,
            "lower_fence":     round(lower_fence, 2),
            "upper_fence":     round(upper_fence, 2),
            "p99":             round(p99, 2),
            "max_value":       round(max_val, 2),
            "is_extreme":      is_extreme,
        }

    # Cap total outlier penalty at the maximum allowed
    outlier_penalty_total = min(MAX_OUTLIER_PENALTY, outlier_penalty_total)

    return {
        "columns":         numeric_cols,
        "skewed_cols":     skewed_cols,
        "outlier_cols":    outlier_cols,
        "details":         details,
        "skew_penalty":    round(skew_penalty_total, 2),
        "outlier_penalty": round(outlier_penalty_total, 2),
    }


def score_dataset(df: pd.DataFrame, col_types: dict) -> dict:
    """
    Run the full quality analysis and return a complete result dictionary.

    This is the main function you call from the Understand tab.
    It runs all three sub-analyses (missing, duplicates, numeric health),
    adds up the penalties, and returns everything in one place.

    Parameters
    ----------
    df        : pd.DataFrame   The dataset to score.
    col_types : dict           Output from detector.detect_column_types().

    Returns
    -------
    dict with keys:
        score           float   Final quality score (0-100)
        grade           str     'Excellent', 'Good', 'Fair', or 'Needs Work'
        grade_color     str     Hex color string for displaying the grade
        missing         dict    Output from _analyze_missing()
        duplicates      dict    Output from _analyze_duplicates()
        numeric         dict    Output from _analyze_numeric_columns()
        n_rows          int     Total number of rows
        n_cols          int     Total number of columns

    Example
    -------
    >>> from engine.detector import detect_column_types
    >>> from engine.quality  import score_dataset
    >>>
    >>> col_types = detect_column_types(df)
    >>> result    = score_dataset(df, col_types)
    >>>
    >>> print(result['score'])    # e.g. 90.0
    >>> print(result['grade'])    # e.g. 'Excellent'
    """
    # ── Run each sub-analysis ──────────────────────────────────────────────────
    # Each function returns a dictionary of findings for that category.
    # We store them all separately so the tab can display each section.
    missing_result   = _analyze_missing(df)
    duplicate_result = _analyze_duplicates(df)
    numeric_result   = _analyze_numeric_columns(df, col_types)

    # ── Calculate the final score ──────────────────────────────────────────────
    # Start at 100 and subtract penalties from each analysis.
    score = 100.0
    score -= missing_result["penalty"]      # subtract missing value penalty
    score -= duplicate_result["penalty"]    # subtract duplicate row penalty
    score -= numeric_result["skew_penalty"] # subtract skewness penalty
    score -= numeric_result["outlier_penalty"]  # subtract outlier penalty

    # Clamp to [0, 100] — the score should never go negative or above 100
    score = max(0.0, min(100.0, round(score, 1)))

    # ── Determine grade and display color ─────────────────────────────────────
    grade       = _grade_from_score(score)
    grade_color = _grade_color(grade)

    # ── Return everything in one dictionary ───────────────────────────────────
    # By returning a single dict, the tab can cache this result in session state
    # and not recalculate it every time the user clicks something.
    return {
        "score":       score,
        "grade":       grade,
        "grade_color": grade_color,
        "missing":     missing_result,
        "duplicates":  duplicate_result,
        "numeric":     numeric_result,
        "n_rows":      len(df),
        "n_cols":      len(df.columns),
    }


def get_quality_report(result: dict) -> list:
    """
    Convert a score_dataset() result into plain-English sentences.

    This function takes the raw numbers from score_dataset() and
    turns them into sentences a non-technical person can read and
    understand immediately — no statistics knowledge required.

    Parameters
    ----------
    result : dict
        Output from score_dataset().

    Returns
    -------
    list of dicts, each with keys:
        level    str   'good', 'warning', or 'problem'
                       Used by the tab to choose green / yellow / red color
        message  str   The plain-English sentence to display

    Example output
    --------------
    [
        {'level': 'good',    'message': 'No missing values — your data is complete.'},
        {'level': 'good',    'message': 'No duplicate rows found.'},
        {'level': 'warning', 'message': "Claim Amount has extreme outliers ..."},
    ]
    """
    # We build a list of findings as we go.
    # Each finding is a dict with a severity level and a message.
    findings = []

    # ── Missing values ─────────────────────────────────────────────────────────
    missing = result["missing"]

    if missing["total_missing"] == 0:
        # No missing values — this is good news
        findings.append({
            "level":   "good",
            "message": "No missing values — your data is complete."
        })
    else:
        # There are missing values — explain how bad it is
        n  = missing["total_missing"]
        pct = missing["missing_pct"]
        col = missing["worst_col"]
        col_pct = missing["worst_col_pct"]

        # Choose severity based on how much data is missing
        level = "warning" if pct < 5 else "problem"

        findings.append({
            "level": level,
            "message": (
                f"{n:,} missing values found across "
                f"{len(missing['cols_affected'])} column(s). "
                f"The most affected column is '{col}' "
                f"({col_pct}% empty). "
                f"Use the Clean tab to fix this."
            )
        })

    # ── Duplicate rows ─────────────────────────────────────────────────────────
    duplicates = result["duplicates"]

    if duplicates["n_duplicates"] == 0:
        findings.append({
            "level":   "good",
            "message": "No duplicate rows found."
        })
    else:
        n   = duplicates["n_duplicates"]
        pct = duplicates["duplicate_pct"]

        level = "warning" if pct < 5 else "problem"

        findings.append({
            "level": level,
            "message": (
                f"{n:,} duplicate rows detected ({pct}% of your data). "
                f"These are identical copies that can skew your results. "
                f"Use the Clean tab to remove them."
            )
        })

    # ── Skewed numeric columns ─────────────────────────────────────────────────
    numeric = result["numeric"]

    if numeric["skewed_cols"]:
        # Explain skewness without using the word "skewness"
        cols_str = ", ".join(f"'{c}'" for c in numeric["skewed_cols"])
        findings.append({
            "level": "warning",
            "message": (
                f"Some numeric columns have very uneven value distributions: "
                f"{cols_str}. "
                f"Most values are clustered at one end while a few are extreme. "
                f"This is common in financial data and can be handled in the "
                f"Clean & Export tab."
            )
        })

    # ── Extreme outliers ───────────────────────────────────────────────────────
    # Report extreme outliers per column (only financial/metric columns)
    for col, detail in numeric["details"].items():
        if detail["is_extreme"]:
            findings.append({
                "level": "warning",
                "message": (
                    f"'{col}' contains extreme values — the largest value "
                    f"({detail['max_value']:,.0f}) is far above the typical range. "
                    f"Charts will automatically adjust to show the data clearly."
                )
            })

    # ── Outlier-heavy columns ──────────────────────────────────────────────────
    for col in numeric["outlier_cols"]:
        detail = numeric["details"][col]

        # Only report if not already reported as extreme
        if not detail["is_extreme"]:
            findings.append({
                "level": "warning",
                "message": (
                    f"'{col}' has {detail['outlier_pct']}% of values "
                    f"outside the normal range "
                    f"({detail['outlier_count']:,} values). "
                    f"These may be legitimate but are worth reviewing."
                )
            })

    return findings


def get_column_health_table(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """
    Build a plain-English health table for every numeric column.

    This is used in the 'Advanced Details' expandable section of the
    Understand tab. It gives technically curious users a column-by-column
    breakdown without putting jargon in the main view.

    Parameters
    ----------
    df        : pd.DataFrame   The dataset.
    col_types : dict           Output from detector.detect_column_types().

    Returns
    -------
    pd.DataFrame with columns:
        Column          Column name
        Type            Plain-English type label (e.g. 'Financial Amount')
        Missing         Number of missing values
        Missing %       Percentage missing
        Outliers        Number of outlier values
        Outlier %       Percentage of values that are outliers
        Value Spread    'Even' / 'Uneven' / 'Very Uneven'
        Status          Plain-English health label
    """
    # Import the plain_english_type helper from the detector module
    # We import here (not at the top) to avoid a circular import issue
    # if detector.py ever imports from quality.py
    from engine.detector import plain_english_type

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    rows = []

    for col in numeric_cols:

        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        # Missing values for this column
        n_missing   = int(df[col].isnull().sum())
        missing_pct = round(n_missing / len(df) * 100, 1)

        # Outliers using IQR
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR    = Q3 - Q1
        outlier_mask  = (col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)
        n_outliers    = int(outlier_mask.sum())
        outlier_pct   = round(n_outliers / len(col_data) * 100, 1)

        # Value spread (using skewness, but expressed in plain terms)
        skew = abs(col_data.skew())
        if skew < 0.5:
            spread = "Even"
        elif skew < 1:
            spread = "Slightly Uneven"
        else:
            spread = "Very Uneven"

        # Overall health status for this column
        # Problems are ranked: missing > outliers > spread
        if missing_pct > 20:
            status = "Needs Attention"
        elif outlier_pct > 15:
            status = "Needs Attention"
        elif skew >= 1:
            status = "Acceptable"
        else:
            status = "Healthy"

        rows.append({
            "Column":      col,
            "Type":        plain_english_type(col_types.get(col, "metric")),
            "Missing":     n_missing,
            "Missing %":   missing_pct,
            "Outliers":    n_outliers,
            "Outlier %":   outlier_pct,
            "Value Spread": spread,
            "Status":      status,
        })

    # Convert the list of rows into a DataFrame so it can be
    # displayed directly with st.dataframe() in the tab
    return pd.DataFrame(rows)


# ── END OF FILE ────────────────────────────────────────────────────────────────
# Nothing should be placed below this line.
# All functions in this file are imported and called by tabs/understand.py.