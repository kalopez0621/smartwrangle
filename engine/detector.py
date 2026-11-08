"""
engine/detector.py
==================
Column type detection engine for SmartWrangle 2.0.

What this file does
-------------------
This is the first module that runs after a file loads. It looks at every
column and gives it a plain-English "type" so the rest of the app knows
how to treat it, without ever asking the user about data types.

The seven types
---------------
    date_column       dates and times                  -> time trend charts
    financial         money amounts (price, total ...) -> dollar formatting
    metric            other numbers                    -> histograms, scaling
    categorical       text with 50 or fewer values     -> bar charts, encoding
    high_cardinality  text with more than 50 values    -> top-10 charts
    text              long free text (notes, comments) -> not charted
    id_column         unique codes (Order ID ...)      -> left out of analysis

How a column gets its type
--------------------------
Two clues are combined:
    1. The column NAME  (for example "Order Date" contains "date")
    2. The column VALUES (are they numbers? text? how many are unique?)

The checks run in a fixed order, and the first one that matches wins.
See detect_column_types() for the exact order.

How to use this file
--------------------
    from engine.detector import detect_column_types, get_columns_of_type

    col_types = detect_column_types(df)            # {"Order Date": "date_column", ...}
    dates     = get_columns_of_type(col_types, "date_column")
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# pandas   : reads column types and converts text to dates
# warnings : hides harmless pandas messages while guessing date formats
import pandas as pd
import warnings


# ── Keyword lists ──────────────────────────────────────────────────────────────
# A column whose NAME contains one of these words is a candidate for that type.
# Matching is case-insensitive and works on part of a word, so "Received Date"
# matches "date" and "TotalSpent" matches "total".

# Words that suggest a date or time column
DATE_KEYWORDS = [
    "date", "time", "timestamp", "datetime",
    "received", "incident", "created", "updated",
    "modified", "opened", "closed", "submitted",
    "reported", "recorded", "logged", "posted"
]

# Words that suggest an identifier (a code that is unique per row)
ID_KEYWORDS = [
    "id", "number", "num", "no", "code", "key",
    "index", "seq", "sequence", "ref", "reference",
    "identifier", "uuid", "guid", "sku", "hash",
    "claim", "ticket", "record", "row", "case"
]

# Words that suggest a money column
FINANCIAL_KEYWORDS = [
    "amount", "price", "cost", "revenue", "salary",
    "pay", "wage", "fee", "value", "charge", "balance",
    "total", "sum", "budget", "spend", "expenditure",
    "income", "profit", "loss", "tax", "discount",
    "refund", "payment", "payout", "settlement", "close"
]

# Words that suggest long free text
TEXT_KEYWORDS = [
    "description", "notes", "note", "comment", "remarks",
    "remark", "detail", "narrative", "summary", "text",
    "message", "feedback", "review", "memo", "body"
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
# A text column with MORE than this many different values is "high cardinality"
# (too many categories for a normal bar chart).
HIGH_CARDINALITY_THRESHOLD = 50

# An ID-named column counts as an identifier only if at least 90% of its
# filled values are unique.
ID_UNIQUENESS_THRESHOLD = 0.90


# ── Utility Functions ─────────────────────────────────────────────────────────

def _has_keyword(col_name: str, keyword_list: list) -> bool:
    """Return True if any keyword appears anywhere in the column name."""
    col_lower = col_name.lower()                        # compare in lowercase
    return any(kw in col_lower for kw in keyword_list)  # True on the first match


def _try_convert_datetime(series: pd.Series) -> pd.Series | None:
    """
    Try to turn a text column into real dates.

    Step 1 reads every value as YYYY-MM-DD (fast and unambiguous).
    Step 2 re-reads only the values Step 1 could not, one by one, in
    whatever format each one uses (MM/DD/YYYY, "March 5, 2024", etc.).

    Before Step 2 existed, a column that was mostly YYYY-MM-DD lost every
    value written another way: those valid dates silently became blanks.

    Values that still cannot be read (like "not a date" or "2024-02-30")
    become blank (NaT). app.py counts them and warns the user.

    Returns the converted column if at least 70% of rows are dates,
    otherwise None (the column is left as text).
    """
    try:
        # Step 1: the standard ISO format
        converted = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")

        # Step 2: values that had text but did not match Step 1.
        # .str.strip() removes stray spaces, which real files often have.
        # format="mixed" reads each value on its own; month comes before
        # day (US style), so 01/02/2024 is January 2.
        leftover = series.notna() & converted.isna()
        if leftover.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                retry = pd.to_datetime(
                    series[leftover].astype(str).str.strip(),
                    format="mixed",
                    errors="coerce",
                )
            converted.loc[leftover] = retry

        # Only treat the column as dates if most rows are real dates
        if converted.notna().mean() >= 0.70:
            return converted

    except Exception:
        # Any unexpected problem means "not a date column"; never crash the app
        pass

    return None


# ── Core Detection Engine ─────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Give every column in the DataFrame one of the seven types.

    IMPORTANT: this function also CHANGES df. Text columns that turn out
    to be dates are converted to real dates inside df, so the rest of the
    app can use them in time charts. That is why app.py passes it the
    working copy and never the original upload.

    Parameters
    ----------
    df : pd.DataFrame   The working dataset.

    Returns
    -------
    dict   {column name: type}, for example {"Order Date": "date_column"}
    """
    col_types = {}

    for col in df.columns:

        # Column names that are not text (for example the number 2024) cannot
        # be checked for keywords, so treat them as simple categories.
        if not isinstance(col, str):
            col_types[col] = "categorical"
            continue

        # ── Facts about this column's values ─────────────────────────────────
        n_non_null = df[col].dropna().__len__()                 # filled cells
        n_unique = df[col].nunique()                            # different values
        uniqueness = n_unique / n_non_null if n_non_null > 0 else 0   # 1.0 = all unique

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        # Text can be stored as "object" (pandas 2) or "str" (pandas 3);
        # this check covers both.
        is_string = (
            pd.api.types.is_string_dtype(df[col])
            or df[col].dtype == object
        )
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])

        # ── The checks, in order. The first match wins. ─────────────────────

        # 1. Already stored as dates (for example from an Excel date cell)
        if is_datetime:
            col_types[col] = "date_column"
            continue

        # 2. Text with a date-like name: try to convert it to real dates
        if is_string and _has_keyword(col, DATE_KEYWORDS):
            converted = _try_convert_datetime(df[col])
            if converted is not None:
                df[col] = converted          # keep the converted dates
                col_types[col] = "date_column"
                continue

        # 3. Text with an ID-like name and mostly unique values: identifier
        if is_string and _has_keyword(col, ID_KEYWORDS) and uniqueness >= ID_UNIQUENESS_THRESHOLD:
            col_types[col] = "id_column"
            continue

        # 4. Numbers with an ID-like name and mostly unique values: identifier
        if is_numeric and _has_keyword(col, ID_KEYWORDS) and uniqueness >= ID_UNIQUENESS_THRESHOLD:
            col_types[col] = "id_column"
            continue

        # 5. Numbers with a money-like name: financial
        if is_numeric and _has_keyword(col, FINANCIAL_KEYWORDS):
            col_types[col] = "financial"
            continue

        # 6. Text with a small number of different values: category
        if is_string and n_unique <= HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "categorical"
            continue

        # 7. Text with many values and a notes-like name: free text
        if is_string and _has_keyword(col, TEXT_KEYWORDS) and n_unique > HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "text"
            continue

        # 8. Any other text with many different values
        if is_string and n_unique > HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "high_cardinality"
            continue

        # 9. Any other numbers
        if is_numeric:
            col_types[col] = "metric"
            continue

        # 10. Anything else (for example True/False columns)
        col_types[col] = "categorical"

    return col_types


# ── Helper Functions ──────────────────────────────────────────────────────────

def count_unreadable_dates(raw_df: pd.DataFrame, converted_df: pd.DataFrame,
                           col_types: dict) -> dict:
    """
    Count values that were lost when text columns were turned into dates.

    detect_column_types() turns values that are not real dates (like
    "not a date" or "2024-02-30") into blanks. This compares the file
    before and after so the Understand tab can warn the user.

    Used by app.py when a file loads, and by Reset to Original.

    Parameters
    ----------
    raw_df       : the data before detect_column_types() ran
    converted_df : the same data after it ran
    col_types    : the result of detect_column_types()

    Returns
    -------
    dict   {column: (how many values were lost, up to 3 example values)}
    """
    warnings_found = {}
    for col, col_type in col_types.items():
        if col_type != "date_column" or col not in raw_df.columns:
            continue
        # Had a value in the file, but is blank after converting
        lost = raw_df[col].notna() & converted_df[col].isna()
        if lost.any():
            examples = raw_df.loc[lost, col].astype(str).unique()[:3]
            warnings_found[col] = (int(lost.sum()), list(examples))
    return warnings_found


def get_columns_of_type(col_types: dict, *types: str) -> list:
    """
    List the columns that have any of the given types.

    Example: get_columns_of_type(col_types, "metric", "financial")
             returns every number column.
    """
    return [col for col, t in col_types.items() if t in types]


def summarize_column_types(col_types: dict) -> dict:
    """Count how many columns have each type, e.g. {"categorical": 4, "metric": 2}."""
    from collections import Counter
    return dict(Counter(col_types.values()))


def plain_english_type(col_type: str) -> str:
    """Turn an internal type name into the label shown to users."""
    labels = {
        "id_column":        "Identifier (not used in analysis)",
        "date_column":      "Date / Time",
        "financial":        "Financial Amount",
        "categorical":      "Category",
        "metric":           "Numeric Measurement",
        "high_cardinality": "Category (many values)",
        "text":             "Free Text",
    }
    return labels.get(col_type, "Other")   # "Other" for anything unexpected


# ── Intelligent Type Expectations & Recommendations ───────────────────────────
# detect_column_types() mostly trusts column NAMES. The functions below look
# only at the VALUES and compare the two opinions. When they disagree, the
# Understand tab shows a recommendation (for example "numbers stored as text").

def infer_expected_type(series: pd.Series) -> str:
    """
    Guess a column's type from its values alone (names are ignored).

    Uses the first 200 filled values. Checks, in order:
        85%+ look like plain numbers          -> "metric"
        60%+ contain "$" or a comma like 1,234 -> "financial"
        75%+ can be read as dates             -> "date_column"
                                                 (each value read on its own)
        95%+ of all values are unique         -> "id_column"
        otherwise                             -> "categorical"
    """
    n_non_null = series.dropna().__len__()
    if n_non_null == 0:
        return "categorical"            # an empty column has nothing to judge

    # A sample keeps this fast on large files
    sample = series.dropna().astype(str).head(200)

    # Plain numbers like "12", "-3.5"
    numeric_match_ratio = sample.str.match(r"^-?\d+(\.\d+)?$").mean()
    if numeric_match_ratio >= 0.85:
        return "metric"

    # Currency-looking text like "$1,234.50": a dollar sign, or a comma
    # between digits (1,234). A plain comma is not enough, or dates like
    # "August 14, 2020" would be mistaken for money.
    currency_ratio = sample.str.contains(r"\$|\d,\d{3}").mean()
    if currency_ratio >= 0.60:
        return "financial"

    # Dates in any common format. format="mixed" reads each value on its own;
    # without it, pandas copies the first value's layout to every row, so in
    # a day-first column "16/01/2024" fails after "01/01/2024" is read as
    # month first.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(sample, format="mixed", errors="coerce")

        if converted.notna().mean() >= 0.75:
            return "date_column"
    except Exception:
        pass

    # Almost every value different: probably an identifier
    uniqueness = series.nunique() / n_non_null if n_non_null > 0 else 0
    if uniqueness >= 0.95:
        return "id_column"

    return "categorical"


def generate_column_recommendations(
    df: pd.DataFrame,
    col_types: dict
) -> list:
    """
    Build the "Intelligent Recommendations" list for the Understand tab.

    A column gets a recommendation when:
        - its detected type and its value-based type disagree in a way
          worth fixing (for example numbers stored as text), or
        - it is 30% or more empty, or
        - it has only one value (it adds nothing to an analysis).
    If more than one applies, the later check's message is the one kept.

    Returns
    -------
    list of dicts with keys: column, detected, expected, recommendation, kind
        kind is "type" when the detected and value-based types disagree,
        "missing" for a mostly empty column, and "single" for a column
        with one value. The Understand tab only says "detected as X but
        behaves like Y" for kind "type".
    """
    recommendations = []

    for col in df.columns:

        detected = col_types.get(col, "categorical")   # type from the name + values
        expected = infer_expected_type(df[col])        # type from the values only

        rec_text = None
        kind = "type"

        # ── Type disagreements ───────────────────────────────────────────────
        if detected != expected:

            if detected in ("categorical", "high_cardinality") and expected == "financial":
                rec_text = (
                    "Values look like money amounts (like $1,234.50) but are stored "
                    "as text, so they cannot be added up or charted. Use Convert Text "
                    "or Currency to Numbers in the Clean & Export tab."
                )

            elif detected in ("categorical", "high_cardinality") and expected == "metric":
                rec_text = (
                    "Values appear numeric but are stored as text. Use "
                    "Convert Text or Currency to Numbers in the Clean & Export tab."
                )

            # Any kind of text column whose values look like dates
            elif detected in ("categorical", "high_cardinality", "text") and expected == "date_column":
                rec_text = (
                    "Values appear to be dates stored as text. Use "
                    "Convert Text to Dates in the Clean & Export tab."
                )

            elif detected == "metric" and expected == "id_column":
                rec_text = "High uniqueness suggests this is an identifier. Consider excluding from analysis."

            elif detected == "financial" and expected == "metric":
                # Guard: if the column name contains financial keywords,
                # the detection is correct. Do not flag it.
                col_name_lower = col.lower()
                name_is_financial = any(kw in col_name_lower for kw in FINANCIAL_KEYWORDS)
                if not name_is_financial:
                    rec_text = "Column appears numeric but may not represent monetary values."

        # ── Mostly empty columns ─────────────────────────────────────────────
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct >= 30:
            rec_text = (
                f"Column is {round(missing_pct,1)}% missing. "
                "Consider filling or removing."
            )
            kind = "missing"

        # ── Columns with a single value ──────────────────────────────────────
        if df[col].nunique() <= 1:
            rec_text = (
                "Every row has the same value, so this column cannot show any "
                "differences. You could remove it."
            )
            kind = "single"

        if rec_text:
            recommendations.append({
                "column": col,
                "detected": detected,
                "expected": expected,
                "recommendation": rec_text,
                "kind": kind,
            })

    return recommendations
