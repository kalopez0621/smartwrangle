"""
engine/detector.py
==================
Column type detection engine for SmartWrangle 2.0.

This is the first module that runs after data loads. It classifies
every column in the dataset into a semantic type so every other part
of the app can adapt its behavior accordingly — without ever asking
the user to understand data types.
"""

import pandas as pd
import numpy as np
import warnings


# ── Keyword lists ──────────────────────────────────────────────────────────────

DATE_KEYWORDS = [
    "date", "time", "timestamp", "datetime",
    "received", "incident", "created", "updated",
    "modified", "opened", "closed", "submitted",
    "reported", "recorded", "logged", "posted"
]

ID_KEYWORDS = [
    "id", "number", "num", "no", "code", "key",
    "index", "seq", "sequence", "ref", "reference",
    "identifier", "uuid", "guid", "sku", "hash",
    "claim", "ticket", "record", "row", "case"
]

FINANCIAL_KEYWORDS = [
    "amount", "price", "cost", "revenue", "salary",
    "pay", "wage", "fee", "value", "charge", "balance",
    "total", "sum", "budget", "spend", "expenditure",
    "income", "profit", "loss", "tax", "discount",
    "refund", "payment", "payout", "settlement", "close"
]

TEXT_KEYWORDS = [
    "description", "notes", "note", "comment", "remarks",
    "remark", "detail", "narrative", "summary", "text",
    "message", "feedback", "review", "memo", "body"
]

HIGH_CARDINALITY_THRESHOLD = 50
ID_UNIQUENESS_THRESHOLD = 0.90


# ── Utility Functions ─────────────────────────────────────────────────────────

def _has_keyword(col_name: str, keyword_list: list) -> bool:
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in keyword_list)


def _try_convert_datetime(series: pd.Series) -> pd.Series | None:
    try:
        converted = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")

        if converted.notna().mean() < 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors="coerce")

        if converted.notna().mean() >= 0.70:
            return converted

    except Exception:
        pass

    return None


# ── Core Detection Engine ─────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> dict:

    col_types = {}

    for col in df.columns:

        if not isinstance(col, str):
            col_types[col] = "categorical"
            continue

        n_non_null = df[col].dropna().__len__()
        n_unique = df[col].nunique()
        uniqueness = n_unique / n_non_null if n_non_null > 0 else 0

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_string = (
            pd.api.types.is_string_dtype(df[col])
            or df[col].dtype == object
        )
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])

        if is_datetime:
            col_types[col] = "date_column"
            continue

        if is_string and _has_keyword(col, DATE_KEYWORDS):
            converted = _try_convert_datetime(df[col])
            if converted is not None:
                df[col] = converted
                col_types[col] = "date_column"
                continue

        if is_string and _has_keyword(col, ID_KEYWORDS) and uniqueness >= ID_UNIQUENESS_THRESHOLD:
            col_types[col] = "id_column"
            continue

        if is_numeric and _has_keyword(col, ID_KEYWORDS) and uniqueness >= ID_UNIQUENESS_THRESHOLD:
            col_types[col] = "id_column"
            continue

        if is_numeric and _has_keyword(col, FINANCIAL_KEYWORDS):
            col_types[col] = "financial"
            continue

        if is_string and n_unique <= HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "categorical"
            continue

        if is_string and _has_keyword(col, TEXT_KEYWORDS) and n_unique > HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "text"
            continue

        if is_string and n_unique > HIGH_CARDINALITY_THRESHOLD:
            col_types[col] = "high_cardinality"
            continue

        if is_numeric:
            col_types[col] = "metric"
            continue

        col_types[col] = "categorical"

    return col_types


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_columns_of_type(col_types: dict, *types: str) -> list:
    return [col for col, t in col_types.items() if t in types]


def summarize_column_types(col_types: dict) -> dict:
    from collections import Counter
    return dict(Counter(col_types.values()))


def plain_english_type(col_type: str) -> str:
    labels = {
        "id_column":        "Identifier (not used in analysis)",
        "date_column":      "Date / Time",
        "financial":        "Financial Amount",
        "categorical":      "Category",
        "metric":           "Numeric Measurement",
        "high_cardinality": "Category (many values)",
        "text":             "Free Text",
    }
    return labels.get(col_type, "Other")


# ── Intelligent Type Expectations & Recommendations ───────────────────────────

def infer_expected_type(series: pd.Series) -> str:
    n_non_null = series.dropna().__len__()
    if n_non_null == 0:
        return "categorical"

    sample = series.dropna().astype(str).head(200)

    numeric_match_ratio = sample.str.match(r"^-?\d+(\.\d+)?$").mean()
    if numeric_match_ratio >= 0.85:
        return "metric"

    currency_ratio = sample.str.contains(r"\$|,").mean()
    if currency_ratio >= 0.60:
        return "financial"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(sample, errors="coerce")

        if converted.notna().mean() >= 0.75:
            return "date_column"
    except Exception:
        pass

    uniqueness = series.nunique() / n_non_null if n_non_null > 0 else 0
    if uniqueness >= 0.95:
        return "id_column"

    return "categorical"


def generate_column_recommendations(
    df: pd.DataFrame,
    col_types: dict
) -> list:

    recommendations = []

    for col in df.columns:

        detected = col_types.get(col, "categorical")
        expected = infer_expected_type(df[col])

        rec_text = None

        if detected != expected:

            if detected == "categorical" and expected in ("metric", "financial"):
                rec_text = "Values appear numeric but are stored as text. Consider converting to numeric."

            elif detected == "categorical" and expected == "date_column":
                rec_text = "Values appear to be dates. Consider converting to datetime."

            elif detected == "metric" and expected == "id_column":
                rec_text = "High uniqueness suggests this is an identifier. Consider excluding from analysis."

            elif detected == "financial" and expected == "metric":
                # Guard: if the column name contains financial keywords,
                # the detection is correct. Do not flag it.
                col_name_lower = col.lower()
                name_is_financial = any(kw in col_name_lower for kw in FINANCIAL_KEYWORDS)
                if not name_is_financial:
                    rec_text = "Column appears numeric but may not represent monetary values."

        missing_pct = df[col].isnull().mean() * 100
        if missing_pct >= 30:
            rec_text = (
                f"Column is {round(missing_pct,1)}% missing. "
                "Consider filling or removing."
            )

        if df[col].nunique() <= 1:
            rec_text = "Column has only one unique value. Consider removing."

        if rec_text:
            recommendations.append({
                "column": col,
                "detected": detected,
                "expected": expected,
                "recommendation": rec_text
            })

    return recommendations