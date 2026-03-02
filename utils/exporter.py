"""
utils/exporter.py
=================
CSV and plain-text summary export for SmartWrangle 2.0.

What this file does
-------------------
This module handles everything related to getting data OUT of the app.
It provides two export options for the Clean & Export tab:

    1. CSV Download
       The cleaned dataset as a .csv file the user can open in Excel,
       Google Sheets, or any other tool.

    2. Text Summary Report
       A plain-English .txt summary of what was found and what was done —
       suitable for sharing with a manager or including in a presentation.

Why not PDF?
------------
PDF generation in Python requires either a headless browser (heavy,
complex to deploy) or a library like ReportLab (requires paid license
for commercial use) or WeasyPrint (has system-level dependencies that
often break on Streamlit Community Cloud's free tier).

A well-formatted plain-text report opens in any text editor, can be
copied into a Word document or email, and deploys with zero additional
dependencies. It is the most reliable choice for a free-tier deployment.

How to use this file
--------------------
    from utils.exporter import (
        df_to_csv_bytes,
        build_summary_report,
        report_to_bytes,
    )

    # In the Clean & Export tab:
    csv_data = df_to_csv_bytes(working_df)
    st.download_button("Download CSV", csv_data, "cleaned_data.csv")

    report_text = build_summary_report(working_df, col_types, quality_result, cleaning_log)
    report_data  = report_to_bytes(report_text)
    st.download_button("Download Report", report_data, "smartwrangle_report.txt")
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# io       : provides in-memory file-like objects (we build files without
#            writing to disk — important for Streamlit Cloud deployment)
# datetime : to add a timestamp to the report so users know when it was made
# pandas   : for DataFrame operations
# numpy    : for statistics in the report
import io
import datetime
import pandas as pd
import numpy as np


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV-formatted bytes ready for download.

    Streamlit's st.download_button() needs the file data as bytes
    (raw binary data), not as a string or a file on disk.
    This function builds the CSV entirely in memory and returns
    the bytes that Streamlit needs.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned working dataset to export.

    Returns
    -------
    bytes
        UTF-8 encoded CSV data. Pass directly to st.download_button().

    Example
    -------
    >>> csv_bytes = df_to_csv_bytes(st.session_state.working_df)
    >>> st.download_button(
    ...     label="Download Cleaned Data",
    ...     data=csv_bytes,
    ...     file_name="cleaned_data.csv",
    ...     mime="text/csv"
    ... )
    """
    # io.StringIO() creates an in-memory text buffer — like an open file,
    # but stored in RAM instead of on disk. This is important for Streamlit
    # Cloud, which has a read-only filesystem.
    buffer = io.StringIO()

    # Write the DataFrame to CSV format into the buffer
    # index=False : don't include the row numbers (0, 1, 2...) as a column
    #               Without this, re-importing the CSV adds an "Unnamed: 0" column
    df.to_csv(buffer, index=False)

    # .getvalue() retrieves the full CSV text from the buffer
    # .encode("utf-8") converts the text string to bytes
    # UTF-8 is the universal encoding that handles all characters correctly,
    # including accented letters, symbols, and international characters
    return buffer.getvalue().encode("utf-8")


def build_summary_report(
    df: pd.DataFrame,
    col_types: dict,
    quality_result: dict,
    cleaning_log: list,
    dataset_name: str = "Dataset"
) -> str:
    """
    Build a plain-English summary report of the dataset and cleaning actions.

    This report is designed to be readable by anyone — not just data analysts.
    A manager, executive, or client should be able to read it and understand
    what the data contains, what was found, and what was done to prepare it.

    The report is structured in five sections:
        1. Header        : App name, date, dataset name
        2. Dataset Overview  : Rows, columns, date range, column inventory
        3. Data Quality      : Score, grade, and plain-English findings
        4. Key Findings      : Most important patterns from the data
        5. Cleaning Actions  : What was changed (from the cleaning log)

    Parameters
    ----------
    df            : pd.DataFrame   The current working dataset.
    col_types     : dict           Output from detector.detect_column_types().
    quality_result: dict           Output from quality.score_dataset().
    cleaning_log  : list of str    Each string is one completed cleaning action.
    dataset_name  : str            A friendly name for the dataset (shown in header).

    Returns
    -------
    str
        The complete report as a single multi-line string.
        Pass to report_to_bytes() to convert for download.
    """
    # Import engine modules here (not at the top) to keep this utility file
    # independent — it doesn't depend on the engine layer being in a specific place
    from engine.detector import get_columns_of_type, plain_english_type
    from engine.quality  import get_quality_report

    # ── Collect column lists by type ───────────────────────────────────────────
    date_cols    = get_columns_of_type(col_types, "date_column")
    fin_cols     = get_columns_of_type(col_types, "financial")
    cat_cols     = get_columns_of_type(col_types, "categorical")
    id_cols      = get_columns_of_type(col_types, "id_column")
    metric_cols  = get_columns_of_type(col_types, "metric")
    hc_cols      = get_columns_of_type(col_types, "high_cardinality")

    # ── Build the report line by line ──────────────────────────────────────────
    # We use a list of strings and join them at the end with newlines.
    # This is more efficient than concatenating strings one by one.
    lines = []

    # ── SECTION 1: Header ─────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")

    lines += [
        "=" * 65,
        "  SMARTWRANGLE 2.0 — DATA ANALYSIS REPORT",
        "=" * 65,
        f"  Dataset   : {dataset_name}",
        f"  Generated : {now}",
        f"  Records   : {len(df):,}",
        f"  Columns   : {len(df.columns)}",
        "=" * 65,
        "",
    ]

    # ── SECTION 2: Dataset Overview ───────────────────────────────────────────
    lines += [
        "SECTION 1 — DATASET OVERVIEW",
        "-" * 40,
        "",
    ]

    # Date range (if date columns exist)
    if date_cols:
        try:
            date_col    = date_cols[0]
            date_series = df[date_col].dropna()
            # Filter to plausible dates
            plausible   = date_series[
                (date_series >= pd.Timestamp("2000-01-01")) &
                (date_series <= pd.Timestamp.now())
            ]
            if len(plausible) > 0:
                date_min = plausible.min().strftime("%B %d, %Y")
                date_max = plausible.max().strftime("%B %d, %Y")
                lines.append(f"  Date Range     : {date_min} to {date_max}")
        except Exception:
            pass

    lines += [
        f"  Total Records  : {len(df):,}",
        f"  Total Columns  : {len(df.columns)}",
        "",
        "  Column Inventory:",
    ]

    # List every column with its plain-English type
    for col in df.columns:
        col_type    = col_types.get(col, "categorical")
        type_label  = plain_english_type(col_type)
        lines.append(f"    - {col} ({type_label})")

    lines.append("")

    # ── SECTION 3: Data Quality ────────────────────────────────────────────────
    lines += [
        "SECTION 2 — DATA QUALITY",
        "-" * 40,
        "",
        f"  Quality Score  : {quality_result['score']} / 100",
        f"  Grade          : {quality_result['grade']}",
        "",
        "  Findings:",
    ]

    # Get the plain-English quality findings
    quality_findings = get_quality_report(quality_result)
    for finding in quality_findings:
        # Add a simple prefix based on severity level
        prefix = "  [OK]     " if finding["level"] == "good" else \
                 "  [NOTE]   " if finding["level"] == "warning" else \
                 "  [ISSUE]  "
        lines.append(f"{prefix}{finding['message']}")

    lines.append("")

    # ── SECTION 4: Key Findings ────────────────────────────────────────────────
    lines += [
        "SECTION 3 — KEY FINDINGS",
        "-" * 40,
        "",
    ]

    # Financial summaries
    if fin_cols:
        lines.append("  Financial Columns:")
        for col in fin_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            total    = col_data.sum()
            median   = col_data.median()
            mean     = col_data.mean()
            n_zero   = int((col_data == 0).sum())
            zero_pct = round(n_zero / len(col_data) * 100, 1)
            lines += [
                f"    {col}:",
                f"      Total   : ${total:>15,.2f}",
                f"      Median  : ${median:>15,.2f}",
                f"      Average : ${mean:>15,.2f}",
                f"      Zero $  : {n_zero:,} records ({zero_pct}%)",
                "",
            ]

    # Categorical summaries (top 3 values per column)
    if cat_cols:
        lines.append("  Category Breakdowns:")
        for col in cat_cols[:4]:     # limit to 4 columns
            vc      = df[col].value_counts()
            total   = len(df)
            lines.append(f"    {col} ({vc.nunique()} categories):")
            for cat, count in vc.head(3).items():
                pct = round(count / total * 100, 1)
                lines.append(f"      {str(cat):35} {count:>7,}  ({pct}%)")
            if len(vc) > 3:
                lines.append(f"      ... and {len(vc)-3} more categories")
            lines.append("")

    # High-cardinality top-5 (airlines, airports, etc.)
    if hc_cols:
        lines.append("  High-Volume Segments (Top 5 each):")
        for col in hc_cols[:2]:
            vc    = df[col].value_counts().head(5)
            total = len(df)
            lines.append(f"    {col}:")
            for cat, count in vc.items():
                pct = round(count / total * 100, 1)
                lines.append(f"      {str(cat):35} {count:>7,}  ({pct}%)")
            lines.append("")

    # ── SECTION 5: Cleaning Actions ───────────────────────────────────────────
    lines += [
        "SECTION 4 — CLEANING ACTIONS PERFORMED",
        "-" * 40,
        "",
    ]

    if cleaning_log:
        for i, action in enumerate(cleaning_log, 1):
            lines.append(f"  {i:2}. {action}")
    else:
        lines.append("  No cleaning actions were performed on this dataset.")

    lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "=" * 65,
        "  Report generated by SmartWrangle 2.0",
        "  For questions about this data, consult your data analyst.",
        "=" * 65,
    ]

    # Join all lines with newline characters to form the complete report string
    return "\n".join(lines)


def report_to_bytes(report_text: str) -> bytes:
    """
    Convert the summary report string to bytes for download.

    Streamlit's st.download_button() needs bytes, not a string.
    This function does the simple conversion.

    Parameters
    ----------
    report_text : str
        The complete report string from build_summary_report().

    Returns
    -------
    bytes
        UTF-8 encoded bytes ready for st.download_button().

    Example
    -------
    >>> report = build_summary_report(df, col_types, quality_result, log)
    >>> st.download_button(
    ...     label="Download Report",
    ...     data=report_to_bytes(report),
    ...     file_name="smartwrangle_report.txt",
    ...     mime="text/plain"
    ... )
    """
    # .encode("utf-8") converts the string to bytes
    # UTF-8 handles all characters, including special symbols and accents
    return report_text.encode("utf-8")


def get_export_filename(prefix: str = "smartwrangle", extension: str = "csv") -> str:
    """
    Generate a timestamped filename for the export.

    Instead of always saving as "data.csv" (which users would overwrite
    every time), we add the date so each export gets a unique name.

    Parameters
    ----------
    prefix    : str   The base filename (default: 'smartwrangle').
    extension : str   The file extension without the dot (default: 'csv').

    Returns
    -------
    str
        A filename like 'smartwrangle_2024-07-15.csv'

    Example
    -------
    >>> get_export_filename("cleaned_data", "csv")
    'cleaned_data_2024-07-15.csv'
    >>> get_export_filename("report", "txt")
    'report_2024-07-15.txt'
    """
    # Get today's date formatted as YYYY-MM-DD
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"{prefix}_{today}.{extension}"


# ── END OF FILE ────────────────────────────────────────────────────────────────
# Nothing should be placed below this line.
# All functions here are imported and called by tabs/clean_export.py.