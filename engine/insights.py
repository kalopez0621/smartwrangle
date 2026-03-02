"""
engine/insights.py
==================
Automatic business insight generation for SmartWrangle 2.0.

What this file does
-------------------
This module looks at a dataset and writes plain-English findings —
the kind of sentences a data analyst would put in a slide deck or
present to a manager.

Instead of showing a chart and leaving the user to figure out what
it means, SmartWrangle 2.0 writes the headline FOR the user, above
every chart. This is what separates the app from a basic chart tool.

Example output
--------------
Instead of:                         SmartWrangle writes:
  "Bar chart: Claim Type"     →    "Passenger Property Loss makes up 64%
                                    of all claims — the single largest
                                    category by far."

  "Line chart: Date Received" →    "Claims peaked in 2005 with 17,097
                                    cases and declined 25% by 2009."

  "Metric: Claim Amount"      →    "The total amount claimed is $344M.
                                    Most claims are under $200, but a
                                    small number exceed $1 million."

The six insight types this module generates
-------------------------------------------
    1. time_trend       How a financial/metric value changes over time
    2. segment          Which categories dominate and by how much
    3. financial        Total, median, and notable patterns in money columns
    4. outcome          Approval/denial/pass/fail rate detection
    5. comparison       How two segments compare against each other
    6. outlier_flag     Plain-English note about extreme values

How to use this file
--------------------
    from engine.insights import generate_all_insights

    insights = generate_all_insights(df, col_types)
    # Returns a list of insight dicts, each with:
    #   type        str     e.g. 'time_trend', 'segment'
    #   headline    str     The one-sentence finding (shown above the chart)
    #   detail      str     A second sentence with more context (optional)
    #   chart_type  str     What kind of chart to draw: 'line','bar','pie'
    #   x_col       str     Column to use on the X axis
    #   y_col       str     Column to use on the Y axis (or None)
    #   data        dict    Pre-computed aggregated data for the chart
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# pandas  : for grouping, aggregating, and filtering data
# numpy   : for math operations like percentiles and rounding
# warnings: to suppress harmless pandas date-parsing messages
import pandas as pd
import numpy as np
import warnings


# ── Constants ──────────────────────────────────────────────────────────────────
# These keywords help identify "outcome" columns — columns that tell us
# whether something was approved, denied, passed, failed, etc.
# We use this to detect columns like Status, Disposition, Result, Decision.
OUTCOME_KEYWORDS = [
    "status", "disposition", "outcome", "result",
    "decision", "verdict", "resolution", "conclusion"
]

# Keywords that indicate a column measures some kind of score or rating.
# These get different treatment from financial columns.
SCORE_KEYWORDS = [
    "score", "rating", "rank", "grade", "index",
    "rate", "ratio", "pct", "percent", "percentage"
]

# The earliest date we consider plausible.
# Any date before this is likely a data entry error.
PLAUSIBLE_DATE_MIN = pd.Timestamp("2000-01-01")


# ── Helper functions ───────────────────────────────────────────────────────────

def _filter_plausible_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Remove rows with implausible dates before doing time-based analysis.

    Real-world datasets often have data entry errors — dates recorded
    in the wrong century (e.g. 1902 instead of 2002) or accidentally
    set far in the future (e.g. 2055). Including these would stretch
    the chart axis so far that the real data becomes invisible.

    Parameters
    ----------
    df       : pd.DataFrame   The full dataset.
    date_col : str            The column containing dates.

    Returns
    -------
    pd.DataFrame
        A copy of the dataset with implausible-date rows removed.
    """
    # pd.Timestamp.now() gives us today's date and time
    today = pd.Timestamp.now()

    # Keep only rows where the date is between year 2000 and today
    mask = (
        (df[date_col] >= PLAUSIBLE_DATE_MIN) &
        (df[date_col] <= today)
    )

    # Return a filtered copy — never modify the original dataset
    return df[mask].copy()


def _choose_time_granularity(date_series: pd.Series) -> str:
    """
    Decide whether to group dates by year or by month.

    Plotting every single day across many years produces a messy
    "spike" chart that is impossible to read. We choose the right
    level of grouping automatically based on how wide the date range is.

    Parameters
    ----------
    date_series : pd.Series   A Series of datetime values (already filtered).

    Returns
    -------
    str
        'yearly'  if the date range spans more than 2 years
        'monthly' if the date range spans between 2 months and 2 years
        'daily'   if the date range is 2 months or less
    """
    # Calculate how many days the date range spans
    date_range_days = (date_series.max() - date_series.min()).days

    if date_range_days > 730:       # more than 2 years → yearly
        return "yearly"
    elif date_range_days > 60:      # more than 60 days → monthly
        return "monthly"
    else:
        return "daily"              # short range → daily is fine


def _aggregate_by_time(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    granularity: str
) -> pd.DataFrame:
    """
    Group the data by a time period and calculate the mean value per period.

    This is what creates the data points for a time-trend line chart.
    Instead of plotting 94,000 individual points (one per claim), we
    group them by year or month and show the average.

    Parameters
    ----------
    df          : pd.DataFrame   The filtered dataset.
    date_col    : str            The date column name.
    value_col   : str            The numeric column to aggregate.
    granularity : str            'yearly', 'monthly', or 'daily'.

    Returns
    -------
    pd.DataFrame
        Two columns: the time period and the mean value per period.
        Also includes 'count' — how many records are in each period.
    """
    # Create a copy so we don't accidentally modify the original
    df_agg = df[[date_col, value_col]].copy().dropna()

    if granularity == "yearly":
        # dt.year extracts just the year number from a datetime
        # e.g. 2024-07-15 → 2024
        df_agg["__period__"] = df_agg[date_col].dt.year

    elif granularity == "monthly":
        # dt.to_period("M") converts to Month period (e.g. "2024-07")
        # .dt.to_timestamp() converts back to a date (first day of month)
        # This keeps it as a proper datetime so Plotly can format the axis
        df_agg["__period__"] = (
            df_agg[date_col].dt.to_period("M").dt.to_timestamp()
        )

    else:
        # Daily — just use the date as-is
        df_agg["__period__"] = df_agg[date_col]

    # Group by the time period and calculate:
    #   mean  → average value (used for the Y axis of the chart)
    #   count → number of records (shown in hover tooltip)
    result = (
        df_agg
        .groupby("__period__")[value_col]
        .agg(mean="mean", count="count")
        .reset_index()
        .rename(columns={"__period__": date_col})
    )

    return result


def _write_trend_headline(
    agg_df: pd.DataFrame,
    date_col: str,
    value_col: str,
    granularity: str
) -> tuple:
    """
    Write a plain-English headline and detail sentence for a time trend.

    This function looks at how the values changed from the beginning
    to the end of the time period and chooses appropriate language.

    Parameters
    ----------
    agg_df      : pd.DataFrame   Aggregated time series (from _aggregate_by_time).
    date_col    : str            The date column name.
    value_col   : str            The value column name.
    granularity : str            'yearly', 'monthly', or 'daily'.

    Returns
    -------
    tuple of (headline: str, detail: str)
    """
    # We need at least 2 data points to describe a trend
    if len(agg_df) < 2:
        return (
            f"{value_col} over time",
            "Not enough data points to describe a trend."
        )

    # Find the period with the highest count (most records)
    peak_idx    = agg_df["count"].idxmax()
    peak_period = agg_df.loc[peak_idx, date_col]
    peak_count  = int(agg_df.loc[peak_idx, "count"])

    # Compare first vs last period for overall direction
    first_count = int(agg_df.iloc[0]["count"])
    last_count  = int(agg_df.iloc[-1]["count"])

    # Percentage change from first to last
    # We protect against division by zero
    if first_count > 0:
        pct_change = (last_count - first_count) / first_count * 100
    else:
        pct_change = 0

    # Format the peak period for display
    # If yearly, just show the year number
    # If monthly, show "Month Year" (e.g. "July 2005")
    if granularity == "yearly":
        peak_label = str(int(peak_period))
    else:
        peak_label = pd.Timestamp(peak_period).strftime("%b %Y")

    # Choose language based on direction and magnitude of change
    if pct_change < -25:
        direction_phrase = f"declined significantly ({abs(pct_change):.0f}%) since then"
    elif pct_change < -10:
        direction_phrase = f"trended downward ({abs(pct_change):.0f}%) since then"
    elif pct_change > 25:
        direction_phrase = f"grown significantly ({pct_change:.0f}%) since then"
    elif pct_change > 10:
        direction_phrase = f"trended upward ({pct_change:.0f}%) since then"
    else:
        direction_phrase = "remained relatively stable"

    headline = (
        f"Activity peaked in {peak_label} "
        f"with {peak_count:,} records and has {direction_phrase}."
    )

    # Second sentence: average value context
    overall_avg = agg_df["mean"].mean()
    detail = (
        f"The average {value_col} across the full period "
        f"was {_format_value(overall_avg, value_col)}."
    )

    return headline, detail


def _format_value(value: float, col_name: str) -> str:
    """
    Format a number for display based on what kind of column it is.

    A financial column gets a dollar sign and comma separators.
    A percentage column gets a % sign.
    Everything else gets comma formatting.

    Parameters
    ----------
    value    : float   The number to format.
    col_name : str     The column name — used to guess the right format.

    Returns
    -------
    str
        A nicely formatted string, e.g. '\\$1,234.56' or '42.3%' or '1,234'

    IMPORTANT — Why \\$ instead of $
    ----------------------------------
    Streamlit's st.markdown() supports LaTeX math notation.
    LaTeX uses the dollar sign as a math delimiter: $x^2$ renders as italic x².
    If a headline contains TWO dollar signs (e.g. "typical is $98.40, average is
    $10,779.00"), Streamlit treats everything between them as a math expression.
    The dollar signs disappear and the numbers render in italic math font.

    The fix is to escape the dollar sign as \\$, which tells the Markdown parser
    to render it as a literal $ character instead of opening a math block.
    In the browser, \\$ displays as $ — the user never sees the backslash.
    """
    col_lower = col_name.lower()

    # Check if this looks like a financial column
    financial_hints = [
        "amount", "price", "cost", "revenue", "salary",
        "pay", "fee", "value", "charge", "total", "close"
    ]
    is_financial = any(h in col_lower for h in financial_hints)

    # Check if this looks like a percentage column
    pct_hints = ["pct", "percent", "percentage", "rate", "ratio"]
    is_pct = any(h in col_lower for h in pct_hints)

    if is_financial:
        # \\$ escapes the dollar sign so Streamlit markdown doesn't treat it
        # as a LaTeX math delimiter. Renders as a literal $ in the browser.
        return f"\\${value:,.2f}"
    elif is_pct:
        # Percentage format
        return f"{value:.1f}%"
    else:
        # Plain number with commas
        return f"{value:,.1f}"


# ── Insight generators ─────────────────────────────────────────────────────────
# Each function below generates one type of insight.
# They all follow the same pattern:
#   - Take the dataframe and column types as input
#   - Do the analysis
#   - Return a dict describing the insight, or None if not applicable


def _insight_time_trend(
    df: pd.DataFrame,
    col_types: dict,
    date_cols: list,
    fin_cols: list,
    metric_cols: list
) -> dict | None:
    """
    Generate a time trend insight.

    Looks at how a financial or metric value changes over time.
    Only runs if the dataset has at least one date column AND
    at least one financial or metric column.

    Returns None if the dataset doesn't have what's needed.
    """
    # We need a date column and something to measure over time
    if not date_cols:
        return None

    # Prefer financial columns; fall back to metric columns
    value_cols = fin_cols if fin_cols else metric_cols
    if not value_cols:
        return None

    # Use the first date column and first value column
    date_col  = date_cols[0]
    value_col = value_cols[0]

    # Remove rows with implausible dates (data entry errors)
    df_filtered = _filter_plausible_dates(df, date_col)

    # Need the date and value column; drop rows where either is missing
    df_filtered = df_filtered[[date_col, value_col]].dropna()

    # If we don't have enough data after filtering, skip this insight
    if len(df_filtered) < 10:
        return None

    # Choose whether to group by year, month, or day
    granularity = _choose_time_granularity(df_filtered[date_col])

    # Aggregate the data into time periods
    agg_df = _aggregate_by_time(df_filtered, date_col, value_col, granularity)

    if len(agg_df) < 2:
        return None

    # Write the plain-English headline and detail sentence
    headline, detail = _write_trend_headline(agg_df, date_col, value_col, granularity)

    # Count how many bad dates we filtered out — shown as a footnote
    n_filtered = len(df) - len(_filter_plausible_dates(df, date_col))

    return {
        "type":        "time_trend",
        "headline":    headline,
        "detail":      detail,
        "chart_type":  "line",
        "x_col":       date_col,
        "y_col":       value_col,
        "granularity": granularity,
        "data":        agg_df,
        "n_filtered":  n_filtered,
        "footnote": (
            f"Note: {n_filtered:,} rows with dates outside "
            f"{PLAUSIBLE_DATE_MIN.year}–present were excluded."
            if n_filtered > 0 else None
        )
    }


def _insight_segment(
    df: pd.DataFrame,
    col_types: dict,
    cat_cols: list,
    fin_cols: list
) -> list:
    """
    Generate segment breakdown insights for categorical columns.

    For each categorical column, we calculate:
    - Which category appears most often (dominant category)
    - What percentage of records it accounts for
    - A plain-English sentence describing the imbalance (if any)

    We also compare financial values across categories when possible
    (e.g. 'Checkpoint claims average 5x more than Checked Baggage').

    Returns a list of insight dicts (one per useful categorical column).
    Returns an empty list if no categorical columns exist.
    """
    insights = []

    # Limit to the 4 most informative categorical columns
    # to avoid overwhelming the Discover tab with too many charts
    for cat_col in cat_cols[:4]:

        # Count how many records are in each category
        value_counts = df[cat_col].value_counts()

        # Skip columns with only one category (no comparison possible)
        if len(value_counts) < 2:
            continue

        # The most common category
        top_category = value_counts.index[0]
        top_count    = int(value_counts.iloc[0])
        top_pct      = round(top_count / len(df) * 100, 1)

        # The least common category
        bottom_category = value_counts.index[-1]
        bottom_count    = int(value_counts.iloc[-1])
        bottom_pct      = round(bottom_count / len(df) * 100, 1)

        # Write a headline based on how dominant the top category is
        if top_pct >= 75:
            headline = (
                f"'{top_category}' dominates — it accounts for "
                f"{top_pct:.0f}% of all {cat_col} records."
            )
        elif top_pct >= 50:
            headline = (
                f"'{top_category}' is the most common {cat_col}, "
                f"making up {top_pct:.0f}% of all records."
            )
        else:
            headline = (
                f"'{top_category}' is the leading {cat_col} "
                f"({top_pct:.0f}%), but no single category dominates."
            )

        # Add a comparison detail if there's a big gap between top and bottom
        ratio = top_count / bottom_count if bottom_count > 0 else 0
        if ratio >= 5:
            detail = (
                f"'{top_category}' has {ratio:.0f}x more records than "
                f"the least common category '{bottom_category}' ({bottom_pct:.1f}%)."
            )
        else:
            detail = (
                f"The least common category is '{bottom_category}' "
                f"with {bottom_count:,} records ({bottom_pct:.1f}%)."
            )

        # ── Financial comparison across categories (bonus insight) ─────────
        fin_comparison = None
        if fin_cols:
            fin_col = fin_cols[0]
            try:
                # Calculate the average financial value per category
                grp = (
                    df.groupby(cat_col)[fin_col]
                    .mean()
                    .sort_values(ascending=False)
                )

                if len(grp) >= 2:
                    top_fin_cat    = grp.index[0]
                    top_fin_avg    = grp.iloc[0]
                    bottom_fin_cat = grp.index[-1]
                    bottom_fin_avg = grp.iloc[-1]

                    # Only report if the difference is meaningful (> 2x)
                    if bottom_fin_avg > 0 and (top_fin_avg / bottom_fin_avg) >= 2:
                        fin_ratio = top_fin_avg / bottom_fin_avg
                        fin_comparison = (
                            f"'{top_fin_cat}' records have "
                            f"{fin_ratio:.1f}x higher average {fin_col} "
                            f"({_format_value(top_fin_avg, fin_col)}) than "
                            f"'{bottom_fin_cat}' "
                            f"({_format_value(bottom_fin_avg, fin_col)})."
                        )
            except Exception:
                # If the comparison fails for any reason, skip it gracefully
                pass

        insights.append({
            "type":           "segment",
            "headline":       headline,
            "detail":         detail,
            "fin_comparison": fin_comparison,
            "chart_type":     "bar",
            "x_col":          cat_col,
            "y_col":          None,
            "data":           value_counts.reset_index().rename(
                                  columns={cat_col: "Category", "count": "Count"}
                              ),
            "col_name":       cat_col,
        })

    return insights


def _insight_financial(
    df: pd.DataFrame,
    col_types: dict,
    fin_cols: list
) -> list:
    """
    Generate financial summary insights.

    For each financial column, calculates total, median, and
    notable patterns (like a high proportion of zero values).
    Writes plain-English sentences describing what the numbers mean.

    Returns a list of insight dicts (one per financial column).
    """
    insights = []

    for fin_col in fin_cols:

        col_data = df[fin_col].dropna()

        if len(col_data) == 0:
            continue

        # Core statistics
        total   = col_data.sum()
        median  = col_data.median()
        mean    = col_data.mean()
        p99     = col_data.quantile(0.99)
        max_val = col_data.max()

        # How many values are zero?
        n_zero   = int((col_data == 0).sum())
        zero_pct = round(n_zero / len(col_data) * 100, 1)

        # Is the column dominated by zeros?
        # (e.g. Close Amount — most denied claims paid $0)
        mostly_zero = zero_pct > 40

        # Is there an extreme outlier?
        is_extreme = p99 > 0 and max_val > p99 * 10

        # Write a headline based on the most notable pattern
        if mostly_zero:
            headline = (
                f"{zero_pct:.0f}% of {fin_col} values are $0 — "
                f"meaning most records resulted in no payout."
            )
            detail = (
                f"Among records with a non-zero amount, "
                f"the median is {_format_value(col_data[col_data > 0].median(), fin_col)}. "
                f"The total across all records is {_format_value(total, fin_col)}."
            )
        elif is_extreme:
            headline = (
                f"The total {fin_col} across all records is "
                f"{_format_value(total, fin_col)}. "
                f"Most values are modest — the typical record is "
                f"{_format_value(median, fin_col)}."
            )
            detail = (
                f"However, a small number of extreme values "
                f"(above {_format_value(p99, fin_col)} at the 99th percentile) "
                f"are pulling the average up to {_format_value(mean, fin_col)}. "
                f"The single largest value is {_format_value(max_val, fin_col)}."
            )
        else:
            headline = (
                f"The total {fin_col} is {_format_value(total, fin_col)}, "
                f"with a typical record value of {_format_value(median, fin_col)}."
            )
            detail = (
                f"The average is {_format_value(mean, fin_col)}, "
                f"and 99% of values fall below {_format_value(p99, fin_col)}."
            )

        insights.append({
            "type":       "financial",
            "headline":   headline,
            "detail":     detail,
            "chart_type": "histogram",
            "x_col":      fin_col,
            "y_col":      None,
            "data": {
                "total":    total,
                "median":   median,
                "mean":     mean,
                "p99":      p99,
                "max_val":  max_val,
                "n_zero":   n_zero,
                "zero_pct": zero_pct,
            },
            "col_name": fin_col,
        })

    return insights


def _insight_outcome(
    df: pd.DataFrame,
    col_types: dict,
    cat_cols: list
) -> dict | None:
    """
    Detect and summarize outcome columns.

    An outcome column is one that records the result of some process —
    like whether a claim was Approved, Denied, or Settled. We detect
    these by looking for columns whose name contains words like
    'status', 'disposition', 'outcome', 'result'.

    This generates the most impactful insight for business datasets
    because decision-makers always want to know: what happened?

    Returns one insight dict for the best outcome column found,
    or None if no outcome column is detected.
    """
    # Search through categorical columns for outcome-related names
    outcome_col = None
    for col in cat_cols:
        if any(kw in col.lower() for kw in OUTCOME_KEYWORDS):
            outcome_col = col
            break   # Use the first one found

    if outcome_col is None:
        return None

    # Count records per outcome
    value_counts = df[outcome_col].value_counts()
    total        = len(df)

    # Build a percentage breakdown
    pct_breakdown = {
        cat: round(cnt / total * 100, 1)
        for cat, cnt in value_counts.items()
    }

    # The most common outcome
    top_outcome     = value_counts.index[0]
    top_pct         = pct_breakdown[top_outcome]

    # Detect whether the dominant outcome is negative
    # (denied, rejected, failed, closed, cancelled)
    negative_words = [
        "deny", "denied", "reject", "rejected", "fail",
        "failed", "close", "closed", "cancel", "cancelled", "lost"
    ]
    is_negative = any(w in top_outcome.lower() for w in negative_words)

    # Write the headline with appropriate framing
    if is_negative and top_pct > 50:
        headline = (
            f"{top_pct:.0f}% of records result in '{top_outcome}' — "
            f"the most common outcome by far."
        )
        # Check if there's a positive counterpart (approval, settle)
        positive_words = ["approve", "approved", "accept", "pass", "settle", "settled", "win"]
        positive_outcomes = [
            (cat, pct)
            for cat, pct in pct_breakdown.items()
            if any(w in cat.lower() for w in positive_words)
        ]
        if positive_outcomes:
            pos_cat, pos_pct = positive_outcomes[0]
            detail = (
                f"Only {pos_pct:.0f}% of records result in "
                f"'{pos_cat}'. "
                f"The remaining {100 - top_pct - pos_pct:.0f}% "
                f"have other outcomes."
            )
        else:
            detail = (
                f"The full breakdown: "
                + ", ".join(
                    f"'{cat}' {pct}%"
                    for cat, pct in list(pct_breakdown.items())[:4]
                )
            )
    else:
        headline = (
            f"The most common {outcome_col} is '{top_outcome}' "
            f"({top_pct:.0f}% of records)."
        )
        detail = (
            "Full breakdown: "
            + ", ".join(
                f"'{cat}' {pct}%"
                for cat, pct in list(pct_breakdown.items())[:5]
            )
        )

    return {
        "type":          "outcome",
        "headline":      headline,
        "detail":        detail,
        "chart_type":    "bar",
        "x_col":         outcome_col,
        "y_col":         None,
        "data":          value_counts.reset_index().rename(
                             columns={outcome_col: "Outcome", "count": "Count"}
                         ),
        "col_name":      outcome_col,
        "pct_breakdown": pct_breakdown,
    }


def _insight_high_cardinality_segment(
    df: pd.DataFrame,
    col_types: dict,
    hc_cols: list,
    fin_cols: list
) -> list:
    """
    Generate top-N breakdown insights for high-cardinality columns.

    High-cardinality columns have too many categories to show all at once
    (e.g. Airline Name has 186 unique values). We show the top 10 instead.

    This is particularly useful for columns like:
    - Airline Name → which airlines have the most claims?
    - Airport Name → which airports are highest risk?
    - Product Name → which products generate the most revenue?

    Returns a list of insight dicts (one per high-cardinality column, max 2).
    """
    insights = []

    # Limit to 2 high-cardinality columns to avoid overloading the tab
    for hc_col in hc_cols[:2]:

        # Get the top 10 categories by record count
        top10 = df[hc_col].value_counts().head(10)

        if len(top10) < 2:
            continue

        top_name  = top10.index[0]
        top_count = int(top10.iloc[0])
        top_pct   = round(top_count / len(df) * 100, 1)

        headline = (
            f"'{top_name}' leads in {hc_col} "
            f"with {top_count:,} records ({top_pct:.0f}% of the total)."
        )

        # Add financial context if available
        detail = f"Showing the top 10 {hc_col} values by record count."

        if fin_cols:
            fin_col = fin_cols[0]
            try:
                # Average financial value per top-10 category
                top10_fin = (
                    df[df[hc_col].isin(top10.index)]
                    .groupby(hc_col)[fin_col]
                    .mean()
                    .sort_values(ascending=False)
                )
                top_fin_cat = top10_fin.index[0]
                top_fin_avg = top10_fin.iloc[0]

                detail = (
                    f"Among the top 10, '{top_fin_cat}' has the highest "
                    f"average {fin_col} at {_format_value(top_fin_avg, fin_col)}."
                )
            except Exception:
                pass

        insights.append({
            "type":       "high_cardinality_segment",
            "headline":   headline,
            "detail":     detail,
            "chart_type": "bar",
            "x_col":      hc_col,
            "y_col":      None,
            "data":       top10.reset_index().rename(
                              columns={hc_col: "Category", "count": "Count"}
                          ),
            "col_name":   hc_col,
            "top_n":      10,
        })

    return insights


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_all_insights(df: pd.DataFrame, col_types: dict) -> list:
    """
    Run all insight generators and return a list of findings.

    This is the only function you need to call from the Discover tab.
    It automatically figures out what kind of data is in the dataset
    and generates the most relevant insights.

    The order of insights in the returned list is the order they will
    appear on the Discover tab:
        1. Outcome   (most impactful for business datasets)
        2. Time trend (shows how things change over time)
        3. Financial  (total amounts and patterns)
        4. Segments   (category breakdowns)
        5. Top-N      (high cardinality breakdowns like airlines)

    Parameters
    ----------
    df        : pd.DataFrame   The dataset to analyze.
                               Date columns must already be converted to
                               datetime — run detector.detect_column_types()
                               before calling this function.
    col_types : dict           Output from detector.detect_column_types().

    Returns
    -------
    list of dicts
        Each dict has at minimum:
            type        str   What kind of insight this is
            headline    str   The one-sentence finding (shown above chart)
            detail      str   A follow-up sentence with more context
            chart_type  str   'line', 'bar', or 'histogram'
            x_col       str   Column for the X axis
            y_col       str   Column for the Y axis (or None)
            data        any   Pre-computed data for the chart
    """
    # ── Import the column-type helper ─────────────────────────────────────────
    # We import here to keep the import at the point of use,
    # making it clear where this dependency comes from.
    from engine.detector import get_columns_of_type

    # ── Pull column lists by type ──────────────────────────────────────────────
    # These lists drive which insights get generated.
    # If a list is empty, the corresponding insight is skipped.
    date_cols   = get_columns_of_type(col_types, "date_column")
    fin_cols    = get_columns_of_type(col_types, "financial")
    cat_cols    = get_columns_of_type(col_types, "categorical")
    metric_cols = get_columns_of_type(col_types, "metric")
    hc_cols     = get_columns_of_type(col_types, "high_cardinality")

    # ── Collect all insights ───────────────────────────────────────────────────
    # We use a list and append to it.
    # None values (when an insight type is not applicable) are filtered out.
    all_insights = []

    # 1. Outcome insight (e.g. denial rate, approval rate)
    #    Goes first because it's typically the most important business question
    outcome = _insight_outcome(df, col_types, cat_cols)
    if outcome:
        all_insights.append(outcome)

    # 2. Time trend (e.g. claims over time)
    trend = _insight_time_trend(df, col_types, date_cols, fin_cols, metric_cols)
    if trend:
        all_insights.append(trend)

    # 3. Financial summaries (one per financial column)
    financial_insights = _insight_financial(df, col_types, fin_cols)
    all_insights.extend(financial_insights)

    # 4. Segment breakdowns (one per categorical column, max 4)
    segment_insights = _insight_segment(df, col_types, cat_cols, fin_cols)
    all_insights.extend(segment_insights)

    # 5. Top-N breakdowns for high-cardinality columns (max 2)
    hc_insights = _insight_high_cardinality_segment(df, col_types, hc_cols, fin_cols)
    all_insights.extend(hc_insights)

    return all_insights


# ── END OF FILE ────────────────────────────────────────────────────────────────
# Nothing should be placed below this line.
# All functions in this file are imported and called by tabs/discover.py.