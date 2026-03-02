"""
tabs/discover.py
================
The "Discover" tab — Tab 2 of SmartWrangle 2.0.

What this tab does
------------------
This tab answers the second question: "What does my data tell me?"

It is the business intelligence layer of SmartWrangle. Every chart
has a plain-English headline above it — the key finding written out
as a sentence, not just a chart title.

A user from any background — HR manager, operations director, student,
small business owner — should be able to look at this tab and
immediately understand what their data is showing them.

Structure of this tab
----------------------
    1. Auto Insights         Charts generated automatically by insights.py,
                             each with a plain-English headline and detail
    2. Explore Your Data     A custom chart builder for users who want to
                             dig into specific relationships themselves

Design principles
-----------------
    - Headline first, chart second.  The sentence is bigger than the chart title.
    - No chart jargon.  "Distribution" is not a word users see here.
    - Deduplication.    If two columns measure the same thing
                        (e.g. Status and Disposition both show outcomes),
                        only the first one is shown.
    - Financial outlier handling.   Histograms auto-cap at the 99th percentile
                                    so extreme values don't collapse the chart.

How to add this tab to app.py
------------------------------
    from tabs.discover import render_discover_tab
    with tab2:
        render_discover_tab()
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st          # web app framework
import pandas as pd             # dataframe operations
import numpy as np              # math and statistics
import plotly.express as px     # chart library

from engine.insights  import generate_all_insights
from engine.detector  import get_columns_of_type
from utils.sanitizer  import sanitize_for_display


def _render_insight_card(insight: dict, df: pd.DataFrame, index: int):
    """
    Render a single insight — headline, detail sentence, and chart.

    This is a helper function called once per insight in the list.
    The underscore prefix is a Python convention meaning this function
    is intended for use only within this file.

    Parameters
    ----------
    insight : dict    One insight dict from generate_all_insights()
    df      : pd.DataFrame   The working dataset (needed for histogram raw data)
    index   : int     The insight number (used to make Streamlit widget keys unique)
    """

    insight_type = insight["type"]

    # ── Headline and detail sentence ──────────────────────────────────────────
    # The headline is shown large above the chart — this is the key finding
    st.markdown(f"#### {insight['headline']}")

    # The detail sentence provides one more level of context
    if insight.get("detail"):
        st.caption(insight["detail"])

    # ── Financial comparison (bonus line for segment insights) ────────────────
    if insight.get("fin_comparison"):
        st.caption(f"💰 {insight['fin_comparison']}")

    # ── Footnote for filtered dates ───────────────────────────────────────────
    if insight.get("footnote"):
        st.caption(f"ℹ {insight['footnote']}")

    # ── Chart rendering ───────────────────────────────────────────────────────
    chart_type = insight["chart_type"]
    data       = insight["data"]
    x_col      = insight["x_col"]

    # ── Line chart (time trend) ────────────────────────────────────────────────
    if chart_type == "line" and isinstance(data, pd.DataFrame):
        y_col = insight.get("y_col")
        if y_col and x_col in data.columns and y_col in data.columns:

            # Format Y axis label nicely
            y_label = f"Average {y_col}"

            fig = px.line(
                data,
                x=x_col,
                y="mean",       # 'mean' is the column name from _aggregate_by_time()
                markers=True,   # show dots at each data point
                labels={x_col: x_col, "mean": y_label},
                # The title is intentionally minimal — the headline IS the title
                title=""
            )

            # Add a secondary trace for record count as a subtle bar
            # This shows volume (how many records) alongside the average value
            fig.add_bar(
                x=data[x_col],
                y=data["count"],
                name="Record Count",
                yaxis="y2",
                opacity=0.2,
                marker_color="#94a3b8",
            )

            # Set up dual Y axis
            fig.update_layout(
                yaxis2=dict(
                    title="Record Count",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                hovermode="x unified",
                showlegend=True,
                plot_bgcolor="white",
                margin=dict(t=10, b=40),
            )

            st.plotly_chart(fig, use_container_width=True, key=f"line_{index}")

    # ── Bar chart (segment, outcome, high_cardinality_segment) ─────────────────
    elif chart_type == "bar" and isinstance(data, pd.DataFrame):

        # The data has two columns: the category and the count
        # We need to find which column is which
        cols = data.columns.tolist()

        # The count column is always named "Count"
        if "Count" in cols:
            cat_col   = [c for c in cols if c != "Count"][0]
            count_col = "Count"
        else:
            # Fallback: use the first two columns
            cat_col, count_col = cols[0], cols[1]

        # Sort by count descending for readability
        data_sorted = data.sort_values(count_col, ascending=True)

        # Add percentage labels to the bars
        total = data_sorted[count_col].sum()
        data_sorted = data_sorted.copy()
        data_sorted["pct"] = (data_sorted[count_col] / total * 100).round(1)
        data_sorted["label"] = data_sorted.apply(
            lambda r: f"{r[count_col]:,} ({r['pct']}%)", axis=1
        )

        fig = px.bar(
            data_sorted,
            x=count_col,
            y=cat_col,
            orientation="h",    # horizontal bars are more readable for categories
            text="label",
            color=count_col,
            color_continuous_scale="Blues",
            title="",
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(t=10, b=40, l=10, r=10),
            xaxis_title="",
            yaxis_title="",
        )

        st.plotly_chart(fig, use_container_width=True, key=f"bar_{index}")

    # ── Histogram (financial columns) ─────────────────────────────────────────
    elif chart_type == "histogram":

        col_name = insight.get("col_name", x_col)
        col_data = df[col_name].dropna()

        # Check for extreme outliers — the main issue with financial data
        p99     = col_data.quantile(0.99)
        max_val = col_data.max()
        is_extreme = (p99 > 0) and (max_val > p99 * 10)

        if is_extreme:
            # Show a checkbox to toggle the cap on/off
            # key= must be unique per chart — we use the column name and index
            cap_chart = st.checkbox(
                f"Cap chart at 99th percentile "
                f"(${p99:,.0f}) for readability — max value is ${max_val:,.0f}",
                value=True,
                key=f"hist_cap_{col_name}_{index}"
            )
            if cap_chart:
                plot_data = col_data[col_data <= p99]
                n_excluded = len(col_data) - len(plot_data)
                st.caption(
                    f"Showing {len(plot_data):,} of {len(col_data):,} values. "
                    f"{n_excluded:,} values above ${p99:,.0f} are not shown here. "
                    f"They are real — use the Explorer below to see the full range."
                )
            else:
                plot_data = col_data
        else:
            plot_data = col_data

        # Determine if the column is financial for dollar formatting
        financial_hints = [
            "amount", "price", "cost", "revenue", "fee",
            "value", "total", "pay", "close"
        ]
        is_financial = any(h in col_name.lower() for h in financial_hints)
        x_label = f"${col_name}" if is_financial else col_name

        fig = px.histogram(
            plot_data,
            nbins=40,
            labels={"value": col_name},
            color_discrete_sequence=["#3b82f6"],
            title="",
        )
        fig.update_layout(
            plot_bgcolor="white",
            xaxis_title=col_name,
            yaxis_title="Number of Records",
            bargap=0.05,
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{index}")


def _render_custom_explorer(df: pd.DataFrame, col_types: dict):
    """
    Render the custom chart builder section.

    This lets users build their own charts by picking columns and
    chart types. It is at the bottom of the Discover tab so it
    doesn't overwhelm users who just want the auto insights.

    Parameters
    ----------
    df        : pd.DataFrame   The working dataset.
    col_types : dict           Column types from detector.py.
    """
    st.markdown("---")
    st.markdown("### 🔍 Explore Your Data")
    st.caption(
        "The charts above were generated automatically. "
        "Use this section to explore any specific relationship you're curious about."
    )

    # Get column lists for the selector dropdowns
    all_cols     = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = (
        get_columns_of_type(col_types, "categorical") +
        get_columns_of_type(col_types, "high_cardinality")
    )

    # Chart type selector
    chart_choice = st.selectbox(
        "What kind of chart?",
        ["Bar — compare categories", "Scatter — compare two numbers",
         "Histogram — see value spread", "Line — values over time"],
        key="explorer_chart_type"
    )

    # ── Bar chart builder ──────────────────────────────────────────────────────
    if chart_choice.startswith("Bar"):
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Category column (X axis)", cat_cols, key="bar_x")
        y_col = col2.selectbox(
            "Value column (Y axis — leave blank to count records)",
            ["Count records"] + numeric_cols,
            key="bar_y"
        )

        if x_col:
            if y_col == "Count records":
                plot_df = df[x_col].value_counts().reset_index()
                plot_df.columns = [x_col, "Count"]
                fig = px.bar(
                    plot_df.head(20),   # top 20 categories
                    x=x_col, y="Count",
                    title=f"Record count by {x_col}",
                    color="Count", color_continuous_scale="Blues"
                )
            else:
                plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(
                    plot_df.head(20),
                    x=x_col, y=y_col,
                    title=f"Average {y_col} by {x_col}",
                    color=y_col, color_continuous_scale="Blues"
                )
            fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, key="explorer_bar")

    # ── Scatter chart builder ──────────────────────────────────────────────────
    elif chart_choice.startswith("Scatter"):
        if len(numeric_cols) < 2:
            st.info("Scatter plots need at least 2 numeric columns.")
        else:
            col1, col2, col3 = st.columns(3)
            x_col   = col1.selectbox("X axis", numeric_cols, key="scatter_x")
            y_col   = col2.selectbox("Y axis", numeric_cols,
                                     index=min(1, len(numeric_cols)-1),
                                     key="scatter_y")
            col_col = col3.selectbox("Color by (optional)",
                                     ["None"] + cat_cols, key="scatter_col")

            color_arg = None if col_col == "None" else col_col
            fig = px.scatter(
                df.sample(min(5000, len(df))),   # sample for performance
                x=x_col, y=y_col, color=color_arg,
                opacity=0.5,
                title=f"{y_col} vs {x_col}",
            )
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, key="explorer_scatter")
            st.caption("Showing up to 5,000 random points for performance.")

    # ── Histogram builder ──────────────────────────────────────────────────────
    elif chart_choice.startswith("Histogram"):
        if not numeric_cols:
            st.info("No numeric columns available for a histogram.")
        else:
            x_col = st.selectbox("Column to show", numeric_cols, key="hist_x")
            log_x = st.checkbox(
                "Use log scale (recommended for financial columns with extreme values)",
                key="hist_log"
            )
            fig = px.histogram(
                df, x=x_col, nbins=50, log_x=log_x,
                color_discrete_sequence=["#3b82f6"],
                title=f"Distribution of {x_col}"
            )
            fig.update_layout(plot_bgcolor="white", bargap=0.05)
            st.plotly_chart(fig, use_container_width=True, key="explorer_hist")

    # ── Line chart builder ─────────────────────────────────────────────────────
    elif chart_choice.startswith("Line"):
        date_cols = get_columns_of_type(col_types, "date_column")
        if not date_cols:
            st.info("No date columns detected. Line charts need a date column.")
        elif not numeric_cols:
            st.info("No numeric columns available.")
        else:
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("Date column", date_cols, key="line_x")
            y_col = col2.selectbox("Value column", numeric_cols, key="line_y")

            df_line = df[[x_col, y_col]].dropna()
            df_line = df_line[
                (df_line[x_col] >= pd.Timestamp("2000-01-01")) &
                (df_line[x_col] <= pd.Timestamp.now())
            ]
            if len(df_line) > 0:
                # Auto aggregate to monthly for readability
                df_line["__month__"] = (
                    df_line[x_col].dt.to_period("M").dt.to_timestamp()
                )
                df_agg = df_line.groupby("__month__")[y_col].mean().reset_index()
                df_agg.columns = [x_col, y_col]
                fig = px.line(
                    df_agg, x=x_col, y=y_col, markers=True,
                    title=f"Monthly average {y_col} over time"
                )
                fig.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True, key="explorer_line")


def render_discover_tab():
    """
    Render the entire Discover tab.

    Called once by app.py whenever the user is on this tab.

    Session state keys this function reads:
        st.session_state.working_df     the current dataset
        st.session_state.col_types      column type dict from detector
    """

    # ── Guard ──────────────────────────────────────────────────────────────────
    if "working_df" not in st.session_state:
        st.info("Upload a dataset using the sidebar to get started.")
        return

    df        = st.session_state.working_df
    col_types = st.session_state.col_types

    # ── Generate (or retrieve cached) insights ─────────────────────────────────
    # Computing insights is the most expensive operation in the app.
    # We cache them in session state so they aren't regenerated on every click.
    if "insights_cache" not in st.session_state:
        with st.spinner("Analyzing your data..."):
            st.session_state.insights_cache = generate_all_insights(df, col_types)

    all_insights = st.session_state.insights_cache

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("### What Your Data Shows")
    st.caption(
        "SmartWrangle analyzed your dataset and found these patterns. "
        "Each chart comes with a plain-English summary of what it means."
    )

    # ── Deduplication ─────────────────────────────────────────────────────────
    # Some datasets have redundant columns that would produce duplicate insights
    # (e.g. both 'Status' and 'Disposition' describe the claim outcome).
    # We track which columns have already been shown to avoid repeats.
    shown_cols = set()

    # ── Render each insight ────────────────────────────────────────────────────
    for i, insight in enumerate(all_insights):
        col_name = insight.get("col_name") or insight.get("x_col")

        # Skip if we already showed an insight for this column
        if col_name in shown_cols:
            continue
        shown_cols.add(col_name)

        # Visual separator between insights
        if i > 0:
            st.markdown("---")

        # Render the headline + chart
        _render_insight_card(insight, df, i)

    # ── If no insights were generated ─────────────────────────────────────────
    if not all_insights:
        st.info(
            "No automatic insights could be generated for this dataset. "
            "Try using the Explorer below to build your own charts."
        )

    # ── Custom explorer at the bottom ─────────────────────────────────────────
    _render_custom_explorer(df, col_types)


# ── END OF FILE ────────────────────────────────────────────────────────────────