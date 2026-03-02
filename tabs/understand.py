"""
tabs/understand.py
==================
The "Understand" tab — Tab 1 of SmartWrangle 2.0.

What this tab does
------------------
This is the first thing a user sees after uploading a dataset.
Its entire job is to answer one question: "What is in my data?"

It does NOT use jargon. A user who has never taken a statistics class
should be able to read this tab and immediately understand:
    - How big their dataset is
    - What time period it covers
    - What kinds of columns it has
    - Whether there are any problems to fix
    - How good the data quality is overall

Structure of this tab
----------------------
    1. Quality Score Banner   A large, colored score card at the top
    2. Dataset at a Glance    Key facts in metric cards (rows, columns, etc.)
    3. What's In Your Data    Column inventory in plain English
    4. Data Findings          Plain-English list of problems (if any)
    5. Advanced Details       Collapsible section for numeric column stats
                              (hidden by default — won't confuse beginners)

How to add this tab to app.py
------------------------------
    from tabs.understand import render_understand_tab
    with tab1:
        render_understand_tab()
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st   # the web app framework — all UI elements come from here
import pandas as pd      # for dataframe operations

# Engine modules — the logic lives here, not in the tab
from engine.detector import get_columns_of_type, plain_english_type, generate_column_recommendations
from engine.quality  import score_dataset, get_quality_report, get_column_health_table

# Utility — sanitize before displaying any DataFrame
from utils.sanitizer import sanitize_for_display


def render_understand_tab():
    """
    Render the entire Understand tab.

    This function is called once by app.py whenever the user is on
    this tab. All Streamlit UI elements (st.metric, st.dataframe, etc.)
    are written here.

    Session state keys this function reads:
        st.session_state.working_df     the current dataset
        st.session_state.col_types      column type dict from detector
        st.session_state.quality_result quality score dict from quality.py
    """

    # ── Guard: make sure we have data before rendering ─────────────────────────
    # If somehow the tab is rendered without data, show a friendly message
    if "working_df" not in st.session_state:
        st.info("Upload a dataset using the sidebar to get started.")
        return

    # ── Pull data from session state ───────────────────────────────────────────
    # We read from session state rather than computing here so that
    # all tabs share the same data and nothing is calculated twice.
    df          = st.session_state.working_df
    col_types   = st.session_state.col_types

    # Compute (or retrieve cached) quality result
    # We cache it so clicking around tabs doesn't re-run the analysis
    if "quality_result" not in st.session_state:
        st.session_state.quality_result = score_dataset(df, col_types)
    result = st.session_state.quality_result

    # ── CSS styling ────────────────────────────────────────────────────────────
    # Streamlit's built-in styling is limited, so we inject a small amount
    # of custom CSS to make the quality score banner and metric cards look
    # more polished. st.markdown() with unsafe_allow_html=True lets us
    # write raw HTML/CSS directly into the page.
    st.markdown("""
    <style>
        /* Quality score banner card */
        .quality-banner {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 28px 36px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 32px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .quality-score-number {
            font-size: 64px;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -2px;
        }
        .quality-grade {
            font-size: 22px;
            font-weight: 600;
            margin-top: 4px;
        }
        .quality-label {
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        /* Finding cards */
        .finding-good    { border-left: 4px solid #22c55e; background: #f0fdf4;
                           padding: 10px 16px; border-radius: 6px; margin: 6px 0;
                           color: #14532d; }
        .finding-warning { border-left: 4px solid #f59e0b; background: #fffbeb;
                           padding: 10px 16px; border-radius: 6px; margin: 6px 0;
                           color: #78350f; }
        .finding-problem { border-left: 4px solid #ef4444; background: #fef2f2;
                           padding: 10px 16px; border-radius: 6px; margin: 6px 0;
                           color: #7f1d1d; }
        /* Column type badge */
        .col-badge {
            display: inline-block;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            background: #e2e8f0;
            color: #475569;
            margin-left: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Quality Score Banner ───────────────────────────────────────
    score       = result["score"]
    grade       = result["grade"]
    grade_color = result["grade_color"]

    # Build the colored score banner using HTML
    # f-strings let us insert Python variables directly into the HTML
    st.markdown(f"""
    <div class="quality-banner">
        <div>
            <div class="quality-label">Data Quality Score</div>
            <div class="quality-score-number" style="color:{grade_color};">{score}</div>
            <div class="quality-grade" style="color:{grade_color};">{grade}</div>
        </div>
        <div style="color:#cbd5e1; font-size:15px; max-width:500px; line-height:1.7;">
            SmartWrangle analyzed your dataset and gave it a score out of 100.
            A higher score means the data is cleaner and more ready to work with.
            See the findings below for details.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: Dataset at a Glance ────────────────────────────────────────
    st.markdown("### Your Dataset at a Glance")

    # st.columns() splits the page into side-by-side columns
    # The numbers [1,1,1,1] mean four equal-width columns
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    # st.metric() shows a big number with a label — clean and readable
    c1.metric("Total Records",  f"{result['n_rows']:,}")
    c2.metric("Total Columns",  f"{result['n_cols']}")
    c3.metric("Missing Values", f"{result['missing']['total_missing']:,}")
    c4.metric("Duplicate Rows", f"{result['duplicates']['n_duplicates']:,}")

    st.markdown("---")

    # ── Date range (shown only if date columns exist) ──────────────────────────
    date_cols = get_columns_of_type(col_types, "date_column")
    if date_cols:
        try:
            date_series = df[date_cols[0]].dropna()
            # Filter plausible dates (remove data-entry errors)
            plausible   = date_series[
                (date_series >= pd.Timestamp("2000-01-01")) &
                (date_series <= pd.Timestamp.now())
            ]
            if len(plausible) > 0:
                date_min = plausible.min().strftime("%B %d, %Y")
                date_max = plausible.max().strftime("%B %d, %Y")
                st.markdown(
                    f"**Date Range:** {date_min} → {date_max} "
                    f"&nbsp;|&nbsp; "
                    f"**Date Column:** {date_cols[0]}"
                )
                st.markdown("---")
        except Exception:
            pass

    # ── SECTION 3: What's in Your Data ────────────────────────────────────────
    st.markdown("### What's in Your Data")
    st.caption(
        "Each column in your dataset has been identified and labeled "
        "so you know what kind of information it contains."
    )

    # Build a clean summary table for column types
    # We group columns by type to avoid a long flat list
    type_groups = {
        "Date / Time":                    get_columns_of_type(col_types, "date_column"),
        "Financial Amount":               get_columns_of_type(col_types, "financial"),
        "Category":                       get_columns_of_type(col_types, "categorical"),
        "Category (many values)":         get_columns_of_type(col_types, "high_cardinality"),
        "Numeric Measurement":            get_columns_of_type(col_types, "metric"),
        "Free Text":                      get_columns_of_type(col_types, "text"),
        "Identifier (not used in analysis)": get_columns_of_type(col_types, "id_column"),
    }

    # Display each group that has at least one column
    for type_label, cols in type_groups.items():
        if not cols:
            continue    # skip empty groups

        # Emoji prefix per type for quick visual scanning
        emoji_map = {
            "Date / Time":                      "📅",
            "Financial Amount":                 "💰",
            "Category":                         "🏷",
            "Category (many values)":           "📋",
            "Numeric Measurement":              "📐",
            "Free Text":                        "📝",
            "Identifier (not used in analysis)":"🔑",
        }
        emoji = emoji_map.get(type_label, "•")

        # Build the column list as a comma-separated string
        col_list = ", ".join(f"**{c}**" for c in cols)
        st.markdown(f"{emoji} **{type_label}** — {col_list}")

    st.markdown("---")

    # ── SECTION 4: Data Findings ───────────────────────────────────────────────
    st.markdown("### Data Findings")
    st.caption(
        "SmartWrangle checked your data for common issues. "
        "Green means no problem. Yellow means something to keep in mind."
    )

    # Get the plain-English findings from quality.py
    findings = get_quality_report(result)

    for finding in findings:
        # Choose the CSS class based on severity level
        css_class = {
            "good":    "finding-good",
            "warning": "finding-warning",
            "problem": "finding-problem",
        }.get(finding["level"], "finding-warning")

        # Render each finding as a colored card
        st.markdown(
            f'<div class="{css_class}">{finding["message"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── SECTION 4B: Intelligent Recommendations ────────────────────────────
    recommendations = generate_column_recommendations(df, col_types)

    if recommendations:
        st.markdown("### Intelligent Recommendations")
        st.caption(
            "SmartWrangle compared detected column types with value patterns "
            "to suggest improvements before analysis."
        )

        for rec in recommendations:

            message = (
                f"<strong>{rec['column']}</strong> detected as "
                f"{plain_english_type(rec['detected'])} "
                f"but appears to behave like "
                f"{plain_english_type(rec['expected'])}.<br><br>"
                f"{rec['recommendation']}"
            )

            # Use warning style to match your design language
            st.markdown(
                f'<div class="finding-warning">{message}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
    
    # ── SECTION 5: Data Preview ────────────────────────────────────────────────
    st.markdown("### Data Preview")
    st.caption("The first 10 rows of your dataset.")

    # Always sanitize before displaying — prevents Arrow serialization errors
    st.dataframe(
        sanitize_for_display(df.head(10)),
        use_container_width=True
    )

    # ── SECTION 6: Advanced Details (collapsed by default) ────────────────────
    # st.expander() creates a collapsible section
    # expanded=False means it starts closed — won't overwhelm beginners
    with st.expander("⚙ Advanced Details — Numeric Column Analysis", expanded=False):
        st.caption(
            "This section shows statistics for your numeric columns. "
            "It is hidden by default because most users won't need it. "
            "Data analysts and advanced users can expand it here."
        )

        # Get the column health table from quality.py
        health_table = get_column_health_table(df, col_types)

        if health_table.empty:
            st.info("No numeric columns found in this dataset.")
        else:
            # Color the Status column for quick visual scanning
            st.dataframe(
                sanitize_for_display(health_table),
                use_container_width=True
            )

        # Show the full column type inventory as a table for technical users
        st.markdown("**Full Column Inventory**")
        col_inventory = pd.DataFrame([
            {
                "Column":      col,
                "Detected As": plain_english_type(col_types.get(col, "categorical")),
                "Unique Values": df[col].nunique(),
                "Missing":     int(df[col].isnull().sum()),
            }
            for col in df.columns
        ])
        st.dataframe(sanitize_for_display(col_inventory), use_container_width=True)


# ── END OF FILE ────────────────────────────────────────────────────────────────