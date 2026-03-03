"""
app.py
======
SmartWrangle 2.0 — Main Application Entry Point

What this file does
-------------------
This is the first file Streamlit runs when someone opens the app.
Its job is to:

    1. Configure the page (title, layout, icon)
    2. Render the sidebar (file uploader + app info)
    3. Handle file uploads — load the data, detect column types,
       initialize session state
    4. Show the landing screen when no file is uploaded
    5. Show the three-tab interface when a file IS uploaded
    6. Route each tab to its corresponding render function

This file contains NO analysis logic — that all lives in the engine/
and utils/ folders. app.py is purely about layout and routing.

Session state managed here
--------------------------
    original_df      the raw uploaded file — NEVER modified after load
    working_df       current version — cleaning tab modifies this
    col_types        dict from detector.detect_column_types()
    quality_result   cached quality score dict
    insights_cache   cached insights list
    cleaning_log     list of plain-English action strings
    version_history  list of df snapshots for undo
    dataset_name     filename shown in the report header

How Streamlit reruns work
--------------------------
Streamlit reruns the ENTIRE script from top to bottom on every user
interaction (button click, file upload, slider change, etc.).

Session state (st.session_state) is the only thing that persists
between reruns. Anything stored as a Python variable without being
saved to session state is lost on the next rerun.

This is why we check "if key not in st.session_state" before computing
expensive operations — we don't want to recalculate the quality score
every time the user scrolls the page.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
# streamlit : the entire web app framework
# pandas    : for loading CSV and Excel files
# os        : to extract the filename from the uploaded file path
import streamlit as st
import pandas as pd
import os

# Engine modules — all the intelligence of the app lives here
from engine.detector import detect_column_types

# Tab render functions — each tab is a separate file
from tabs.understand   import render_understand_tab
from tabs.discover     import render_discover_tab
from tabs.clean_export import render_clean_export_tab

# Utility modules
from utils.sanitizer import sanitize_column_names


# ── Page configuration ─────────────────────────────────────────────────────────
# st.set_page_config() MUST be the very first Streamlit call in the file.
# It sets the browser tab title, icon, and layout.
# layout="wide" uses the full browser width instead of a narrow center column.
st.set_page_config(
    page_title="SmartWrangle 2.0",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ─────────────────────────────────────────────────────────────────
# A small amount of custom CSS to make the app look more polished.
# This runs on every page load, so it always applies.
st.markdown("""
<style>
    /* Import a clean, professional font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* Apply the font to the whole app */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Style the tab bar — make active tab more distinct */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 500;
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #f8fafc;
        color: #1e293b;
        font-weight: 700;
        border-bottom: 3px solid #3b82f6;
    }

    /* Clean up metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }

    /* Expanders — make them look like accordion sections */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 14px;
        color: #374151;
        background: #f9fafb;
        border-radius: 8px;
        padding: 12px 16px !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #e5e7eb;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 16px;
    }

    /* Download buttons */
    [data-testid="stDownloadButton"] button {
        background: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    /* Target only text we control — headings, labels, buttons.
       We removed the wildcard * selector because it was overriding
       Streamlit's internal file uploader text, making "Drag and drop
       file here" and "Limit 200MB per file" unreadable on the dark bg. */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #e2e8f0 !important;
    }

    /* Markdown paragraphs and bold text */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #cbd5e1 !important;
        font-size: 13px;
    }

    /* Caption / small text */
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }

    /* File uploader box — white background so text is always legible */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: ##1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 4px;
    }

    /* "Upload your dataset" label sits ABOVE the white uploader box,
       against the dark sidebar — needs an explicit bright color rule.
       Streamlit renders widget labels as a <p> inside stWidgetLabel. */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #f1f5f9 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    /* Make section headers stand out */
    h3 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    /* Toast notifications */
    [data-testid="stToast"] {
        background: #1e293b;
        color: white;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
# The sidebar is always visible regardless of which tab is active.
# It contains the file uploader and basic app information.
with st.sidebar:

    # App name and tagline
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
        <svg width="32" height="32" viewBox="0 0 88 88" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="88" height="88" rx="22" fill="#1e3a5f"/>
          <ellipse cx="42" cy="44" rx="32" ry="26" fill="#3b82f6" opacity="0.08"/>
          <path d="M11 27 C11 19 17 14 26 14 L42 14 C52 14 57 20 57 28 C57 36 51 40 42 40 L26 40 C18 40 13 45 13 52 C13 60 19 65 28 65 L45 65 C55 65 60 59 60 51"
                stroke="#60a5fa" stroke-width="6.5" stroke-linecap="round" fill="none"/>
          <path d="M57 16 L63 54 L70 33 L77 54 L83 16"
                stroke="#93c5fd" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
        <span style="font-size:20px; font-weight:800; color:#f1f5f9; letter-spacing:-0.5px;">SmartWrangle</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:12px; margin-top:0; margin-bottom:0;'>AI-powered data wrangling</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────────────
    # st.file_uploader() shows a drag-and-drop zone.
    # type=["csv","xlsx"] restricts to spreadsheet files.
    # When a file is uploaded, this returns a file-like object.
    # When nothing is uploaded, it returns None.
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV (.csv) and Excel (.xlsx)"
    )

    st.markdown("---")

    # ── Show current dataset info if one is loaded ─────────────────────────────
    if "working_df" in st.session_state:
        df = st.session_state.working_df

        st.markdown("**Current Dataset**")
        st.markdown(
            f"📄 {st.session_state.get('dataset_name', 'Dataset')}"
        )
        st.markdown(
            f"**{len(df):,}** rows · **{len(df.columns)}** columns"
        )

        # Show quality score in the sidebar for quick reference
        if "quality_result" in st.session_state:
            score = st.session_state.quality_result["score"]
            grade = st.session_state.quality_result["grade"]
            st.markdown(
                f"Quality score: **{score}/100** ({grade})"
            )

        st.markdown("---")

        # Reset button — lets the user upload a different file
        if st.button("Upload a different file", use_container_width=True):
            # Clear all session state keys related to the current dataset
            keys_to_clear = [
                "original_df", "working_df", "col_types",
                "quality_result", "insights_cache",
                "cleaning_log", "version_history", "dataset_name"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # st.rerun() triggers a full page refresh
            st.rerun()

    st.markdown("---")
    st.caption("SmartWrangle 2.0")
    st.caption("Built with Streamlit · Python")


# ── File loading ────────────────────────────────────────────────────────────────
# This block runs every time a file is uploaded (or re-uploaded).
# We check if the uploaded file is different from the one already loaded
# by comparing filenames — if they match, we skip re-loading to avoid
# resetting the user's cleaning work.

if uploaded_file is not None:

    # Has this specific file already been loaded?
    # We compare the filename to avoid reloading on every rerun.
    already_loaded = (
        "dataset_name" in st.session_state and
        st.session_state.dataset_name == os.path.splitext(uploaded_file.name)[0]
    )

    if not already_loaded:

        # ── Load the file ──────────────────────────────────────────────────────
        # Show a spinner while loading — large files can take a moment
        with st.spinner(f"Loading {uploaded_file.name}..."):
            try:
                # Determine file type by extension
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()

                if file_ext == ".csv":
                    # low_memory=False: tells pandas to read the whole file
                    # before inferring column types. This avoids the common
                    # "DtypeWarning: mixed types" error on large files.
                    raw_df = pd.read_csv(uploaded_file, low_memory=False)

                elif file_ext == ".xlsx":
                    # engine="openpyxl" is the modern Excel reader.
                    # Without this, newer .xlsx files may fail to load.
                    raw_df = pd.read_excel(uploaded_file, engine="openpyxl")

                else:
                    st.error(
                        f"Unsupported file type: {file_ext}. "
                        f"Please upload a CSV or Excel file."
                    )
                    st.stop()   # stop execution — don't try to render tabs

                # ── Clean up column names ──────────────────────────────────────
                # Strip any leading/trailing spaces from column headers.
                # This prevents subtle bugs where " Amount" != "Amount".
                raw_df = sanitize_column_names(raw_df)

                # ── Remove string 'None' artifact ─────────────────────────────
                # A previous bug allowed users to type 'None' as a constant
                # fill value for date columns. This left the 4-character string
                # 'None' stored as a category in those columns. This sanitizer
                # removes that artifact on every file load before anything else
                # in the app reads the data.
                # We only check object (text) columns — numeric and datetime
                # columns are never affected.
                for _col in raw_df.select_dtypes(include="object").columns:
                    if (raw_df[_col] == "None").any():
                        raw_df[_col] = raw_df[_col].replace("None", pd.NA)

                # ── Store original (never to be modified) ──────────────────────
                # original_df is kept for the "Reset to original" feature
                # and for the export summary to show the starting state.
                st.session_state.original_df = raw_df.copy()

                # ── Create working copy ────────────────────────────────────────
                # working_df starts as a copy of original_df.
                # All cleaning operations modify working_df, not original_df.
                working_df = raw_df.copy()

                # ── Detect column types ────────────────────────────────────────
                # detect_column_types() reads every column and classifies it.
                # It ALSO converts date columns to datetime in-place,
                # which is why we pass working_df and not raw_df —
                # we want the date conversion to persist in working_df.
                #
                # The StringDtype fix (v1.0 bug) lives inside detect_column_types():
                # it uses pd.api.types.is_string_dtype() instead of checking
                # dtype == "object", which caused date columns to never convert.
                col_types = detect_column_types(working_df)

                # ── Save everything to session state ───────────────────────────
                st.session_state.working_df      = working_df
                st.session_state.col_types       = col_types
                st.session_state.cleaning_log    = []   # empty — no actions yet
                st.session_state.version_history = []   # empty undo history
                st.session_state.dataset_name    = (
                    os.path.splitext(uploaded_file.name)[0]
                )

                # ── Clear any cached analysis from a previous file ─────────────
                for key in ["quality_result", "insights_cache"]:
                    if key in st.session_state:
                        del st.session_state[key]

                # ── Force a clean rerender ─────────────────────────────────────
                # Streamlit renders the sidebar BEFORE this file loading block
                # runs, so without this call the sidebar still shows the previous
                # dataset name, row count, and quality score after a new file
                # is uploaded. st.rerun() triggers a fresh top-to-bottom render
                # so sidebar and main content both read the updated session state.
                st.rerun()

            except Exception as e:
                # If loading fails, show a helpful error message
                st.error(
                    f"Could not load this file. "
                    f"Make sure it is a valid CSV or Excel file.\n\n"
                    f"Technical details: {str(e)}"
                )
                st.stop()


# ── Main content area ──────────────────────────────────────────────────────────
# Everything below this line is what shows in the main page area.

if "working_df" not in st.session_state:

    # ── Landing screen (no file uploaded yet) ─────────────────────────────────
    st.markdown("""
    <div style="
        max-width: 700px;
        margin: 60px auto 0 auto;
        text-align: center;
        padding: 0 24px;
    ">
        <div style="margin-bottom: 24px; display: flex; justify-content: center;">
            <svg width="96" height="96" viewBox="0 0 88 88" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- Rounded square background -->
              <rect width="88" height="88" rx="22" fill="#1e293b"/>
              <!-- Subtle inner glow -->
              <ellipse cx="42" cy="44" rx="32" ry="26" fill="#3b82f6" opacity="0.06"/>
              <!-- S letterform — bold, left side, blue-500 -->
              <path d="M11 27 C11 19 17 14 26 14 L42 14 C52 14 57 20 57 28
                       C57 36 51 40 42 40 L26 40 C18 40 13 45 13 52
                       C13 60 19 65 28 65 L45 65 C55 65 60 59 60 51"
                    stroke="#3b82f6" stroke-width="6.5" stroke-linecap="round" fill="none"/>
              <!-- W letterform — slightly lighter, right side, blue-400 -->
              <path d="M57 16 L63 54 L70 33 L77 54 L83 16"
                    stroke="#60a5fa" stroke-width="5.5" stroke-linecap="round"
                    stroke-linejoin="round" fill="none"/>
            </svg>
        </div>
        <h1 style="
            font-size: 42px;
            font-weight: 800;
            color: #1e293b;
            letter-spacing: -1px;
            margin-bottom: 12px;
        ">SmartWrangle 2.0</h1>
        <p style="
            font-size: 19px;
            color: #64748b;
            line-height: 1.7;
            margin-bottom: 40px;
        ">
            Upload any dataset and SmartWrangle tells you what's in it,
            what it shows, and how to clean it — in plain English,
            without requiring any data science knowledge.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Three feature cards
    col1, col2, col3 = st.columns(3)
    card_style = (
        "background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; "
        "padding:24px; height:100%;"
    )

    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:32px; margin-bottom:12px;">📋</div>
            <div style="font-size:17px; font-weight:700; color:#1e293b; margin-bottom:8px;">
                Understand
            </div>
            <div style="font-size:14px; color:#64748b; line-height:1.6;">
                Instantly see what's in your data — column types, date ranges,
                quality score, and any problems — in plain English.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:32px; margin-bottom:12px;">📊</div>
            <div style="font-size:17px; font-weight:700; color:#1e293b; margin-bottom:8px;">
                Discover
            </div>
            <div style="font-size:14px; color:#64748b; line-height:1.6;">
                Auto-generated charts with plain-English headlines
                telling you exactly what each chart means.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:32px; margin-bottom:12px;">⚙</div>
            <div style="font-size:17px; font-weight:700; color:#1e293b; margin-bottom:8px;">
                Clean &amp; Export
            </div>
            <div style="font-size:14px; color:#64748b; line-height:1.6;">
                One-click fixes, full undo support, and download your
                cleaned data plus a plain-English summary report.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#94a3b8; font-size:14px;'>"
        "← Upload a CSV or Excel file using the sidebar to get started"
        "</p>",
        unsafe_allow_html=True
    )

else:

    # ── Three-tab interface (file is loaded) ───────────────────────────────────
    # st.tabs() creates the clickable tab bar at the top of the page.
    # Each "with tab:" block only renders when that tab is active.
    tab1, tab2, tab3 = st.tabs([
        "📋  Understand",
        "📊  Discover",
        "⚙  Clean & Export",
    ])

    with tab1:
        # Tab 1: What is in my data?
        render_understand_tab()

    with tab2:
        # Tab 2: What does my data show?
        render_discover_tab()

    with tab3:
        # Tab 3: Fix and download.
        render_clean_export_tab()


# ── END OF FILE ────────────────────────────────────────────────────────────────