"""
utils/pdf_report.py
===================
Builds the PDF report offered in the Clean & Export tab.

What the report contains
------------------------
    1. Cover summary      File name, date, size, and the quality score
    2. Before and after   The numbers at upload next to the numbers now
    3. Findings           The plain-English findings from the Understand tab
    4. Charts             The automatic Discover charts, each with its headline
    5. Key numbers        Totals and averages for number columns, top categories
    6. Cleaning steps     Every action taken, in order

The report repackages what SmartWrangle has already found. It does not
run any new analysis, and the AI Analyst conversation is not included.

Why matplotlib and fpdf2
------------------------
The app's on-screen charts are Plotly. Saving Plotly charts as images
needs Google Chrome installed, which Streamlit Community Cloud does not
have. So the report redraws its charts with matplotlib, which needs no
browser, and builds the PDF with fpdf2, which is pure Python.

Both libraries need the same font for text with accents and symbols.
matplotlib ships with the DejaVu Sans font, so the PDF uses that file
too, and nothing extra has to be installed.

Main function
-------------
    build_pdf_report(original_df, working_df, col_types, cleaning_log,
                     dataset_name, date_warnings) -> bytes
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import datetime
import io
import warnings
from pathlib import Path

from utils.wording import plural          # "1 value" / "3 values"

import numpy as np
import pandas as pd

# matplotlib must use the "Agg" backend on a server: it draws to memory
# instead of trying to open a window. This line has to come before pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF
from fpdf.enums import XPos, YPos


# ── Look and feel ──────────────────────────────────────────────────────────────
# Colors as (red, green, blue), matching the app's dark navy and blue theme
NAVY   = (30, 41, 59)
BLUE   = (59, 130, 246)
GRAY   = (100, 116, 139)
LIGHT  = (241, 245, 249)
GREEN  = (22, 163, 74)
AMBER  = (202, 138, 4)
RED    = (220, 38, 38)

# The same blue and gray for matplotlib, which wants hex strings
BLUE_HEX = "#3b82f6"
GRAY_HEX = "#94a3b8"

# Font files that come with matplotlib (regular and bold)
FONT_DIR     = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD    = FONT_DIR / "DejaVuSans-Bold.ttf"

MAX_CHARTS = 6          # keeps the report short enough to read
MAX_BARS   = 15         # bars per chart; more than this gets unreadable on paper


# ── Small helpers ──────────────────────────────────────────────────────────────

def _short(text, limit: int = 45) -> str:
    """
    Shorten long values (like a 300-character title) so tables stay tidy.
    Line breaks inside a value (some files have them) become spaces.
    """
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _plain(text) -> str:
    """
    Undo the screen-only escaping in insight text. The app writes "\\$" so
    Streamlit does not read "$...$" as a math formula; a PDF needs plain "$".
    """
    text = str(text).replace("\\$", "$")
    return " ".join(text.split())        # line breaks inside values -> spaces


def _money_or_number(value: float, is_money: bool) -> str:
    """Format a number with commas, and a $ sign for money columns."""
    if pd.isna(value):
        return "-"
    if is_money:
        return f"${value:,.2f}"
    # Whole numbers without decimals, others with two
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _change(before, after, decimals: int = 0) -> str:
    """Show the difference between two numbers, like '+12' or '-3.5'."""
    diff = round(after - before, decimals)
    if diff == 0:
        return "No change"
    text = f"{diff:+,.{decimals}f}"
    return text


def _chart_text(text) -> str:
    """
    Make text safe for chart labels. Like Streamlit, matplotlib reads text
    between two $ signs as a math formula, so '$99K-$132K' turned into
    '99K - 132K' in italics. A backslash before each $ keeps it literal.
    """
    return str(text).replace("$", r"\$")


def _fig_to_png(fig) -> io.BytesIO:
    """Save a matplotlib chart into memory as a PNG image and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)                     # free the memory the chart used
    buffer.seek(0)
    return buffer


def _new_axes():
    """A clean, wide chart area with the app's light styling."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)    # no box around the chart
    ax.grid(axis="both", color="#e2e8f0", linewidth=0.6)
    ax.set_axisbelow(True)                    # grid lines behind the data
    ax.tick_params(labelsize=8)
    return fig, ax


# ── Charts (redrawn from the Discover tab's automatic insights) ────────────────

def _chart_line(insight: dict):
    """Time trend: average value per period, with record counts behind it."""
    data, x_col, y_col = insight["data"], insight["x_col"], insight.get("y_col")
    if not isinstance(data, pd.DataFrame) or "mean" not in data.columns:
        return None, None
    fig, ax = _new_axes()
    # Record counts as a light shaded area on a second axis (right side)
    ax2 = ax.twinx()
    ax2.fill_between(data[x_col], data["count"], step="mid", color=GRAY_HEX, alpha=0.25)
    ax2.set_ylabel("Record count", fontsize=8, color=GRAY_HEX)
    ax2.tick_params(labelsize=8, colors=GRAY_HEX)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    # The average line on top
    ax.plot(data[x_col], data["mean"], marker="o", markersize=3, color=BLUE_HEX)
    ax.set_ylabel(_chart_text(f"Average {y_col}"), fontsize=8)
    ax.set_zorder(ax2.get_zorder() + 1)       # keep the line in front
    ax.patch.set_visible(False)
    fig.autofmt_xdate()                       # tilt dates so they don't overlap
    return _fig_to_png(fig), None


def _chart_bar(insight: dict):
    """Category breakdown: horizontal bars with count and percent labels."""
    data = insight["data"]
    if not isinstance(data, pd.DataFrame) or len(data.columns) < 2:
        return None, None
    cols = data.columns.tolist()
    if "Count" in cols:
        cat_col, count_col = [c for c in cols if c != "Count"][0], "Count"
    else:
        cat_col, count_col = cols[0], cols[1]
    total = data[count_col].sum()        # percentages use every category
    note = None
    if len(data) > MAX_BARS:
        note = f"Showing the top {MAX_BARS} of {len(data):,} values."
        data = data.nlargest(MAX_BARS, count_col)
    data = data.sort_values(count_col, ascending=True)   # biggest bar at the top

    fig, ax = _new_axes()
    fig.set_size_inches(7.5, max(2.2, 0.32 * len(data) + 0.8))
    labels = [_chart_text(_short(v, 30)) for v in data[cat_col]]
    bars = ax.barh(labels, data[count_col], color=BLUE_HEX)
    ax.grid(axis="y", visible=False)
    # Write "1,234 (12.3%)" at the end of each bar
    for bar, count in zip(bars, data[count_col]):
        pct = count / total * 100 if total else 0
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {count:,.0f} ({pct:.1f}%)", va="center", fontsize=7, color="#334155")
    ax.set_xlim(0, data[count_col].max() * 1.25)        # room for the labels
    return _fig_to_png(fig), note


def _chart_histogram(insight: dict, df: pd.DataFrame):
    """Value spread for a number column, capped at the 99th percentile if needed."""
    col_name = insight.get("col_name", insight["x_col"])
    if col_name not in df.columns:
        return None, None
    values = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if values.empty:
        return None, None

    note = None
    # Same rule as the Discover tab: a few huge values squash the chart
    p99, top = values.quantile(0.99), values.max()
    if p99 > 0 and top > p99 * 10:
        shown = values[values <= p99]
        note = (f"Chart shows {len(shown):,} of {len(values):,} values. "
                f"{plural(len(values) - len(shown), 'value')} above {p99:,.0f} are left out "
                f"so the rest can be seen. They are still in the data.")
        values = shown

    fig, ax = _new_axes()
    ax.hist(values, bins=40, color=BLUE_HEX, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(_chart_text(col_name), fontsize=8)
    ax.set_ylabel("Number of records", fontsize=8)
    return _fig_to_png(fig), note


def _draw_chart(insight: dict, df: pd.DataFrame):
    """Pick the right chart for an insight. Returns (png or None, note or None)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # e.g. a symbol the font lacks
            kind = insight.get("chart_type")
            if kind == "line":
                return _chart_line(insight)
            if kind == "bar":
                return _chart_bar(insight)
            if kind == "histogram":
                return _chart_histogram(insight, df)
    except Exception:
        plt.close("all")
    # Any chart that cannot be drawn is skipped; its headline is still shown
    return None, None


# ── The PDF document ───────────────────────────────────────────────────────────

class _ReportPDF(FPDF):
    """An FPDF page with the SmartWrangle footer on every page."""

    def footer(self):
        """Page number and app name at the bottom of each page."""
        self.set_y(-12)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 6, f"SmartWrangle report  |  Page {self.page_no()} of {{nb}}",
                  align="C")


def _section_title(pdf: FPDF, number: int, title: str, room: float = 60):
    """
    A numbered section heading with a blue rule under it.
    room = how much space (mm) the start of the section needs; if less is
    left on the page, the heading moves to a new page instead of sitting
    alone at the bottom.
    """
    if pdf.get_y() > pdf.h - room:
        pdf.add_page()
    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, f"{number}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


def _paragraph(pdf: FPDF, text: str, size: float = 9.5, color=NAVY, bold: bool = False):
    """A block of wrapped text that uses the full page width."""
    pdf.set_font("DejaVu", "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.52, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _table(pdf: FPDF, header: list, rows: list, widths: tuple, align: tuple):
    """A simple table with a navy header row and light striped rows."""
    pdf.set_font("DejaVu", "", 8.5)
    pdf.set_text_color(*NAVY)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_fill_color(*LIGHT)       # the stripe color (the cover banner left it navy)
    with pdf.table(col_widths=widths, text_align=align, line_height=5.5,
                   headings_style=_heading_style(), cell_fill_color=LIGHT,
                   cell_fill_mode="ROWS", first_row_as_headings=True) as table:
        for row in [header] + rows:
            table_row = table.row()
            for value in row:
                table_row.cell(str(value))
    pdf.ln(2)


def _heading_style():
    """Bold white text on navy for table header rows."""
    from fpdf.fonts import FontFace
    return FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=NAVY)


def _quality_numbers(df: pd.DataFrame, col_types: dict) -> dict:
    """Score a table and pull out the numbers the report compares."""
    from engine.quality import score_dataset
    q = score_dataset(df, col_types)
    return {
        "result":     q,
        "score":      q["score"],
        "grade":      q["grade"],
        "rows":       len(df),
        "columns":    len(df.columns),
        "missing":    int(q["missing"]["total_missing"]),
        "duplicates": int(q["duplicates"]["n_duplicates"]),
    }


# ── Main function ──────────────────────────────────────────────────────────────

def build_pdf_report(
    original_df: pd.DataFrame,
    working_df: pd.DataFrame,
    col_types: dict,
    cleaning_log: list,
    dataset_name: str = "Dataset",
    date_warnings: dict | None = None,
) -> bytes:
    """
    Build the six-section PDF report and return it as bytes for download.

    Parameters
    ----------
    original_df  : pd.DataFrame  The file exactly as uploaded (never changed).
    working_df   : pd.DataFrame  The data as it looks now, after cleaning.
    col_types    : dict          Column types for working_df (from detector.py).
    cleaning_log : list of str   Every action taken, in order.
    dataset_name : str           File name shown on the cover.
    date_warnings: dict          {column: (count, examples)} of values that
                                 could not be read as dates on upload.

    Returns
    -------
    bytes   The finished PDF file.
    """
    # Engine imports are here, like in exporter.py, so this file can be
    # imported without loading the whole engine first.
    from engine.detector import detect_column_types, get_columns_of_type
    from engine.quality import get_quality_report
    from engine.insights import generate_all_insights
    from engine.cleaner import find_placeholders

    # ── Gather the numbers ─────────────────────────────────────────────────────
    # The original upload is scored the same way the app scored it on upload.
    # detect_column_types() changes the table it is given (dates), so it
    # gets a copy and the real original is never touched.
    original_copy = original_df.copy()
    before = _quality_numbers(original_copy, detect_column_types(original_copy))
    now = _quality_numbers(working_df, col_types)

    # ── Set up the document ────────────────────────────────────────────────────
    pdf = _ReportPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 16, 16)
    pdf.add_font("DejaVu", "", str(FONT_REGULAR))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.set_title(f"SmartWrangle report: {dataset_name}")
    pdf.set_author("SmartWrangle")
    pdf.add_page()

    # ── SECTION 1: Cover summary ───────────────────────────────────────────────
    # A navy banner with the file name and the score, like the Understand tab
    banner_top = pdf.get_y()
    pdf.set_fill_color(*NAVY)
    pdf.rect(pdf.l_margin, banner_top, pdf.epw, 42, style="F")
    pdf.set_xy(pdf.l_margin + 6, banner_top + 5)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "SMARTWRANGLE DATA REPORT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin + 6)
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(110, 9, _short(dataset_name, 40))
    # Score on the right side of the banner
    grade_color = {"Excellent": GREEN, "Good": GREEN, "Fair": AMBER}.get(now["grade"], RED)
    pdf.set_xy(pdf.w - pdf.r_margin - 56, banner_top + 6)
    pdf.set_font("DejaVu", "B", 26)
    pdf.set_text_color(*grade_color)
    pdf.cell(50, 12, f"{now['score']:.1f}", align="R")
    pdf.set_xy(pdf.w - pdf.r_margin - 56, banner_top + 19)
    pdf.set_font("DejaVu", "", 9)
    pdf.cell(50, 5, f"Quality score  |  {now['grade']}", align="R")
    # Size and date under the name
    pdf.set_xy(pdf.l_margin + 6, banner_top + 22)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(0, 5, f"{now['rows']:,} rows  |  {now['columns']} columns")
    pdf.set_xy(pdf.l_margin + 6, banner_top + 29)
    created = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    pdf.cell(0, 5, f"Created {created}")
    pdf.set_y(banner_top + 46)
    _paragraph(pdf,
               "This report sums up what SmartWrangle found in the file and what "
               "was done to clean it. The quality score runs from 0 to 100 and "
               "drops for missing values, duplicate rows, very uneven number "
               "columns, and extreme values.", size=9, color=GRAY)

    # ── SECTION 2: Before and after ────────────────────────────────────────────
    _section_title(pdf, 1, "Before and After Cleaning")
    rows = [
        ["Quality score", f"{before['score']:.1f} ({before['grade']})",
         f"{now['score']:.1f} ({now['grade']})", _change(before["score"], now["score"], 1)],
        ["Rows", f"{before['rows']:,}", f"{now['rows']:,}", _change(before["rows"], now["rows"])],
        ["Columns", before["columns"], now["columns"], _change(before["columns"], now["columns"])],
        ["Missing cells", f"{before['missing']:,}", f"{now['missing']:,}",
         _change(before["missing"], now["missing"])],
        ["Duplicate rows", f"{before['duplicates']:,}", f"{now['duplicates']:,}",
         _change(before["duplicates"], now["duplicates"])],
    ]
    _table(pdf, ["Measure", "When uploaded", "Now", "Change"], rows,
           widths=(34, 26, 26, 20), align=("LEFT", "RIGHT", "RIGHT", "RIGHT"))
    if not cleaning_log:
        _paragraph(pdf, "No cleaning steps have been taken yet, so both columns match.",
                   size=8.5, color=GRAY)
    elif now["missing"] > before["missing"]:
        # Explain the one result that looks wrong but is not
        _paragraph(pdf,
                   "Missing cells went up. This is expected when placeholder words "
                   "like ERROR or UNKNOWN are marked as blank, or when text that is "
                   "not a date or number is cleared during a conversion: those cells "
                   "were already missing information, and now they are counted.",
                   size=8.5, color=GRAY)

    # ── SECTION 3: Findings ────────────────────────────────────────────────────
    _section_title(pdf, 2, "What SmartWrangle Found")
    findings = [(f["level"], f["message"]) for f in get_quality_report(now["result"])]

    placeholders = find_placeholders(working_df)
    if placeholders:
        n_ph = sum(p["count"] for p in placeholders)
        shown = ", ".join(f"'{p}'" for p in sorted({p["shown_as"] for p in placeholders})[:4])
        findings.append(("warning",
                         f"{plural(n_ph, 'placeholder value')} such as {shown} may stand for "
                         f"missing data but are not counted as missing."))
    for col, (n_bad, examples) in (date_warnings or {}).items():
        findings.append(("warning",
                         f"{plural(n_bad, 'value')} in '{col}' could not be read as dates when "
                         f"the file was uploaded (for example: "
                         f"{', '.join(repr(e) for e in examples)})."))

    labels = {"good": ("OK", GREEN), "warning": ("NOTE", AMBER)}
    for level, message in findings:
        tag, color = labels.get(level, ("ISSUE", RED))
        pdf.set_font("DejaVu", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(14, 5, tag)
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(0, 5, message, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    # ── SECTION 4: Charts with headlines ───────────────────────────────────────
    _section_title(pdf, 3, "Charts and What They Show", room=110)
    try:
        insights = generate_all_insights(working_df, col_types)[:MAX_CHARTS]
    except Exception:
        insights = []
    if not insights:
        _paragraph(pdf, "No automatic charts could be made for this data.", size=9, color=GRAY)
    for insight in insights:
        png, note = _draw_chart(insight, working_df)
        # Keep a headline and its chart on the same page when possible
        if pdf.get_y() > pdf.h - 95:
            pdf.add_page()
        _paragraph(pdf, _plain(insight.get("headline", "")), size=10.5, bold=True)
        if insight.get("detail"):
            _paragraph(pdf, _plain(insight["detail"]), size=8.5, color=GRAY)
        if insight.get("fin_comparison"):
            _paragraph(pdf, _plain(insight["fin_comparison"]), size=8.5, color=GRAY)
        if png is not None:
            pdf.image(png, w=pdf.epw * 0.9, x=pdf.l_margin + pdf.epw * 0.05)
        for extra in (note, insight.get("footnote")):
            if extra:
                _paragraph(pdf, _plain(extra), size=7.5, color=GRAY)
        pdf.ln(4)

    # ── SECTION 5: Key numbers ─────────────────────────────────────────────────
    _section_title(pdf, 4, "Key Numbers")
    money_cols = get_columns_of_type(col_types, "financial")
    number_cols = money_cols + get_columns_of_type(col_types, "metric")
    number_rows = []
    for col in number_cols[:10]:
        values = pd.to_numeric(working_df[col], errors="coerce").dropna()
        if values.empty:
            continue
        is_money = col in money_cols
        number_rows.append([
            _short(col, 28),
            f"{len(values):,}",
            _money_or_number(values.sum(), True) if is_money else "-",   # totals only make sense for money
            _money_or_number(values.mean(), is_money),
            _money_or_number(values.median(), is_money),
            _money_or_number(values.min(), is_money),
            _money_or_number(values.max(), is_money),
        ])
    if number_rows:
        _paragraph(pdf, "Number columns", size=9.5, bold=True)
        pdf.ln(1)
        _table(pdf, ["Column", "Filled", "Total", "Average", "Median", "Lowest", "Highest"],
               number_rows, widths=(30, 13, 22, 19, 19, 19, 22),
               align=("LEFT",) + ("RIGHT",) * 6)
    else:
        _paragraph(pdf, "There are no number columns to sum up yet. If numbers are "
                        "stored as text, convert them in the Clean & Export tab first.",
                   size=9, color=GRAY)

    category_cols = get_columns_of_type(col_types, "categorical")[:4]
    if category_cols:
        pdf.ln(2)
        _paragraph(pdf, "Most common values", size=9.5, bold=True)
        pdf.ln(1)
        rows = []
        for col in category_cols:
            counts = working_df[col].value_counts()
            total = counts.sum()
            for value, count in counts.head(3).items():
                rows.append([_short(col, 25), _short(value, 35), f"{count:,}",
                             f"{count / total * 100:.1f}%"])
        _table(pdf, ["Column", "Value", "Count", "Share"], rows,
               widths=(30, 45, 15, 15), align=("LEFT", "LEFT", "RIGHT", "RIGHT"))

    # ── SECTION 6: Cleaning steps ──────────────────────────────────────────────
    _section_title(pdf, 5, "Cleaning Steps Taken")
    if cleaning_log:
        for i, step in enumerate(cleaning_log, start=1):
            pdf.set_font("DejaVu", "B", 9)
            pdf.set_text_color(*BLUE)
            pdf.cell(8, 5, f"{i}.")
            pdf.set_font("DejaVu", "", 9)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(0, 5, step, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(0.5)
    else:
        _paragraph(pdf, "No cleaning steps were taken. The data is exactly as uploaded.",
                   size=9, color=GRAY)

    # fpdf2 returns a bytearray; Streamlit's download button wants bytes
    return bytes(pdf.output())


# ── END OF FILE ────────────────────────────────────────────────────────────────
