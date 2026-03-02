# SmartWrangle 2.0

**AI-powered data wrangling assistant — upload any dataset and understand it instantly.**

[![Live App](https://img.shields.io/badge/Live%20App-smartwrangle.streamlit.app-3b82f6?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartwrangle.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-1e293b?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## What It Does

SmartWrangle takes any CSV or Excel file and gives you three things immediately:

- **A plain-English explanation** of what is in your data — column types, date ranges, data quality score, and any problems found
- **Auto-generated charts** with AI-written headlines that tell you what each visualization actually means
- **One-click cleaning tools** with undo support — remove duplicates, fill missing values, fix formats, and download your cleaned dataset

No data science knowledge required. Upload a file, read the findings, take action.

---

## Live Demo

**[smartwrangle.streamlit.app](https://smartwrangle.streamlit.app)**

Upload any CSV or Excel file to try it. Test datasets that work well:
- Any sales or transaction dataset
- HR or employee records
- Survey results
- Public datasets from Kaggle

---

## Screenshots

### Understand Tab — Quality Score and Data Findings
The app scores your dataset out of 100 and explains every finding in plain English.

### Discover Tab — Auto-Generated Charts
SmartWrangle detects what type of data you have and generates the most relevant charts automatically, with a headline for each one explaining what it shows.

### Clean & Export Tab — Guided Cleaning
Suggested actions tell you what to fix. Every tool shows a preview before making changes. Full undo support on every action.

---

## Features

### Understand Tab
- Dataset quality score (0–100) with grade and explanation
- Automatic column type classification (numeric, categorical, date, identifier, financial)
- Plain-English data findings — missing values, duplicates, outliers, skewness
- Date range detection
- Column inventory with type badges
- Data preview

### Discover Tab
- Auto-generated charts — time trends, correlations, distributions, categorical breakdowns
- AI-written headline for every chart explaining what it means
- Smart handling of high-cardinality columns (top 10, horizontal bars)
- Log scale suggestion for skewed distributions
- Custom chart builder (bar, scatter, line, box, histogram)

### Clean & Export Tab
- Suggested actions — SmartWrangle detects issues and recommends fixes
- **Duplicate preview** — see all duplicate rows grouped side-by-side before removing anything
- Missing value strategies — median, mean, mode, or constant fill
- Date column protection — prevents filling date columns with non-date values
- Currency column converter — converts formatted strings like `$1,234.56` or `(2,350.00)` to numeric
- Drop columns, rename, type conversion
- Full undo system — every action can be reversed
- Download cleaned CSV
- Download plain-English summary report

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Streamlit |
| Data | Pandas 2.x, NumPy |
| Visualization | Plotly |
| ML / Preprocessing | scikit-learn |
| Statistical Analysis | statsmodels (VIF) |
| File Support | openpyxl (Excel), pyarrow (Arrow serialization) |
| Language | Python 3.12 |

---

## Project Structure

```
SmartWrangle2/
│
├── app.py                  # Main entry point — routing, sidebar, session state
├── requirements.txt        # All dependencies
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration
│
├── engine/                 # All analysis and transformation logic
│   ├── detector.py         # Column type classification
│   ├── quality.py          # Dataset quality scoring
│   ├── insights.py         # Chart generation and AI headlines
│   └── cleaner.py          # Cleaning and transformation operations
│
├── tabs/                   # Tab render functions
│   ├── understand.py       # Understand tab
│   ├── discover.py         # Discover tab
│   └── clean_export.py     # Clean & Export tab
│
└── utils/                  # Shared utilities
    ├── sanitizer.py        # Column name cleaning, dtype safety
    └── exporter.py         # Download and report generation
```

---

## Run Locally

**Requirements:** Python 3.10 or higher

```bash
# Clone the repository
git clone https://github.com/kalopez0621/smartwrangle.git
cd smartwrangle

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Architecture Notes

**Engine layer** (`engine/`) contains all analysis and transformation logic with no Streamlit dependencies. Every cleaning function returns `(new_df, log_message)` — a modified DataFrame and a plain-English description of what was done. The original uploaded file is never modified.

**Session state** persists the working dataset, undo history, cleaning log, column types, and quality score across Streamlit reruns. Column types are refreshed after every cleaning action so suggestions stay accurate.

**General purpose** — no hardcoded column names anywhere. The app works on any structured dataset by using pattern matching on column names and statistical properties to classify and analyze data.

---

## About

Built as a portfolio project for a dual bachelor's program in Data Analytics and Applied AI.

The goal was to build a production-quality tool that demonstrates the full analytics workflow — from raw data assessment through cleaning and export — while making every step accessible to someone without a data science background.

---

*SmartWrangle 2.0 — Built with Python and Streamlit*