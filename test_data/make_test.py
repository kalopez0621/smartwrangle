# Builds the SmartWrangle messy test spreadsheet.
# Problems planted on purpose: blanks, exact duplicate rows, numbers stored as text,
# currency text, bad and mixed-format dates, extra spaces, inconsistent capitalization,
# an extreme outlier, and a column with only one value.
import pandas as pd, numpy as np
rng = np.random.default_rng(42)
n = 200
dates = pd.date_range("2024-01-01", periods=n, freq="2D")
df = pd.DataFrame({
    "Order ID": [f"ORD-{1000+i}" for i in range(n)],
    "Order Date": [d.strftime("%Y-%m-%d") if i % 7 else d.strftime("%m/%d/%Y") for i, d in enumerate(dates)],
    "Customer": rng.choice(["  Maria Lopez", "maria lopez", "John Smith  ", "JOHN SMITH", "Ana Diaz", "ana diaz "], n),
    "Region": rng.choice(["North", "South", "East", "West", "north ", "SOUTH"], n),
    "Product": rng.choice(["Laptop", "Monitor", "Keyboard", "Mouse", "Desk Chair"], n),
    "Amount": [f"${v:,.2f}" for v in rng.gamma(2.0, 250, n)],
    "Quantity": rng.integers(1, 12, n).astype(str),
    "Unit Price": rng.normal(80, 15, n).round(2),
    "Rating": rng.integers(1, 6, n).astype(float),
    "Notes": rng.choice(["", "rush order", "  gift wrap ", "call first", "none"], n),
    "Store": ["Main Street"] * n,
})
# Bad dates and blanks in dates
for i, v in [(5, "not a date"), (17, "13/45/2024"), (33, "2024-02-30"), (48, "yesterday")]:
    df.loc[i, "Order Date"] = v
df.loc[[9, 60, 120], "Order Date"] = np.nan
# Accounting negatives and blanks in Amount
df.loc[[11, 90], "Amount"] = ["($1,250.00)", "($75.50)"]
df.loc[[14, 70, 140, 141], "Amount"] = np.nan
# Words and blanks inside a number column
df.loc[[21, 77], "Quantity"] = ["seven", "N/A"]
df.loc[[25, 99, 150], "Quantity"] = np.nan
# Extreme outlier and blanks in a real number column
df.loc[rng.choice(n, 25, replace=False), "Unit Price"] = np.nan
df.loc[rng.choice(n, 40, replace=False), "Rating"] = np.nan
df.loc[rng.choice(n, 30, replace=False), "Region"] = np.nan
# Outlier placed last so the random blanks above cannot overwrite it
df.loc[40, "Unit Price"] = 99999.0
# Exact duplicate rows (8 copies)
df = pd.concat([df, df.iloc[[2, 3, 4, 50, 51, 100, 101, 180]]], ignore_index=True)
# Extra spaces in a header too
df = df.rename(columns={"Product": " Product "})
df.to_csv("smartwrangle_messy_test.csv", index=False)
df.to_excel("smartwrangle_messy_test.xlsx", index=False)
print(df.shape); print(df.head(6).to_string())
