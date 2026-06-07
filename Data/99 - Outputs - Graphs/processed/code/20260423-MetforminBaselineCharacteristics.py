# %%
"""
20260423-MetforminBaselineCharacteristics.py
============================================
Generates Table 1: Baseline characteristics of metformin NDC–year observations,
by country of manufacture.

Unit of analysis : NDC–year (one row per unique NDC × test sweep year)
Columns          : Overall | India (IND) | China (CHN) | USA | Not linked
Output           : Console + Excel file  (same directory as this script)

Run as a Jupyter notebook cell (%%  markers) or plain Python.
"""

# %%
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (same roots as 20260408-MetforminJAMAGraphs.py)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/06 - Metformin Data/Derived"
)
DF_FILE = BASE_DIR / "Q&As1234_v8_v02.xlsx"

OUT_DIR = Path(__file__).parent if "__file__" in dir() else Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/99 - Outputs - Graphs/processed/code"
)
OUT_XLSX = OUT_DIR.parent / "20260423_Table1_BaselineCharacteristics.xlsx"

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DMF_COL  = "DMF (ng/DAY) Valisure"
NDMA_COL = "NDMA (ng/DAY) Valisure"
DISS_COL = "Difference Factor"

SCORE_ORDER  = [0.0, 1.5, 3.5]
SCORE_LABELS = {0.0: "NAI", 1.5: "VAI", 3.5: "OAI"}

# Country groups shown in table columns
# "Not linked" = CountryCode is NaN OR not in the three primary countries
PRIMARY_COUNTRIES = ["IND", "CHN", "USA"]
COUNTRY_LABELS    = {"IND": "India", "CHN": "China", "USA": "USA"}


# ──────────────────────────────────────────────────────────────────────────────
# NDC NORMALISATION (verbatim from 20260408 script)
# ──────────────────────────────────────────────────────────────────────────────
def _digits_only(x) -> str:
    if pd.isna(x):
        return ""
    return "".join(ch for ch in str(x).strip() if ch.isdigit())


def ndc10_to_ndc11_and_542(ndc10_digits: str):
    s = ndc10_digits
    if len(s) != 10:
        return (np.nan, np.nan)
    if s[0] == "0":
        a, b, c = s[:4], s[4:8], s[8:]
        ndc11 = a.zfill(5) + b + c
    else:
        a, b, c = s[:5], s[5:8], s[8:]
        ndc11 = a + b.zfill(4) + c
        if len(ndc11) != 11:
            a, b, c = s[:5], s[5:9], s[9:]
            ndc11 = a + b + c.zfill(2)
    if len(ndc11) != 11:
        return (np.nan, np.nan)
    ndc_542 = f"{ndc11[:5]}-{ndc11[5:9]}-{ndc11[9:]}"
    return (ndc11, ndc_542)


def ndc542_to_ndc11(ndc_542: str):
    if pd.isna(ndc_542):
        return np.nan
    s = str(ndc_542).strip().replace(" ", "")
    parts = s.split("-")
    if len(parts) != 3:
        return np.nan
    a = _digits_only(parts[0])
    b = _digits_only(parts[1])
    c = _digits_only(parts[2])
    if not a or not b or not c:
        return np.nan
    return a.zfill(5) + b.zfill(4) + c.zfill(2)


def ensure_ndc542_and_ndc11(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    candidate_10 = None
    for c in ["NDC", "ndc", "product_ndc", "Product NDC", "productNDC"]:
        if c in out.columns:
            candidate_10 = c
            break

    if "NDC11" in out.columns:
        out["NDC11"] = out["NDC11"].astype(str).str.replace(r"\D", "", regex=True)
        out.loc[out["NDC11"].str.len() != 11, "NDC11"] = np.nan
    else:
        out["NDC11"] = np.nan

    if "NDC_542" in out.columns:
        out["NDC11"] = out["NDC11"].fillna(out["NDC_542"].apply(ndc542_to_ndc11))

    if candidate_10 is not None:
        raw10 = out[candidate_10].apply(_digits_only)
        raw10 = raw10.where(raw10.str.len() == 10, "")
        ndc11_542 = raw10.apply(
            lambda s: ndc10_to_ndc11_and_542(s) if s else (np.nan, np.nan)
        )
        out["NDC11"] = out["NDC11"].fillna(ndc11_542.apply(lambda t: t[0]))
        if "NDC_542" not in out.columns:
            out["NDC_542"] = ndc11_542.apply(lambda t: t[1])
        else:
            out["NDC_542"] = out["NDC_542"].fillna(ndc11_542.apply(lambda t: t[1]))

    if "NDC_542" not in out.columns:
        out["NDC_542"] = np.nan
    out.loc[out["NDC_542"].isna() & out["NDC11"].notna(), "NDC_542"] = (
        out.loc[out["NDC_542"].isna() & out["NDC11"].notna(), "NDC11"]
        .apply(lambda s: f"{str(s)[:5]}-{str(s)[5:9]}-{str(s)[9:]}")
    )
    return out


def pick_price_series(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(np.nan, index=df.index)
    if "nadac_price" in df.columns:
        s = df["nadac_price"].copy()
    if "sdud_price_total_per_unit" in df.columns:
        s = s.fillna(df["sdud_price_total_per_unit"])
    return s


# ──────────────────────────────────────────────────────────────────────────────
# BUILD NDC–YEAR TABLE  (verbatim logic from 20260408 script)
# ──────────────────────────────────────────────────────────────────────────────
def build_ndc_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw rows to one row per (NDC11, Year).
    Key rules:
      • NDMA in 2024 → NaN  (not measured that year)
      • Dissolution in 2020/2022 → NaN  (only measured in 2024)
      • PriorScore = most recent inspection score ≤ Dec 31 of test year,
        snapped to {0.0=NAI, 1.5=VAI, 3.5=OAI}
    """
    d = df.copy()
    if "Event Start Date" in d.columns:
        d["Event Start Date"] = pd.to_datetime(d["Event Start Date"], errors="coerce")

    if "ndc11" in d.columns and "NDC11" not in d.columns:
        d = d.rename(columns={"ndc11": "NDC11"})
    d = ensure_ndc542_and_ndc11(d)
    d = d.dropna(subset=["NDC11"]).copy()

    d["Year"] = pd.to_numeric(d.get("Year"), errors="coerce")
    d = d.dropna(subset=["Year"]).copy()
    d["Year"] = d["Year"].astype(int)

    d["TestDate"] = pd.to_datetime(d["Year"].astype(str) + "-12-31")
    d["price"]    = pd.to_numeric(pick_price_series(d), errors="coerce")
    d["volume"]   = pd.to_numeric(d.get("iqvia_extended_units"), errors="coerce")

    d["DMF"]  = pd.to_numeric(d.get(DMF_COL),  errors="coerce")
    d["NDMA"] = pd.to_numeric(d.get(NDMA_COL), errors="coerce")
    d.loc[d["Year"] == 2024, "NDMA"] = np.nan

    diss_source_col = (
        DISS_COL if DISS_COL in d.columns
        else ("Dissolution" if "Dissolution" in d.columns else None)
    )
    d["Dissolution"] = pd.to_numeric(d.get(diss_source_col), errors="coerce")
    d.loc[d["Year"].isin([2020, 2022]), "Dissolution"] = np.nan

    if "Event Score" in d.columns or "Score" in d.columns:
        d["ScoreUsed"] = pd.to_numeric(d.get("Event Score"), errors="coerce")
        d.loc[d["ScoreUsed"].isna(), "ScoreUsed"] = pd.to_numeric(
            d.get("Score"), errors="coerce"
        )
    else:
        d["ScoreUsed"] = np.nan

    keys = ["NDC11", "Year"]

    def prior_score_for_group(g: pd.DataFrame):
        test_date = g["TestDate"].iloc[0]
        gg = g.dropna(subset=["Event Start Date"])
        gg = gg[gg["Event Start Date"] <= test_date]
        if gg.empty:
            return (np.nan, pd.NaT)
        last = gg.sort_values("Event Start Date").iloc[-1]
        return (
            last["ScoreUsed"] if pd.notna(last["ScoreUsed"]) else np.nan,
            last["Event Start Date"],
        )

    agg = (
        d.groupby(keys, as_index=False)
         .agg(
            NDC_542     =("NDC_542",          "first"),
            NDC         =("NDC",              "first"),
            NDC8        =("NDC8",             "first"),
            Strength    =("Strength",         "first"),
            FEI         =("FEI",              "first"),
            Firm        =("Firm",             "first"),
            CountryCode =("CountryCode",      "first"),
            CountryName =("CountryName",      "first"),
            TestDate    =("TestDate",         "first"),
            price       =("price",            "mean"),
            volume      =("volume",           "mean"),
            DMF         =("DMF",              "mean"),
            NDMA        =("NDMA",             "mean"),
            Dissolution =("Dissolution",      "mean"),
            n_event_rows=("Event Start Date", "size"),
         )
    )

    prior_raw, prior_date = [], []
    for _, g in d.groupby(keys, sort=False):
        s, dt = prior_score_for_group(g)
        prior_raw.append(s)
        prior_date.append(dt)

    agg["PriorScore_raw"] = prior_raw
    agg["PriorEventDate"] = prior_date

    def nearest_score(x):
        if pd.isna(x): return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    agg["PriorScore"]     = agg["PriorScore_raw"].apply(nearest_score)
    agg["PriorScore_cat"] = pd.Categorical(
        agg["PriorScore"], categories=SCORE_ORDER, ordered=True
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# COUNTRY GROUP ASSIGNMENT
# ──────────────────────────────────────────────────────────────────────────────
def assign_country_group(cc):
    """Map CountryCode to one of: IND, CHN, USA, or NL (Not Linked)."""
    if pd.isna(cc) or str(cc).strip() not in PRIMARY_COUNTRIES:
        return "NL"
    return str(cc).strip()


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _n_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "–"
    return f"{num} ({100*num/denom:.1f}%)"


def _mean_sd(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "–"
    return f"{s.mean():,.2f} ± {s.std(ddof=1):,.2f}"


def _median_iqr(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "–"
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    return f"{s.median():,.2f} [{p25:,.2f}–{p75:,.2f}]"


def _median_iqr_units(series: pd.Series) -> str:
    """Same as _median_iqr but formats large numbers with comma thousands."""
    s = series.dropna()
    if len(s) == 0:
        return "–"
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    med = s.median()
    return f"{med:,.0f} [{p25:,.0f}–{p75:,.0f}]"


# ──────────────────────────────────────────────────────────────────────────────
# BUILD ONE COLUMN OF TABLE 1
# ──────────────────────────────────────────────────────────────────────────────
def build_column(df_subset: pd.DataFrame, total_ndc_overall: int) -> dict:
    """
    Compute all Table 1 statistics for a subset (one country column or Overall).
    Returns an ordered dict of {row_label: formatted_string}.
    """
    d = df_subset.copy()
    n_obs   = len(d)
    n_ndc   = d["NDC11"].nunique()

    # ── NDC–year observations by test type ───────────────────────────────────
    n_dmf   = int(d["DMF"].notna().sum())
    n_ndma  = int(d["NDMA"].notna().sum())
    n_diss  = int(d["Dissolution"].notna().sum())

    # ── Sweep year counts ────────────────────────────────────────────────────
    year_counts = d["Year"].value_counts().sort_index()
    n_2020 = int(year_counts.get(2020, 0))
    n_2022 = int(year_counts.get(2022, 0))
    n_2024 = int(year_counts.get(2024, 0))

    # ── FDA inspection outcomes ──────────────────────────────────────────────
    # Denominator = observations with a known prior inspection score
    has_prior = d["PriorScore"].notna()
    n_with_prior = int(has_prior.sum())
    n_nai = int((d["PriorScore"] == 0.0).sum())
    n_vai = int((d["PriorScore"] == 1.5).sum())
    n_oai = int((d["PriorScore"] == 3.5).sum())
    n_no_prior = int((~has_prior).sum())

    # ── Quality measures ─────────────────────────────────────────────────────
    dmf_str  = _mean_sd(d["DMF"])
    ndma_str = _mean_sd(d["NDMA"])
    diss_str = _mean_sd(d["Dissolution"])

    # ── Market measures ──────────────────────────────────────────────────────
    price_str  = _median_iqr(d["price"])
    vol_str    = _median_iqr_units(d.loc[d["volume"] > 0, "volume"])

    # ── Assemble rows ─────────────────────────────────────────────────────────
    rows = {}

    # Header counts
    rows["NDC–year observations, n"]     = str(n_obs)
    rows["  DMF tested, n"]              = str(n_dmf)
    rows["  NDMA tested, n"]             = str(n_ndma)
    rows["  Dissolution tested, n"]      = str(n_diss)

    rows["Unique NDCs, n (%)"]           = _n_pct(n_ndc, total_ndc_overall)

    # Sweep year distribution
    rows["Sweep year, n"]                = ""
    rows["  2020"]                       = str(n_2020)
    rows["  2022"]                       = str(n_2022)
    rows["  2024"]                       = str(n_2024)

    # Inspection outcomes (% of NDC–year obs with a known prior inspection)
    rows[f"FDA inspection outcome (n={n_with_prior} with prior), n (%)"] = ""
    rows["  NAI (No Action Indicated)"]  = _n_pct(n_nai, n_with_prior)
    rows["  VAI (Voluntary Action)"]     = _n_pct(n_vai, n_with_prior)
    rows["  OAI (Official Action)"]      = _n_pct(n_oai, n_with_prior)
    rows["  No prior inspection"]        = str(n_no_prior)

    # Quality measures
    rows["Quality measures, mean ± SD"]  = ""
    rows["  DMF (ng/day)"]               = dmf_str
    rows["  NDMA (ng/day)"]              = ndma_str
    rows["  Dissolution difference"]     = diss_str

    # Market measures
    rows["Market measures, median [IQR]"] = ""
    rows["  NADAC price ($/unit)"]        = price_str
    rows["  Annual volume (ext. units)"]  = vol_str

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

# Load raw data
print(f"Loading: {DF_FILE}")
df_raw = pd.read_excel(DF_FILE)
print(f"  rows={len(df_raw):,}  cols={len(df_raw.columns)}")

# Build NDC–year table
ndc_year_df = build_ndc_year_table(df_raw)
print(f"\nNDC–year table: {len(ndc_year_df):,} rows | "
      f"{ndc_year_df['NDC11'].nunique()} unique NDC11s | "
      f"Years: {sorted(ndc_year_df['Year'].unique())}")

# Assign country group
ndc_year_df["CountryGroup"] = ndc_year_df["CountryCode"].apply(assign_country_group)

# Overall = IND + CHN + USA only (excludes Not linked)
linked_df = ndc_year_df[ndc_year_df["CountryGroup"].isin(PRIMARY_COUNTRIES)]

subsets = {
    "Overall": linked_df,
    "India":   ndc_year_df[ndc_year_df["CountryGroup"] == "IND"],
    "China":   ndc_year_df[ndc_year_df["CountryGroup"] == "CHN"],
    "USA":     ndc_year_df[ndc_year_df["CountryGroup"] == "USA"],
}

# Total unique NDCs = unique NDCs in linked set (denominator for % calculations)
total_ndc_overall = linked_df["NDC11"].nunique()

# Build each column
cols_data = {}
for col_name, subset in subsets.items():
    cols_data[col_name] = build_column(subset, total_ndc_overall)

# Assemble into a DataFrame (rows = characteristics, cols = country groups)
row_labels = list(cols_data["Overall"].keys())
table = pd.DataFrame(
    {col: [cols_data[col].get(lbl, "") for lbl in row_labels]
     for col in subsets.keys()},
    index=row_labels,
)
table.index.name = "Characteristic"

# ──────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("Table 1. Baseline characteristics of metformin NDC–year observations, by country of manufacture")
print("=" * 110)
col_widths = {c: max(len(c), 20) for c in table.columns}
row_w = 50
header = f"{'Characteristic':<{row_w}}" + "".join(
    f"  {c:^{col_widths[c]}}" for c in table.columns
)
print(header)
print("-" * len(header))
for lbl, row in table.iterrows():
    line = f"{str(lbl):<{row_w}}" + "".join(
        f"  {str(v):^{col_widths[c]}}" for c, v in row.items()
    )
    print(line)
print("=" * 110)

print("\nNotes:")
print("  • Overall = India + China + USA only. NDCs with no linked facility are excluded.")
print("  • NDMA tested only in sweep years 2020 and 2022 (not 2024).")
print("  • Dissolution tested only in sweep year 2024 (not 2020 or 2022).")
print("  • DMF tested in all three sweep years.")
print("  • FDA inspection outcome = most recent inspection ≤ Dec 31 of test year,")
print("    snapped to NAI (score=0), VAI (score≈1.5), OAI (score≈3.5).")
print("  • Price = NADAC (preferred) or SDUD total-per-unit; volume = IQVIA extended units.")
print("  • % for Unique NDCs uses total unique NDCs in linked set as denominator.")
print("  • % for inspection outcomes uses NDC–year obs with a known prior inspection as denominator.")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE TO EXCEL
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSaving to: {OUT_XLSX}")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    # Main table
    table.to_excel(writer, sheet_name="Table1", startrow=2)

    ws = writer.sheets["Table1"]
    ws["A1"] = (
        "Table 1. Baseline characteristics of metformin NDC–year observations, "
        "by country of manufacture"
    )
    ws["A1"].font = __import__("openpyxl.styles", fromlist=["Font"]).Font(bold=True)

    # Notes sheet
    notes = pd.DataFrame({"Notes": [
        "Overall = India + China + USA only. NDCs with no linked facility are excluded.",
        "NDMA tested only in sweep years 2020 and 2022 (not 2024).",
        "Dissolution tested only in sweep year 2024 (not 2020 or 2022).",
        "DMF tested in all three sweep years (2020, 2022, 2024).",
        "FDA inspection outcome = most recent inspection <= Dec 31 of test year, "
        "snapped to NAI/VAI/OAI.",
        "Price = NADAC (preferred) or SDUD total-per-unit.",
        "Volume = IQVIA extended units (annual, positive values only for median/IQR).",
        "% for Unique NDCs uses total unique NDCs in linked set as denominator.",
        "% for inspection outcomes uses NDC-year obs with a known prior inspection as denominator.",
    ]})
    notes.to_excel(writer, sheet_name="Notes", index=False)

print("Done.")

# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC: per-group row counts
# ──────────────────────────────────────────────────────────────────────────────
print("\nDiagnostic: NDC–year obs and unique NDCs per country group")
for grp, subset in subsets.items():
    print(
        f"  {grp:<14}: {len(subset):>4} obs | "
        f"{subset['NDC11'].nunique():>3} unique NDCs | "
        f"Years: {sorted(subset['Year'].unique())}"
    )

# %%
