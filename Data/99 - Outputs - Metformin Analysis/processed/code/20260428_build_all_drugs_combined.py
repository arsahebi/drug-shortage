# %%
# =============================================================================
# 0. IMPORTS & PATHS
# =============================================================================
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage"
)

VALISURE_FILE = BASE_DIR / "Data/08 - Valisure/raw/Testing Data_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx"
NDC_FEI_FILE  = BASE_DIR / "Data/17 - NDC, FEI Mapping/ndc_fei_from_labels.csv"
IQVIA_FILE    = BASE_DIR / "Data/04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)/processed/2025-12-18-iqvia_with_sdud_nadac.cleaned.csv"
REDICA_FILE   = BASE_DIR / "Data/07 - Redica/processed/SITE_RED_FLAG_EVENTS.xlsx"
FAERS_FILE    = BASE_DIR / "Data/15 - FDA - Adverse Event/processed/faers_all_drugs_anda_linked.csv"

OUT_DIR  = BASE_DIR / "Data/99 - Outputs - Graphs/processed"
OUT_CSV  = OUT_DIR / "QAs_all_drugs_combined_v01.csv"
OUT_XLSX = OUT_DIR / "QAs_all_drugs_combined_v01.xlsx"

DRUG_SHEETS = [
    "Metformin", "Lisinopril", "Potassium chloride", "Metoprolol",
    "Magnesium sulfate", "Tacrolimus", "Ampicillin", "Atorvastatin",
    "Calcium Gluconate", "Vancomycin", "Bupropion", "Metronidazole", "Pantoprazole",
]
SCORE_ORDER = [0.0, 1.5, 3.5]   # NAI, VAI, OAI


# %%
# =============================================================================
# 1. NDC HELPER FUNCTIONS  (copied from 20260408-MetforminJAMAGraphs.py)
# =============================================================================
def _digits(x) -> str:
    return "" if pd.isna(x) else re.sub(r"\D", "", str(x))

def ndc10_to_ndc11(s10: str) -> str:
    """10-digit NDC (no hyphens) → 11-digit NDC11."""
    if len(s10) != 10:
        return ""
    if s10[0] == "0":
        return s10[:4].zfill(5) + s10[4:8] + s10[8:]
    ndc11 = s10[:5] + s10[5:8].zfill(4) + s10[8:]
    if len(ndc11) != 11:
        ndc11 = s10[:5] + s10[5:9] + s10[9:].zfill(2)
    return ndc11 if len(ndc11) == 11 else ""

def ndc_to_ndc11(x) -> str:
    """Parse any NDC representation (hyphenated or digit string) → 11-digit string, or ''."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Already 11 digits
    d = _digits(s)
    if len(d) == 11:
        return d
    # Hyphenated: zero-pad each segment to 5-4-2
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 3:
            ndc11 = _digits(parts[0]).zfill(5) + _digits(parts[1]).zfill(4) + _digits(parts[2]).zfill(2)
            return ndc11 if len(ndc11) == 11 else ""
    # 10-digit bare
    if len(d) == 10:
        return ndc10_to_ndc11(d)
    return ""

def snap_score(x) -> float:
    """Snap a raw inspection score to the nearest of {0.0=NAI, 1.5=VAI, 3.5=OAI}."""
    if pd.isna(x):
        return np.nan
    arr = np.array(SCORE_ORDER, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(x)))])


# %%
# =============================================================================
# 2. VALISURE – load all 13 drug sheets
# =============================================================================
# Two-row header: use header=1 (metric names are in row index 1).
# "Elemental" and "Carcinogens" columns are blank section-header placeholders → drop.
# Column names are standardized to consistent Valisure metric names.
# Sweep year is inferred from Sample ID prefix (e.g. "23-" → 2023).

RENAME_QUALITY = {
    # carcinogen impurities
    "LCMS DMF (ng/day)":  "DMF (ng/DAY) Valisure",
    "LCMS NDMA (ng/day)": "NDMA (ng/DAY) Valisure",
    "LCMS NMBA (ng/day)": "NMBA (ng/DAY) Valisure",
    "LCMS NDEA (ng/day)": "NDEA (ng/DAY) Valisure",
    "LCMS NMPA (ng/day)": "NMPA (ng/DAY) Valisure",
    # elemental impurities (ug/day as reported by Valisure)
    "As (ug/day)": "As (ug/day) Valisure",
    "Pb (ug/day)": "Pb (ug/day) Valisure",
    "Tl (ug/day)": "Tl (ug/day) Valisure",
    "Cd (ug/day)": "Cd (ug/day) Valisure",
    "Hg (ug/day)": "Hg (ug/day) Valisure",
    "Li (ug/day)": "Li (ug/day) Valisure",
    "Cr (ug/day)": "Cr (ug/day) Valisure",
    "Ni (ug/day)": "Ni (ug/day) Valisure",
}
DROP_COLS = {"Elemental", "Carcinogens", "Dissolution"}   # blank section-header placeholders

sheets = []
xf = pd.ExcelFile(VALISURE_FILE)
for drug in DRUG_SHEETS:
    sheet = next((s for s in xf.sheet_names if s.lower() == drug.lower()), None)
    if sheet is None:
        print(f"WARNING: sheet '{drug}' not found"); continue

    df = pd.read_excel(VALISURE_FILE, sheet_name=sheet, header=1)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS or c.startswith("Unnamed")])
    df = df.rename(columns=RENAME_QUALITY)
    df["Drug"] = drug

    # Set a fixed year
    df["Year"] = 2024


    # Convert "<LOQ" and other non-numeric text in metric columns to NaN
    for col in RENAME_QUALITY.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sheets.append(df)
    print(f"  {drug}: {len(df)} rows, year={sorted(df['Year'].dropna().unique().tolist())}")

valisure = pd.concat(sheets, ignore_index=True, sort=False)
print(f"\nValisure combined: {len(valisure)} rows, {valisure['Drug'].nunique()} drugs")


# %%
# =============================================================================
# 3. NDC NORMALIZATION
# =============================================================================
# NDC11 = 11-digit string.  NDC9 = first 9 digits (labeler + product, no package).
# NDC9 is the join key to the FEI mapping file (which stores labeler+product only).

valisure["NDC11"] = valisure["NDC"].apply(ndc_to_ndc11).replace("", np.nan)
valisure["NDC9"]  = valisure["NDC11"].apply(lambda x: str(x)[:9] if pd.notna(x) else np.nan)
valisure["NDC8"]  = valisure["NDC11"].apply(lambda x: str(x)[:8] if pd.notna(x) else np.nan)
valisure["NDC_542"] = valisure["NDC11"].apply(
    lambda x: f"{x[:5]}-{x[5:9]}-{x[9:]}" if pd.notna(x) else np.nan
)

print(f"NDC11 coverage: {valisure['NDC11'].notna().sum()} / {len(valisure)}")


# %%
# =============================================================================
# 4. NDC → FEI + COUNTRY  (primary mapping file)
# =============================================================================
# manufacture_ndc in this file is 9-digit (labeler+product, no package segment).
# → normalize to bare 9 digits and join on NDC9.
# Country is parsed from the parenthetical country code at the end of ADDRESS,
# e.g. "…India (IND)" → CountryCode = IND.

ndc_fei_raw = pd.read_csv(NDC_FEI_FILE, dtype=str, low_memory=False)

ndc_fei_raw["NDC9"] = ndc_fei_raw["manufacture_ndc"].apply(_digits)
ndc_fei_raw = ndc_fei_raw[ndc_fei_raw["NDC9"].str.len() == 9].copy()

def parse_country(addr):
    if pd.isna(addr): return ("OTHER", "Other")
    m = re.search(r"\(([A-Z]{2,3})\)\s*$", str(addr))
    if not m: return ("OTHER", "Other")
    code = m.group(1)
    names = {"IND": "India", "CHN": "China", "USA": "United States",
             "BGD": "Bangladesh", "CAN": "Canada", "DEU": "Germany",
             "GBR": "United Kingdom", "AUS": "Australia", "IRL": "Ireland",
             "JPN": "Japan", "KOR": "South Korea"}
    return (code, names.get(code, code))

ndc_fei_raw["CountryCode"] = ndc_fei_raw["ADDRESS"].apply(lambda a: parse_country(a)[0])
ndc_fei_raw["CountryName"] = ndc_fei_raw["ADDRESS"].apply(lambda a: parse_country(a)[1])

# Deduplicate on NDC9 (keep first occurrence)
ndc_map = (
    ndc_fei_raw[["NDC9", "FEI_NUMBER", "FIRM_NAME", "CountryCode", "CountryName"]]
    .drop_duplicates("NDC9", keep="first")
    .rename(columns={"FEI_NUMBER": "FEI", "FIRM_NAME": "Firm_from_map"})
)

valisure = valisure.merge(ndc_map, on="NDC9", how="left")
valisure["Firm"] = valisure["Labeler"].fillna(valisure["Firm_from_map"])

print(f"FEI linked: {valisure['FEI'].notna().sum()} / {len(valisure)}")
print(valisure["CountryCode"].value_counts())


# %%
# =============================================================================
# 5. MARKET DATA  (annual IQVIA volume + NADAC price)
# =============================================================================
# Monthly panel → aggregate to calendar year.
# NOTE: current panel file contains only Metformin; other drugs will be NaN
#       until the multi-drug panel is built.

iqvia_raw = pd.read_csv(IQVIA_FILE, dtype={"ndc11": str}, low_memory=False)
iqvia_raw["NDC11"] = iqvia_raw["ndc11"].apply(ndc_to_ndc11).replace("", np.nan)
iqvia_raw["Year"]  = pd.to_datetime(iqvia_raw["date"], errors="coerce").dt.year

market = (
    iqvia_raw
    .groupby(["NDC11", "Year"], as_index=False)
    .agg(
        iqvia_extended_units        =("iqvia_extended_units",       "sum"),
        iqvia_trx                   =("iqvia_trx",                  "sum"),
        nadac_price                 =("nadac_price",                "mean"),
        sdud_price_total_per_unit   =("sdud_price_total_per_unit",  "mean"),
        sdud_num_prescriptions      =("sdud_num_prescriptions",     "sum"),
    )
)

valisure["Year_int"] = pd.to_numeric(valisure["Year"], errors="coerce").astype("Int64")
market["Year"]       = market["Year"].astype("Int64")

valisure = valisure.merge(
    market.rename(columns={"Year": "Year_int"}),
    on=["NDC11", "Year_int"], how="left"
)
print(f"Market data joined: {valisure['iqvia_extended_units'].notna().sum()} / {len(valisure)}")


# %%
# =============================================================================
# 6. REDICA INSPECTION EVENTS
# =============================================================================
# Red Flag Score 1 → snap to {0.0=NAI, 1.5=VAI, 3.5=OAI}.

redica_raw = pd.read_excel(REDICA_FILE, sheet_name=0, dtype=str)
redica_raw.columns = [str(c).strip() for c in redica_raw.columns]

redica_raw = redica_raw.rename(columns={
    next(c for c in redica_raw.columns if "fei" in c.lower()): "FEI",
    next(c for c in redica_raw.columns if "event start" in c.lower()): "Event Start Date",
    next(c for c in redica_raw.columns if "score" in c.lower()): "Score_raw",
})
redica_raw["FEI"]             = redica_raw["FEI"].astype(str).str.strip()
redica_raw["Event Start Date"] = pd.to_datetime(redica_raw["Event Start Date"], errors="coerce").astype("datetime64[us]")
redica_raw["Score_raw"]        = pd.to_numeric(redica_raw["Score_raw"], errors="coerce").astype("float64")
redica_raw["Score"]            = redica_raw["Score_raw"].apply(snap_score)

events = (
    redica_raw[["FEI", "Event Start Date", "Score"]]
    .dropna(subset=["FEI", "Event Start Date"])
    .drop_duplicates(subset=["FEI", "Event Start Date"])
    .sort_values(["FEI", "Event Start Date"])
    .reset_index(drop=True)
)
print(f"Redica events: {len(events)} rows, {events['FEI'].nunique()} FEIs")
print(events["Score"].value_counts().sort_index())


# %%
# =============================================================================
# 7. ATTACH PRIOR INSPECTION SCORE  (per-FEI merge_asof)
# =============================================================================
# For each sample row, find the most recent inspection at its FEI
# with Event Start Date <= Dec 31 of the sweep year.

valisure["TestDate"] = pd.to_datetime(
    valisure["Year_int"].astype(str).str.replace("<NA>", "NaT") + "-12-31",
    format="%Y-%m-%d", errors="coerce"
).astype("datetime64[us]")

events_by_fei = {fei: g.sort_values("Event Start Date").reset_index(drop=True)
                 for fei, g in events.groupby("FEI")}

prior_score = []
prior_date  = []
for _, row in valisure.iterrows():
    fei = str(row["FEI"]) if pd.notna(row["FEI"]) else None
    td  = row["TestDate"]
    ev  = events_by_fei.get(fei)
    if ev is None or pd.isna(td):
        prior_score.append(np.nan); prior_date.append(pd.NaT); continue
    past = ev[ev["Event Start Date"] <= td]
    if past.empty:
        prior_score.append(np.nan); prior_date.append(pd.NaT)
    else:
        last = past.iloc[-1]
        prior_score.append(last["Score"]); prior_date.append(last["Event Start Date"])

valisure["Score"]            = prior_score
valisure["Event Score"]      = valisure["Score"]
valisure["Event Start Date"] = prior_date
valisure["EventYear"]        = pd.to_datetime(valisure["Event Start Date"], errors="coerce").dt.year
valisure["NAI"] = np.where(valisure["Score"].notna(), (valisure["Score"] == 0.0).astype(float), np.nan)
valisure["VAI"] = np.where(valisure["Score"].notna(), (valisure["Score"] == 1.5).astype(float), np.nan)
valisure["OAI"] = np.where(valisure["Score"].notna(), (valisure["Score"] == 3.5).astype(float), np.nan)

print(f"Inspection score attached: {valisure['Score'].notna().sum()} / {len(valisure)}")


# %%
# =============================================================================
# 8. FAERS ADVERSE EVENTS
# =============================================================================
# Join on ANDA number (Application Number in Valisure → appl_no in FAERS).
# Compute per-ANDA: faers_total_reports, faers_serious_reports, faers_serious_event_rate.

faers_raw = pd.read_csv(FAERS_FILE, dtype=str, low_memory=False)

# Normalize ANDA: strip "ANDA"/"NDA" prefix and leading zeros
def norm_anda(x):
    if pd.isna(x): return None
    s = re.sub(r"^(ANDA|NDA|BLA)\s*", "", str(x).strip().upper())
    s = re.sub(r"\D", "", s).lstrip("0")
    return s or None

faers_raw["appl_no_norm"]    = faers_raw["appl_no"].apply(norm_anda)
faers_raw["serious_flag_bool"] = faers_raw["serious_flag"].str.strip().str.upper().isin(
    ["TRUE", "1", "YES", "Y", "T"]
)

faers_agg = (
    faers_raw.dropna(subset=["appl_no_norm"])
    .groupby("appl_no_norm", as_index=False)
    .agg(
        faers_total_reports   =("appl_no_norm",      "size"),
        faers_serious_reports =("serious_flag_bool", "sum"),
    )
)
faers_agg["faers_serious_event_rate"] = (
    faers_agg["faers_serious_reports"] / faers_agg["faers_total_reports"]
)

valisure["appl_no_norm"] = valisure["Application Number"].apply(norm_anda)
valisure = valisure.merge(faers_agg, on="appl_no_norm", how="left")

print(f"FAERS joined: {valisure['faers_serious_event_rate'].notna().sum()} / {len(valisure)}")


# %%
# =============================================================================
# 9. FINAL COLUMN ORDER & CLEANUP
# =============================================================================
valisure["Year"] = valisure["Year_int"]

QUALITY_COLS = [
    "DoD Drug Score",
    "DMF (ng/DAY) Valisure", "NDMA (ng/DAY) Valisure",
    "NMBA (ng/DAY) Valisure", "NDEA (ng/DAY) Valisure", "NMPA (ng/DAY) Valisure",
    "Dissolution", "Difference Factor",
    "As (ug/day) Valisure", "Pb (ug/day) Valisure", "Tl (ug/day) Valisure",
    "Cd (ug/day) Valisure", "Hg (ug/day) Valisure",
    "Li (ug/day) Valisure", "Cr (ug/day) Valisure", "Ni (ug/day) Valisure",
]

ORDERED_COLS = (
    ["Drug", "Firm", "Year", "NDC", "NDC_542", "NDC8", "NDC11", "Strength",
     "FEI", "CountryCode", "CountryName"]
    + [c for c in QUALITY_COLS if c in valisure.columns]
    + [c for c in ["iqvia_extended_units", "iqvia_trx", "nadac_price",
                   "sdud_price_total_per_unit", "sdud_num_prescriptions"] if c in valisure.columns]
    + [c for c in ["Event Start Date", "EventYear", "Score", "Event Score",
                   "NAI", "VAI", "OAI"] if c in valisure.columns]
    + [c for c in ["faers_total_reports", "faers_serious_reports",
                   "faers_serious_event_rate"] if c in valisure.columns]
    # append any remaining columns not yet listed
    + [c for c in valisure.columns if c not in (
        set(["Drug", "Firm", "Year", "NDC", "NDC_542", "NDC8", "NDC11", "Strength",
             "FEI", "CountryCode", "CountryName",
             "iqvia_extended_units", "iqvia_trx", "nadac_price",
             "sdud_price_total_per_unit", "sdud_num_prescriptions",
             "Event Start Date", "EventYear", "Score", "Event Score", "NAI", "VAI", "OAI",
             "faers_total_reports", "faers_serious_reports", "faers_serious_event_rate"])
        | set(QUALITY_COLS)
    )]
)
# deduplicate while preserving order
seen = set(); ORDERED_COLS = [c for c in ORDERED_COLS if not (c in seen or seen.add(c))]

combined = valisure[[c for c in ORDERED_COLS if c in valisure.columns]].copy()
print(f"\nFinal dataset: {len(combined)} rows × {len(combined.columns)} columns")


# %%
# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================
print(f"\n{'Drug':<22} {'N_NDC':>6} {'FEI%':>6}  {'IND':>5} {'CHN':>5} {'USA':>5}  "
      f"{'Mkt':>5} {'Insp':>5} {'FAERS':>6}")
print("-" * 75)
for drug, g in combined.groupby("Drug", sort=True):
    uq       = g.drop_duplicates("NDC11")
    n_ndc    = uq["NDC11"].notna().sum()
    n_fei    = uq["FEI"].notna().sum()
    cc       = uq["CountryCode"]
    print(
        f"{drug:<22} {n_ndc:>6} {100*n_fei/max(n_ndc,1):>5.1f}%  "
        f"{(cc=='IND').sum():>5} {(cc=='CHN').sum():>5} {(cc=='USA').sum():>5}  "
        f"{g['iqvia_extended_units'].notna().sum():>5} "
        f"{g['Score'].notna().sum():>5} "
        f"{g['faers_serious_event_rate'].notna().sum():>6}"
    )


# %%
# =============================================================================
# 11. SAVE
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

combined.to_csv(OUT_CSV, index=False)
print(f"Saved CSV:   {OUT_CSV}")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    combined.to_excel(writer, sheet_name="All_Drugs", index=False)
    for drug in sorted(combined["Drug"].unique()):
        combined[combined["Drug"] == drug].to_excel(writer, sheet_name=drug[:31], index=False)

print(f"Saved Excel: {OUT_XLSX}")
