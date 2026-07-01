# %%
"""
Build Metformin Inspection Panel
=================================
Reconstructs a Sheet1-style panel for ALL 112 Valisure-tested metformin NDCs
(vs the 82 used in the original paper) using:
  - Corrected FEI mapping (Sheet1 + Amir's col H for new NDCs)
  - Redica inspection events (one row per NDC × inspection event)
  - Redica site-level aggregate scores

Output columns
--------------
  FEI_in_old_Redica         — True if FEI appears in METFORMIN_old Redica export (old source)
  FEI_in_new_Redica         — True if FEI appears in Valisure14 Redica export (current source)
  Insp_coverage             — per inspection row: "both" | "old only" | "new only" | null
  NDC, NDC11, NDC8          — three NDC formats
  Firm                      — manufacturer name
  Strength                  — dosage strength
  CountryCode               — 3-letter ISO country code
  FEI                       — facility establishment identifier
  Site Display Name         — Redica site label
  Valisure Years            — which sweeps tested this NDC (e.g. "2020+2022")
  Event Start Date          — FDA inspection start date (from METFORMIN_old; null for new-only events)
  Event End Date            — inspection end date (from both sources)
  EventYear                 — calendar year of inspection
  Classification            — NAI / VAI / OAI
  NAI, VAI, OAI             — binary flags
  483, No 483               — binary flags
  483 critical/major/other  — Redica observation severity counts
  Warning Letter            — binary flag
  [site-level aggregates from Redica Data Availability]
  Total Inspections, FDA Inspections, 483s Issued,
  Total Observations, Warning Letters Issued, Import Alerts Issued
  OAI Rate                  — site-level OAI % computed from event data
  Inspections per Year      — FDA Inspections / years covered

Sources
-------
  Valisure raw : Data/08 - Valisure/raw/Valisure_2024_raw.xlsx
  Sheet1       : Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx
  Amir sheet   : same file, sheet "Amir-Unique NDC from Valisure ("
  Insp history : Data/07 - Redica/processed/valisure_fei_inspection_history.csv
                   (FDA-sole + DQA filtered; combines METFORMIN_old + Valisure14)
  Redica agg   : Data/07 - Redica/processed/redica_all_drugs_combined.csv
                   (site-level aggregate stats only: Total Inspections etc.)
                 Data/07 - Redica/raw/Site List.xlsx
  FDA Insp     : Data/14 - FDA - Inspection/raw/Inspections Details.xlsx
"""

import re
from typing import Optional
import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE      = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage"
)
VAL_RAW      = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
QA_FILE      = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST    = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV   = BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
INSP_HISTORY = BASE / "Data/07 - Redica/processed/valisure_fei_inspection_history.csv"
FDA_INSP     = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"
OUT_DIR      = BASE / "Data/99 - Outputs - Metformin Analysis/processed"
OUT_FILE     = OUT_DIR / "metformin_panel_v1.csv"

# =============================================================================
# HELPERS
# =============================================================================
# Country name → 3-letter ISO code (for FDA Inspection Details)
COUNTRY_MAP = {
    "India": "IND", "China": "CHN", "United States": "USA",
    "Canada": "CAN", "Bangladesh": "BGD", "United Kingdom": "GBR",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Japan": "JPN", "Israel": "ISR", "Ireland": "IRL", "Netherlands": "NLD",
    "Australia": "AUS", "Singapore": "SGP", "South Korea": "KOR",
}


def to_ndc11(x) -> Optional[str]:
    """Normalise any NDC variant to 11-digit NDC11 (no hyphens, 5+4+2)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    parts = [p for p in s.replace(" ", "").split("-") if p]
    if len(parts) == 3:
        lab, prod, pkg = parts
        return lab.zfill(5) + prod.zfill(4) + pkg.zfill(2)[-2:]
    raw = s.replace("-", "").replace(" ", "")
    if len(raw) == 10:
        return raw[:5] + "0" + raw[5:]
    if len(raw) == 11:
        return raw
    return None


def ndc11_to_display(n11: str) -> tuple[str, str, str]:
    """Return (NDC 5-3-2, NDC11 5-4-2 with hyphens, NDC8 5-3) from bare NDC11."""
    lab, prod4, pkg = n11[:5], n11[5:9], n11[9:]
    prod3 = prod4.lstrip("0").zfill(3) if prod4.lstrip("0") else "000"
    ndc    = f"{lab}-{prod3}-{pkg}"
    ndc11h = f"{lab}-{prod4}-{pkg}"
    ndc8   = f"{lab}-{prod3}"
    return ndc, ndc11h, ndc8


def clean_fei(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return None


# =============================================================================
# 1. VALISURE — NDC universe + Firm + Strength per NDC11
# =============================================================================
print("Loading Valisure source data...")
xls_val = pd.ExcelFile(VAL_RAW)
df20 = xls_val.parse("2020 Testing Data")
df22 = xls_val.parse("2022 Testing Data - Actual")
df24 = xls_val.parse("2024 Testing Data", header=1)

# Build NDC11 → {ndc_raw, firm, strength, years} from all three sweeps
val_info: dict[str, dict] = {}

for sweep_year, df, ndc_col, ndc11_col in [
    ("2020", df20, "NDC",   None),
    ("2022", df22, "NDC",   None),
    ("2024", df24, "NDC",   "NDC11"),
]:
    col = ndc11_col if ndc11_col and ndc11_col in df.columns else ndc_col
    ndc_raw_col = ndc_col
    for _, row in df.iterrows():
        if sweep_year == "2024":
            n11 = to_ndc11(row.get("NDC11"))
            ndc_raw = str(row.get("NDC", "")).strip()
        else:
            n11 = to_ndc11(row.get("NDC"))
            ndc_raw = str(row.get("NDC", "")).strip()
        if not n11:
            continue
        if n11 not in val_info:
            val_info[n11] = {
                "ndc_raw": ndc_raw,
                "firm": str(row.get("Firm", "")).strip(),
                "strength": str(row.get(
                    "Strength",
                    row.get("Dosage (mg)", "")
                )).strip(),
                "years": [],
            }
        if sweep_year not in val_info[n11]["years"]:
            val_info[n11]["years"].append(sweep_year)
        # prefer non-blank values
        if not val_info[n11]["firm"] and str(row.get("Firm", "")).strip():
            val_info[n11]["firm"] = str(row["Firm"]).strip()

val_df = pd.DataFrame([
    {
        "ndc11_norm": n11,
        "val_firm": d["firm"],
        "val_strength": d["strength"],
        "val_ndc_raw": d["ndc_raw"],
        "Valisure Years": "+".join(sorted(d["years"])),
    }
    for n11, d in val_info.items()
])

print(f"  Valisure NDC11s: {len(val_df)}")

# =============================================================================
# 2. SHEET1 — FEI + Firm + Strength + CountryCode for the original 88 NDCs
# =============================================================================
print("Loading Sheet1...")
df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1")
df_s1["ndc11_norm"] = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_norm"]   = df_s1["FEI"].apply(clean_fei)

# One record per NDC11 from Sheet1 (FEI, Firm, Country, Strength)
s1_meta = (
    df_s1[["ndc11_norm", "fei_norm", "Firm", "Strength", "CountryCode"]]
    .dropna(subset=["ndc11_norm"])
    .drop_duplicates("ndc11_norm")
    .rename(columns={"fei_norm": "FEI", "Firm": "s1_firm", "Strength": "s1_strength"})
)
sheet1_ndcs = set(s1_meta["ndc11_norm"])
# FEIs that appeared in the original Sheet1 analysis (the 18 original facilities)
sheet1_feis = set(df_s1["FEI"].apply(clean_fei).dropna().unique())

# =============================================================================
# 3. AMIR'S COL H — FEI for new NDCs not in Sheet1
# =============================================================================
print("Loading Amir's sheet...")
raw = pd.read_excel(
    QA_FILE,
    sheet_name="Amir-Unique NDC from Valisure (",
    header=None,
)
raw.columns = ["NDC", "NDC11", "c", "d", "e", "NDC11_F",
               "FEI_G", "Found_FEI_H", "Notes1", "Notes2", "extra"]
df_amir = raw.iloc[1:].reset_index(drop=True)
df_amir["ndc11_norm"] = (
    df_amir["NDC11_F"].apply(to_ndc11)
    .fillna(df_amir["NDC11"].apply(to_ndc11))
)
df_amir["fei_H"] = df_amir["Found_FEI_H"].apply(clean_fei)

# New NDCs (not in Sheet1) that Amir found FEIs for via col H
new_fei_map = (
    df_amir[df_amir["ndc11_norm"].notna() & df_amir["fei_H"].notna()]
    [["ndc11_norm", "fei_H"]]
    .drop_duplicates("ndc11_norm")
    .rename(columns={"fei_H": "FEI"})
)

# =============================================================================
# 4. BUILD MASTER NDC → FEI MAPPING
# =============================================================================
print("Building NDC→FEI mapping...")

# Start with all 112 Valisure NDC11s
ndc_master = val_df.copy()

# Merge Sheet1 metadata (FEI, firm, country)
ndc_master = ndc_master.merge(s1_meta, on="ndc11_norm", how="left")
ndc_master["In Sheet1"] = ndc_master["ndc11_norm"].isin(sheet1_ndcs)

# For NDCs not in Sheet1: fill FEI from Amir col H
no_fei_mask = ndc_master["FEI"].isna()
ndc_master = ndc_master.merge(
    new_fei_map.rename(columns={"FEI": "FEI_H"}),
    on="ndc11_norm", how="left"
)
ndc_master.loc[no_fei_mask, "FEI"] = ndc_master.loc[no_fei_mask, "FEI_H"]
ndc_master.drop(columns=["FEI_H"], inplace=True)

# Firm: prefer Sheet1, fall back to Valisure
ndc_master["Firm"] = ndc_master["s1_firm"].fillna(ndc_master["val_firm"])
ndc_master["Strength"] = ndc_master["s1_strength"].fillna(ndc_master["val_strength"])

# =============================================================================
# 5. COUNTRY CODE FOR NEW NDCs (from FDA Inspection Details by FEI)
# =============================================================================
new_feis = set(ndc_master.loc[~ndc_master["In Sheet1"], "FEI"].dropna())
if new_feis:
    print(f"  Loading FDA Inspection Details for {len(new_feis)} new FEIs...")
    df_fda = pd.read_excel(FDA_INSP)
    df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
    fei_country = (
        df_fda[df_fda["FEI Number"].isin(new_feis)]
        [["FEI Number", "Country/Area"]]
        .drop_duplicates("FEI Number")
        .rename(columns={"FEI Number": "FEI", "Country/Area": "fda_country"})
    )
    fei_country["CountryCode_new"] = fei_country["fda_country"].map(COUNTRY_MAP)
    ndc_master = ndc_master.merge(fei_country[["FEI", "CountryCode_new"]], on="FEI", how="left")
    mask = ndc_master["CountryCode"].isna() & ndc_master["CountryCode_new"].notna()
    ndc_master.loc[mask, "CountryCode"] = ndc_master.loc[mask, "CountryCode_new"]
    ndc_master.drop(columns=["CountryCode_new"], inplace=True)

# For NDCs still missing Firm or CountryCode but with a known FEI,
# fill from another NDC sharing the same FEI (same facility → same country/firm)
fei_firm    = ndc_master.dropna(subset=["FEI", "Firm"]).drop_duplicates("FEI").set_index("FEI")["Firm"]
fei_country = ndc_master.dropna(subset=["FEI", "CountryCode"]).drop_duplicates("FEI").set_index("FEI")["CountryCode"]
for col, lookup in [("Firm", fei_firm), ("CountryCode", fei_country)]:
    mask = ndc_master[col].isna() & ndc_master["FEI"].notna()
    ndc_master.loc[mask, col] = ndc_master.loc[mask, "FEI"].map(lookup)

# =============================================================================
# 6. ADD NDC DISPLAY FORMATS
# =============================================================================
ndc_master[["NDC", "NDC11", "NDC8"]] = pd.DataFrame(
    ndc_master["ndc11_norm"].apply(
        lambda n: ndc11_to_display(n) if isinstance(n, str) else ("", "", "")
    ).tolist(),
    index=ndc_master.index,
)
# Prefer Sheet1 NDC format (hyphenated) when available
s1_ndc = df_s1.drop_duplicates("ndc11_norm")[["ndc11_norm", "NDC", "NDC8"]].copy()
ndc_master = ndc_master.merge(
    s1_ndc.rename(columns={"NDC": "s1_ndc", "NDC8": "s1_ndc8"}),
    on="ndc11_norm", how="left"
)
ndc_master["NDC"]  = ndc_master["s1_ndc"].fillna(ndc_master["NDC"])
ndc_master["NDC8"] = ndc_master["s1_ndc8"].fillna(ndc_master["NDC8"])
ndc_master.drop(columns=["s1_ndc", "s1_ndc8"], inplace=True)

# Clean up clearly invalid firm values ("0", "nan")
ndc_master["Firm"] = ndc_master["Firm"].replace({"0": None, "nan": None, "": None})
ndc_master["Strength"] = ndc_master["Strength"].replace({"0": None, "nan": None, "": None})

print(f"  NDC master rows: {len(ndc_master)}  (all 112 Valisure NDCs)")

# =============================================================================
# 7. INSPECTION HISTORY + REDICA SITE AGGREGATES
# =============================================================================
print("Loading inspection history and Redica aggregate stats...")
df_site = pd.read_excel(SITE_LIST)
df_site["FEI"] = df_site["FEI"].astype(str).str.strip()
site_name_map = df_site.set_index("FEI")["Site Display Name"].to_dict()

# Site-level aggregate totals (Total Inspections, FDA Inspections, etc.)
# from the full Redica export; used for site-profile columns only.
df_redica = pd.read_csv(REDICA_CSV)
df_redica["FEI"] = df_redica["FEI"].astype(str).str.strip()
site_agg = (
    df_redica[["FEI", "Total Inspections", "FDA Inspections", "483s Issued",
               "Total Observations", "Warning Letters Issued", "Import Alerts Issued"]]
    .drop_duplicates("FEI")
)

# Per-inspection history: FDA-sole + DQA filtered, METFORMIN_old and Valisure14 combined.
# "Insp_coverage" and "Source" columns are set during history file construction.
df_hist = pd.read_csv(INSP_HISTORY)
df_hist["FEI"]             = df_hist["FEI"].astype(str).str.strip()
df_hist["Event End Date"]  = pd.to_datetime(df_hist["Event End Date"],  errors="coerce")
df_hist["Event Start Date"]= pd.to_datetime(df_hist["Event Start Date"], errors="coerce")
df_hist["EventYear"]       = df_hist["EventYear"].astype(pd.Int64Dtype())

# OAI Rate and Inspections per Year computed from the filtered history.
hist_stats = (
    df_hist.groupby("FEI").agg(
        oai_count    = ("Classification", lambda x: (x == "OAI").sum()),
        total_events = ("Classification", "count"),
        min_year     = ("EventYear", "min"),
        max_year     = ("EventYear", "max"),
    ).reset_index()
)
hist_stats["OAI Rate"] = hist_stats["oai_count"] / hist_stats["total_events"]
hist_stats["Inspections per Year"] = hist_stats["total_events"] / (
    hist_stats["max_year"] - hist_stats["min_year"] + 1
)

# Per-FEI membership: which Redica source contains each FEI?
_fei_in_old = set(
    df_hist.loc[df_hist["Insp_coverage"].isin(["both", "METFORMIN_old only"]), "FEI"].unique()
)
_fei_in_new = set(
    df_hist.loc[df_hist["Insp_coverage"].isin(["both", "Valisure14 only"]), "FEI"].unique()
)

# =============================================================================
# 8. BUILD PANEL: NDC × inspection event
# =============================================================================
print("Building panel (NDC × inspection event)...")

hist_feis        = set(df_hist["FEI"].unique())
ndc_with_hist    = ndc_master[ndc_master["FEI"].isin(hist_feis)].copy()
ndc_without_hist = ndc_master[~ndc_master["FEI"].isin(hist_feis)].copy()

# Drop history columns that overlap with ndc_master (Firm, CountryCode, FEI_in_old
# are already set from NDC master; prefer those values).
_hist_merge = df_hist.drop(
    columns=[c for c in ["Firm", "CountryCode", "FEI_in_old"] if c in df_hist.columns]
)

# Each NDC × all inspection events for its FEI from the history file.
# History columns: Event Start Date, Event End Date, EventYear, Classification,
# NAI, VAI, OAI, 483, No 483, Warning Letter, 483 critical/major/other,
# Site Display Name, Source, Insp_coverage.
panel_with = ndc_with_hist.merge(_hist_merge, on="FEI", how="left")
panel_with["Insp Source"] = panel_with["Source"]

# NDCs with no history coverage: one blank row each
panel_without = ndc_without_hist.copy()
for col in ["Event Start Date", "Event End Date", "EventYear", "Classification",
            "NAI", "VAI", "OAI", "483", "No 483",
            "483 critical", "483 major", "483 other", "Warning Letter",
            "Site Display Name", "Source", "Insp_coverage"]:
    panel_without[col] = None
panel_without["Insp Source"] = None

panel = pd.concat([panel_with, panel_without], ignore_index=True)

# Fill Site Display Name from Site List where history has no entry (METFORMIN-only FEIs).
panel["Site Display Name"] = panel.apply(
    lambda r: r.get("Site Display Name") or
              site_name_map.get(str(r["FEI"]) if pd.notna(r.get("FEI")) else "", None),
    axis=1,
)

# Merge site aggregate stats and computed rates.
panel = panel.merge(site_agg, on="FEI", how="left")
panel = panel.merge(hist_stats[["FEI", "OAI Rate", "Inspections per Year"]], on="FEI", how="left")

panel["Year"] = panel["EventYear"]

# =============================================================================
# 9. SELECT AND ORDER FINAL COLUMNS
# =============================================================================

# Three coverage flags replacing the old five (Dataset, FEI_in_old, Insp Source,
# Insp_coverage with long names, In Sheet1):
panel["FEI_in_old_Redica"] = panel["FEI"].isin(_fei_in_old)  # per FEI
panel["FEI_in_new_Redica"] = panel["FEI"].isin(_fei_in_new)  # per FEI
panel["Insp_coverage"] = panel["Insp_coverage"].map({         # per row
    "both":               "both",
    "METFORMIN_old only": "old only",
    "Valisure14 only":    "new only",
})

# NDC_origin: per-NDC provenance — how was the FEI mapping obtained?
# All 112 NDCs were always in Valisure's testing data; what differs is whether
# the NDC→FEI match was in the original Sheet1 or newly resolved by Amir.
#   "Originally matched"        — FEI was in the original Sheet1 analysis
#   "Newly matched – known FEI" — Amir found the FEI; facility already in old METFORMIN data
#   "Newly matched – new FEI"   — Amir found the FEI; facility not in old Redica data at all
#   "Unmatched"                 — no FEI mapping found
def _ndc_origin(row):
    if pd.isna(row.get("FEI")):
        return "Unmatched"
    if row.get("In Sheet1", False):
        return "Originally matched"
    if row.get("FEI_in_old_Redica", False):
        return "Newly matched – known FEI"
    return "Newly matched – new FEI"

panel["NDC_origin"] = panel.apply(_ndc_origin, axis=1)

FINAL_COLS = [
    # Provenance / coverage flags
    "NDC_origin",         # per NDC: where NDC/FEI came from
    "FEI_in_old_Redica",  # per FEI: in METFORMIN_old Redica export?
    "FEI_in_new_Redica",  # per FEI: in current Valisure14 Redica export?
    "Insp_coverage",      # per row: "both" | "old only" | "new only" | null
    # NDC identity
    "Firm", "Year", "NDC", "NDC11", "NDC8", "Strength", "CountryCode",
    # Facility
    "FEI", "Site Display Name",
    # Valisure context
    "Valisure Years",
    # Inspection event
    "Event Start Date", "Event End Date", "EventYear", "Classification",
    "NAI", "VAI", "OAI", "483", "No 483",
    "483 critical", "483 major", "483 other", "Warning Letter",
    # Site-level aggregates
    "Total Inspections", "FDA Inspections", "483s Issued",
    "Total Observations", "Warning Letters Issued", "Import Alerts Issued",
    "OAI Rate", "Inspections per Year",
]

# Keep only columns that exist
FINAL_COLS = [c for c in FINAL_COLS if c in panel.columns]
panel_out = panel[FINAL_COLS].sort_values(
    ["NDC11", "EventYear"], na_position="last"
).reset_index(drop=True)

# =============================================================================
# 10. SAVE
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
panel_out.to_csv(OUT_FILE, index=False)
print(f"\nSaved: {OUT_FILE}  ({len(panel_out):,} rows)")

# ── Summary stats ─────────────────────────────────────────────────────────────
ndc_level = panel_out.drop_duplicates("NDC11")

print("\n── NDC & FEI counts by origin ──────────────────────────────────────────")
origin_summary = (
    ndc_level.groupby("NDC_origin", dropna=False)
    .agg(n_NDCs=("NDC11", "count"), n_FEIs=("FEI", "nunique"))
    .reindex(["Originally matched", "Newly matched – known FEI", "Newly matched – new FEI", "Unmatched"])
    .fillna(0).astype(int)
)
origin_summary["n_FEIs"] = origin_summary["n_FEIs"].where(
    origin_summary.index != "Unmatched", 0
)
print(origin_summary.to_string())

print("\n── FEI Redica membership (unique FEIs with mapping) ────────────────────")
fei_level = panel_out.dropna(subset=["FEI"]).drop_duplicates("FEI")
both_redica  = (fei_level["FEI_in_old_Redica"] &  fei_level["FEI_in_new_Redica"]).sum()
old_only     = (fei_level["FEI_in_old_Redica"] & ~fei_level["FEI_in_new_Redica"]).sum()
new_only     = (~fei_level["FEI_in_old_Redica"] &  fei_level["FEI_in_new_Redica"]).sum()
neither      = (~fei_level["FEI_in_old_Redica"] & ~fei_level["FEI_in_new_Redica"]).sum()
print(f"  In both old & new Redica : {both_redica}")
print(f"  Old Redica only          : {old_only}")
print(f"  New Redica only          : {new_only}")
print(f"  Neither (no Redica data) : {neither}")

print("\n── Inspection event coverage (rows with an event) ──────────────────────")
ev = panel_out[panel_out["Insp_coverage"].notna()]
cov_by_origin = (
    ev.groupby(["NDC_origin", "Insp_coverage"])
    .size().unstack(fill_value=0)
)
col_order = [c for c in ["both", "new only", "old only"] if c in cov_by_origin.columns]
print(cov_by_origin[col_order].to_string())

# %%
