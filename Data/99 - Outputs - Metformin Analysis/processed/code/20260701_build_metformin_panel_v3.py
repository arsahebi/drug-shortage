# %%
"""
Build Metformin Inspection Panel v3
====================================
Uses the new NDC→FEI master (Amir & Amirreza full review, 112 NDCs) as the
FEI mapping source, replacing the Sheet1 + old Amir tab used in panel_v1.

Two-FEI NDCs get two separate sets of inspection rows (one per FEI).

Output columns are identical to panel_v1 (metformin_panel_v1.csv) plus:
  fei_rank          — "primary" | "secondary_two_duns" | "primary_from_note" | "not_found"
  ndc_fei_origin    — how this NDC→FEI assignment was derived vs Sheet1

Comparison stats at end:
  - Records added because of new/changed FEI assignments
  - Records added because inspection data has more events for same FEI
  - Records only in old panel (FEI no longer assigned to that NDC)
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
MASTER    = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_master.csv"
VAL_RAW   = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
QA_FILE   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV= BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
INSP_HIST = BASE / "Data/07 - Redica/processed/valisure_fei_inspection_history.csv"
FDA_INSP  = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"
OLD_PANEL = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v1.csv"
OUT_FILE  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v3.csv"

COUNTRY_MAP = {
    "India": "IND", "China": "CHN", "United States": "USA",
    "Canada": "CAN", "Bangladesh": "BGD", "United Kingdom": "GBR",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Japan": "JPN", "Israel": "ISR", "Ireland": "IRL", "Netherlands": "NLD",
    "Australia": "AUS", "Singapore": "SGP", "South Korea": "KOR",
}

# =============================================================================
# HELPERS
# =============================================================================
def to_ndc11(x) -> Optional[str]:
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


def ndc11_to_display(n11: str):
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
# 1. LOAD NDC→FEI MASTER  (Amir & Amirreza review, 112 NDCs)
# =============================================================================
print("Loading NDC→FEI master...")
master = pd.read_csv(MASTER, dtype=str)
master["ndc11"] = master["ndc11"].apply(to_ndc11)
master["fei"]   = master["fei"].apply(clean_fei)
# Keep only rows with a valid NDC11; include not_found so we carry all 112 NDCs
master = master.dropna(subset=["ndc11"]).reset_index(drop=True)

print(f"  Master rows         : {len(master)}")
print(f"  Unique NDC11s       : {master['ndc11'].nunique()}")
print(f"  NDC×FEI pairs w/ FEI: {master['fei'].notna().sum()}")
print(f"  FEI rank breakdown  :")
print(master["fei_rank"].value_counts().to_string())

# =============================================================================
# 2. SHEET1 — for NDC_origin classification and CountryCode fallback
# =============================================================================
print("\nLoading Sheet1 for origin classification...")
df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1", dtype=str)
df_s1["ndc11"] = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_s1"] = df_s1["FEI"].apply(clean_fei)
s1_fei = (
    df_s1.dropna(subset=["ndc11", "fei_s1"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["fei_s1"]
)
s1_country = (
    df_s1.dropna(subset=["ndc11"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["CountryCode"]
)
s1_firm = (
    df_s1.dropna(subset=["ndc11"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["Firm"]
)
s1_strength = (
    df_s1.dropna(subset=["ndc11"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["Strength"]
)
sheet1_ndcs = set(s1_fei.index)

def _origin(row):
    ndc, fei, rank = row["ndc11"], row["fei"], row["fei_rank"]
    if rank == "not_found":
        return "Unmatched"
    if rank == "secondary_two_duns":
        return "Two FEIs – secondary"
    if ndc in sheet1_ndcs:
        old_fei = s1_fei.get(ndc)
        if old_fei and old_fei == fei:
            return "Sheet1 confirmed"
        elif old_fei:
            return "FEI updated"
        else:
            return "Sheet1 – newly matched"
    return "New assignment"

master["ndc_fei_origin"] = master.apply(_origin, axis=1)

# =============================================================================
# 3. VALISURE — NDC metadata (Firm, Strength, Valisure Years)
# =============================================================================
print("Loading Valisure metadata...")
xls_val = pd.ExcelFile(VAL_RAW)
df20 = xls_val.parse("2020 Testing Data")
df22 = xls_val.parse("2022 Testing Data - Actual")
df24 = xls_val.parse("2024 Testing Data", header=1)

val_info: dict[str, dict] = {}
for sweep_year, df in [("2020", df20), ("2022", df22), ("2024", df24)]:
    for _, row in df.iterrows():
        n11 = to_ndc11(row.get("NDC11") if sweep_year == "2024" else row.get("NDC"))
        if not n11:
            continue
        if n11 not in val_info:
            val_info[n11] = {
                "val_firm": str(row.get("Firm", "")).strip(),
                "val_strength": str(row.get("Strength", row.get("Dosage (mg)", ""))).strip(),
                "val_ndc_raw": str(row.get("NDC", "")).strip(),
                "years": [],
            }
        if sweep_year not in val_info[n11]["years"]:
            val_info[n11]["years"].append(sweep_year)
        if not val_info[n11]["val_firm"] and str(row.get("Firm", "")).strip():
            val_info[n11]["val_firm"] = str(row["Firm"]).strip()

val_df = pd.DataFrame([
    {
        "ndc11": n11,
        "val_firm": d["val_firm"],
        "val_strength": d["val_strength"],
        "Valisure Years": "+".join(sorted(d["years"])),
    }
    for n11, d in val_info.items()
])
print(f"  Valisure NDC11s: {len(val_df)}")

# =============================================================================
# 4. REDICA SITE LIST + AGGREGATE
# =============================================================================
print("Loading Redica site list and aggregates...")
sl = pd.read_excel(SITE_LIST, dtype=str)
sl["FEI"] = sl["FEI"].str.strip()
sl["redica_firm"] = sl["Site Display Name"].str.split("[").str[0].str.strip()
FEI_TO_SITE   = sl.set_index("FEI")["Site Display Name"].to_dict()
FEI_TO_FIRM   = sl.set_index("FEI")["redica_firm"].to_dict()
REDICA_FEIS   = set(sl["FEI"])

df_redica = pd.read_csv(REDICA_CSV)
df_redica["FEI"] = df_redica["FEI"].astype(str).str.strip()
site_agg = (
    df_redica[["FEI", "Total Inspections", "FDA Inspections", "483s Issued",
               "Total Observations", "Warning Letters Issued", "Import Alerts Issued"]]
    .drop_duplicates("FEI")
)

# =============================================================================
# 5. FDA INSPECTION DETAILS — country codes for all FEIs
# =============================================================================
print("Loading FDA inspection details for country codes...")
df_fda = pd.read_excel(FDA_INSP)
df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
fei_country = (
    df_fda[["FEI Number", "Country/Area"]]
    .drop_duplicates("FEI Number")
    .rename(columns={"FEI Number": "FEI", "Country/Area": "fda_country"})
)
fei_country["CountryCode_fda"] = fei_country["fda_country"].map(COUNTRY_MAP)
FEI_TO_COUNTRY = fei_country.set_index("FEI")["CountryCode_fda"].to_dict()

# =============================================================================
# 6. BUILD NDC-LEVEL METADATA TABLE
# =============================================================================
print("Building NDC metadata table...")
# One row per (ndc11, fei) from master; merge Valisure metadata
ndc_meta = master.copy()
ndc_meta = ndc_meta.merge(val_df, on="ndc11", how="left")

# Firm: Sheet1 manual → Redica by FEI → Valisure
ndc_meta["s1_firm"]     = ndc_meta["ndc11"].map(s1_firm)
ndc_meta["redica_firm"] = ndc_meta["fei"].map(FEI_TO_FIRM)
ndc_meta["Firm"] = (
    ndc_meta["s1_firm"]
    .fillna(ndc_meta["redica_firm"])
    .fillna(ndc_meta["val_firm"])
    .replace({"0": None, "nan": None, "": None})
)

# Strength: Sheet1 → Valisure
ndc_meta["s1_strength"] = ndc_meta["ndc11"].map(s1_strength)
ndc_meta["Strength"] = ndc_meta["s1_strength"].fillna(ndc_meta["val_strength"]).replace({"0": None, "nan": None, "": None})

# CountryCode: Sheet1 → FDA inspection by FEI
ndc_meta["s1_country"]  = ndc_meta["ndc11"].map(s1_country)
ndc_meta["fdi_country"] = ndc_meta["fei"].map(FEI_TO_COUNTRY)
ndc_meta["CountryCode"] = ndc_meta["s1_country"].fillna(ndc_meta["fdi_country"])

# Firm mismatch flag
def _mismatch(row):
    vf = str(row.get("val_firm", "")).strip().lower()
    rf = str(row.get("redica_firm", "")).strip().lower()
    if not vf or vf in ("nan", "0", "") or not rf:
        return 0
    return int(vf != rf)
ndc_meta["firm_valisure_mismatch"] = ndc_meta.apply(_mismatch, axis=1)

# NDC display formats
ndc_meta[["NDC", "NDC11", "NDC8"]] = pd.DataFrame(
    ndc_meta["ndc11"].apply(
        lambda n: ndc11_to_display(n) if isinstance(n, str) else ("", "", "")
    ).tolist(),
    index=ndc_meta.index,
)

# Site Display Name from Redica site list
ndc_meta["Site Display Name"] = ndc_meta["fei"].map(FEI_TO_SITE)
ndc_meta["FEI_in_Redica"] = ndc_meta["fei"].isin(REDICA_FEIS)

print(f"  NDC metadata rows: {len(ndc_meta)}  ({ndc_meta['fei'].notna().sum()} with FEI)")

# =============================================================================
# 7. INSPECTION HISTORY
# =============================================================================
print("Loading inspection history...")
df_hist = pd.read_csv(INSP_HIST)
df_hist["FEI"]              = df_hist["FEI"].astype(str).str.strip()
df_hist["Event End Date"]   = pd.to_datetime(df_hist["Event End Date"],  errors="coerce")
df_hist["Event Start Date"] = pd.to_datetime(df_hist["Event Start Date"], errors="coerce")
df_hist["EventYear"]        = df_hist["EventYear"].astype(pd.Int64Dtype())

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

_fei_in_old = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "METFORMIN_old only"]), "FEI"].unique())
_fei_in_new = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "Valisure14 only"]),    "FEI"].unique())
hist_feis   = set(df_hist["FEI"].unique())

# =============================================================================
# 8. BUILD PANEL: (NDC × FEI) × inspection event
# =============================================================================
print("Building panel...")

ndc_with    = ndc_meta[ndc_meta["fei"].notna()  & ndc_meta["fei"].isin(hist_feis)].copy()
ndc_no_hist = ndc_meta[ndc_meta["fei"].notna()  & ~ndc_meta["fei"].isin(hist_feis)].copy()
ndc_no_fei  = ndc_meta[ndc_meta["fei"].isna()].copy()

_hist_drop = [c for c in ["Firm", "CountryCode"] if c in df_hist.columns]
_hist_merge = df_hist.drop(columns=_hist_drop)

# Rename fei → FEI for merge key consistency
ndc_with = ndc_with.rename(columns={"fei": "FEI"})
ndc_no_hist = ndc_no_hist.rename(columns={"fei": "FEI"})
ndc_no_fei  = ndc_no_fei.rename(columns={"fei": "FEI"})

panel_with = ndc_with.merge(_hist_merge, on="FEI", how="left")

blank_hist_cols = [
    "Event Start Date", "Event End Date", "EventYear", "Classification",
    "NAI", "VAI", "OAI", "483", "No 483",
    "483 critical", "483 major", "483 other", "Warning Letter",
    "Site Display Name", "Source", "Insp_coverage",
]
for col in blank_hist_cols:
    ndc_no_hist[col] = None
    ndc_no_fei[col]  = None

panel = pd.concat([panel_with, ndc_no_hist, ndc_no_fei], ignore_index=True)

# Fill Site Display Name from Redica site list when history row lacks it
panel["Site Display Name"] = panel.apply(
    lambda r: (r.get("Site Display Name") or FEI_TO_SITE.get(str(r.get("FEI", "")) or "", None)),
    axis=1,
)

# Merge site aggregate stats and computed rates
panel = panel.merge(site_agg, on="FEI", how="left")
panel = panel.merge(hist_stats[["FEI", "OAI Rate", "Inspections per Year"]], on="FEI", how="left")
panel["Year"] = panel["EventYear"]

# Redica coverage flags
panel["FEI_in_old_Redica"] = panel["FEI"].isin(_fei_in_old)
panel["FEI_in_new_Redica"] = panel["FEI"].isin(_fei_in_new)
panel["Insp_coverage"] = panel["Insp_coverage"].map({
    "both":               "both",
    "METFORMIN_old only": "old only",
    "Valisure14 only":    "new only",
})

# =============================================================================
# 9. SELECT AND ORDER FINAL COLUMNS
# =============================================================================
FINAL_COLS = [
    "ndc_fei_origin",
    "fei_rank",
    "FEI_in_old_Redica",
    "FEI_in_new_Redica",
    "Insp_coverage",
    "Firm", "firm_valisure_mismatch", "Year",
    "NDC", "NDC11", "NDC8", "Strength", "CountryCode",
    "FEI", "Site Display Name",
    "Valisure Years",
    "Event Start Date", "Event End Date", "EventYear", "Classification",
    "NAI", "VAI", "OAI", "483", "No 483",
    "483 critical", "483 major", "483 other", "Warning Letter",
    "Total Inspections", "FDA Inspections", "483s Issued",
    "Total Observations", "Warning Letters Issued", "Import Alerts Issued",
    "OAI Rate", "Inspections per Year",
]
FINAL_COLS = [c for c in FINAL_COLS if c in panel.columns]

panel_out = (
    panel[FINAL_COLS]
    .sort_values(["NDC11", "FEI", "EventYear"], na_position="last")
    .reset_index(drop=True)
)

panel_out.to_csv(OUT_FILE, index=False)
print(f"\nSaved: {OUT_FILE}  ({len(panel_out):,} rows)")

# =============================================================================
# 10. SUMMARY STATS
# =============================================================================
print("\n── Panel summary (new v3) ──────────────────────────────────────────────")

# Per (NDC, FEI) pair level
pair_level = panel_out.drop_duplicates(["NDC11", "FEI"])

print(f"\nNDC_FEI origin breakdown (unique NDC×FEI pairs):")
print(pair_level["ndc_fei_origin"].value_counts().to_string())

cov = panel_out[panel_out["Insp_coverage"].notna()]["Insp_coverage"].value_counts()
print(f"\nInspection event coverage:")
print(cov.to_string())

# FEI Redica membership
fei_level = panel_out.dropna(subset=["FEI"]).drop_duplicates("FEI")
print(f"\nFEI Redica membership ({len(fei_level)} unique FEIs with mapping):")
print(f"  In both old & new Redica : {(fei_level['FEI_in_old_Redica'] &  fei_level['FEI_in_new_Redica']).sum()}")
print(f"  Old Redica only          : {(fei_level['FEI_in_old_Redica'] & ~fei_level['FEI_in_new_Redica']).sum()}")
print(f"  New Redica only          : {(~fei_level['FEI_in_old_Redica'] &  fei_level['FEI_in_new_Redica']).sum()}")
print(f"  Neither                  : {(~fei_level['FEI_in_old_Redica'] & ~fei_level['FEI_in_new_Redica']).sum()}")

# =============================================================================
# 11. COMPARE WITH PANEL_V1
# =============================================================================
print("\n── Comparison with panel_v1 ────────────────────────────────────────────")
try:
    old = pd.read_csv(OLD_PANEL, dtype=str, low_memory=False)
    old["ndc11_norm"] = old["NDC11"].apply(to_ndc11) if "NDC11" in old.columns else old.get("NDC11")
    old["FEI"] = old["FEI"].astype(str).str.strip() if "FEI" in old.columns else None

    # (NDC11, FEI) pairs in old vs new
    old_pairs = set(
        zip(
            old["ndc11_norm"].dropna(),
            old["FEI"].dropna()
        )
    )
    new_pairs = set(
        zip(
            panel_out["NDC11"].apply(to_ndc11).dropna(),
            panel_out["FEI"].dropna()
        )
    )

    added_pairs   = new_pairs - old_pairs
    removed_pairs = old_pairs - new_pairs
    kept_pairs    = new_pairs & old_pairs

    print(f"\n(NDC, FEI) pair-level changes:")
    print(f"  Kept    (in both v1 & v3)  : {len(kept_pairs):>4} pairs")
    print(f"  Added   (new in v3)        : {len(added_pairs):>4} pairs")
    print(f"  Removed (lost from v3)     : {len(removed_pairs):>4} pairs")

    # Inspection row counts for kept pairs
    def _row_count(df, pairs):
        key = list(zip(df["NDC11"].apply(to_ndc11), df["FEI"].astype(str).str.strip()))
        mask = pd.Series(key, index=df.index).isin(pairs)
        return df[mask]

    old_kept = _row_count(old, kept_pairs)
    new_kept = _row_count(panel_out, kept_pairs)
    print(f"\nInspection rows for kept pairs:")
    print(f"  Old panel_v1 : {len(old_kept):,} rows")
    print(f"  New panel_v3 : {len(new_kept):,} rows")
    delta = len(new_kept) - len(old_kept)
    print(f"  Delta        : {delta:+,} rows (new inspection data for same FEIs)")

    print(f"\nRows from ADDED pairs   : {len(_row_count(panel_out, added_pairs)):,}")
    print(f"Rows from REMOVED pairs : {len(_row_count(old, removed_pairs)):,}")

except FileNotFoundError:
    print("  panel_v1 not found — skipping comparison")

# %%
