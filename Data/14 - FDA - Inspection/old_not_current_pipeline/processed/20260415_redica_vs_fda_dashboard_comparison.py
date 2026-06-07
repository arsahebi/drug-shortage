# %%
"""
Redica vs. FDA Dashboard Data Completeness Comparison
======================================================
Compares per-FEI inspection counts and 483 counts between:
  - Redica (Valisure_Sites_Data_Availability.xlsx + Site List.xlsx)
  - FDA Dashboard downloads (Inspections Details.xlsx + Published 483s.xlsx)

Goal: determine whether Redica has access to more data than the FDA Dashboard.
"""

import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage"
)
REDICA_RAW = BASE / "Data/07 - Redica/raw"
FDA_RAW     = BASE / "Data/14 - FDA - Inspection/raw"
OUT_DIR     = BASE / "Data/14 - FDA - Inspection/processed"

# ── Load Redica data ─────────────────────────────────────────────────────────
print("Loading Redica files...")
site_list = pd.read_excel(
    REDICA_RAW / "Site List.xlsx",
    sheet_name="Site List",
    dtype={"FEI": str},
)
# Normalise FEI: strip whitespace, remove trailing ".0" from numeric reads
site_list["FEI"] = site_list["FEI"].str.strip().str.replace(r"\.0$", "", regex=True)

data_avail = pd.read_excel(
    REDICA_RAW / "Valisure_Sites_Data_Availability.xlsx",
    sheet_name="Data Availability",
)

# Join on Site Redica Id to get FEI alongside Redica counts
redica = site_list.merge(
    data_avail[["Site Redica Id", "FDA Inspections", "483s Issued"]],
    on="Site Redica Id",
    how="left",
)
redica = redica.rename(columns={
    "FDA Inspections": "redica_fda_inspections",
    "483s Issued":     "redica_483s_issued",
})
print(f"  Redica: {len(redica)} sites, {redica['FEI'].nunique()} unique FEIs")

# ── Load FDA Dashboard – Inspection Details ───────────────────────────────────
print("Loading FDA Dashboard files...")
insp = pd.read_excel(
    FDA_RAW / "Inspections Details.xlsx",
    sheet_name="Sheet1",
    dtype={"FEI Number": str},
)
insp["FEI Number"] = insp["FEI Number"].str.strip().str.replace(r"\.0$", "", regex=True)

fda_insp_counts = (
    insp.groupby("FEI Number")["Inspection ID"]
    .nunique()
    .reset_index()
    .rename(columns={"FEI Number": "FEI", "Inspection ID": "fda_dash_inspections"})
)

# ── Load FDA Dashboard – Published 483s ──────────────────────────────────────
p483 = pd.read_excel(
    FDA_RAW / "Published 483s.xlsx",
    sheet_name="Sheet1",
    dtype={"FEI Number": str},
)
p483["FEI Number"] = p483["FEI Number"].str.strip().str.replace(r"\.0$", "", regex=True)

fda_483_counts = (
    p483.groupby("FEI Number")["Record ID"]
    .nunique()
    .reset_index()
    .rename(columns={"FEI Number": "FEI", "Record ID": "fda_dash_483s"})
)

# ── Merge everything on FEI (left join keeps only Redica's 127 sites) ────────
df = (
    redica
    .merge(fda_insp_counts, on="FEI", how="left")
    .merge(fda_483_counts,  on="FEI", how="left")
)
df[["redica_fda_inspections", "redica_483s_issued",
    "fda_dash_inspections",   "fda_dash_483s"]] = (
    df[["redica_fda_inspections", "redica_483s_issued",
        "fda_dash_inspections",   "fda_dash_483s"]]
    .fillna(0).astype(int)
)

# ── Derived comparison columns ────────────────────────────────────────────────
df["insp_diff"]       = df["redica_fda_inspections"] - df["fda_dash_inspections"]   # >0 means Redica has more
df["p483_diff"]       = df["redica_483s_issued"]     - df["fda_dash_483s"]          # >0 means Redica has more
df["insp_match"]      = df["insp_diff"] == 0
df["p483_match"]      = df["p483_diff"] == 0
df["redica_more_insp"] = df["insp_diff"] > 0
df["fda_more_insp"]   = df["insp_diff"] < 0
df["redica_more_483"]  = df["p483_diff"] > 0
df["fda_more_483"]    = df["p483_diff"] < 0

# ── Summary statistics ────────────────────────────────────────────────────────
total_sites = len(df)

print("\n" + "="*65)
print("REDICA vs. FDA DASHBOARD — DATA COMPLETENESS SUMMARY")
print("="*65)

print(f"\nTotal sites (union of both sources): {total_sites}")

# Inspection counts
print("\n── Inspections ─────────────────────────────────────────────")
print(f"  Redica total FDA inspections (sum):   {df['redica_fda_inspections'].sum():>6}")
print(f"  FDA Dashboard inspections (sum):      {df['fda_dash_inspections'].sum():>6}")
print(f"  Sites where counts match:             {df['insp_match'].sum():>6} / {total_sites}")
print(f"  Sites where Redica has MORE:          {df['redica_more_insp'].sum():>6}")
print(f"  Sites where FDA Dashboard has MORE:   {df['fda_more_insp'].sum():>6}")

# 483 counts
print("\n── Published 483s ───────────────────────────────────────────")
print(f"  Redica total 483s issued (sum):       {df['redica_483s_issued'].sum():>6}")
print(f"  FDA Dashboard 483s (sum):             {df['fda_dash_483s'].sum():>6}")
print(f"  Sites where counts match:             {df['p483_match'].sum():>6} / {total_sites}")
print(f"  Sites where Redica has MORE:          {df['redica_more_483'].sum():>6}")
print(f"  Sites where FDA Dashboard has MORE:   {df['fda_more_483'].sum():>6}")

# Sites only in one source
only_redica = df[df["fda_dash_inspections"] == 0]["FEI"].tolist()
only_fda    = df[df["redica_fda_inspections"] == 0]["FEI"].tolist()
print(f"\n  FEIs only in Redica (not in FDA Dashboard): {len(only_redica)}")
print(f"  FEIs only in FDA Dashboard (not in Redica): {len(only_fda)}")

# ── Per-FEI detail table ──────────────────────────────────────────────────────
cols_out = [
    "FEI", "Site Display Name",
    "redica_fda_inspections", "fda_dash_inspections", "insp_diff",
    "redica_483s_issued",     "fda_dash_483s",         "p483_diff",
]
out_df = df[cols_out].sort_values("FEI").reset_index(drop=True)

out_path = OUT_DIR / "20260415_redica_vs_fda_dashboard_comparison.csv"
out_df.to_csv(out_path, index=False)
print(f"\nPer-FEI comparison table saved to:\n  {out_path}")

# ── Show sites with discrepancies ─────────────────────────────────────────────
discrepancies = out_df[(out_df["insp_diff"] != 0) | (out_df["p483_diff"] != 0)]
print(f"\nSites with ANY discrepancy ({len(discrepancies)}):")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
print(discrepancies.to_string(index=False))
