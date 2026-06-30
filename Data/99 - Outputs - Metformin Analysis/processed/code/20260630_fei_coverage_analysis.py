# %%
"""
FEI Coverage Analysis for Metformin NDCs
=========================================
Compares two NDC→FEI mapping methods:
  - Sheet1 ("old method"): the original FEI assignments used in the submitted paper
  - Amir's sheet ("Amir-Unique NDC from Valisure ("): Amir's manual re-mapping
    using cols F (NDC11) and G (FEI) with fallback col I (Found FEI)

For each method, reports how many unique FEIs can be matched to the Redica
combined inspection dataset (redica_all_drugs_combined.csv, 127 FEIs).
"""

import re
import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage"
)
QA_FILE      = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST    = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV   = BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"

# =============================================================================
# HELPERS
# =============================================================================
def clean_fei(x) -> str | None:
    """Convert any FEI-like value to a plain integer string; return None if non-numeric."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):   # notes/text → skip
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return None


def sep(title: str = "", w: int = 70):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
sep("Loading data")

# Amir's manual NDC→FEI mapping (col F = NDC11.1, col G = FEI, col I = Found FEI)
df_amir = pd.read_excel(QA_FILE, sheet_name="Amir-Unique NDC from Valisure (")
df_amir["fei_primary"] = df_amir["FEI"].apply(clean_fei)
df_amir["fei_found"]   = df_amir["Found FEI"].apply(clean_fei)
df_amir["fei_best"]    = df_amir["fei_primary"].fillna(df_amir["fei_found"])

# Sheet1 – original FEI assignments used in the paper
df_sheet1 = pd.read_excel(QA_FILE, sheet_name="Sheet1")
df_sheet1["FEI"] = df_sheet1["FEI"].apply(clean_fei)

# Redica FEI universe (Site List and combined CSV should be identical)
df_site   = pd.read_excel(SITE_LIST)
df_redica = pd.read_csv(REDICA_CSV)
redica_feis_site = set(df_site["FEI"].dropna().astype(str).str.strip())
redica_feis_csv  = set(df_redica["FEI"].dropna().astype(str).str.strip())
assert redica_feis_site == redica_feis_csv, "Site List and combined CSV FEIs disagree!"
REDICA_FEIS = redica_feis_site
print(f"Redica FEI universe: {len(REDICA_FEIS)} sites")

# =============================================================================
# 2. AMIR'S SHEET COVERAGE
# =============================================================================
sep("Amir's NDC→FEI mapping (new method)")

n_rows_amir      = len(df_amir)
n_ndc_amir       = df_amir["NDC11.1"].nunique()
n_with_fei_prim  = df_amir["fei_primary"].notna().sum()
n_with_fei_found = df_amir["fei_found"].notna().sum()
n_with_any_fei   = df_amir["fei_best"].notna().sum()
n_no_fei         = df_amir["fei_best"].isna().sum()
amir_feis        = set(df_amir["fei_best"].dropna())
amir_in_redica   = amir_feis & REDICA_FEIS
amir_not_redica  = amir_feis - REDICA_FEIS

print(f"  Total NDC rows in Amir's sheet         : {n_rows_amir}")
print(f"  Unique NDC11s                          : {n_ndc_amir}")
print(f"  NDCs with FEI in col G (primary)       : {n_with_fei_prim}")
print(f"  NDCs with FEI in col I (Found FEI)     : {n_with_fei_found}")
print(f"  NDCs with any valid FEI                : {n_with_any_fei}")
print(f"  NDCs without any FEI                   : {n_no_fei}")
print(f"  Unique FEIs identified                 : {len(amir_feis)}")
print(f"  FEIs matched in Redica                 : {len(amir_in_redica)}/{len(amir_feis)}"
      f" ({len(amir_in_redica)/len(amir_feis)*100:.1f}%)")
print(f"  FEIs NOT in Redica                     : {sorted(amir_not_redica)}")

# NDCs whose FEI is in Redica
n_ndc_matched = df_amir[df_amir["fei_best"].isin(amir_in_redica)].shape[0]
print(f"  NDC rows whose FEI is in Redica        : {n_ndc_matched}/{n_rows_amir}"
      f" ({n_ndc_matched/n_rows_amir*100:.1f}%)")

# Detail: FEIs not in Redica and which NDCs they belong to
print("\n  FEIs not in Redica (Amir method):")
miss_amir = df_amir[df_amir["fei_best"].isin(amir_not_redica)][
    ["NDC", "NDC11.1", "fei_best"]
].drop_duplicates().sort_values("fei_best")
for _, row in miss_amir.iterrows():
    print(f"    FEI {row['fei_best']}  NDC {row['NDC']}  ({row['NDC11.1']})")

# =============================================================================
# 3. SHEET1 COVERAGE (old method)
# =============================================================================
sep("Sheet1 FEI coverage (old / submitted method)")

n_ndc_s1        = df_sheet1["NDC11"].nunique()
sheet1_feis     = set(df_sheet1["FEI"].dropna())
s1_in_redica    = sheet1_feis & REDICA_FEIS
s1_not_redica   = sheet1_feis - REDICA_FEIS

print(f"  Unique NDC11s in Sheet1                : {n_ndc_s1}")
print(f"  Unique FEIs in Sheet1                  : {len(sheet1_feis)}")
print(f"  FEIs matched in Redica                 : {len(s1_in_redica)}/{len(sheet1_feis)}"
      f" ({len(s1_in_redica)/len(sheet1_feis)*100:.1f}%)")
print(f"  FEIs NOT in Redica                     : {sorted(s1_not_redica)}")

print("\n  FEIs not in Redica (Sheet1 method):")
for fei in sorted(s1_not_redica):
    firms = df_sheet1[df_sheet1["FEI"] == fei][["Firm", "CountryCode"]].drop_duplicates()
    for _, row in firms.iterrows():
        print(f"    FEI {fei}  {row['Firm']} [{row['CountryCode']}]")

# =============================================================================
# 4. COMPARISON: WHAT AMIR'S METHOD ADDS
# =============================================================================
sep("Comparison: Amir (new) vs Sheet1 (old)")

new_feis       = amir_feis - sheet1_feis          # in Amir only
dropped_feis   = sheet1_feis - amir_feis          # in Sheet1 only
common_feis    = amir_feis & sheet1_feis

print(f"  FEIs shared by both methods            : {len(common_feis)}")
print(f"  FEIs added by Amir (new, all in Redica): {len(new_feis)}")
print(f"    {sorted(new_feis)}")

# Firm names for the new FEIs (from Redica site list)
print("\n  New FEIs and their Redica display names:")
for fei in sorted(new_feis):
    name_row = df_site[df_site["FEI"].astype(str).str.strip() == fei]
    name = name_row["Site Display Name"].values[0] if len(name_row) else "NOT IN REDICA"
    print(f"    {fei}  {name}")

print(f"\n  FEIs in Sheet1 dropped by Amir          : {len(dropped_feis)}")
for fei in sorted(dropped_feis):
    firms = df_sheet1[df_sheet1["FEI"] == fei][["Firm", "CountryCode"]].drop_duplicates()
    for _, row in firms.iterrows():
        in_redica = "IN Redica" if fei in REDICA_FEIS else "NOT in Redica"
        print(f"    {fei}  {row['Firm']} [{row['CountryCode']}]  ({in_redica})")

# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================
sep("Summary table")

summary = pd.DataFrame([
    {
        "Method"            : "Sheet1 (old/submitted)",
        "Unique NDC11s"     : n_ndc_s1,
        "Unique FEIs"       : len(sheet1_feis),
        "FEIs in Redica"    : len(s1_in_redica),
        "FEIs NOT in Redica": len(s1_not_redica),
        "Redica match rate" : f"{len(s1_in_redica)/len(sheet1_feis)*100:.1f}%",
    },
    {
        "Method"            : "Amir (new/revised)",
        "Unique NDC11s"     : n_ndc_amir,
        "Unique FEIs"       : len(amir_feis),
        "FEIs in Redica"    : len(amir_in_redica),
        "FEIs NOT in Redica": len(amir_not_redica),
        "Redica match rate" : f"{len(amir_in_redica)/len(amir_feis)*100:.1f}%",
    },
])
print(summary.to_string(index=False))
print()
print("Notes:")
print("  - Redica universe = 127 FEIs (Site List.xlsx = redica_all_drugs_combined.csv)")
print("  - Amir's FEI is col G of 'Amir-Unique NDC from Valisure (' tab,")
print("    with fallback to col I (Found FEI) where col G is blank")
print("  - Amir's sheet has 128 rows; 23 rows have no valid FEI at all")
print("  - The 3 Amir FEIs not in Redica correspond to:")
print("    Inventia Healthcare (IND), Marksans Pharma (IND), Qingdao BAHEAL (CHN)")
print("  - The 1 Sheet1 FEI dropped by Amir = Apotex Corp (CAN) — also not in Redica")

# %%
