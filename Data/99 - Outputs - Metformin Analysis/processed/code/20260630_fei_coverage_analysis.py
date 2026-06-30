# %%
"""
FEI Coverage Analysis for Metformin NDCs
=========================================
Compares NDC→FEI mapping coverage before and after Amir's manual review.

Source: Q&As1234_v8_v02.xlsx, sheet "Amir-Unique NDC from Valisure ("
  Col G (FEI)       — original FEI carried over from Sheet1 (old method)
  Col H (Found FEI) — FEI Amir found manually via DailyMed / ProPublica

For each method, reports how many unique FEIs can be matched against:
  1. Redica combined dataset (redica_all_drugs_combined.csv, 127 FEIs)
  2. FDA Inspections Details (14 - FDA - Inspection/raw) — fallback
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
QA_FILE   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV= BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
FDA_INSP  = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"

# =============================================================================
# HELPERS
# =============================================================================
def clean_fei(x) -> str | None:
    if pd.isna(x): return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s): return None
    try: return str(int(float(s)))
    except: return None


def sep(title: str = "", w: int = 70):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
sep("Loading data")

# Amir's sheet — read with header=None to get correct column positions:
#   col 6 (G) = FEI       → original FEI from Sheet1 (old method)
#   col 7 (H) = Found FEI → FEI Amir found manually (new)
raw = pd.read_excel(QA_FILE, sheet_name="Amir-Unique NDC from Valisure (", header=None)
raw.columns = ["NDC","NDC11","c","d","e","NDC11_F","FEI_G","Found_FEI_H","Notes1","Notes2","extra"]
df_amir = raw.iloc[1:].reset_index(drop=True)
df_amir["fei_G"] = df_amir["FEI_G"].apply(clean_fei)       # old / Sheet1
df_amir["fei_H"] = df_amir["Found_FEI_H"].apply(clean_fei) # Amir's new finds
df_amir["fei_best"] = df_amir["fei_G"].fillna(df_amir["fei_H"])

# Sheet1 — original paper data
df_sheet1 = pd.read_excel(QA_FILE, sheet_name="Sheet1")
df_sheet1["FEI"] = df_sheet1["FEI"].apply(clean_fei)
sheet1_feis = set(df_sheet1["FEI"].dropna())

# Redica FEI universe
df_site   = pd.read_excel(SITE_LIST)
df_redica = pd.read_csv(REDICA_CSV)
redica_site = set(df_site["FEI"].dropna().astype(str).str.strip())
redica_csv  = set(df_redica["FEI"].dropna().astype(str).str.strip())
assert redica_site == redica_csv, "Site List and combined CSV FEIs disagree!"
REDICA_FEIS = redica_site
print(f"Redica FEI universe        : {len(REDICA_FEIS)} sites")

# FDA Inspection Details — drug rows only (fallback)
print("Loading FDA Inspection Details (large file)...")
df_fda = pd.read_excel(FDA_INSP)
df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
df_fda_drugs = df_fda[df_fda["Product Type"].str.lower().str.contains("drug", na=False)]
FDA_FEIS = set(df_fda_drugs["FEI Number"].unique())
ALL_SOURCES = REDICA_FEIS | FDA_FEIS
print(f"FDA Inspection Details FEIs: {len(FDA_FEIS):,} (drug inspections only)")

# =============================================================================
# 2. COL G — OLD / SHEET1 FEIs (subset visible in Amir's 128-NDC review)
# =============================================================================
sep("Col G (FEI) — old method, same as Sheet1")

old_feis = set(df_amir["fei_G"].dropna())
print(f"  Unique FEIs in col G              : {len(old_feis)}")
print(f"  All contained in Sheet1?          : {old_feis <= sheet1_feis}")
print(f"  Sheet1 FEIs absent from col G     : {sorted(sheet1_feis - old_feis)}")
print(f"    (those NDCs are not in Amir's 128-NDC review list)")

# =============================================================================
# 3. COL H — AMIR'S NEW FINDS
# =============================================================================
sep("Col H (Found FEI) — Amir's manual lookup via DailyMed / ProPublica")

found_feis    = set(df_amir["fei_H"].dropna())
confirmed     = found_feis & sheet1_feis      # also in Sheet1 (Amir confirmed)
genuinely_new = found_feis - sheet1_feis      # brand-new FEIs not in Sheet1

print(f"  NDC rows where Amir found a FEI   : {df_amir['fei_H'].notna().sum()} / {len(df_amir)}")
print(f"  Unique FEIs in col H              : {len(found_feis)}")
print(f"  Col H FEIs already in Sheet1      : {sorted(confirmed)}")
print(f"  Col H FEIs genuinely new ({len(genuinely_new)})     :")
for fei in sorted(genuinely_new):
    name_row = df_site[df_site["FEI"].astype(str).str.strip() == fei]
    name = name_row["Site Display Name"].values[0] if len(name_row) else "not in Redica"
    in_r = "Redica ✓" if fei in REDICA_FEIS else ("FDA ✓" if fei in FDA_FEIS else "neither")
    print(f"    {fei}  {name}  [{in_r}]")

# =============================================================================
# 4. COVERAGE COMPARISON
# =============================================================================
sep("Coverage comparison: old (Sheet1) vs combined (G + H)")

all_amir_feis = set(df_amir["fei_best"].dropna())   # col G ∪ col H

rows = []
for label, feis in [("Sheet1 (old)", sheet1_feis), ("Amir G+H (revised)", all_amir_feis)]:
    rows.append({
        "Method"              : label,
        "Unique FEIs"         : len(feis),
        "In Redica"           : len(feis & REDICA_FEIS),
        "FDA Details only"    : len(feis & (FDA_FEIS - REDICA_FEIS)),
        "In neither"          : len(feis - ALL_SOURCES),
        "Redica %"            : f"{len(feis & REDICA_FEIS)/len(feis)*100:.1f}%",
        "Redica+FDA %"        : f"{len(feis & ALL_SOURCES)/len(feis)*100:.1f}%",
    })
print(pd.DataFrame(rows).to_string(index=False))

# =============================================================================
# 5. FDA FALLBACK — DETAIL FOR FEIs MISSING FROM REDICA
# =============================================================================
sep("FDA Inspection Details for FEIs absent from Redica")

for label, feis in [("Sheet1 (old)", sheet1_feis), ("Amir G+H (revised)", all_amir_feis)]:
    missing = feis - REDICA_FEIS
    print(f"\n  {label} — {len(missing)} FEIs not in Redica:")
    for fei in sorted(missing):
        sub = df_fda_drugs[df_fda_drugs["FEI Number"] == fei]
        if sub.empty:
            print(f"    {fei}: NOT in FDA Details either")
        else:
            name    = sub["Legal Name"].iloc[0]
            country = sub["Country/Area"].iloc[0]
            n       = len(sub)
            classes = sub["Classification"].value_counts().to_dict()
            print(f"    {fei}: {name} [{country}]  {n} drug insp.  {classes}")

# =============================================================================
# 6. NDC-LEVEL SUMMARY
# =============================================================================
sep("NDC-level coverage (Amir G+H combined)")

total_ndcs = df_amir["NDC"].nunique()
for label, mask in [
    ("No FEI at all"          , df_amir["fei_best"].isna()),
    ("FEI in Redica"          , df_amir["fei_best"].isin(REDICA_FEIS)),
    ("FEI in FDA Details only", df_amir["fei_best"].isin(FDA_FEIS - REDICA_FEIS)),
    ("FEI in neither source"  , ~df_amir["fei_best"].isin(ALL_SOURCES) & df_amir["fei_best"].notna()),
]:
    n = df_amir[mask]["NDC"].nunique()
    print(f"  {label:30s}: {n:3d} / {total_ndcs}  ({n/total_ndcs*100:.1f}%)")

print()
print("Notes:")
print("  - Col G = FEI col in Amir's sheet = old method (identical to Sheet1 for matched NDCs)")
print("  - Col H = Found FEI col = Amir's manual lookup results")
print("  - 8 genuinely new FEIs from col H vs Sheet1; all 8 are in Redica")
print("  - The one new FEI not previously identified: Mylan Laboratories Ltd (IND) — 3005587313")
print("  - 3 FEIs (Marksans, Inventia, BAHEAL) absent from Redica but present in FDA Details")
print("  - Apotex Corp (3012378179) from Sheet1 absent from Amir's review and from Redica;")
print("    also NOT found in FDA Inspection Details")

# %%
