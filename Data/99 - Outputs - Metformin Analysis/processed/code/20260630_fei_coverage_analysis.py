# %%
"""
Metformin NDC & FEI Coverage Analysis
======================================
Ground-truth NDC count from Valisure raw sweep data (3 years),
then traces how many reach the final analysis (82 NDCs) and
how old vs. Amir's revised FEI matching compares against
Redica (127 FEIs) and FDA Inspection Details (fallback).

Sources
-------
  Valisure raw:  Data/08 - Valisure/raw/Valisure_2024_raw.xlsx
                   sheets: "2020 Testing Data", "2022 Testing Data - Actual",
                            "2024 Testing Data"
  Old analysis:  Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx
                   sheets: Sheet1 (old FEI match), "Amir-Unique NDC from Valisure ("
  Redica:        Data/07 - Redica/raw/Site List.xlsx  /  processed/redica_all_drugs_combined.csv
  FDA Inspection:Data/14 - FDA - Inspection/raw/Inspections Details.xlsx
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
VALISURE_RAW = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
QA_FILE      = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST    = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV   = BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
FDA_INSP     = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"

# =============================================================================
# HELPERS
# =============================================================================
def to_ndc11(x) -> str | None:
    """Normalize any NDC variant to 11-digit string (5-4-2, no hyphens)."""
    if pd.isna(x):
        return None
    s = str(x).strip().replace("-", "").replace(" ", "")
    if len(s) == 10:
        return s[:5] + "0" + s[5:]   # 5-3-2 → 5-4-2
    elif len(s) == 11:
        return s
    return None   # unexpected length


def clean_fei(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return None


def sep(title: str = "", w: int = 70):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


# =============================================================================
# 1. VALISURE GROUND TRUTH — unique NDC11s tested across all 3 sweeps
# =============================================================================
sep("STEP 1: Valisure ground-truth NDC count (3 sweep years)")

xls_val = pd.ExcelFile(VALISURE_RAW)
df20 = xls_val.parse("2020 Testing Data")
df22 = xls_val.parse("2022 Testing Data - Actual")
df24 = xls_val.parse("2024 Testing Data", header=1)

# 2020 & 2022: NDC column is in 5-3-2 format → to_ndc11 converts
# 2024: NDC11 column is already in 5-4-2 format → use directly
ndc20 = set(filter(None, df20["NDC"].apply(to_ndc11)))
ndc22 = set(filter(None, df22["NDC"].apply(to_ndc11)))
ndc24 = set(filter(None, df24["NDC11"].apply(to_ndc11)))

val_union = ndc20 | ndc22 | ndc24

print(f"  2020 sweep unique NDC11s         : {len(ndc20)}")
print(f"  2022 sweep unique NDC11s         : {len(ndc22)}")
print(f"  2024 sweep unique NDC11s         : {len(ndc24)}")
print(f"  Overlap 2020∩2022                : {len(ndc20 & ndc22)}")
print(f"  Overlap 2022∩2024                : {len(ndc22 & ndc24)}")
print(f"  Overlap 2020∩2024                : {len(ndc20 & ndc24)}")
print(f"  All-3-year overlap               : {len(ndc20 & ndc22 & ndc24)}")
print(f"  TOTAL unique Valisure NDC11s     : {len(val_union)}")

# =============================================================================
# 2. SHEET1 — OLD ANALYSIS COVERAGE
# =============================================================================
sep("STEP 2: Sheet1 (old analysis) — which Valisure NDCs were included?")

df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1")
df_s1["ndc11_norm"] = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_norm"]   = df_s1["FEI"].apply(clean_fei)

sheet1_ndcs = set(df_s1["ndc11_norm"].dropna())
sheet1_feis = set(df_s1["fei_norm"].dropna())

in_val   = sheet1_ndcs & val_union   # Sheet1 NDCs that Valisure tested
not_in_val = sheet1_ndcs - val_union  # Sheet1 NDCs NOT in Valisure (unexpected)

val_not_in_s1 = val_union - sheet1_ndcs  # Valisure-tested NDCs missing from Sheet1

print(f"  Sheet1 unique NDC11s             : {len(sheet1_ndcs)}")
print(f"  Of those, Valisure-tested        : {len(in_val)}")
print(f"  Sheet1 NDCs NOT in Valisure      : {len(not_in_val)}  ← unexpected")
print(f"  Valisure NDCs absent from Sheet1 : {len(val_not_in_s1)}  (no IQVIA match or excluded)")

print()
print("  Country breakdown of Sheet1 NDC11s:")
s1_by_country = df_s1.drop_duplicates("ndc11_norm").groupby("CountryCode").size()
for cc, n in s1_by_country.items():
    print(f"    {cc}: {n}")

print()
excluded_cc = ["CAN", "BGD"]
used_ndcs = (
    df_s1[~df_s1["CountryCode"].isin(excluded_cc)]
    ["ndc11_norm"].dropna().nunique()
)
excl_n = df_s1[df_s1["CountryCode"].isin(excluded_cc)]["ndc11_norm"].dropna().nunique()
print(f"  NDCs used in final analysis      : {used_ndcs}  (CAN+BGD excluded: {excl_n})")
print(f"  Sheet1 unique FEIs               : {len(sheet1_feis)}")

# =============================================================================
# 3. AMIR'S MANUAL FEI LOOKUP — col G (old) vs col H (new finds)
# =============================================================================
sep("STEP 3: Amir's sheet — FEI matching (old col G vs new col H)")

raw = pd.read_excel(
    QA_FILE,
    sheet_name="Amir-Unique NDC from Valisure (",
    header=None,
)
raw.columns = ["NDC", "NDC11", "c", "d", "e", "NDC11_F",
               "FEI_G", "Found_FEI_H", "Notes1", "Notes2", "extra"]
df_amir = raw.iloc[1:].reset_index(drop=True)
df_amir["fei_G"]    = df_amir["FEI_G"].apply(clean_fei)        # old (same as Sheet1)
df_amir["fei_H"]    = df_amir["Found_FEI_H"].apply(clean_fei)  # Amir's manual finds
df_amir["fei_best"] = df_amir["fei_G"].fillna(df_amir["fei_H"])

old_feis  = set(df_amir["fei_G"].dropna())
new_feis  = set(df_amir["fei_H"].dropna())
confirmed      = new_feis & sheet1_feis   # Amir's col H already in Sheet1
genuinely_new  = new_feis - sheet1_feis   # truly new FEIs

print(f"  Rows in Amir's sheet (NDC universe): {len(df_amir)}")
print(f"  Rows with col G FEI (old)          : {df_amir['fei_G'].notna().sum()}")
print(f"  Rows with col H FEI (new find)     : {df_amir['fei_H'].notna().sum()}")
print(f"  Unique FEIs col G                  : {len(old_feis)}  (same as Sheet1)")
print(f"  Unique FEIs col H                  : {len(new_feis)}")
print(f"    Already in Sheet1                : {len(confirmed)}")
print(f"    Genuinely new FEIs               : {len(genuinely_new)}")
print(f"  Combined unique FEIs (G ∪ H)       : {len(old_feis | new_feis)}")

df_site = pd.read_excel(SITE_LIST)

print(f"\n  Genuinely new FEIs from Amir's col H:")
for fei in sorted(genuinely_new):
    name_row = df_site[df_site["FEI"].astype(str).str.strip() == fei]
    name = name_row["Site Display Name"].values[0] if len(name_row) else "not in Redica"
    print(f"    {fei}  {name}")

# =============================================================================
# 4. FEI COVERAGE vs REDICA + FDA DETAILS
# =============================================================================
sep("STEP 4: FEI coverage — Redica vs FDA Inspection Details")

df_redica = pd.read_csv(REDICA_CSV)
redica_feis = set(df_site["FEI"].dropna().astype(str).str.strip())
assert redica_feis == set(df_redica["FEI"].dropna().astype(str).str.strip()), \
    "Site List and combined CSV FEIs disagree!"
print(f"  Redica FEI universe              : {len(redica_feis)} sites")

print("  Loading FDA Inspection Details (large file)...")
df_fda      = pd.read_excel(FDA_INSP)
df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
df_fda_drugs = df_fda[df_fda["Product Type"].str.lower().str.contains("drug", na=False)]
fda_feis    = set(df_fda_drugs["FEI Number"].unique())
all_sources = redica_feis | fda_feis
print(f"  FDA Inspection Details FEIs      : {len(fda_feis):,} (drug rows only)")

all_amir_feis = set(df_amir["fei_best"].dropna())

rows = []
for label, feis in [("Sheet1 (old)", sheet1_feis), ("Amir G+H (revised)", all_amir_feis)]:
    rows.append({
        "Method"           : label,
        "Unique FEIs"      : len(feis),
        "In Redica"        : len(feis & redica_feis),
        "FDA Details only" : len(feis & (fda_feis - redica_feis)),
        "In neither"       : len(feis - all_sources),
        "Redica %"         : f"{len(feis & redica_feis)/len(feis)*100:.1f}%",
        "Redica+FDA %"     : f"{len(feis & all_sources)/len(feis)*100:.1f}%",
    })
print()
print(pd.DataFrame(rows).to_string(index=False))

# =============================================================================
# 5. NDC-LEVEL COVERAGE SUMMARY (Amir's combined G+H)
# =============================================================================
sep("STEP 5: NDC-level FEI coverage (Amir G+H combined)")

total_ndcs = df_amir["NDC"].nunique()
for label, mask in [
    ("No FEI at all",           df_amir["fei_best"].isna()),
    ("FEI in Redica",           df_amir["fei_best"].isin(redica_feis)),
    ("FEI in FDA Details only", df_amir["fei_best"].isin(fda_feis - redica_feis)),
    ("FEI in neither source",  ~df_amir["fei_best"].isin(all_sources) & df_amir["fei_best"].notna()),
]:
    n = df_amir[mask]["NDC"].nunique()
    print(f"  {label:30s}: {n:3d} / {total_ndcs}  ({n/total_ndcs*100:.1f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
sep("SUMMARY")
print(f"""
  Valisure testing (all 3 sweeps, 2020+2022+2024):
    Total unique NDC11s tested        : {len(val_union)}
      2020-only                       : {len(ndc20 - ndc22 - ndc24)}
      2022-only                       : {len(ndc22 - ndc20 - ndc24)}
      2024-only                       : {len(ndc24 - ndc20 - ndc22)}
      Multiple years                  : {len(val_union) - len(ndc20 - ndc22 - ndc24) - len(ndc22 - ndc20 - ndc24) - len(ndc24 - ndc20 - ndc22)}

  From Valisure → Sheet1 (old analysis):
    Included in Sheet1                : {len(in_val)} / {len(val_union)}
    Missing from Sheet1               : {len(val_not_in_s1)}  (no IQVIA match or otherwise excluded)
    Sheet1 total unique NDCs          : {len(sheet1_ndcs)}
      IND + CHN + USA (used)          : {used_ndcs}
      CAN + BGD (excluded)            : {excl_n}

  FEI matching:
    Old method (Sheet1)               : {len(sheet1_feis)} unique FEIs  ({len(sheet1_feis & redica_feis)} in Redica)
    Amir's revised (G+H)              : {len(all_amir_feis)} unique FEIs  ({len(all_amir_feis & redica_feis)} in Redica)
    Net gain from Amir's review       : {len(genuinely_new)} genuinely new FEIs
""")

# %%
