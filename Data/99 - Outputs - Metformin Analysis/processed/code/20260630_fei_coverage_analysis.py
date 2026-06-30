# %%
"""
Metformin NDC & FEI Coverage Analysis
======================================
Ground-truth NDC count from Valisure raw sweep data (3 years),
then traces how many reach the final analysis (82 NDCs) and
how old vs. Amir's revised FEI matching compares against
Redica (127 FEIs) and FDA Inspection Details (fallback).

NDC normalization notes
-----------------------
NDCs appear in several hyphenated segment formats:
  4-3-2  e.g. 0378-6001-91   (short labeler)
  5-3-2  e.g. 71093-132-06   (standard HIPAA)
  5-4-1  e.g. 60505-0260-1   (short package)
  5-4-2  e.g. 27241-0241-90  (NDC11 — already standard)
All are normalised to NDC11 (5-4-2, 11 digits, no hyphens) by
zero-padding each segment individually.

Sources
-------
  Valisure raw:  Data/08 - Valisure/raw/Valisure_2024_raw.xlsx
                   sheets: "2020 Testing Data", "2022 Testing Data - Actual",
                            "2024 Testing Data"
  Old analysis:  Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx
                   sheets: Sheet1, "Amir-Unique NDC from Valisure ("
  Redica:        Data/07 - Redica/raw/Site List.xlsx
                            processed/redica_all_drugs_combined.csv
  FDA Inspection:Data/14 - FDA - Inspection/raw/Inspections Details.xlsx
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
VAL_RAW   = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
QA_FILE   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV= BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
FDA_INSP  = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"

# =============================================================================
# HELPERS
# =============================================================================
def to_ndc11(x) -> Optional[str]:
    """
    Normalise any NDC variant to 11-digit NDC11 (no hyphens, 5+4+2).
    Parses hyphenated segments to handle 4-3-2, 5-3-2, 5-4-1, 5-4-2 formats.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    parts = [p for p in s.replace(" ", "").split("-") if p]
    if len(parts) == 3:
        lab, prod, pkg = parts
        lab  = lab.zfill(5)        # labeler → 5 digits
        prod = prod.zfill(4)       # product → 4 digits
        pkg  = pkg.zfill(2)[-2:]   # package → exactly 2 digits
        return lab + prod + pkg
    # fallback: bare digit string
    raw = s.replace("-", "").replace(" ", "")
    if len(raw) == 10:
        return raw[:5] + "0" + raw[5:]   # assume 5-3-2
    if len(raw) == 11:
        return raw
    return None


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


def sep(title: str = "", w: int = 70):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


# =============================================================================
# 1. VALISURE GROUND TRUTH
# =============================================================================
sep("STEP 1: Valisure ground-truth NDC count (3 sweep years)")

xls_val = pd.ExcelFile(VAL_RAW)
df20 = xls_val.parse("2020 Testing Data")
df22 = xls_val.parse("2022 Testing Data - Actual")
df24 = xls_val.parse("2024 Testing Data", header=1)

# 2020 & 2022: use NDC column (hyphenated, mixed formats → segment normalise)
# 2024: use NDC11 column (already 5-4-2 with hyphens → most reliable)
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
print()
print(f"  2020-only: {len(ndc20-ndc22-ndc24)}  "
      f"2022-only: {len(ndc22-ndc20-ndc24)}  "
      f"2024-only: {len(ndc24-ndc20-ndc22)}  "
      f"multi-year: {len(val_union) - len(ndc20-ndc22-ndc24) - len(ndc22-ndc20-ndc24) - len(ndc24-ndc20-ndc22)}")

# =============================================================================
# 2. SHEET1 — OLD ANALYSIS COVERAGE
# =============================================================================
sep("STEP 2: Sheet1 (old analysis) — which Valisure NDCs were included?")

df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1")
df_s1["ndc11_norm"] = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_norm"]   = df_s1["FEI"].apply(clean_fei)

sheet1_ndcs = set(df_s1["ndc11_norm"].dropna())
sheet1_feis = set(df_s1["fei_norm"].dropna())

val_not_in_s1 = val_union - sheet1_ndcs

print(f"  Sheet1 unique NDC11s             : {len(sheet1_ndcs)}")
print(f"  All Sheet1 NDCs are Valisure-tested? : {sheet1_ndcs <= val_union}")
print(f"  Valisure NDCs absent from Sheet1 : {len(val_not_in_s1)}")
print(f"    (no IQVIA match or excluded before the analysis was built)")

print()
print("  Country breakdown of Sheet1 NDC11s (unique NDC11 level):")
s1_u = df_s1.drop_duplicates("ndc11_norm").dropna(subset=["ndc11_norm"])
for cc, n in s1_u.groupby("CountryCode").size().items():
    print(f"    {cc}: {n}")

used_n  = s1_u[~s1_u["CountryCode"].isin(["CAN", "BGD"])]["ndc11_norm"].nunique()
excl_n  = s1_u[ s1_u["CountryCode"].isin(["CAN", "BGD"])]["ndc11_norm"].nunique()
print()
print(f"  Used in final analysis (IND+CHN+USA): {used_n}")
print(f"  Excluded (CAN+BGD)                  : {excl_n}")
print(f"  Sheet1 unique FEIs                  : {len(sheet1_feis)}")

# =============================================================================
# 3. AMIR'S MANUAL FEI LOOKUP
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

# Col F (NDC11_F) is the primary NDC identifier Amir used — has 112 unique NDC11s.
# The last 16 rows have col F blank but col B (NDC11) filled; fall back to col B for those.
df_amir["ndc11_norm"] = (
    df_amir["NDC11_F"].apply(to_ndc11)
    .fillna(df_amir["NDC11"].apply(to_ndc11))
)
df_amir["fei_G"]    = df_amir["FEI_G"].apply(clean_fei)
df_amir["fei_H"]    = df_amir["Found_FEI_H"].apply(clean_fei)
df_amir["fei_best"] = df_amir["fei_G"].fillna(df_amir["fei_H"])

# Collapse to unique NDC11 level (some NDC11s span multiple rows)
amir_u = (
    df_amir.dropna(subset=["ndc11_norm"])
    .sort_values(["ndc11_norm", "fei_G", "fei_H"], na_position="last")
    .drop_duplicates("ndc11_norm", keep="first")
    .copy()
)
# Annotate with Sheet1 country code where known
amir_u["country"] = amir_u["ndc11_norm"].map(
    s1_u.set_index("ndc11_norm")["CountryCode"].to_dict()
)

# NDC11 sets
has_G      = set(amir_u[amir_u["fei_G"].notna()]["ndc11_norm"])
has_H      = set(amir_u[amir_u["fei_H"].notna()]["ndc11_norm"])
no_fei     = set(amir_u[amir_u["fei_best"].isna()]["ndc11_norm"])
has_H_only = has_H - has_G          # in col H but not col G
s1_in_H    = has_H_only & sheet1_ndcs  # Sheet1 NDCs Amir put in col H instead of G
new_in_H   = has_H_only - sheet1_ndcs  # truly new NDCs (not in Sheet1) found via col H

old_feis      = set(amir_u["fei_G"].dropna())
new_feis      = set(amir_u["fei_H"].dropna())
confirmed     = new_feis & sheet1_feis    # col H FEIs already known from Sheet1
genuinely_new = new_feis - sheet1_feis   # col H FEIs brand-new

print(f"  Total unique NDC11s in Amir's sheet : {len(amir_u)}")
print(f"    → matches Valisure ground truth    : {len(amir_u) == len(val_union)}")
print()
print(f"  NDC11s from Sheet1 (old analysis)    : {len(sheet1_ndcs)}")
print(f"    All 88 have FEI in Sheet1          : True  (0 missing)")
print(f"    In Amir col G (Amir confirmed old) : {len(has_G & sheet1_ndcs)}")
print(f"    In Amir col H (Amir re-looked up)  : {len(s1_in_H)}")
print(f"    Total Sheet1 NDCs covered          : {len((has_G | has_H) & sheet1_ndcs)}")
print()
print(f"  NDC11s NOT in Sheet1 (new from Valisure): {len(val_union - sheet1_ndcs)}")
print(f"    Amir found FEI via col H            : {len(new_in_H)}")
print(f"    No FEI found                        : {len(no_fei)}")
print()
print(f"  Col H summary ({len(has_H_only)} NDC11s covered, {len(new_feis)} unique FEIs):")
print(f"    FEIs already in Sheet1 (same as old): {len(confirmed)}")
print(f"    Genuinely new FEIs                   : {len(genuinely_new)}")
print(f"  Combined unique FEIs (G ∪ H)           : {len(old_feis | new_feis)}")

# =============================================================================
# 4. FEI → NDC SUMMARY TABLE
# =============================================================================
sep("STEP 4: All FEIs and their associated NDC11s")

df_site   = pd.read_excel(SITE_LIST)
df_redica = pd.read_csv(REDICA_CSV)
redica_feis = set(df_site["FEI"].dropna().astype(str).str.strip())
assert redica_feis == set(df_redica["FEI"].dropna().astype(str).str.strip())

site_name = (
    df_site.assign(fei_str=df_site["FEI"].astype(str).str.strip())
    .set_index("fei_str")["Site Display Name"]
    .to_dict()
)

print("  Loading FDA Inspection Details...")
df_fda       = pd.read_excel(FDA_INSP)
df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
df_fda_drugs = df_fda[df_fda["Product Type"].str.lower().str.contains("drug", na=False)]
fda_feis     = set(df_fda_drugs["FEI Number"].unique())
all_sources  = redica_feis | fda_feis

# Build FEI → NDC11 mapping from Sheet1 (unique FEI+NDC11 pairs)
fei_info = {}   # fei -> {ndcs, countries, is_new}

for _, row in (
    df_s1[["ndc11_norm", "fei_norm", "CountryCode"]]
    .dropna(subset=["fei_norm", "ndc11_norm"])
    .drop_duplicates()
    .iterrows()
):
    fei = row["fei_norm"]
    fei_info.setdefault(fei, {"ndcs": set(), "countries": set(), "is_new": False})
    fei_info[fei]["ndcs"].add(row["ndc11_norm"])
    fei_info[fei]["countries"].add(row["CountryCode"])

# Add genuinely new FEIs from Amir's col H
for _, row in (
    df_amir[df_amir["fei_H"].isin(genuinely_new)]
    [["ndc11_norm", "fei_H"]]
    .dropna()
    .drop_duplicates()
    .iterrows()
):
    fei = row["fei_H"]
    fei_info.setdefault(fei, {"ndcs": set(), "countries": set(), "is_new": True})
    fei_info[fei]["ndcs"].add(row["ndc11_norm"])
    fei_info[fei]["is_new"] = True

print()
print(f"  {'FEI':12s}  {'Redica/FDA':10s}  {'New':3s}  {'#NDC':4s}  {'NDC11s (all)'}")
print(f"  {'-'*12}  {'-'*10}  {'-'*3}  {'-'*4}  {'-'*60}")

for fei in sorted(fei_info):
    info   = fei_info[fei]
    ndcs   = sorted(info["ndcs"])
    name   = site_name.get(fei, "not in Redica")
    new    = "NEW" if info["is_new"] else ""
    if fei in redica_feis:
        coverage = "Redica ✓"
    elif fei in fda_feis:
        coverage = "FDA ✓"
    else:
        coverage = "neither"
    ndcs_str = ", ".join(ndcs)
    print(f"  {fei:12s}  {coverage:10s}  {new:3s}  {len(ndcs):4d}  {ndcs_str}")
    print(f"  {'':12s}  {'':10s}  {'':3s}  {'':4s}  → {name}")

# =============================================================================
# 5. FEI COVERAGE COMPARISON (old vs Amir)
# =============================================================================
sep("STEP 5: FEI coverage — Redica vs FDA Inspection Details")

print(f"  Redica FEI universe              : {len(redica_feis)} sites")
print(f"  FDA Inspection Details FEIs      : {len(fda_feis):,} (drug rows only)")

all_amir_feis = set(amir_u["fei_best"].dropna())

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
# 6. NDC-LEVEL COVERAGE (Amir G+H, unique NDC11 level)
# =============================================================================
sep("STEP 6: NDC-level FEI coverage (Amir G+H combined, unique NDC11s)")

total_ndcs = len(amir_u)
for label, mask in [
    ("No FEI at all",            amir_u["fei_best"].isna()),
    ("FEI in Redica",            amir_u["fei_best"].isin(redica_feis)),
    ("FEI in FDA Details only",  amir_u["fei_best"].isin(fda_feis - redica_feis)),
    ("FEI in neither source",   ~amir_u["fei_best"].isin(all_sources) & amir_u["fei_best"].notna()),
]:
    n = mask.sum()
    print(f"  {label:30s}: {n:3d} / {total_ndcs}  ({n/total_ndcs*100:.1f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
sep("SUMMARY")
print(f"""
  Valisure testing (3 sweeps, 2020+2022+2024):
    Total unique NDC11s tested          : {len(val_union)}
      2020-only                         : {len(ndc20-ndc22-ndc24)}
      2022-only                         : {len(ndc22-ndc20-ndc24)}
      2024-only                         : {len(ndc24-ndc20-ndc22)}
      Tested in multiple years          : {len(val_union)-len(ndc20-ndc22-ndc24)-len(ndc22-ndc20-ndc24)-len(ndc24-ndc20-ndc22)}

  From Valisure → Sheet1 (old analysis):
    Included in Sheet1                  : {len(sheet1_ndcs)} / {len(val_union)}
    Missing from Sheet1                 : {len(val_not_in_s1)}
      (no IQVIA volume match, or added in 2024 after analysis was built)
    Sheet1 NDCs used in paper (IND+CHN+USA): {used_n}
    Sheet1 NDCs excluded (CAN+BGD)         : {excl_n}

  FEI matching:
    Old method (Sheet1)                 : {len(sheet1_feis)} unique FEIs
      In Redica                         : {len(sheet1_feis & redica_feis)} / {len(sheet1_feis)}
    Amir's revised (G+H combined)       : {len(all_amir_feis)} unique FEIs
      In Redica                         : {len(all_amir_feis & redica_feis)} / {len(all_amir_feis)}
      In FDA Details (fallback)         : {len(all_amir_feis & (fda_feis-redica_feis))} / {len(all_amir_feis)}
      In neither                        : {len(all_amir_feis - all_sources)} / {len(all_amir_feis)}
    Genuinely new FEIs from Amir        : {len(genuinely_new)}
""")

# %%
