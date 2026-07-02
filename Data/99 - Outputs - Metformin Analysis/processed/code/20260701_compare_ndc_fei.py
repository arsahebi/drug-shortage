# %%
"""
NDC→FEI Comparison: Sheet1 (old) vs Amir & Amirreza tab (new)
==============================================================
Old FEI source : Sheet1 tab in Q&As1234_v8_v02.xlsx (88 NDCs)
New FEI source : "Amir and Amirreza Review all ND" tab, col D (112 NDCs)

Output: one row per NDC with fei_old, fei_new, and status
  same           — Sheet1 FEI confirmed by new review
  changed        — FEI corrected to a different one
  new_assignment — not in Sheet1, new review found a FEI
  lost           — Sheet1 had a FEI, new review returned Not Found
  both_missing   — neither source has a FEI
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
DRLS = BASE / "Data/05 - Firm Level/drls_reg.xlsx"
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_comparison.csv"


# =============================================================================
# HELPERS
# =============================================================================
def clean_fei(x):
    if pd.isna(x) or str(x).strip().lower() in ("nan", "", "not found"):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return None


def to_ndc11(x):
    if pd.isna(x) or str(x).strip() in ("", "nan"):
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


def has_val(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def get_status(n, o):
    n_ok, o_ok = has_val(n), has_val(o)
    if n_ok and o_ok:      return "same" if n == o else "changed"
    if n_ok and not o_ok:  return "new_assignment"
    if not n_ok and o_ok:  return "lost"
    return "both_missing"


def fmt_ndc(n):
    return f"{n[:5]}-{n[5:9]}-{n[9:]}" if n and len(n) == 11 else n


# =============================================================================
# LOAD DATA
# =============================================================================
xls  = pd.ExcelFile(QA)
drls = pd.read_excel(DRLS, dtype=str)
drls["FEI_NUMBER"] = drls["FEI_NUMBER"].str.strip()
FEI_FIRM = drls.drop_duplicates("FEI_NUMBER").set_index("FEI_NUMBER")["FIRM_NAME"].to_dict()

# OLD: Sheet1
s1 = pd.read_excel(xls, sheet_name="Sheet1", dtype=str)
s1["ndc11"] = s1["NDC11"].apply(to_ndc11)
s1["fei"]   = s1["FEI"].apply(clean_fei)
S1_FEI = (
    s1.dropna(subset=["ndc11", "fei"])
      .drop_duplicates("ndc11")
      .set_index("ndc11")["fei"]
)
print(f"Sheet1: {len(S1_FEI)} NDCs with FEI")

# NEW: Amir and Amirreza tab (col D = fei_new for all 112 NDCs)
nt = pd.read_excel(xls, sheet_name="Amir and Amirreza Review all ND", header=None, dtype=str)
nd = nt.iloc[1:].copy()
nd.columns = ["a", "ndc11_raw", "dm_friendly", "fei_new_raw", "e", "f",
              "assigned_to", "dm_r", "fei_r", "how_found", "propublica", "notes"]
nd["ndc11"]   = nd["ndc11_raw"].apply(to_ndc11)
nd["fei_new"] = nd["fei_new_raw"].apply(clean_fei)
# Pull method/reviewer from the right-side columns (keyed by DailyMed Friendly)
how_map      = nd.dropna(subset=["dm_r"]).drop_duplicates("dm_r").set_index("dm_r")
nd["method"]   = nd["dm_friendly"].map(how_map["how_found"])
nd["reviewer"] = nd["dm_friendly"].map(how_map["assigned_to"])

new_df = (
    nd[["ndc11", "ndc11_raw", "dm_friendly", "fei_new", "method", "reviewer"]]
    .dropna(subset=["ndc11"])
    .drop_duplicates("ndc11")
    .reset_index(drop=True)
)
print(f"Amir & Amirreza tab: {len(new_df)} NDCs, "
      f"{new_df['fei_new'].apply(has_val).sum()} with FEI\n")


# =============================================================================
# COMPARE
# =============================================================================
new_df["fei_old"]  = new_df["ndc11"].map(S1_FEI)
new_df["status"]   = new_df.apply(lambda r: get_status(r["fei_new"], r["fei_old"]), axis=1)
new_df["firm_old"] = new_df["fei_old"].map(FEI_FIRM)
new_df["firm_new"] = new_df["fei_new"].map(FEI_FIRM)
new_df["ndc_display"] = new_df["ndc11"].apply(fmt_ndc)

cols = ["ndc_display", "ndc11", "dm_friendly",
        "fei_old", "firm_old",
        "fei_new", "firm_new",
        "reviewer", "method", "status"]
new_df[cols].to_csv(OUT, index=False)
print(f"Saved: {OUT}\n")


# =============================================================================
# REPORT
# =============================================================================
print("=" * 60)
print(f"TOTAL NDCs : {len(new_df)}")
print(f"  Old (Sheet1) had FEI : {new_df['fei_old'].apply(has_val).sum()}  "
      f"({new_df['fei_old'].dropna().nunique()} unique FEIs)")
print(f"  New (A&A)  found FEI : {new_df['fei_new'].apply(has_val).sum()}  "
      f"({new_df['fei_new'].dropna().nunique()} unique FEIs)")
print()
print("STATUS BREAKDOWN:")
print(new_df["status"].value_counts().to_string())
print()

print("── SAME (Sheet1 FEI confirmed) " + "─" * 29)
print(f"  {(new_df['status']=='same').sum()} NDCs")
print()

print("── CHANGED (FEI corrected to a different one) " + "─" * 14)
ch = new_df[new_df["status"] == "changed"][
    ["ndc_display", "fei_old", "firm_old", "fei_new", "firm_new"]]
print(ch.to_string(index=False) if len(ch) else "  None")
print()

print("── NEW ASSIGNMENT (not in Sheet1, now found) " + "─" * 15)
na = new_df[new_df["status"] == "new_assignment"][
    ["ndc_display", "fei_new", "firm_new", "reviewer"]]
print(na.to_string(index=False) if len(na) else "  None")
print()

print("── LOST (Sheet1 had FEI, new review = Not Found) " + "─" * 10)
lo = new_df[new_df["status"] == "lost"][
    ["ndc_display", "fei_old", "firm_old"]]
print(lo.to_string(index=False) if len(lo) else "  None")
print()

print("── BOTH MISSING (unresolvable) " + "─" * 29)
bm = new_df[new_df["status"] == "both_missing"][["ndc_display", "reviewer"]]
print(bm.to_string(index=False) if len(bm) else "  None")
# %%
