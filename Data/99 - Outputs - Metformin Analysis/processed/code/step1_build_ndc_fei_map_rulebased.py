# %%
"""
Step 1 (rule-based) — Build NDC→FEI Map
========================================
Replaces the manual-search map (step1_build_ndc_fei_map_manual.py).

Source: step1_ndc_fei_map_rulebased.xlsx
  Tab 1 "NDCs from Valisure Data"   — the NDC universe (Valisure-tested NDCs)
      col A Valisure NDC (5-3-2)  col B ndc_11 (5-4-2)  col C ndc_9 (5-4)
      col D "manufacture found"   — 1 if a manufacture-operation FEI exists
      col E "any found"           — 1 if any DailyMed establishment operation exists
  Tab 2 "Filtered Linkage NDC-FEI" — DailyMed establishment operations per ndc_9
      opr_type ∈ {manufacture, analysis, pack, label, repack, relabel}
      FEI = establishment FEI for that operation
  Tab 3 "Manufacture Only"        — tab 2 pre-filtered to opr_type == "manufacture"
                                    (verified identical to our filter; not read)

Rule
----
Assign an FEI to an NDC ONLY when the DailyMed establishment operation for that
NDC's ndc_9 is opr_type == "manufacture". Analysis / pack / label / repack /
relabel sites are dropped — they are not the manufacturing facility whose
inspection history we want to attribute to the product.

Output: step1_ndc_fei_map_rulebased.csv
  NDC, NDC11, NDC8, NDC9, FEI, manufacturer_name, fei_count, facility_distance_km

NDC8 and facility_distance_km are carried for schema compatibility with the
manual map consumed by step2. facility_distance_km came from manual lookup and
has no rule-based equivalent, so it is empty.
"""

import re
from typing import Optional

import pandas as pd
from pathlib import Path

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
SRC  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map_rulebased.xlsx"
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map_rulebased.csv"

TAB_NDC  = "NDCs from Valisure Data"
TAB_LINK = "Filtered Linkage NDC-FEI"
MANUFACTURE = "manufacture"


# ── helpers ───────────────────────────────────────────────────────────────────
def clean_fei(x) -> Optional[str]:
    if pd.isna(x) or str(x).strip().lower() in ("nan", "", "not found"):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return None


def norm_ndc11(x) -> Optional[str]:
    """5-4-2 with zero-padded segments."""
    if pd.isna(x):
        return None
    parts = [p for p in str(x).strip().replace(" ", "").split("-") if p]
    if len(parts) != 3:
        return None
    lab, prod, pkg = parts
    return f"{lab.zfill(5)}-{prod.zfill(4)}-{pkg.zfill(2)}"


def norm_ndc9(x) -> Optional[str]:
    """5-4 with zero-padded segments (the join key between the two tabs)."""
    if pd.isna(x):
        return None
    parts = [p for p in str(x).strip().replace(" ", "").split("-") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0].zfill(5)}-{parts[1].zfill(4)}"


def ndc_display(n11: str) -> str:
    """5-3-2 display form used as the `NDC` column downstream."""
    lab, prod, pkg = n11.split("-")
    return f"{lab}-{prod.lstrip('0').zfill(3)}-{pkg}"


def ndc8(n11: str) -> str:
    """5-3 labeler-product key (legacy column kept for step2 compatibility)."""
    lab, prod, _ = n11.split("-")
    return f"{lab}-{prod.lstrip('0').zfill(3)}"


# ── load ──────────────────────────────────────────────────────────────────────
xls  = pd.ExcelFile(SRC)
ndcs = pd.read_excel(xls, sheet_name=TAB_NDC,  dtype=str)
link = pd.read_excel(xls, sheet_name=TAB_LINK, dtype=str)

ndcs["NDC11"] = ndcs["ndc_11"].apply(norm_ndc11)
ndcs["NDC9"]  = ndcs["ndc_9"].apply(norm_ndc9)
ndcs = ndcs.dropna(subset=["NDC11"]).drop_duplicates("NDC11").reset_index(drop=True)

link["NDC9"]    = link["ndc_9"].apply(norm_ndc9)
link["FEI"]     = link["FEI"].apply(clean_fei)
link["op"]      = link["opr_type"].astype(str).str.strip().str.lower()

# ── rule: manufacture operations only ────────────────────────────────────────
mfg = (
    link[(link["op"] == MANUFACTURE) & link["FEI"].notna() & link["NDC9"].notna()]
    .drop_duplicates(["NDC9", "FEI"])
    [["NDC9", "FEI", "manufacturer_name"]]
    .sort_values(["NDC9", "FEI"])
)

dropped = link[(link["op"] != MANUFACTURE) & link["FEI"].notna()]
print(f"Establishment operations in tab 2 : {link['FEI'].notna().sum()}")
print(f"  kept (opr_type == manufacture)  : {len(mfg)}")
print(f"  dropped (non-manufacture)       : {len(dropped)}")
print(dropped["op"].value_counts().to_string())

# sanity check against col D of tab 1
has_mfg = set(mfg["NDC9"])
flag    = pd.to_numeric(ndcs["manufacture found"], errors="coerce").fillna(0).astype(int)
mismatch = ndcs[(flag == 1) != ndcs["NDC9"].isin(has_mfg)]
if len(mismatch):
    print(f"\n!! {len(mismatch)} NDC(s) disagree with col D 'manufacture found':")
    print(mismatch[["ndc_11", "ndc_9", "manufacture found"]].to_string(index=False))
else:
    print("\nCol D 'manufacture found' agrees with the tab-2 manufacture filter for all NDCs.")

# ── emit one row per (NDC11, manufacturer FEI) ───────────────────────────────
out = ndcs[["NDC11", "NDC9"]].merge(mfg, on="NDC9", how="left")
out["NDC"]  = out["NDC11"].apply(ndc_display)
out["NDC8"] = out["NDC11"].apply(ndc8)

n_per_ndc = out[out["FEI"].notna()].groupby("NDC11")["FEI"].nunique()
out["fei_count"] = out["NDC11"].map(
    lambda n: "Not Applicable" if n not in n_per_ndc
    else ("Single - Manufacture" if n_per_ndc[n] == 1 else "Multi FEI - Manufacture")
)
out["facility_distance_km"] = None

COLS = ["NDC", "NDC11", "NDC8", "NDC9", "FEI",
        "manufacturer_name", "fei_count", "facility_distance_km"]
out = (out[COLS]
       .drop_duplicates(["NDC11", "FEI"])
       .sort_values(["NDC11", "FEI"], na_position="last")
       .reset_index(drop=True))

out.to_csv(OUT, index=False)

print(f"\nSaved: {OUT}")
print(f"Rows                  : {len(out)}")
print(f"Unique NDC11          : {out['NDC11'].nunique()}")
print(f"NDC11 with an FEI     : {out[out['FEI'].notna()]['NDC11'].nunique()}")
print(f"NDC11 with no FEI     : {out[out['FEI'].isna()]['NDC11'].nunique()}")
print(f"Unique FEIs           : {out['FEI'].dropna().nunique()}")
print(f"\nfei_count breakdown (per NDC11):")
print(out.drop_duplicates("NDC11")["fei_count"].value_counts().to_string())
print(f"\nMulti-FEI NDC11s:")
print(out[out["NDC11"].isin(n_per_ndc[n_per_ndc > 1].index)]
      [["NDC11", "FEI", "manufacturer_name"]].to_string(index=False))
# %%
