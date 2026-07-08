# %%
"""
Step 1 — Build NDC→FEI Map
============================
Two-step algorithm using only col H and col I from the Q&A tab:

  Step 1: Build col H (DailyMed-friendly NDC) → set of FEIs from col I.
          One col H can have multiple FEIs (multi-FEI NDCs) → multiple rows later.

  Step 2: For each col B NDC11, extract the "up to second dash" key (= NDC8,
          the 5-3 labeler-product portion, e.g. 71093-0132-04 → 71093-132).
          Look up that key in the col H map to get FEI(s).
          Emit one row per unique (NDC11, FEI) pair.

Output: step1_ndc_fei_map.csv
  NDC, NDC11, NDC8, FEI, fei_count, facility_distance_km
"""

import re
from collections import defaultdict
from typing import Optional

import pandas as pd
from pathlib import Path

BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA_FILE = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
OUT     = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map.csv"

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

def to_ndc11(x) -> Optional[str]:
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

def ndc_formats(n11: str):
    """Return (display 5-3-2, NDC11 5-4-2, NDC8 5-3) for a bare 11-digit string."""
    lab, prod4, pkg = n11[:5], n11[5:9], n11[9:]
    prod3 = prod4.lstrip("0").zfill(3) if prod4.lstrip("0") else "000"
    return f"{lab}-{prod3}-{pkg}", f"{lab}-{prod4}-{pkg}", f"{lab}-{prod3}"

def dm_key(s: str) -> Optional[str]:
    """Canonical DailyMed-friendly key: strip leading zeros from labeler AND product.
    Matches col H format (e.g. '00904-7164' and '0904-7164' both → '904-7164')."""
    if not s:
        return None
    parts = s.split("-")
    if len(parts) < 2:
        return None
    lab  = parts[0].lstrip("0") or "0"
    prod = parts[1].lstrip("0").zfill(3) if parts[1].lstrip("0") else "000"
    return f"{lab}-{prod}"

def norm(x) -> Optional[str]:
    return str(x).strip() if pd.notna(x) and str(x).strip() else None

# ── load tab ──────────────────────────────────────────────────────────────────
xls = pd.ExcelFile(QA_FILE)
nt  = pd.read_excel(xls, sheet_name="Amir and Amirreza Review all ND",
                    header=None, dtype=str)
nd  = nt.iloc[1:].copy()
nd.columns = ["a", "ndc11_raw", "dm_friendly", "fei_primary_raw",
              "e", "f", "assigned_to", "dm_r", "fei_r", "fei_old_raw",
              "same", "how_found", "propublica", "prop_status", "fei_count",
              "facility_distance_km", "notes"]
nd = nd.dropna(subset=["ndc11_raw"]).drop_duplicates("ndc11_raw").reset_index(drop=True)

# ── Step 1: col H → {FEIs, fei_count, distance} ──────────────────────────────
h_feis: dict[str, set]  = defaultdict(set)
h_meta: dict[str, dict] = {}

for _, r in nd.iterrows():
    h = dm_key(norm(r["dm_r"]) or "")   # col H, normalized
    if not h:
        continue
    fei  = clean_fei(r["fei_r"])         # col I
    dist = r["facility_distance_km"]
    dist = None if pd.isna(dist) else dist
    if fei:
        h_feis[h].add(fei)
    if h not in h_meta:
        h_meta[h] = {"fei_count": norm(r["fei_count"]), "dist": dist}

# ── Step 2: match col B NDC11 → col H via NDC8 (up to second dash) ───────────
rows = []
for _, r in nd.iterrows():
    n11 = to_ndc11(r["ndc11_raw"])
    if not n11:
        continue
    ndc_display, ndc11_fmt, ndc8 = ndc_formats(n11)
    key  = dm_key(ndc8)   # normalized "up to second dash" — matches col H canonical form

    feis = sorted(h_feis.get(key, set())) or [None]
    meta = h_meta.get(key, {"fei_count": norm(r["fei_count"]), "dist": None})

    for fei in feis:
        rows.append({
            "NDC":                  ndc_display,
            "NDC11":                ndc11_fmt,
            "NDC8":                 ndc8,
            "FEI":                  fei,
            "fei_count":            meta["fei_count"],
            "facility_distance_km": meta["dist"],
        })

out = pd.DataFrame(rows).drop_duplicates(["NDC11", "FEI"])
out.to_csv(OUT, index=False)

print(f"Saved: {OUT}")
print(f"\nTotal rows       : {len(out)}  ({out['FEI'].notna().sum()} with FEI)")
print(f"Unique FEIs      : {out['FEI'].dropna().nunique()}")
print(f"NDCs with no FEI : {out['FEI'].isna().sum()}")
print(f"\nSample — 71093 rows:")
print(out[out["NDC11"].str.contains("71093", na=False)]
      [["NDC","NDC11","NDC8","FEI","fei_count"]].to_string(index=False))
print(f"\nNDC11s with multiple FEIs (multi-FEI via col H group):")
dups = out[out.duplicated("NDC11", keep=False) & out["FEI"].notna()]
print(dups[["NDC","NDC11","FEI","fei_count","facility_distance_km"]].head(12).to_string(index=False))
# %%
