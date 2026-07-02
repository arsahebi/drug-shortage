# %%
"""
Build NDC→FEI Master Mapping (Amir & Amirreza review)
======================================================
Source: "Amir and Amirreza Review all ND" tab in Q&As1234_v8_v02.xlsx
  - Col D (left side) = primary FEI per NDC11
  - Notes col L: when it says "There are two DUNS number ... The other FEI is X"
    → create a second row for that NDC with FEI = X

Output: 20260701_ndc_fei_master.csv
  ndc11, ndc_display, fei, fei_rank (primary/secondary), reviewer, method, notes
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_master.csv"

# ── helpers ──────────────────────────────────────────────────────────────────
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

def fmt_ndc(n):
    return f"{n[:5]}-{n[5:9]}-{n[9:]}" if n and len(n) == 11 else n

def extract_other_fei(note):
    """Return FEI string from 'The other FEI is XXXXXXX' in notes, or None."""
    if not isinstance(note, str):
        return None
    m = re.search(r"[Tt]he other FEI is\s+(\d{5,10})", note)
    return m.group(1) if m else None

# ── load tab ─────────────────────────────────────────────────────────────────
xls = pd.ExcelFile(QA)
nt  = pd.read_excel(xls, sheet_name="Amir and Amirreza Review all ND",
                    header=None, dtype=str)
nd  = nt.iloc[1:].copy()
nd.columns = ["a", "ndc11_raw", "dm_friendly", "fei_primary_raw",
              "e", "f", "assigned_to", "dm_r", "fei_r", "how_found",
              "propublica", "notes"]

nd["ndc11"]       = nd["ndc11_raw"].apply(to_ndc11)
nd["fei_primary"] = nd["fei_primary_raw"].apply(clean_fei)
nd["fei_other"]   = nd["notes"].apply(extract_other_fei)


def _has(x):
    """True only for non-None, non-NaN values (avoids np.nan truthy bug)."""
    return x is not None and not (isinstance(x, float) and pd.isna(x))

# inherit reviewer / method from right-side (keyed by dm_friendly)
how_map    = nd.dropna(subset=["dm_r"]).drop_duplicates("dm_r").set_index("dm_r")
nd["reviewer"] = nd["dm_friendly"].map(how_map["assigned_to"])
nd["method"]   = nd["dm_friendly"].map(how_map["how_found"])

nd = nd.dropna(subset=["ndc11"]).drop_duplicates("ndc11").reset_index(drop=True)

# ── expand to one row per (NDC, FEI) ─────────────────────────────────────────
rows = []
for _, r in nd.iterrows():
    base = {
        "ndc11":       r["ndc11"],
        "ndc_display": fmt_ndc(r["ndc11"]),
        "dm_friendly": r["dm_friendly"],
        "reviewer":    r["reviewer"],
        "method":      r["method"],
        "notes":       r["notes"],
    }
    p = r["fei_primary"]
    o = r["fei_other"]
    if _has(p):
        rows.append({**base, "fei": p, "fei_rank": "primary"})
        if _has(o) and o != p:
            rows.append({**base, "fei": o, "fei_rank": "secondary_two_duns"})
    elif _has(o):
        # primary is Not Found but secondary exists — treat as sole assignment
        rows.append({**base, "fei": o, "fei_rank": "primary_from_note"})
    else:
        rows.append({**base, "fei": None, "fei_rank": "not_found"})

master = pd.DataFrame(rows)
master.to_csv(OUT, index=False)

# ── stats ─────────────────────────────────────────────────────────────────────
total_ndcs   = nd["ndc11"].nunique()
found        = master[master["fei"].notna()]
two_fei_ndcs = master[(master["fei_rank"] == "secondary_two_duns") & master["fei"].notna()]["ndc11"].nunique()
not_found    = master[master["fei"].isna()]["ndc11"].nunique()

print(f"Saved: {OUT}")
print()
print(f"Total NDCs          : {total_ndcs}")
print(f"NDCs with ≥1 FEI    : {found['ndc11'].nunique()}")
print(f"NDCs with 2 FEIs    : {two_fei_ndcs}")
print(f"NDCs with no FEI    : {not_found}")
print(f"Total NDC×FEI rows  : {len(master[master['fei'].notna()])}")
print(f"Unique FEIs         : {master['fei'].nunique()}")
print()
print("FEI rank breakdown:")
print(master["fei_rank"].value_counts().to_string())
print()
print("Two-FEI NDCs (primary + secondary):")
two = master[master["fei_rank"] == "secondary_two_duns"][["ndc_display", "ndc11"]]
for ndc11 in two["ndc11"]:
    sub = master[master["ndc11"] == ndc11][["ndc_display", "fei", "fei_rank"]]
    print(f"  {sub.to_string(index=False)}")
# %%
