# %%
"""
Step 1 — Build NDC→FEI Map
============================
Source: "Amir and Amirreza Review all ND" tab in Q&As1234_v8_v02.xlsx
  - Col D = primary FEI per NDC11
  - Col L notes: "The other FEI is XXXXXXX" → secondary row for that NDC

Origin classification vs Sheet1 (original paper mapping):
  Sheet1 confirmed       — same FEI as Sheet1
  FEI updated            — FEI changed from Sheet1
  Sheet1 – newly matched — NDC was in Sheet1 but had no FEI; now found
  New assignment         — NDC not in Sheet1 at all
  Two FEIs – secondary   — second manufacturing FEI for same NDC
  Unmatched              — no FEI found in either source

Output: step1_ndc_fei_map.csv
  NDC, FEI, ndc_fei_origin, fei_rank
"""

import re
import pandas as pd
from pathlib import Path

BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA_FILE = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
OUT     = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map.csv"

# ── helpers ───────────────────────────────────────────────────────────────────
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

def fmt_ndc(n11):
    return f"{n11[:5]}-{n11[5:9]}-{n11[9:]}" if n11 and len(n11) == 11 else n11

def extract_other_fei(note):
    if not isinstance(note, str):
        return None
    m = re.search(r"[Tt]he other FEI is\s+(\d{5,10})", note)
    return m.group(1) if m else None

def _has(x):
    return x is not None and not (isinstance(x, float) and pd.isna(x))

# ── load Amir & Amirreza tab ──────────────────────────────────────────────────
xls = pd.ExcelFile(QA_FILE)
nt  = pd.read_excel(xls, sheet_name="Amir and Amirreza Review all ND",
                    header=None, dtype=str)
nd  = nt.iloc[1:].copy()
nd.columns = ["a", "ndc11_raw", "dm_friendly", "fei_primary_raw",
              "e", "f", "assigned_to", "dm_r", "fei_r", "how_found",
              "propublica", "notes"]

nd["ndc11"]       = nd["ndc11_raw"].apply(to_ndc11)
nd["fei_primary"] = nd["fei_primary_raw"].apply(clean_fei)
nd["fei_other"]   = nd["notes"].apply(extract_other_fei)

# reviewer / method from right-side columns (keyed by dm_friendly)
how_map      = nd.dropna(subset=["dm_r"]).drop_duplicates("dm_r").set_index("dm_r")
nd["reviewer"] = nd["dm_friendly"].map(how_map["assigned_to"])
nd["method"]   = nd["dm_friendly"].map(how_map["how_found"])

nd = nd.dropna(subset=["ndc11"]).drop_duplicates("ndc11").reset_index(drop=True)

# ── expand: one row per (NDC, FEI) ───────────────────────────────────────────
rows = []
for _, r in nd.iterrows():
    p, o = r["fei_primary"], r["fei_other"]
    base = {"ndc11": r["ndc11"]}
    if _has(p):
        rows.append({**base, "fei": p, "fei_rank": "primary"})
        if _has(o) and o != p:
            rows.append({**base, "fei": o, "fei_rank": "secondary_two_duns"})
    elif _has(o):
        rows.append({**base, "fei": o, "fei_rank": "primary_from_note"})
    else:
        rows.append({**base, "fei": None, "fei_rank": "not_found"})

master = pd.DataFrame(rows)

# ── ndc_fei_origin vs Sheet1 ──────────────────────────────────────────────────
df_s1 = pd.read_excel(xls, sheet_name="Sheet1", dtype=str)
df_s1["ndc11"]  = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_s1"] = df_s1["FEI"].apply(clean_fei)
s1_fei = (
    df_s1.dropna(subset=["ndc11", "fei_s1"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["fei_s1"]
)
sheet1_ndcs = set(s1_fei.index)

def _origin(row):
    ndc, fei, rank = row["ndc11"], row["fei"], row["fei_rank"]
    if rank == "not_found":
        return "Unmatched"
    if rank == "secondary_two_duns":
        return "Two FEIs – secondary"
    if ndc in sheet1_ndcs:
        old = s1_fei.get(ndc)
        if old and old == fei:  return "Sheet1 confirmed"
        if old:                 return "FEI updated"
        return "Sheet1 – newly matched"
    return "New assignment"

master["ndc_fei_origin"] = master.apply(_origin, axis=1)
master["NDC"]            = master["ndc11"].apply(fmt_ndc)

# ── save ──────────────────────────────────────────────────────────────────────
out = master[["NDC", "fei", "ndc_fei_origin", "fei_rank"]].rename(columns={"fei": "FEI"})
out.to_csv(OUT, index=False)

print(f"Saved: {OUT}")
print(f"\nTotal rows : {len(out)}  ({out['FEI'].notna().sum()} with FEI)")
print(f"\nndc_fei_origin:")
print(out["ndc_fei_origin"].value_counts().to_string())
print(f"\nfei_rank:")
print(out["fei_rank"].value_counts().to_string())
# %%
