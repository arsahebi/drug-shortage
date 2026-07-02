# %%
"""
NDC→FEI Coverage Builder
==========================
Produces two layered output files from the Amir & Amirreza master mapping.

FILE 1 — 20260701_ndc_fei_map.csv
  One row per (NDC, FEI) pair. Pure mapping layer; no Redica data.
  Columns: NDC, FEI, ndc_fei_origin, fei_rank

FILE 2 — 20260701_ndc_fei_redica_coverage.csv
  One row per (NDC, FEI, inspection event). Redica coverage added;
  origin and rank columns dropped.
  Columns: NDC, FEI, EventYear, Classification,
           fei_redica   — whether FEI appears in old / new / both / neither Redica
           insp_coverage — whether this inspection row came from old / new / both

The pair of files supports the paper claim: "we obtained updated Redica data
covering X% of records that were not available in the original export."
"""

import re
import pandas as pd
from pathlib import Path

BASE       = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
MASTER     = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_master.csv"
QA_FILE    = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
INSP_HIST  = BASE / "Data/07 - Redica/processed/valisure_fei_inspection_history.csv"
OUT_MAP    = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_map.csv"
OUT_COV    = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_redica_coverage.csv"


# ── helpers ───────────────────────────────────────────────────────────────────
def to_ndc11(x):
    if pd.isna(x): return None
    s = str(x).strip()
    parts = [p for p in s.replace(" ", "").split("-") if p]
    if len(parts) == 3:
        lab, prod, pkg = parts
        return lab.zfill(5) + prod.zfill(4) + pkg.zfill(2)[-2:]
    raw = s.replace("-", "").replace(" ", "")
    if len(raw) == 10: return raw[:5] + "0" + raw[5:]
    if len(raw) == 11: return raw
    return None

def clean_fei(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s): return None
    try: return str(int(float(s)))
    except: return None

def fmt_ndc(n11):
    return f"{n11[:5]}-{n11[5:9]}-{n11[9:]}" if n11 and len(n11) == 11 else n11


# ── load master ───────────────────────────────────────────────────────────────
master = pd.read_csv(MASTER, dtype=str)
master["ndc11"] = master["ndc11"].apply(to_ndc11)
master["fei"]   = master["fei"].apply(clean_fei)
master["NDC"]   = master["ndc11"].apply(fmt_ndc)
master = master.dropna(subset=["ndc11"]).reset_index(drop=True)


# ── ndc_fei_origin: compare with Sheet1 ──────────────────────────────────────
df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1", dtype=str)
df_s1["ndc11"]   = df_s1["NDC11"].apply(to_ndc11)
df_s1["fei_s1"]  = df_s1["FEI"].apply(clean_fei)
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


# =============================================================================
# FILE 1 — pure mapping layer
# =============================================================================
map_out = master[["NDC", "fei", "ndc_fei_origin", "fei_rank"]].rename(columns={"fei": "FEI"})
map_out.to_csv(OUT_MAP, index=False)
print(f"FILE 1 saved: {OUT_MAP.name}")
print(f"  {len(map_out)} rows  ({map_out['FEI'].notna().sum()} with FEI)\n")

print("  ndc_fei_origin breakdown:")
print(map_out["ndc_fei_origin"].value_counts().to_string())
print()
print("  fei_rank breakdown:")
print(map_out["fei_rank"].value_counts().to_string())


# =============================================================================
# FILE 2 — Redica coverage layer
# =============================================================================
df_hist = pd.read_csv(INSP_HIST, dtype=str)
df_hist["FEI"]      = df_hist["FEI"].str.strip()
df_hist["EventYear"] = pd.to_numeric(df_hist["EventYear"], errors="coerce").astype(pd.Int64Dtype())

# FEI-level: which Redica sources cover each FEI?
fei_in_old = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "METFORMIN_old only"]), "FEI"])
fei_in_new = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "Valisure14 only"]),    "FEI"])

def _fei_redica(fei):
    if pd.isna(fei): return None
    old = fei in fei_in_old
    new = fei in fei_in_new
    if old and new:  return "both"
    if old:          return "old only"
    if new:          return "new only"
    return "neither"

# Normalise Insp_coverage label to match fei_redica vocabulary
COV_MAP = {
    "both":               "both",
    "METFORMIN_old only": "old only",
    "Valisure14 only":    "new only",
}

# Join: (NDC, FEI) pairs from master × inspection history
pairs = master[master["fei"].notna()][["NDC", "ndc11", "fei"]].rename(columns={"fei": "FEI"})
hist_cols = ["FEI", "EventYear", "Classification", "Insp_coverage"]
panel = pairs.merge(df_hist[hist_cols], on="FEI", how="left")

# NDCs with no FEI — keep as stub rows (no inspection data)
no_fei = master[master["fei"].isna()][["NDC", "ndc11"]].copy()
no_fei["FEI"] = None
for c in ["EventYear", "Classification", "Insp_coverage"]:
    no_fei[c] = None
panel = pd.concat([panel, no_fei], ignore_index=True)

panel["fei_redica"]    = panel["FEI"].apply(_fei_redica)
panel["insp_coverage"] = panel["Insp_coverage"].map(COV_MAP)

cov_out = (
    panel[["NDC", "FEI", "EventYear", "Classification", "fei_redica", "insp_coverage"]]
    .sort_values(["NDC", "FEI", "EventYear"], na_position="last")
    .reset_index(drop=True)
)
cov_out.to_csv(OUT_COV, index=False)
print(f"\n\nFILE 2 saved: {OUT_COV.name}")
print(f"  {len(cov_out)} rows  ({cov_out['FEI'].notna().sum()} with FEI)\n")

insp_rows = cov_out[cov_out["insp_coverage"].notna()]
total_insp = len(insp_rows)
print(f"  Inspection rows by source ({total_insp} total):")
cov_tbl = insp_rows["insp_coverage"].value_counts()
for label, n in cov_tbl.items():
    print(f"    {label:<12} {n:>4}  ({n/total_insp*100:.1f}%)")

print(f"\n  FEI Redica coverage (unique FEIs with mapping):")
fei_tbl = cov_out.dropna(subset=["FEI"]).drop_duplicates("FEI")["fei_redica"].value_counts()
for label, n in fei_tbl.items():
    print(f"    {label:<12} {n:>3} FEIs")
# %%
