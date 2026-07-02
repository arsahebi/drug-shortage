# %%
"""
Step 2 — Build Inspection Panel
=================================
Reads step1_ndc_fei_map.csv and joins all Redica inspection data to produce
one row per (NDC × FEI × inspection event).

Coverage columns added:
  fei_redica    — whether this FEI appears in old / new / both / neither Redica export
  insp_coverage — whether this inspection row came from old / new / both

Output: step2_panel.csv
"""

import re
from typing import Optional
import pandas as pd
from pathlib import Path

BASE       = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP1      = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map.csv"
VAL_RAW    = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
QA_FILE    = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST  = BASE / "Data/07 - Redica/raw/Site List.xlsx"
REDICA_CSV = BASE / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
INSP_HIST  = BASE / "Data/07 - Redica/processed/valisure_fei_inspection_history.csv"
FDA_INSP   = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"
OUT        = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step2_panel.csv"

COUNTRY_MAP = {
    "India": "IND", "China": "CHN", "United States": "USA",
    "Canada": "CAN", "Bangladesh": "BGD", "United Kingdom": "GBR",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Japan": "JPN", "Israel": "ISR", "Ireland": "IRL", "Netherlands": "NLD",
    "Australia": "AUS", "Singapore": "SGP", "South Korea": "KOR",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def to_ndc11(x) -> Optional[str]:
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

def ndc11_to_display(n11: str):
    lab, prod4, pkg = n11[:5], n11[5:9], n11[9:]
    prod3 = prod4.lstrip("0").zfill(3) if prod4.lstrip("0") else "000"
    return f"{lab}-{prod3}-{pkg}", f"{lab}-{prod4}-{pkg}", f"{lab}-{prod3}"

def clean_fei(x) -> Optional[str]:
    if pd.isna(x): return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s): return None
    try: return str(int(float(s)))
    except: return None

# ── 1. Step 1 map ─────────────────────────────────────────────────────────────
print("Loading step1_ndc_fei_map...")
step1 = pd.read_csv(STEP1, dtype=str)
step1["ndc11"] = step1["NDC"].apply(to_ndc11)
step1["FEI"]   = step1["FEI"].apply(clean_fei)

# ── 2. Valisure metadata ──────────────────────────────────────────────────────
print("Loading Valisure metadata...")
xls_val = pd.ExcelFile(VAL_RAW)
val_info: dict[str, dict] = {}
for sweep_year, df in [
    ("2020", xls_val.parse("2020 Testing Data")),
    ("2022", xls_val.parse("2022 Testing Data - Actual")),
    ("2024", xls_val.parse("2024 Testing Data", header=1)),
]:
    for _, row in df.iterrows():
        n11 = to_ndc11(row.get("NDC11") if sweep_year == "2024" else row.get("NDC"))
        if not n11: continue
        if n11 not in val_info:
            val_info[n11] = {
                "val_firm":     str(row.get("Firm", "")).strip(),
                "val_strength": str(row.get("Strength", row.get("Dosage (mg)", ""))).strip(),
                "years": [],
            }
        if sweep_year not in val_info[n11]["years"]:
            val_info[n11]["years"].append(sweep_year)
        if not val_info[n11]["val_firm"] and str(row.get("Firm", "")).strip():
            val_info[n11]["val_firm"] = str(row["Firm"]).strip()

val_df = pd.DataFrame([
    {"ndc11": n11,
     "val_firm": d["val_firm"],
     "val_strength": d["val_strength"],
     "Valisure Years": "+".join(sorted(d["years"]))}
    for n11, d in val_info.items()
])

# ── 3. Sheet1 firm / strength / country fallback ──────────────────────────────
print("Loading Sheet1 metadata...")
df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1", dtype=str)
df_s1["ndc11"] = df_s1["NDC11"].apply(to_ndc11)
s1_meta = (
    df_s1.dropna(subset=["ndc11"]).drop_duplicates("ndc11")
    .set_index("ndc11")[["Firm", "Strength", "CountryCode"]]
    .rename(columns={"Firm": "s1_firm", "Strength": "s1_strength", "CountryCode": "s1_country"})
)

# ── 4. Redica site list + aggregates ──────────────────────────────────────────
print("Loading Redica site list and aggregates...")
sl = pd.read_excel(SITE_LIST, dtype=str)
sl["FEI"] = sl["FEI"].str.strip()
sl["redica_firm"] = sl["Site Display Name"].str.split("[").str[0].str.strip()
FEI_TO_SITE = sl.set_index("FEI")["Site Display Name"].to_dict()
FEI_TO_FIRM = sl.set_index("FEI")["redica_firm"].to_dict()

df_redica = pd.read_csv(REDICA_CSV)
df_redica["FEI"] = df_redica["FEI"].astype(str).str.strip()
site_agg = (
    df_redica[["FEI", "Total Inspections", "FDA Inspections", "483s Issued",
               "Total Observations", "Warning Letters Issued", "Import Alerts Issued"]]
    .drop_duplicates("FEI")
)

# ── 5. FDA country codes by FEI ───────────────────────────────────────────────
print("Loading FDA inspection details...")
df_fda = pd.read_excel(FDA_INSP)
df_fda["FEI Number"] = df_fda["FEI Number"].astype(str).str.strip()
FEI_TO_COUNTRY = (
    df_fda[["FEI Number", "Country/Area"]]
    .drop_duplicates("FEI Number")
    .assign(CountryCode=lambda d: d["Country/Area"].map(COUNTRY_MAP))
    .set_index("FEI Number")["CountryCode"]
    .to_dict()
)

# ── 6. Inspection history ─────────────────────────────────────────────────────
print("Loading inspection history...")
df_hist = pd.read_csv(INSP_HIST)
df_hist["FEI"]              = df_hist["FEI"].astype(str).str.strip()
df_hist["Event End Date"]   = pd.to_datetime(df_hist["Event End Date"],  errors="coerce")
df_hist["Event Start Date"] = pd.to_datetime(df_hist["Event Start Date"], errors="coerce")
df_hist["EventYear"]        = df_hist["EventYear"].astype(pd.Int64Dtype())

fei_in_old = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "METFORMIN_old only"]), "FEI"])
fei_in_new = set(df_hist.loc[df_hist["Insp_coverage"].isin(["both", "Valisure14 only"]),    "FEI"])

hist_stats = (
    df_hist.groupby("FEI").agg(
        oai_count    = ("Classification", lambda x: (x == "OAI").sum()),
        total_events = ("Classification", "count"),
        min_year     = ("EventYear", "min"),
        max_year     = ("EventYear", "max"),
    ).reset_index()
)
hist_stats["OAI Rate"]            = hist_stats["oai_count"] / hist_stats["total_events"]
hist_stats["Inspections per Year"] = hist_stats["total_events"] / (
    hist_stats["max_year"] - hist_stats["min_year"] + 1
)

# ── 7. Build NDC metadata ─────────────────────────────────────────────────────
print("Building NDC metadata...")
meta = step1.merge(val_df, on="ndc11", how="left")
meta = meta.join(s1_meta, on="ndc11", how="left")

meta["redica_firm"] = meta["FEI"].map(FEI_TO_FIRM)
meta["Firm"] = (
    meta["s1_firm"].fillna(meta["redica_firm"]).fillna(meta["val_firm"])
    .replace({"0": None, "nan": None, "": None})
)
meta["Strength"] = (
    meta["s1_strength"].fillna(meta["val_strength"])
    .replace({"0": None, "nan": None, "": None})
)
meta["CountryCode"] = meta["s1_country"].fillna(meta["FEI"].map(FEI_TO_COUNTRY))

def _mismatch(row):
    vf = str(row.get("val_firm", "")).strip().lower()
    rf = str(row.get("redica_firm", "")).strip().lower()
    return 0 if not vf or vf in ("nan","0","") or not rf else int(vf != rf)
meta["firm_valisure_mismatch"] = meta.apply(_mismatch, axis=1)

meta[["NDC_display", "NDC11_fmt", "NDC8"]] = pd.DataFrame(
    meta["ndc11"].apply(
        lambda n: ndc11_to_display(n) if isinstance(n, str) else ("","","")
    ).tolist(), index=meta.index
)
meta["NDC"]  = meta["NDC"].fillna(meta["NDC_display"])
meta["NDC11"] = meta["NDC11_fmt"]
meta["Site Display Name"] = meta["FEI"].map(FEI_TO_SITE)

def _fei_redica(fei):
    if pd.isna(fei): return None
    old, new = fei in fei_in_old, fei in fei_in_new
    if old and new: return "both"
    if old:         return "old only"
    if new:         return "new only"
    return "neither"
meta["fei_redica"] = meta["FEI"].apply(_fei_redica)

# ── 8. Build panel: (NDC × FEI) × inspection event ───────────────────────────
print("Building panel...")
hist_feis = set(df_hist["FEI"].unique())
_hist_drop = [c for c in ["Firm", "CountryCode"] if c in df_hist.columns]
hist_merge = df_hist.drop(columns=_hist_drop)

with_hist    = meta[meta["FEI"].notna() & meta["FEI"].isin(hist_feis)]
no_hist      = meta[meta["FEI"].notna() & ~meta["FEI"].isin(hist_feis)].copy()
no_fei       = meta[meta["FEI"].isna()].copy()

panel_with = with_hist.merge(hist_merge, on="FEI", how="left")

blank_cols = [
    "Event Start Date", "Event End Date", "EventYear", "Classification",
    "NAI", "VAI", "OAI", "483", "No 483",
    "483 critical", "483 major", "483 other", "Warning Letter",
    "Site Display Name", "Source", "Insp_coverage",
]
for col in blank_cols:
    no_hist[col] = None
    no_fei[col]  = None

panel = pd.concat([panel_with, no_hist, no_fei], ignore_index=True)

COV_MAP = {"both": "both", "METFORMIN_old only": "old only", "Valisure14 only": "new only"}
panel["insp_coverage"] = panel["Insp_coverage"].map(COV_MAP)

# Fill Site Display Name where history row lacks it
panel["Site Display Name"] = panel.apply(
    lambda r: r.get("Site Display Name") or FEI_TO_SITE.get(str(r.get("FEI") or ""), None),
    axis=1,
)

panel = panel.merge(site_agg, on="FEI", how="left")
panel = panel.merge(hist_stats[["FEI", "OAI Rate", "Inspections per Year"]], on="FEI", how="left")
panel["Year"] = panel["EventYear"]

# ── 9. Final column order ─────────────────────────────────────────────────────
FINAL_COLS = [
    # Mapping layer
    "ndc_fei_origin", "fei_rank",
    # Redica coverage
    "fei_redica", "insp_coverage",
    # NDC identity
    "Firm", "firm_valisure_mismatch", "Year",
    "NDC", "NDC11", "NDC8", "Strength", "CountryCode",
    # Facility
    "FEI", "Site Display Name",
    # Valisure context
    "Valisure Years",
    # Inspection event
    "Event Start Date", "Event End Date", "EventYear", "Classification",
    "NAI", "VAI", "OAI", "483", "No 483",
    "483 critical", "483 major", "483 other", "Warning Letter",
    # Site-level aggregates
    "Total Inspections", "FDA Inspections", "483s Issued",
    "Total Observations", "Warning Letters Issued", "Import Alerts Issued",
    "OAI Rate", "Inspections per Year",
]
FINAL_COLS = [c for c in FINAL_COLS if c in panel.columns]
panel_out = (
    panel[FINAL_COLS]
    .sort_values(["NDC", "FEI", "EventYear"], na_position="last")
    .reset_index(drop=True)
)

panel_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}  ({len(panel_out):,} rows)")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\nndc_fei_origin (unique NDC×FEI pairs):")
print(panel_out.drop_duplicates(["NDC","FEI"])["ndc_fei_origin"].value_counts().to_string())

insp = panel_out[panel_out["insp_coverage"].notna()]
total = len(insp)
print(f"\nInspection rows by source ({total} total):")
for lbl, n in insp["insp_coverage"].value_counts().items():
    print(f"  {lbl:<12} {n:>4}  ({n/total*100:.1f}%)")

print(f"\nfei_redica (unique FEIs with mapping):")
print(panel_out.dropna(subset=["FEI"]).drop_duplicates("FEI")["fei_redica"].value_counts().to_string())
# %%
