# %%
"""
Step 2 (July 2026 refresh) — Build Metformin Inspection Panel
=============================================================
Uses fresh Redica July 2026 data for all 29 metformin FEIs.
Joins NDC→FEI map (step1) with inspection events to produce one row
per (NDC × FEI × inspection event).

Sources
-------
  step1_ndc_fei_map.csv                         — NDC→FEI map
  MetfrmoinValisure_FEI_RedicaID_Mapping_...    — Redica ID ↔ FEI
  MetfrmoinValisure_Red_Flag_Events_...         — inspection events
  Q&As1234_v8_v02.xlsx Sheet1                   — firm / strength / country fallback

Output columns (matching Q&A layout)
-------------------------------------
  Firm, Year, NDC, NDC11, NDC8, Strength        (Q&A cols A–F)
  FEI                                            (Q&A col J)
  CountryName, CountryCode                       (Q&A cols T–U)
  Event Start Date, Event End Date, EventYear    (Q&A cols V–X)
  483, No 483, NAI, VAI, OAI                     (Q&A cols Y–AC)
  Inspections per Year                           (Q&A col AD)

Output: step2_panel_july26.csv
"""

import ast
import re
from typing import Optional

import pandas as pd
from pathlib import Path

BASE     = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP1    = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map.csv"
QA_FILE  = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
RAW      = BASE / "Data/07 - Redica/raw"
FEI_MAP  = RAW  / "MetfrmoinValisure_FEI_RedicaID_Mapping_RedicaJuly26.xlsx"
EVENTS   = RAW  / "MetfrmoinValisure_Red_Flag_Events_RedicaJuly26.xlsx"
OUT      = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step2_panel_july26.csv"

COUNTRY_MAP = {
    "India": "IND", "China": "CHN", "United States": "USA",
    "United States of America": "USA",
    "Canada": "CAN", "Bangladesh": "BGD", "United Kingdom": "GBR",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Japan": "JPN", "Israel": "ISR", "Ireland": "IRL", "Netherlands": "NLD",
    "Australia": "AUS", "Singapore": "SGP", "South Korea": "KOR",
}

DQA_PROGRAMS = {
    "VAI: Drug Quality Assurance",
    "NAI: Drug Quality Assurance",
    "OAI: Drug Quality Assurance",
}
NON_DQA_PROGRAMS = {
    "VAI: Generic Drug Evaluation", "NAI: Generic Drug Evaluation", "OAI: Generic Drug Evaluation",
    "VAI: Bioresearch Monitoring", "NAI: Bioresearch Monitoring", "OAI: Bioresearch Monitoring",
    "VAI: Postmarketing Surveillance and Epidemiology: Human Drugs",
    "NAI: Postmarketing Surveillance and Epidemiology: Human Drugs",
    "OAI: Postmarketing Surveillance and Epidemiology: Human Drugs",
    "VAI: New Drug Evaluation", "NAI: New Drug Evaluation", "OAI: New Drug Evaluation",
    "VAI: Compliance: Medical Devices",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_list(x) -> list:
    try:
        return ast.literal_eval(x) if pd.notna(x) and str(x).strip().startswith("[") else []
    except Exception:
        return []

def clean_fei(x) -> Optional[str]:
    if pd.isna(x): return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s): return None
    try: return str(int(float(s)))
    except: return None

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

def extract_country(site_display_name: str):
    """Parse 'Firm [City / Country]' → (country_name, country_code)."""
    if not isinstance(site_display_name, str):
        return None, None
    m = re.search(r'\[.+?\s*/\s*(.+?)\]', site_display_name)
    if not m:
        return None, None
    country = m.group(1).strip()
    return country, COUNTRY_MAP.get(country)

def dqa_classification(outcome_vals: list) -> Optional[str]:
    """NAI/VAI/OAI: DQA takes priority; fallback highest severity."""
    candidates = []
    for v in outcome_vals:
        if not isinstance(v, str): continue
        for cls in ("OAI", "VAI", "NAI"):
            if v.startswith(cls):
                prog = v.split(":", 1)[1].strip() if ":" in v else ""
                candidates.append((cls, prog))
    dqa = next((cls for cls, prog in candidates if prog == "Drug Quality Assurance"), None)
    if dqa: return dqa
    for priority in ("OAI", "VAI", "NAI"):
        m = next((cls for cls, _ in candidates if cls == priority), None)
        if m: return m
    return None

# ── 1. FEI mapping (Redica ID → FEI) ─────────────────────────────────────────
print("Loading FEI mapping...")
fei_map  = pd.read_excel(FEI_MAP, dtype=str)
id_to_fei = dict(zip(fei_map["Redica ID"].str.strip(), fei_map["All FEIs"].str.strip()))
all_feis  = set(id_to_fei.values())
print(f"  {len(all_feis)} FEIs in scope")

# ── 2. Parse inspection events (new July26 format = Valisure14 format) ────────
print("Parsing inspection events...")
df = pd.read_excel(EVENTS)
df["FEI"]      = df["Site Redica Id"].map(id_to_fei)
df["agency"]   = df["Agency List"].apply(parse_list)
df["industry"] = df["Industry List"].apply(parse_list)
df["attr"]     = df["Risk Event Attribute"].apply(parse_list)
df["vals"]     = df["Risk Event Attribute Value"].apply(parse_list)
df["event_dt"] = pd.to_datetime(df["Event Date"], errors="coerce")

# Keep only FDA inspection rows
df_fda = df[
    (df["Event Type"] == "Inspection") &
    (df["agency"].apply(lambda x: "US - FDA" in x)) &
    (df["event_dt"].notna())
].copy()

# FEI → country from Site Display Name
fei_to_site    = {}
fei_to_country = {}
fei_to_country_code = {}
for _, row in df_fda.drop_duplicates("FEI").iterrows():
    fei = row["FEI"]
    if pd.isna(fei): continue
    fei_to_site[fei] = row["Site Display Name"]
    cn, cc = extract_country(row["Site Display Name"])
    fei_to_country[fei] = cn
    fei_to_country_code[fei] = cc

# One row per (FEI, inspection date)
insp_rows = []
for (fei, site_name, end_dt), grp in df_fda.groupby(
        ["FEI", "Site Display Name", "event_dt"]):
    outcome_vals = []
    for _, row in grp.iterrows():
        if "Inspection Outcome" in row["attr"]:
            outcome_vals.extend(row["vals"])

    outcome_set = set(v for v in outcome_vals if isinstance(v, str))

    # Skip if only non-DQA programs and no 483/No 483 signal
    has_dqa   = bool(outcome_set & DQA_PROGRAMS)
    has_483   = "483" in outcome_set or "No 483" in outcome_set
    has_not_provided = "Not Provided" in outcome_set
    pure_non_dqa = bool(outcome_set & NON_DQA_PROGRAMS) and not has_dqa
    if pure_non_dqa and not has_483 and not has_not_provided:
        continue

    classification = dqa_classification(outcome_vals)
    is_483 = 1 if "483" in outcome_set else 0

    insp_rows.append({
        "FEI":              fei,
        "Site Display Name": site_name,
        "Event Start Date": None,          # not available in this format
        "Event End Date":   end_dt,
        "EventYear":        end_dt.year,
        "Classification":   classification,
        "NAI": 1 if classification == "NAI" else 0,
        "VAI": 1 if classification == "VAI" else 0,
        "OAI": 1 if classification == "OAI" else 0,
        "483":    is_483,
        "No 483": 1 if is_483 == 0 else 0,
    })

df_insp = pd.DataFrame(insp_rows)
print(f"  {len(df_insp)} inspection events across {df_insp['FEI'].nunique()} FEIs")

# ── 3. Inspections per Year per FEI ──────────────────────────────────────────
print("Computing Inspections per Year...")
insp_stats = (
    df_insp.groupby("FEI")["EventYear"]
    .agg(["count", "min", "max"])
    .rename(columns={"count": "n_events", "min": "min_year", "max": "max_year"})
    .reset_index()
)
insp_stats["Inspections per Year"] = insp_stats["n_events"] / (
    insp_stats["max_year"] - insp_stats["min_year"] + 1
)

# ── 4. NDC metadata ───────────────────────────────────────────────────────────
print("Loading NDC metadata...")
step1 = pd.read_csv(STEP1, dtype=str)
step1["ndc11"] = step1["NDC"].apply(to_ndc11)
step1["FEI"]   = step1["FEI"].apply(clean_fei)

# Strength / Country from Sheet1 (Firm comes from Site Display Name instead)
df_s1 = pd.read_excel(QA_FILE, sheet_name="Sheet1", dtype=str)
df_s1["ndc11"] = df_s1["NDC11"].apply(to_ndc11)
s1_meta = (
    df_s1.dropna(subset=["ndc11"]).drop_duplicates("ndc11")
    .set_index("ndc11")[["Strength", "CountryCode"]]
    .rename(columns={"Strength": "s1_strength", "CountryCode": "s1_country"})
)

meta = step1.merge(s1_meta, on="ndc11", how="left")
meta["Strength"] = meta["s1_strength"].replace({"0": None, "nan": None, "": None})
# NDC, NDC11, NDC8, fei_count, facility_distance_km already provided by step1

# ── 5. Build panel (NDC × FEI × inspection event) ────────────────────────────
print("Building panel...")
hist_feis = set(df_insp["FEI"].dropna())

with_hist = meta[meta["FEI"].notna() & meta["FEI"].isin(hist_feis)]
no_hist   = meta[meta["FEI"].notna() & ~meta["FEI"].isin(hist_feis)].copy()
no_fei    = meta[meta["FEI"].isna()].copy()

panel_with = with_hist.merge(df_insp, on="FEI", how="left")

blank_insp = ["Event Start Date", "Event End Date", "EventYear", "Classification",
              "NAI", "VAI", "OAI", "483", "No 483", "Site Display Name"]
for col in blank_insp:
    no_hist[col] = None
    no_fei[col]  = None

panel = pd.concat([panel_with, no_hist, no_fei], ignore_index=True)

# Year column
panel["Year"] = panel["EventYear"]

# Country from Redica site display name; Sheet1 country as fallback
panel["CountryName"] = panel["FEI"].map(fei_to_country)
panel["CountryCode"] = panel["FEI"].map(fei_to_country_code).fillna(panel["s1_country"])

# Inspections per Year
panel = panel.merge(insp_stats[["FEI", "Inspections per Year"]], on="FEI", how="left")

# Site Display Name: fill from FEI lookup where not already set
panel["Site Display Name"] = panel.apply(
    lambda r: r.get("Site Display Name") or fei_to_site.get(str(r.get("FEI") or ""), None),
    axis=1,
)

# Firm: parse from Site Display Name (text before the '['); blank for NDCs with no FEI
def parse_firm(site_name):
    if not isinstance(site_name, str):
        return None
    parts = site_name.split("[")
    name = parts[0].strip().title()
    return name if name else None

panel["Firm"] = panel["Site Display Name"].apply(parse_firm)

# ── 6. Final column order ─────────────────────────────────────────────────────
FINAL_COLS = [
    "Firm", "Year",
    "NDC", "NDC11", "NDC8", "Strength",
    "FEI",
    "fei_count", "facility_distance_km",
    "CountryName", "CountryCode",
    "Event Start Date", "Event End Date", "EventYear",
    "483", "No 483", "NAI", "VAI", "OAI",
    "Inspections per Year",
    "Site Display Name",
]
FINAL_COLS = [c for c in FINAL_COLS if c in panel.columns]
panel_out = (
    panel[FINAL_COLS]
    .sort_values(["NDC", "FEI", "EventYear"], na_position="last")
    .reset_index(drop=True)
)

# Ensure FEI is stored as string, not float
panel_out["FEI"] = panel_out["FEI"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else None
)
panel_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}  ({len(panel_out):,} rows)")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\nFEIs in scope : {all_feis.__len__()}")
print(f"FEIs with inspections : {panel_out[panel_out['EventYear'].notna()]['FEI'].nunique()}")
print(f"Inspection rows : {panel_out['EventYear'].notna().sum()}")

print(f"\nInspection outcome breakdown (NAI/VAI/OAI):")
insp_rows_dedup = panel_out.dropna(subset=["EventYear"]).drop_duplicates(["FEI","Event End Date"])
print(f"  NAI : {insp_rows_dedup['NAI'].sum():.0f}")
print(f"  VAI : {insp_rows_dedup['VAI'].sum():.0f}")
print(f"  OAI : {insp_rows_dedup['OAI'].sum():.0f}")
print(f"  No classification (Not Provided / other) : {(insp_rows_dedup[['NAI','VAI','OAI']].sum(axis=1)==0).sum()}")

print(f"\nUnique NDC×FEI pairs : {panel_out.drop_duplicates(['NDC','FEI']).shape[0]}")
print(f"NDCs with no FEI     : {panel_out[panel_out['FEI'].isna()]['NDC'].nunique()}")
# %%
