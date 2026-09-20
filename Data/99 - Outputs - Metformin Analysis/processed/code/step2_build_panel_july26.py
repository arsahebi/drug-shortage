# %%
"""
Step 2 (July 2026 refresh) — Build Metformin Inspection Panel
=============================================================
Uses fresh Redica July 2026 data for all 29 metformin FEIs.
Joins NDC→FEI map (step1) with inspection events to produce one row
per (NDC × FEI × inspection event).

Sources
-------
  step1_ndc_fei_map_rulebased.csv               — NDC→FEI map (manufacture-only rule)
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

Sample exclusions (applied here, inherited by steps 3-6)
--------------------------------------------------------
  1. Canada and Bangladesh facilities are dropped from the entire analysis.
  2. Facilities with no Redica inspection history are dropped from the entire
     analysis, as are NDCs that have no FEI at all.
Both apply to every downstream figure, including those not about country or
inspection history, so that all reported results describe one constant sample.

Output: step2_panel_july26.csv
"""

import ast
import re
from typing import Optional

import pandas as pd
from pathlib import Path

BASE     = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
import os
STEP1    = Path(os.environ.get("STEP1_OVERRIDE",
           str(BASE / "Data/99 - Outputs - Metformin Analysis/processed/step1_ndc_fei_map_rulebased.csv")))
QA_FILE  = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
RAW      = BASE / "Data/07 - Redica/raw"
FEI_MAP  = RAW  / "MetfrmoinValisure_FEI_RedicaID_Mapping_RedicaJuly26.xlsx"
EVENTS   = RAW  / "MetfrmoinValisure_Red_Flag_Events_RedicaJuly26.xlsx"
OUT      = Path(os.environ.get("STEP2_OUT_OVERRIDE",
           str(BASE / "Data/99 - Outputs - Metformin Analysis/processed/step2_panel_july26.csv")))

# ── Sample exclusions (apply to the ENTIRE analysis, every downstream figure) ──
# Enforced here in step 2 so steps 3-6 inherit one filtered panel and every
# reported result describes the same sample. Do not re-filter per figure.
EXCLUDE_COUNTRIES     = {"Canada", "Bangladesh"}   # drop these facilities outright
REQUIRE_REDICA_HISTORY = os.environ.get("REQUIRE_REDICA_HISTORY", "1") == "1"
DROP_NDCS_WITHOUT_FEI  = os.environ.get("DROP_NDCS_WITHOUT_FEI", "1") == "1"

COUNTRY_MAP = {
    "India": "IND", "China": "CHN", "United States": "USA",
    "United States of America": "USA",
    "Canada": "CAN", "Bangladesh": "BGD", "United Kingdom": "GBR",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Japan": "JPN", "Israel": "ISR", "Ireland": "IRL", "Netherlands": "NLD",
    "Australia": "AUS", "Singapore": "SGP", "South Korea": "KOR",
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
    """Return NAI/VAI/OAI only for Drug Quality Assurance outcomes; None otherwise."""
    for v in outcome_vals:
        if not isinstance(v, str): continue
        if v == "OAI: Drug Quality Assurance": return "OAI"
        if v == "VAI: Drug Quality Assurance": return "VAI"
        if v == "NAI: Drug Quality Assurance": return "NAI"
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

    classification = dqa_classification(outcome_vals)
    if classification is None:
        continue  # only DQA-classified inspections enter the panel

    outcome_set = set(v for v in outcome_vals if isinstance(v, str))
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

# ── 2b. EXCLUSION 1 — drop Canada / Bangladesh facilities ────────────────────
excluded_feis = {f for f, c in fei_to_country.items() if c in EXCLUDE_COUNTRIES}
if excluded_feis:
    print(f"\nEXCLUSION 1 — {sorted(EXCLUDE_COUNTRIES)} facilities dropped:")
    for f in sorted(excluded_feis):
        n = (df_insp["FEI"] == f).sum()
        print(f"  {f}  {fei_to_country[f]:<12} {fei_to_site.get(f, '')}  ({n} events)")
    df_insp = df_insp[~df_insp["FEI"].isin(excluded_feis)].reset_index(drop=True)
    for f in excluded_feis:
        fei_to_site.pop(f, None)
        fei_to_country.pop(f, None)
        fei_to_country_code.pop(f, None)
    print(f"  -> {len(df_insp)} events across {df_insp['FEI'].nunique()} FEIs remain")

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

# EXCLUSION 1 is absolute: an excluded-country facility is dropped from every
# NDC regardless of REQUIRE_REDICA_HISTORY, so it can never re-enter the panel
# via the no-history bucket with just a blank country field.
if excluded_feis:
    n_excl = meta[meta["FEI"].isin(excluded_feis)]["NDC11"].nunique()
    if n_excl:
        print(f"  (dropping {n_excl} NDC11(s) matched to an excluded-country "
              f"facility, unconditionally)")
    meta = meta[~meta["FEI"].isin(excluded_feis)].copy()

with_hist = meta[meta["FEI"].notna() & meta["FEI"].isin(hist_feis)]
no_hist   = meta[meta["FEI"].notna() & ~meta["FEI"].isin(hist_feis)].copy()
no_fei    = meta[meta["FEI"].isna()].copy()

# ── EXCLUSION 2 — facilities with no Redica inspection history ───────────────
# Includes FEIs dropped by EXCLUSION 1, since they are no longer in df_insp.
if REQUIRE_REDICA_HISTORY and len(no_hist):
    dropped = sorted(no_hist["FEI"].dropna().unique())
    print(f"\nEXCLUSION 2 — {len(dropped)} FEI(s) with no Redica history dropped "
          f"({no_hist['NDC11'].nunique()} NDC11s affected):")
    for f in dropped:
        why = "excluded country" if f in excluded_feis else "no Redica events"
        print(f"  {f}  ({why})  NDCs: {sorted(no_hist[no_hist['FEI'] == f]['NDC11'].unique())}")
    no_hist = no_hist.iloc[0:0]

# Same logic applied to NDCs that never had an FEI: no facility, no history.
if DROP_NDCS_WITHOUT_FEI and len(no_fei):
    print(f"\nEXCLUSION 2b — {no_fei['NDC11'].nunique()} NDC11(s) with no FEI dropped")
    no_fei = no_fei.iloc[0:0]

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

# EXCLUSION 1 also applies to the Sheet1 fallback: an NDC with no matched FEI
# can still carry a Canada/Bangladesh country from the old Q&A spreadsheet.
_excl_codes = {"CAN": "Canada", "BGD": "Bangladesh"}
_excl_codes = {c for c, name in _excl_codes.items() if name in EXCLUDE_COUNTRIES}
if _excl_codes:
    _leak = panel["FEI"].isna() & panel["CountryCode"].isin(_excl_codes)
    if _leak.any():
        print(f"\n  (dropping {panel.loc[_leak, 'NDC11'].nunique()} more NDC11(s) whose only "
              f"country signal is the Sheet1 fallback and reads Canada/Bangladesh)")
        panel = panel[~_leak].reset_index(drop=True)

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
print(f"\nFEIs in Redica scope (pre-exclusion) : {len(all_feis)}")
print(f"FEIs in analysis sample (post-exclusion) : {panel_out['FEI'].nunique()}")
print(f"NDC11s in analysis sample : {panel_out['NDC11'].nunique()}")
print(f"Inspection rows : {panel_out['EventYear'].notna().sum()}")
print(f"Countries in sample : {sorted(panel_out['CountryName'].dropna().unique())}")
assert not (set(panel_out['CountryName'].dropna()) & EXCLUDE_COUNTRIES), "excluded country leaked into panel (CountryName)"
assert not (set(panel_out['CountryCode'].dropna()) & _excl_codes), "excluded country leaked into panel (CountryCode / Sheet1 fallback)"
if REQUIRE_REDICA_HISTORY and DROP_NDCS_WITHOUT_FEI:
    assert panel_out['EventYear'].notna().all(), "row without inspection history leaked into panel"

print(f"\nInspection outcome breakdown (NAI/VAI/OAI):")
insp_rows_dedup = panel_out.dropna(subset=["EventYear"]).drop_duplicates(["FEI","Event End Date"])
print(f"  NAI : {insp_rows_dedup['NAI'].sum():.0f}")
print(f"  VAI : {insp_rows_dedup['VAI'].sum():.0f}")
print(f"  OAI : {insp_rows_dedup['OAI'].sum():.0f}")
print(f"  No classification (Not Provided / other) : {(insp_rows_dedup[['NAI','VAI','OAI']].sum(axis=1)==0).sum()}")

print(f"\nUnique NDC×FEI pairs : {panel_out.drop_duplicates(['NDC','FEI']).shape[0]}")
print(f"NDCs with no FEI     : {panel_out[panel_out['FEI'].isna()]['NDC'].nunique()}")
# %%
