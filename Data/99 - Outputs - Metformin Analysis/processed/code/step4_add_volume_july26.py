# %%
"""
Step 4 (July 2026 refresh) — Add Volume Data to Panel
=======================================================
Extends step3_panel_july26.csv with annual volume for each Valisure
test year (2020, 2022, 2024).

Sources (raw only)
------------------
  IQVIA: Metformin Jul 2019 - Jun 2025 NDC Level.xlsx
         Sheet "TRx"            → iqvia_trx (total prescriptions)
         Sheet "Extended Units" → iqvia_extended_units (used as volume in paper)
  SDUD:  SDUD_2020/2022/2024.csv (raw state × quarter files)
         Both FFSU and MCOU (= total Medicaid) summed across all states

Aggregation
-----------
  IQVIA: sum monthly values Jan–Dec of each test year per NDC11
  SDUD:  sum all quarters of each test year per NDC11 (FFSU + MCOU)

Join key: (NDC11 bare 11-digit, TestYear)

Missing coverage
----------------
  3 NDCs have no SDUD volume (42291-0497-90, 42291-0498-01, 71205-0884-60)
  → their SDUD columns will be null

Output: step4_panel_july26.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

BASE   = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP3  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step3_panel_july26.csv"
IQVIA  = BASE / "Data/06 - IQVIA/raw/Metformin Jul 2019 - Jun 2025 NDC Level.xlsx"
SDUD   = BASE / "Data/04 - Medicaid - SDUD/raw"
OUT    = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step4_panel_july26.csv"

TEST_YEARS = [2020, 2022, 2024]

# ── helpers ───────────────────────────────────────────────────────────────────
def to_ndc11_bare(x) -> Optional[str]:
    if pd.isna(x):
        return None
    d = re.sub(r'[^0-9]', '', str(x).strip())
    if len(d) == 10:
        return d[:5] + '0' + d[5:]
    if len(d) == 11:
        return d
    return None


def parse_iqvia_month(col: str) -> Optional[tuple]:
    """'TRx\\nJan 2020' or 'EUTRx\\nJan 2020' → (year, month_num)."""
    m = re.search(r'(\w{3})\s+(\d{4})', col)
    if not m:
        return None
    month_abbr, year = m.group(1), int(m.group(2))
    months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    mn = months.get(month_abbr)
    return (year, mn) if mn else None


# ── 1. IQVIA — both sheets ────────────────────────────────────────────────────
def load_iqvia_annual(sheet: str, value_name: str) -> pd.DataFrame:
    """Load one IQVIA sheet, melt to long, sum Jan–Dec per test year per NDC."""
    print(f"  Loading IQVIA sheet '{sheet}'...")
    raw = pd.ExcelFile(IQVIA).parse(sheet, dtype=str)

    # NDC column: "71093013404 TAB 1000MG 100" → first token
    raw['ndc11_bare'] = raw['NDC'].str.split().str[0].apply(to_ndc11_bare)
    raw = raw.dropna(subset=['ndc11_bare'])

    # Monthly value columns
    month_cols = [c for c in raw.columns if '\n' in c]
    id_cols    = ['ndc11_bare']

    melted = raw[id_cols + month_cols].melt(
        id_vars='ndc11_bare', var_name='col', value_name='val'
    )
    melted['val'] = pd.to_numeric(melted['val'], errors='coerce')

    # Parse year from column name
    parsed = melted['col'].apply(parse_iqvia_month)
    melted['year']  = parsed.apply(lambda x: x[0] if x else None)
    melted['month'] = parsed.apply(lambda x: x[1] if x else None)
    melted = melted.dropna(subset=['year', 'month'])
    melted['year'] = melted['year'].astype(int)

    # Filter to test years, sum Jan–Dec
    melted = melted[melted['year'].isin(TEST_YEARS)]
    annual = (
        melted.groupby(['ndc11_bare', 'year'], as_index=False)['val']
        .sum()
        .rename(columns={'val': value_name, 'year': 'TestYear'})
    )
    annual['TestYear'] = annual['TestYear'].astype('Int64')
    print(f"    → {len(annual)} (NDC, year) rows | "
          f"{annual['ndc11_bare'].nunique()} unique NDC11s")
    return annual


print("=" * 60)
print("IQVIA volume")
print("=" * 60)
iq_trx  = load_iqvia_annual('TRx',            'iqvia_trx')
iq_ext  = load_iqvia_annual('Extended Units',  'iqvia_extended_units')

# Merge both metrics into one lookup
iqvia_lookup = iq_trx.merge(iq_ext, on=['ndc11_bare', 'TestYear'], how='outer')
print(f"IQVIA lookup: {len(iqvia_lookup)} rows")
for yr in TEST_YEARS:
    sub = iqvia_lookup[iqvia_lookup['TestYear'] == yr]
    print(f"  {yr}: {len(sub)} NDC11s | "
          f"trx non-null={sub['iqvia_trx'].notna().sum()} | "
          f"ext non-null={sub['iqvia_extended_units'].notna().sum()}")


# ── 2. SDUD — raw annual CSVs ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SDUD volume")
print("=" * 60)

# Get the step3 NDC list for filtering (avoid scanning 5M+ rows per year)
step3_ndcs = set(
    pd.read_csv(STEP3, dtype=str)['NDC11']
    .str.replace('-', '', regex=False)
    .dropna()
    .unique()
)
print(f"Filtering SDUD to {len(step3_ndcs)} step3 NDC11s")

sdud_chunks = []
for yr in TEST_YEARS:
    fpath = SDUD / f"SDUD_{yr}.csv"
    if not fpath.exists():
        print(f"  {yr}: file not found — skipping")
        continue
    print(f"  Loading SDUD {yr}...", end=' ')
    df = pd.read_csv(fpath, dtype=str)

    # Normalize NDC to 11-digit bare (SDUD NDC is already 11 digits, no dashes)
    df['ndc11_bare'] = df['NDC'].apply(to_ndc11_bare)

    # Filter to step3 NDCs
    df = df[df['ndc11_bare'].isin(step3_ndcs)]
    print(f"{len(df)} metformin rows")

    # Convert numeric columns
    df['Number of Prescriptions'] = pd.to_numeric(df['Number of Prescriptions'], errors='coerce')
    df['Units Reimbursed']        = pd.to_numeric(df['Units Reimbursed'],        errors='coerce')

    # Sum FFSU + MCOU across all states and all quarters of the year
    annual = (
        df.groupby('ndc11_bare', as_index=False)
        .agg(
            sdud_num_prescriptions=('Number of Prescriptions', 'sum'),
            sdud_units_reimbursed =('Units Reimbursed',        'sum'),
        )
    )
    annual['TestYear'] = yr
    sdud_chunks.append(annual)
    print(f"    → {len(annual)} unique NDC11s in {yr}")

sdud_lookup = pd.concat(sdud_chunks, ignore_index=True)
sdud_lookup['TestYear'] = sdud_lookup['TestYear'].astype('Int64')
print(f"\nSDUD lookup: {len(sdud_lookup)} rows")
for yr in TEST_YEARS:
    sub = sdud_lookup[sdud_lookup['TestYear'] == yr]
    print(f"  {yr}: {len(sub)} NDC11s with SDUD data")


# ── 3. Load step3 and join volume ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Joining to step3 panel")
print("=" * 60)

panel = pd.read_csv(STEP3, dtype=str)
panel['ndc11_bare'] = panel['NDC11'].str.replace('-', '', regex=False)
panel['TestYear']   = pd.to_numeric(panel['TestYear'], errors='coerce').astype('Int64')
print(f"step3 rows: {len(panel):,}")

panel = panel.merge(iqvia_lookup, on=['ndc11_bare', 'TestYear'], how='left')
panel = panel.merge(sdud_lookup,  on=['ndc11_bare', 'TestYear'], how='left')

# Drop working column
panel.drop(columns=['ndc11_bare'], inplace=True)

# ── 4. Final column order and save ───────────────────────────────────────────
FINAL_COLS = [
    'redica_firm', 'valisure_firm', 'valisure_labeler', 'TestYear',
    'NDC', 'NDC11', 'NDC8', 'Strength',
    'valisure_tested_years', 'n_lots',
    'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor',
    'iqvia_trx', 'iqvia_extended_units',
    'sdud_num_prescriptions', 'sdud_units_reimbursed',
    'FEI', 'fei_count', 'facility_distance_km',
    'CountryName', 'CountryCode',
    'Event Start Date', 'Event End Date', 'EventYear',
    '483', 'No 483', 'NAI', 'VAI', 'OAI',
    'Inspections per Year',
    'Site Display Name',
]
FINAL_COLS = [c for c in FINAL_COLS if c in panel.columns]

panel_out = (
    panel[FINAL_COLS]
    .sort_values(['NDC', 'TestYear', 'FEI', 'EventYear'], na_position='last')
    .reset_index(drop=True)
)
panel_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"  {len(panel_out):,} rows | {panel_out['NDC11'].nunique()} NDC11s")

# ── 5. Coverage summary ───────────────────────────────────────────────────────
print("\n── Volume coverage by test year ──")
for col in ['iqvia_trx', 'iqvia_extended_units', 'sdud_num_prescriptions', 'sdud_units_reimbursed']:
    panel_out[col] = pd.to_numeric(panel_out[col], errors='coerce')

for yr in TEST_YEARS:
    sub = panel_out[panel_out['TestYear'] == yr].drop_duplicates('NDC11')
    print(f"  {yr}:")
    for col in ['iqvia_trx', 'iqvia_extended_units', 'sdud_num_prescriptions', 'sdud_units_reimbursed']:
        n = sub[col].notna().sum()
        print(f"    {col}: {n}/{len(sub)} NDC11s")

print("\n── Sample: one NDC across all 3 years ──")
sample_ndc = panel_out.dropna(subset=['iqvia_extended_units'])['NDC11'].iloc[0]
sample = (
    panel_out[panel_out['NDC11'] == sample_ndc]
    .drop_duplicates('TestYear')
    [['NDC11', 'TestYear', 'iqvia_trx', 'iqvia_extended_units',
      'sdud_num_prescriptions', 'sdud_units_reimbursed']]
)
print(sample.to_string(index=False))
# %%
