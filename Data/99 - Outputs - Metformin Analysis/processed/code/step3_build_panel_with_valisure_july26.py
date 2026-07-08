# %%
"""
Step 3 (July 2026 refresh) — Add Valisure Data to Panel
=========================================================
Extends step2_panel_july26.csv (NDC × FEI × inspection event) to:

  NDC × FEI × inspection event × Valisure test year (2020, 2022, 2024)

Valisure raw sources
--------------------
  Valisure_2024_raw.xlsx  "2020 Testing Data"         → NDMA + DMF; multiple lots → MAX
  Valisure_2024_raw.xlsx  "2022 Testing Data - Actual" → NDMA + DMF; multiple lots → MAX
  Valisure_2024_raw.xlsx  "2024 Testing Data"          → DMF only (no NDMA in 2024)
  Testing Data_DoD ...    "Metformin"                  → Difference Factor (2024 only)

ND  → 0   |   <LOQ / BLOQ → 151.54   (consistent with original Sheet1 treatment)

Steps
-----
  1. Parse each raw Valisure sheet → per-NDC quality table per year.
  2. Expand step2 panel × [2020, 2022, 2024] → ValisureYear column.
  3. Left-join quality on (NDC11 bare, ValisureYear).
     Unmatched (NDC not tested that year) → Valisure columns null.
  4. Strength: Valisure year-specific Strength when available; else step2 Strength.
  5. Year = ValisureYear; EventYear retains the original inspection year.

Output: step3_panel_july26.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP2   = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step2_panel_july26.csv"
RAW24   = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
DOD     = BASE / "Data/08 - Valisure/raw/Testing Data_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx"
OUT     = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step3_panel_july26.csv"

LOQ_VAL = 151.54   # sentinel for <LOQ / BLOQ results

# ── helpers ───────────────────────────────────────────────────────────────────
def to_ndc11_bare(x) -> Optional[str]:
    """Any NDC format → bare 11-digit string (5-4-2 no hyphens)."""
    if pd.isna(x):
        return None
    d = re.sub(r'[^0-9]', '', str(x).strip())
    if len(d) == 10:
        return d[:5] + '0' + d[5:]
    if len(d) == 11:
        return d
    return None


def parse_qual(x, nd_val=0.0, loq_val=LOQ_VAL) -> float:
    """Convert raw Valisure value: ND→0, <LOQ/<LOD/BLOQ→151.54, else float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ('ND', 'N/D'):
        return nd_val
    if s in ('<LOQ', 'LOQ', '<LOD', 'LOD', 'BLOQ', '<BLOQ'):
        return loq_val
    if s in ('--', '-', 'N/A', 'NA', ''):
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def safe_max(series: pd.Series) -> float:
    """Max of parse_qual values; NaN if all NaN."""
    vals = series.apply(parse_qual).dropna()
    return vals.max() if len(vals) else np.nan


# ── 1a. Valisure 2020 ─────────────────────────────────────────────────────────
print("Loading Valisure 2020...")
xls = pd.ExcelFile(RAW24)
df20 = xls.parse('2020 Testing Data', dtype=str)
df20['ndc11_bare'] = df20['NDC'].apply(to_ndc11_bare)
df20['strength']   = df20['Dosage (mg)'].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)

qual20 = (
    df20.dropna(subset=['ndc11_bare'])
    .groupby('ndc11_bare', as_index=False)
    .agg(
        ndma            =('NDMA Daily Intake (ng/day)', safe_max),
        dmf             =('DMF Daily Intake (ng/day)',  safe_max),
        strength        =('strength', 'first'),
        valisure_firm   =('Firm',        'first'),
        valisure_labeler=('Distributor', 'first'),
        n_lots          =('Lot Number',  'count'),
    )
)
qual20['ValisureYear'] = 2020
qual20['diff_factor']  = np.nan
print(f"  2020: {len(qual20)} NDC11s")

# ── 1b. Valisure 2022 ─────────────────────────────────────────────────────────
print("Loading Valisure 2022...")
df22 = xls.parse('2022 Testing Data - Actual', dtype=str)
# Use NDC11 column (5-4-2 with hyphens) preferentially; fall back to NDC (5-3-2)
df22['ndc11_bare'] = df22['NDC11'].apply(to_ndc11_bare).fillna(df22['NDC'].apply(to_ndc11_bare))

qual22 = (
    df22.dropna(subset=['ndc11_bare'])
    .groupby('ndc11_bare', as_index=False)
    .agg(
        ndma            =('NDMA (ng/Max Daily Dose)', safe_max),
        dmf             =('DMF (ng/Max Daily Dose)',  safe_max),
        strength        =('Strength', 'first'),
        valisure_firm   =('Firm',     'first'),
        valisure_labeler=('Labeler',  'first'),
        n_lots          =('Lot',      'count'),
    )
)
qual22['ValisureYear'] = 2022
qual22['diff_factor']  = np.nan
print(f"  2022: {len(qual22)} NDC11s")

# ── 1c. Valisure 2024 ─────────────────────────────────────────────────────────
print("Loading Valisure 2024...")
# Row 0 is a merged header label row; row 1 is the true column header
df24 = xls.parse('2024 Testing Data', header=1, dtype=str)
df24['ndc11_bare'] = df24['NDC11'].apply(to_ndc11_bare).fillna(df24['NDC'].apply(to_ndc11_bare))

qual24 = (
    df24.dropna(subset=['ndc11_bare'])
    .groupby('ndc11_bare', as_index=False)
    .agg(
        dmf             =('DMF (ng/DAY) Valisure', safe_max),
        strength        =('Strength',  'first'),
        valisure_firm   =('Firm',      'first'),
        valisure_labeler=('Labeler',   'first'),
        n_lots          =('Sample ID', 'count'),
    )
)
qual24['ndma'] = np.nan  # not measured in 2024
qual24['ValisureYear'] = 2024

# Difference Factor from DoD supplemental file
df_dod = pd.ExcelFile(DOD).parse('Metformin', header=1, dtype=str)
df_dod['ndc11_bare'] = df_dod['NDC'].apply(to_ndc11_bare)
dod_diff = (
    df_dod.dropna(subset=['ndc11_bare'])
    [['ndc11_bare', 'Difference Factor']]
    .drop_duplicates('ndc11_bare')
)
dod_diff['diff_factor'] = pd.to_numeric(dod_diff['Difference Factor'], errors='coerce')

qual24 = qual24.merge(dod_diff[['ndc11_bare', 'diff_factor']], on='ndc11_bare', how='left')
QUAL_COLS = ['ndc11_bare', 'ValisureYear', 'ndma', 'dmf', 'diff_factor', 'strength',
             'valisure_firm', 'valisure_labeler', 'n_lots']
qual24 = qual24[[c for c in QUAL_COLS if c in qual24.columns]]
print(f"  2024: {len(qual24)} NDC11s")

# ── 1d. Stack into quality lookup table ──────────────────────────────────────
qual = pd.concat([
    qual20[[c for c in QUAL_COLS if c in qual20.columns]],
    qual22[[c for c in QUAL_COLS if c in qual22.columns]],
    qual24[[c for c in QUAL_COLS if c in qual24.columns]],
], ignore_index=True)

# NDC-level firm/labeler: prefer most recent year (2024 > 2022 > 2020)
firm_lookup = (
    pd.concat([qual24, qual22, qual20])[['ndc11_bare', 'valisure_firm', 'valisure_labeler']]
    .dropna(subset=['ndc11_bare'])
    .drop_duplicates('ndc11_bare', keep='first')
)

# Rename to final column names (drop firm/labeler here — joined separately at NDC level)
qual = qual.rename(columns={
    'ndma':        'NDMA (ng/DAY) Valisure',
    'dmf':         'DMF (ng/DAY) Valisure',
    'diff_factor': 'Difference Factor',
    'strength':    'val_strength',
}).drop(columns=['valisure_firm', 'valisure_labeler'], errors='ignore')

# Round to 2 decimal places
for col in ['NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor']:
    qual[col] = pd.to_numeric(qual[col], errors='coerce').round(2)

print(f"\nQuality lookup: {len(qual)} rows across 3 years")
for yr in [2020, 2022, 2024]:
    sub = qual[qual['ValisureYear'] == yr]
    print(f"  {yr}: {len(sub)} NDC11s  |  "
          f"NDMA populated: {sub['NDMA (ng/DAY) Valisure'].notna().sum()}  |  "
          f"DMF populated: {sub['DMF (ng/DAY) Valisure'].notna().sum()}  |  "
          f"Diff populated: {sub['Difference Factor'].notna().sum()}")

# ── 2. Load step2 panel and expand by Valisure test year ─────────────────────
print("\nLoading step2 panel...")
panel = pd.read_csv(STEP2, dtype=str)
for col in ['EventYear', 'NAI', 'VAI', 'OAI', '483', 'No 483']:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')

panel['ndc11_bare'] = panel['NDC11'].apply(to_ndc11_bare)
print(f"  step2 rows: {len(panel):,}")

# NDC-level summary: which test years each NDC was covered
tested_years = (
    qual.groupby('ndc11_bare')['ValisureYear']
    .apply(lambda s: '+'.join(str(y) for y in sorted(s.dropna())))
    .rename('valisure_tested_years')
    .reset_index()
)

# Cross-join with the three test years
ALL_YEARS = pd.DataFrame({'TestYear': pd.array([2020, 2022, 2024], dtype='Int64')})
panel_exp = panel.merge(ALL_YEARS, how='cross')
print(f"  After expansion (×3): {len(panel_exp):,} rows")

# ── 3. Join Valisure quality ──────────────────────────────────────────────────
print("\nJoining Valisure quality...")
qual = qual.rename(columns={'ValisureYear': 'TestYear'})
qual['TestYear'] = qual['TestYear'].astype('Int64')
panel_exp = panel_exp.merge(qual, on=['ndc11_bare', 'TestYear'], how='left')

# NDC-level "tested years" string (join separately since it's NDC-level, not year-level)
panel_exp = panel_exp.merge(tested_years, on='ndc11_bare', how='left')
panel_exp['valisure_tested_years'] = panel_exp['valisure_tested_years'].fillna('Not tested')

# Strength: year-specific Valisure Strength when available; fall back to step2 Strength
panel_exp['Strength'] = panel_exp['val_strength'].fillna(panel_exp['Strength'])
panel_exp.drop(columns=['val_strength'], inplace=True)

# NDC-level firm/labeler from Valisure (constant across test years)
panel_exp = panel_exp.merge(firm_lookup, on='ndc11_bare', how='left')

# Redica firm: text before '[' in Site Display Name
def parse_redica_firm(site_name):
    if not isinstance(site_name, str):
        return None
    name = site_name.split('[')[0].strip().title()
    return name if name else None

panel_exp['redica_firm'] = panel_exp['Site Display Name'].apply(parse_redica_firm)

print(f"  NDMA populated: {panel_exp['NDMA (ng/DAY) Valisure'].notna().sum():,}")
print(f"  DMF  populated: {panel_exp['DMF (ng/DAY) Valisure'].notna().sum():,}")
print(f"  Diff populated: {panel_exp['Difference Factor'].notna().sum():,}")

# ── 5. Final column order and save ───────────────────────────────────────────
FINAL_COLS = [
    'redica_firm', 'valisure_firm', 'valisure_labeler', 'TestYear',
    'NDC', 'NDC11', 'NDC8', 'Strength',
    'valisure_tested_years', 'n_lots',
    'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor',
    'FEI', 'fei_count', 'facility_distance_km',
    'CountryName', 'CountryCode',
    'Event Start Date', 'Event End Date', 'EventYear',
    '483', 'No 483', 'NAI', 'VAI', 'OAI',
    'Inspections per Year',
    'Site Display Name',
]
FINAL_COLS = [c for c in FINAL_COLS if c in panel_exp.columns]

panel_out = (
    panel_exp[FINAL_COLS]
    .sort_values(['NDC', 'TestYear', 'FEI', 'EventYear'], na_position='last')
    .reset_index(drop=True)
)

panel_out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"  {len(panel_out):,} rows")
print(f"  Unique NDC11s        : {panel_out['NDC11'].nunique()}")
print(f"  Unique FEIs          : {panel_out['FEI'].dropna().nunique()}")

print("\n── Valisure coverage by year ──")
for yr in [2020, 2022, 2024]:
    sub = panel_out[panel_out['TestYear'] == yr]
    n_dmf   = sub['DMF (ng/DAY) Valisure'].notna().sum()
    n_ndcs  = sub[sub['DMF (ng/DAY) Valisure'].notna()]['NDC11'].nunique()
    print(f"  {yr}: {n_dmf:,}/{len(sub):,} rows with DMF  ({n_ndcs} unique NDC11s)")

print("\n── valisure_tested_years distribution ──")
ndc_tested = panel_out.drop_duplicates('NDC11')[['NDC11','valisure_tested_years']]
print(ndc_tested['valisure_tested_years'].value_counts().to_string())

print("\n── Sample: first Valisure-covered NDC across all 3 years ──")
covered = panel_out.dropna(subset=['DMF (ng/DAY) Valisure'])
if not covered.empty:
    first_ndc = covered['NDC11'].iloc[0]
    sample = (
        panel_out[panel_out['NDC11'] == first_ndc]
        .drop_duplicates('TestYear')
        [['NDC11', 'TestYear', 'valisure_tested_years', 'Strength',
          'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor']]
    )
    print(sample.to_string(index=False))

# ── 6. Compare against Sheet1 Q&A ────────────────────────────────────────────
print("\n── Comparison with Sheet1 Q&A (sanity check) ──")
QA_FILE = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
s1 = pd.read_excel(QA_FILE, sheet_name='Sheet1', dtype=str)
s1['ndc11_bare'] = s1['NDC11'].apply(to_ndc11_bare)
s1['TestYear'] = pd.to_numeric(s1['Year'], errors='coerce').astype('Int64')
for col in ['NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor']:
    s1[col] = pd.to_numeric(s1[col], errors='coerce')

s1_dedup = s1.drop_duplicates(['ndc11_bare', 'TestYear'])

# Build our deduped lookup from the new step3
our_dedup = (
    panel_out[['NDC11', 'TestYear',
               'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor']]
    .copy()
)
our_dedup['ndc11_bare'] = our_dedup['NDC11'].apply(to_ndc11_bare)
our_dedup = our_dedup.drop_duplicates(['ndc11_bare', 'TestYear'])

comp = s1_dedup.merge(our_dedup, on=['ndc11_bare', 'TestYear'],
                      suffixes=('_s1', '_new'), how='inner')

def match_pct(a, b, tol=0.1):
    mask = a.notna() & b.notna()
    n = mask.sum()
    if n == 0: return '0/0'
    m = ((a[mask] - b[mask]).abs() <= tol).sum()
    return f"{m}/{n}"

print(f"  (NDC, Year) pairs in both: {len(comp)}")
for col in ['DMF (ng/DAY) Valisure', 'NDMA (ng/DAY) Valisure', 'Difference Factor']:
    s1c, nc = f'{col}_s1', f'{col}_new'
    if s1c in comp.columns and nc in comp.columns:
        print(f"  {col}: {match_pct(comp[s1c], comp[nc])} match within 0.1")
# %%
