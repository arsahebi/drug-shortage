# %%
"""
Build Metformin Panel v2
=========================
Extends metformin_panel_v1.csv (NDC × inspection event) to the full
Sheet1-equivalent structure:

  NDC × Valisure test year × inspection event

Steps
-----
  1. Extract Valisure quality (DMF, NDMA, Difference Factor) per NDC × year
       2020 : Valisure_2024_raw.xlsx  "2020 Testing Data"  – MAX across lots
       2022 : Valisure_2022.xlsx      "Sheet1"             – MAX across lots
       2024 : Valisure_2024_raw.xlsx  "2024 Testing Data"  – already per-NDC
              + DoD file for Difference Factor (joined by NDC)
     ND → 0,  <LOQ → 151.54  (consistent with original Sheet1 treatment)

  2. Explode v1 panel on Valisure test years (one copy per test year per row)

  3. Join quality columns on (NDC11, ValisureYear)

  4. Join annual volume from IQVIA+SDUD+NADAC monthly panel
     (sum flow variables Jan–Dec of ValisureYear; mean for nadac_price)

  5. Save metformin_panel_v2.csv

  6. Compare against original Sheet1 for the 82 original NDCs / 3 test years
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
V1      = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v1.csv"
RAW24   = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw.xlsx"
RAW22   = BASE / "Data/08 - Valisure/raw/Valisure_2022.xlsx"
DOD     = BASE / "Data/08 - Valisure/raw/Testing Data_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx"
MONTHLY = BASE / "Data/04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)/processed/2026-02-24-iqvia_with_sdud_nadac.cleaned.csv"
SHEET1  = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
OUT_V2  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v2.csv"

LOQ_VAL = 151.54   # <LOQ sentinel used in original Sheet1

# =============================================================================
# HELPERS
# =============================================================================
def clean11(x):
    """Normalise any NDC format to bare 11-digit string."""
    if pd.isna(x):
        return None
    d = re.sub(r'[^0-9]', '', str(x).strip())
    if len(d) == 10:
        return d[:5] + '0' + d[5:]   # 5-3-2 → 5-4-2
    if len(d) == 11:
        return d
    if 8 <= len(d) < 11:
        return d.zfill(11)
    return None


def parse_qual(x, nd_val=0.0, loq_val=LOQ_VAL):
    """Convert raw Valisure quality value to float: ND→0, <LOQ→151.54."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s == 'ND':
        return nd_val
    if s in ('<LOQ', 'LOQ', '<LOD', 'BLOQ'):
        return loq_val
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def safe_max(s):
    """Max after applying parse_qual; returns NaN if all NaN."""
    vals = s.apply(parse_qual).dropna()
    return vals.max() if len(vals) else np.nan


# =============================================================================
# 1. VALISURE QUALITY — 2020
#    Source: Valisure_2024_raw.xlsx → "2020 Testing Data"
#    Multiple lots per NDC → MAX
# =============================================================================
print("Loading Valisure 2020...")
xls24 = pd.ExcelFile(RAW24)
df20_raw = xls24.parse('2020 Testing Data')
df20_raw['ndc11'] = df20_raw['NDC'].apply(clean11)

df20 = (
    df20_raw.dropna(subset=['ndc11'])
    .groupby('ndc11', as_index=False)
    .agg(
        dmf_ng_day =('DMF Daily Intake (ng/day)',  'max'),
        ndma_ng_day=('NDMA Daily Intake (ng/day)', 'max'),
    )
)
df20['valisure_year'] = 2020
df20['diff_factor']  = np.nan
print(f"  2020: {len(df20)} NDCs")

# =============================================================================
# 2. VALISURE QUALITY — 2022
#    Source: Valisure_2022.xlsx → "Sheet1"
#    Multiple lots per NDC → MAX (after ND→0, <LOQ→151.54)
# =============================================================================
print("Loading Valisure 2022...")
xls22  = pd.ExcelFile(RAW22)
df22_raw = xls22.parse('Sheet1')
# NDC13 is already 5-4-2 with hyphens (= NDC11); prefer over NDC (5-3-2)
df22_raw['ndc11'] = df22_raw['NDC13'].apply(clean11).fillna(
    df22_raw['NDC'].apply(clean11)
)

df22 = (
    df22_raw.dropna(subset=['ndc11'])
    .groupby('ndc11', as_index=False)
    .agg(
        dmf_ng_day =('DMF (ng/Max Daily Dose)',  safe_max),
        ndma_ng_day=('NDMA (ng/Max Daily Dose)', safe_max),
    )
)
df22['valisure_year'] = 2022
df22['diff_factor']  = np.nan
print(f"  2022: {len(df22)} NDCs")

# =============================================================================
# 3. VALISURE QUALITY — 2024
#    Source: Valisure_2024_raw.xlsx → "2024 Testing Data"  (already per-NDC)
#    Difference Factor: DoD file → "Metformin" sheet
#    NDMA not measured in 2024
# =============================================================================
print("Loading Valisure 2024...")
df24_raw = xls24.parse('2024 Testing Data', header=1)
df24_raw['ndc11']      = df24_raw['NDC11'].apply(clean11)
df24_raw['dmf_ng_day'] = df24_raw['DMF (ng/DAY) Valisure'].apply(parse_qual)

df24 = (
    df24_raw.dropna(subset=['ndc11'])
    [['ndc11', 'dmf_ng_day']]
    .drop_duplicates('ndc11')
    .copy()
)
df24['ndma_ng_day'] = np.nan  # not measured in 2024

# Difference Factor from DoD supplemental file
xls_dod = pd.ExcelFile(DOD)
df_dod  = xls_dod.parse('Metformin', header=1)
df_dod['ndc11'] = df_dod['NDC'].apply(clean11)
dod_diff = (
    df_dod.dropna(subset=['ndc11'])
    [['ndc11', 'Difference Factor']]
    .rename(columns={'Difference Factor': 'diff_factor'})
    .drop_duplicates('ndc11')
)
df24 = df24.merge(dod_diff, on='ndc11', how='left')
df24['valisure_year'] = 2024
print(f"  2024: {len(df24)} NDCs")

# =============================================================================
# 4. STACK INTO QUALITY TABLE: ndc11 × valisure_year
# =============================================================================
qual = pd.concat([df20, df22, df24], ignore_index=True)
qual['ndc11'] = qual['ndc11'].astype(str)
# Round to 1 decimal place — matching Sheet1
for c in ['dmf_ng_day', 'ndma_ng_day']:
    qual[c] = qual[c].round(1)

print(f"\nQuality table: {len(qual)} rows")
print(qual.groupby('valisure_year')[['dmf_ng_day','ndma_ng_day','diff_factor']].describe().T)

# =============================================================================
# 5. LOAD v1 PANEL AND EXPAND BY VALISURE TEST YEAR
# =============================================================================
print("\nLoading v1 panel...")
v1 = pd.read_csv(V1)
v1['ndc11_bare'] = v1['NDC11'].apply(clean11)

# All NDCs get rows for all three test years (2020, 2022, 2024).
# Quality columns will be NaN for years the NDC was not actually tested.
ALL_VAL_YEARS = pd.DataFrame({'ValisureYear': pd.array([2020, 2022, 2024], dtype='Int64')})

v2 = v1.rename(columns={'Year': 'InspYear'}).merge(ALL_VAL_YEARS, how='cross')
v2['Year'] = v2['ValisureYear']

print(f"v1 rows: {len(v1):,}  →  v2 after expansion: {len(v2):,}")
print("Rows per Valisure year:")
print(v2['ValisureYear'].value_counts().sort_index())

# =============================================================================
# 6. JOIN QUALITY DATA
# =============================================================================
v2 = v2.merge(
    qual.rename(columns={'ndc11': 'ndc11_bare', 'valisure_year': 'ValisureYear'}),
    on=['ndc11_bare', 'ValisureYear'],
    how='left',
)
print(f"\nAfter quality join: {len(v2):,} rows")
print(f"  dmf_ng_day  populated: {v2['dmf_ng_day'].notna().sum():,}")
print(f"  ndma_ng_day populated: {v2['ndma_ng_day'].notna().sum():,}")
print(f"  diff_factor populated: {v2['diff_factor'].notna().sum():,}")

# =============================================================================
# 7. ANNUAL VOLUME FROM MONTHLY PANEL
#    Sum flow variables Jan–Dec of ValisureYear; mean for nadac_price
# =============================================================================
print("\nLoading monthly volume panel...")
monthly = pd.read_csv(MONTHLY)
monthly['ndc11'] = monthly['ndc11'].astype(str).str.zfill(11)
monthly['year']  = pd.to_datetime(monthly['date'], errors='coerce').dt.year

SUM_COLS  = ['iqvia_trx', 'iqvia_extended_units', 'sdud_num_prescriptions',
             'sdud_units_reimbursed', 'total_amount_reimbursed', 'medicaid_amount_reimbursed']
MEAN_COLS = ['nadac_price']

for c in SUM_COLS + MEAN_COLS:
    monthly[c] = pd.to_numeric(monthly.get(c, np.nan), errors='coerce')

annual = monthly.groupby(['ndc11', 'year'], as_index=False).agg(
    **{c: (c, 'sum')  for c in SUM_COLS},
    **{c: (c, 'mean') for c in MEAN_COLS},
)
annual['sdud_price_total_per_unit'] = (
    annual['total_amount_reimbursed'] /
    annual['sdud_units_reimbursed'].replace(0, np.nan)
)
annual['sdud_price_medicaid_per_unit'] = (
    annual['medicaid_amount_reimbursed'] /
    annual['sdud_units_reimbursed'].replace(0, np.nan)
)
annual = annual.rename(columns={'ndc11': 'ndc11_bare', 'year': 'ValisureYear'})

v2 = v2.merge(annual, on=['ndc11_bare', 'ValisureYear'], how='left')
print(f"After volume join: {len(v2):,} rows")
print(f"  iqvia_trx populated: {v2['iqvia_trx'].notna().sum():,}")
print(f"  nadac_price populated: {v2['nadac_price'].notna().sum():,}")

# =============================================================================
# 8. FINAL COLUMN ORDER AND SAVE
# =============================================================================
FINAL_COLS = [
    # Provenance
    'NDC_origin', 'FEI_in_old_Redica', 'FEI_in_new_Redica', 'Insp_coverage',
    # Identity
    'Firm', 'Year', 'NDC', 'NDC11', 'NDC8', 'Strength', 'CountryCode',
    'FEI', 'Site Display Name', 'Valisure Years',
    # Valisure quality
    'dmf_ng_day', 'ndma_ng_day', 'diff_factor',
    # Inspection event
    'Event Start Date', 'Event End Date', 'EventYear', 'Classification',
    'NAI', 'VAI', 'OAI', '483', 'No 483',
    '483 critical', '483 major', '483 other', 'Warning Letter',
    # Site-level aggregates
    'Total Inspections', 'FDA Inspections', '483s Issued',
    'Total Observations', 'Warning Letters Issued', 'Import Alerts Issued',
    'OAI Rate', 'Inspections per Year',
    # Volume / price
    'iqvia_trx', 'iqvia_extended_units',
    'sdud_num_prescriptions', 'sdud_units_reimbursed',
    'total_amount_reimbursed', 'medicaid_amount_reimbursed',
    'sdud_price_total_per_unit', 'sdud_price_medicaid_per_unit',
    'nadac_price',
]
FINAL_COLS = [c for c in FINAL_COLS if c in v2.columns]
v2_out = (
    v2[FINAL_COLS]
    .sort_values(['NDC11', 'Year', 'EventYear'], na_position='last')
    .reset_index(drop=True)
)

v2_out.to_csv(OUT_V2, index=False)
print(f"\nSaved: {OUT_V2}")
print(f"  {len(v2_out):,} rows | {v2_out['NDC11'].nunique()} unique NDCs | "
      f"{v2_out['FEI'].nunique()} unique FEIs")

# =============================================================================
# 9. COMPARE WITH SHEET1
# =============================================================================
print("\n" + "=" * 70)
print("SHEET1 COMPARISON")
print("=" * 70)

df_s1 = pd.read_excel(SHEET1, sheet_name='Sheet1')
df_s1['ndc11_bare'] = df_s1['NDC11'].apply(clean11)
df_s1['Year'] = pd.to_numeric(df_s1['Year'], errors='coerce').astype(pd.Int64Dtype())

# Reattach ndc11_bare to v2_out for comparison (not saved to CSV)
v2_out = v2_out.copy()
v2_out['ndc11_bare'] = v2_out['NDC11'].apply(clean11)

# ── Row counts ──────────────────────────────────────────────────────────────
s1_ndc_years  = df_s1[['ndc11_bare','Year']].drop_duplicates()
v2_ndc_years  = v2_out[v2_out['ndc11_bare'].isin(s1_ndc_years['ndc11_bare'])][['ndc11_bare','Year']].drop_duplicates()

in_both = s1_ndc_years.merge(v2_ndc_years, on=['ndc11_bare','Year'])
only_s1 = s1_ndc_years.merge(v2_ndc_years, on=['ndc11_bare','Year'], how='left', indicator=True)
only_s1 = only_s1[only_s1['_merge']=='left_only']
only_v2 = v2_ndc_years.merge(s1_ndc_years, on=['ndc11_bare','Year'], how='left', indicator=True)
only_v2 = only_v2[only_v2['_merge']=='left_only']

print(f"\nSheet1 unique (NDC, Year) pairs : {len(s1_ndc_years)}")
print(f"v2     unique (NDC, Year) pairs  : {len(v2_ndc_years)}  (filtered to Sheet1 NDCs)")
print(f"  In both                        : {len(in_both)}")
print(f"  Only in Sheet1                 : {len(only_s1)}")
if len(only_s1):
    print("    →", only_s1[['ndc11_bare','Year']].values.tolist())
print(f"  Only in v2                     : {len(only_v2)}")
if len(only_v2):
    print("    →", only_v2[['ndc11_bare','Year']].values.tolist())

# ── Quality column comparison ───────────────────────────────────────────────
# Merge on (NDC11, Year) at NDC-year level (drop_duplicates to avoid inflation)
s1_qual = (
    df_s1[['ndc11_bare','Year','DMF (ng/DAY) Valisure','NDMA (ng/DAY) Valisure','Difference Factor']]
    .drop_duplicates(['ndc11_bare','Year'])
    .rename(columns={
        'DMF (ng/DAY) Valisure'  : 's1_dmf',
        'NDMA (ng/DAY) Valisure' : 's1_ndma',
        'Difference Factor'      : 's1_diff',
    })
)
v2_qual = (
    v2_out[['ndc11_bare','Year','dmf_ng_day','ndma_ng_day','diff_factor']]
    .drop_duplicates(['ndc11_bare','Year'])
    .rename(columns={
        'dmf_ng_day'  : 'v2_dmf',
        'ndma_ng_day' : 'v2_ndma',
        'diff_factor' : 'v2_diff',
    })
)
comp_qual = s1_qual.merge(v2_qual, on=['ndc11_bare','Year'], how='inner')

def pct_match(a, b, tol=0.1):
    mask = a.notna() & b.notna()
    n = mask.sum()
    if n == 0: return 0, 0
    match = (a[mask] - b[mask]).abs() <= tol
    return int(match.sum()), int(n)

for col, s1c, v2c in [('DMF','s1_dmf','v2_dmf'),
                       ('NDMA','s1_ndma','v2_ndma'),
                       ('Diff Factor','s1_diff','v2_diff')]:
    m, n = pct_match(comp_qual[s1c], comp_qual[v2c])
    print(f"\n{col}: {m}/{n} match within 0.1 ng/day")
    # Show mismatches
    bad = comp_qual[(comp_qual[s1c].notna()) & (comp_qual[v2c].notna()) &
                    ((comp_qual[s1c] - comp_qual[v2c]).abs() > 0.1)]
    if not bad.empty:
        print(f"  Mismatches ({len(bad)}):")
        print(bad[['ndc11_bare','Year',s1c,v2c]].to_string(index=False))

# ── Volume column comparison ─────────────────────────────────────────────────
print("\n── Volume column comparison (mean absolute difference, Sheet1 NDCs) ──")
VOL_MAP = {
    'iqvia_trx'                  : 'iqvia_trx',
    'iqvia_extended_units'       : 'iqvia_extended_units',
    'sdud_num_prescriptions'     : 'sdud_num_prescriptions',
    'sdud_units_reimbursed'      : 'sdud_units_reimbursed',
    'total_amount_reimbursed'    : 'total_amount_reimbursed',
    'medicaid_amount_reimbursed' : 'medicaid_amount_reimbursed',
    'nadac_price'                : 'nadac_price',
}
s1_vol = (
    df_s1[['ndc11_bare','Year'] + list(VOL_MAP.keys())]
    .drop_duplicates(['ndc11_bare','Year'])
)
v2_vol = (
    v2_out[['ndc11_bare','Year'] + list(VOL_MAP.values())]
    .drop_duplicates(['ndc11_bare','Year'])
)
comp_vol = s1_vol.merge(v2_vol, on=['ndc11_bare','Year'], how='inner',
                        suffixes=('_s1','_v2'))

for col in VOL_MAP:
    c_s1 = f'{col}_s1'
    c_v2 = f'{col}_v2'
    both  = comp_vol[c_s1].notna() & comp_vol[c_v2].notna()
    n     = both.sum()
    if n == 0:
        print(f"  {col:40s}: no overlap")
        continue
    corr = comp_vol.loc[both, c_s1].corr(comp_vol.loc[both, c_v2])
    pct  = ((comp_vol.loc[both, c_s1] - comp_vol.loc[both, c_v2]).abs()
            / comp_vol.loc[both, c_s1].replace(0, np.nan)).median()
    print(f"  {col:40s}: n={n:3d}  corr={corr:.4f}  median_rel_diff={pct:.4f}")

# ── Row-count comparison per NDC-Year ───────────────────────────────────────
print("\n── Inspection-row counts per (NDC, Year): Sheet1 vs v2 ──")
s1_rc = df_s1.groupby(['ndc11_bare','Year']).size().reset_index(name='n_s1')
v2_rc = (
    v2_out[v2_out['ndc11_bare'].isin(df_s1['ndc11_bare'])]
    .groupby(['ndc11_bare','Year']).size().reset_index(name='n_v2')
)
rc_comp = s1_rc.merge(v2_rc, on=['ndc11_bare','Year'], how='outer').fillna(0)
rc_comp['n_s1'] = rc_comp['n_s1'].astype(int)
rc_comp['n_v2'] = rc_comp['n_v2'].astype(int)
match_rc = (rc_comp['n_s1'] == rc_comp['n_v2']).sum()
print(f"  (NDC, Year) pairs with matching row count: {match_rc}/{len(rc_comp)}")
bad_rc = rc_comp[rc_comp['n_s1'] != rc_comp['n_v2']].sort_values('n_s1', ascending=False)
if not bad_rc.empty:
    print(f"  Mismatches ({len(bad_rc)}) — first 15:")
    print(bad_rc.head(15).to_string(index=False))

# %%
