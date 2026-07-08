# %%
"""
Step 5 (July 2026 refresh) — Build Analysis-Ready Panel
=========================================================
Collapses step4_panel_july26.csv from
  (NDC11 × FEI × InspectionEvent × TestYear)
to the estimation unit:
  (NDC11 × TestYear) — one row per NDC per Valisure test year

Prior inspection logic (bug-fixed vs old pre-revision code):
  For each (NDC11, TestYear):
    1. Keep rows where FEI is non-null, EventYear ≤ TestYear, NAI+VAI+OAI == 1
    2. Select the row with maximum EventYear (most recent classified inspection)
    3. Tie-break on same EventYear: worst outcome wins (OAI > VAI > NAI)
  CountryCode / CountryName: from the FEI that provided the prior inspection

Bugs fixed vs old build_ndc_year_table():
  - sort-order mismatch: old agg used sort=True, prior-score loop used sort=False
    → PriorScore was assigned to wrong (NDC11, Year) rows
  - FEI column stored "first" FEI but score came from a different FEI's inspection

Output: step5_analysis_panel_july26.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE  = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP4 = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step4_panel_july26.csv"
OUT   = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step5_analysis_panel_july26.csv"

OUTCOME_SCORE = {'NAI': 0.0, 'VAI': 1.5, 'OAI': 3.5}
OUTCOME_RANK  = {'OAI': 2,  'VAI': 1,   'NAI': 0}   # higher = worse, for tie-breaking

# ── 1. Load step4 ─────────────────────────────────────────────────────────────
print("Loading step4 panel...")
df = pd.read_csv(STEP4, dtype=str)
for col in ['NAI', 'VAI', 'OAI', 'EventYear']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['TestYear'] = pd.to_numeric(df['TestYear'], errors='coerce').astype('Int64')
print(f"  {len(df):,} rows | {df['NDC11'].nunique()} NDC11s")


# ── 2. Derive outcome label per inspection row ────────────────────────────────
def row_outcome(row) -> str | None:
    """Return 'OAI'/'VAI'/'NAI' for a classified inspection row, else None."""
    if row['OAI'] == 1:
        return 'OAI'
    if row['VAI'] == 1:
        return 'VAI'
    if row['NAI'] == 1:
        return 'NAI'
    return None

df['_outcome'] = df.apply(row_outcome, axis=1)
df['_outcome_rank'] = df['_outcome'].map(OUTCOME_RANK)


# ── 3. Find prior inspection per (NDC11, TestYear) ────────────────────────────
print("Finding prior inspection per (NDC11, TestYear)...")

# Inspection-eligible rows only: non-null FEI, EventYear <= TestYear, classified
insp = df[
    df['FEI'].notna() &
    df['EventYear'].notna() &
    (df['EventYear'] <= df['TestYear']) &
    df['_outcome'].notna()
].copy()

print(f"  Eligible inspection rows (EventYear <= TestYear, classified): {len(insp):,}")

# Within each (NDC11, TestYear), select most recent; tie-break: worst outcome
insp_sorted = insp.sort_values(
    ['NDC11', 'TestYear', 'EventYear', '_outcome_rank'],
    ascending=[True, True, False, False]   # EventYear desc (most recent first), outcome desc (worst first)
)
prior = (
    insp_sorted
    .drop_duplicates(subset=['NDC11', 'TestYear'], keep='first')
    [['NDC11', 'TestYear', '_outcome', 'EventYear', 'FEI', 'CountryCode', 'CountryName', 'Site Display Name']]
    .rename(columns={
        '_outcome':          'prior_outcome',
        'EventYear':         'prior_event_year',
        'FEI':               'prior_fei',
        'CountryCode':       'prior_country_code',
        'CountryName':       'prior_country_name',
        'Site Display Name': 'prior_site',
    })
)
prior['prior_score'] = prior['prior_outcome'].map(OUTCOME_SCORE)

print(f"  Prior inspection found for {len(prior)} (NDC11, TestYear) pairs")
print("  Prior outcome distribution:")
print(prior['prior_outcome'].value_counts().to_string())


# ── 4. Collapse NDC-level columns ─────────────────────────────────────────────
print("Collapsing to (NDC11, TestYear) level...")

# These columns are constant within (NDC11, TestYear) — take first non-null
NDC_COLS = [
    'NDC', 'NDC8', 'Strength',
    'valisure_tested_years', 'n_lots',
    'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor',
    'iqvia_trx', 'iqvia_extended_units',
    'sdud_num_prescriptions', 'sdud_units_reimbursed',
    'redica_firm', 'valisure_firm', 'valisure_labeler',
    'fei_count',
]

agg = (
    df.groupby(['NDC11', 'TestYear'], sort=True)[NDC_COLS]
    .first()
    .reset_index()   # promotes NDC11, TestYear from index back to columns
)

# Count distinct non-null FEIs per NDC11
n_feis = (
    df[df['FEI'].notna()]
    .groupby('NDC11')['FEI']
    .nunique()
    .rename('n_feis')
    .reset_index()
)
agg = agg.merge(n_feis, on='NDC11', how='left')
agg['n_feis'] = agg['n_feis'].fillna(0).astype(int)

# Country: prefer country from prior inspection; fallback to first non-null in step4
fallback_country = (
    df[df['CountryCode'].notna()]
    .groupby('NDC11')[['CountryCode', 'CountryName']]
    .first()
    .reset_index()
    .rename(columns={'CountryCode': '_fallback_cc', 'CountryName': '_fallback_cn'})
)
agg = agg.merge(fallback_country, on='NDC11', how='left')


# ── 5. Merge prior inspection ─────────────────────────────────────────────────
agg = agg.merge(prior, on=['NDC11', 'TestYear'], how='left')

# Fill country from prior inspection; otherwise use fallback
agg['CountryCode'] = agg['prior_country_code'].fillna(agg['_fallback_cc'])
agg['CountryName'] = agg['prior_country_name'].fillna(agg['_fallback_cn'])
agg.drop(columns=['prior_country_code', 'prior_country_name', '_fallback_cc', '_fallback_cn'],
         inplace=True)


# ── 6. Final column order and types ───────────────────────────────────────────
FINAL_COLS = [
    'NDC', 'NDC11', 'NDC8', 'Strength',
    'TestYear',
    'valisure_tested_years', 'n_lots',
    'NDMA (ng/DAY) Valisure', 'DMF (ng/DAY) Valisure', 'Difference Factor',
    'iqvia_trx', 'iqvia_extended_units',
    'sdud_num_prescriptions', 'sdud_units_reimbursed',
    'redica_firm', 'valisure_firm', 'valisure_labeler',
    'CountryCode', 'CountryName',
    'fei_count', 'n_feis',
    'prior_fei', 'prior_event_year', 'prior_outcome', 'prior_score', 'prior_site',
]
FINAL_COLS = [c for c in FINAL_COLS if c in agg.columns]

panel = (
    agg[FINAL_COLS]
    .sort_values(['NDC11', 'TestYear'])
    .reset_index(drop=True)
)

panel.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"  {len(panel):,} rows | {panel['NDC11'].nunique()} NDC11s")


# ── 7. Coverage summary ───────────────────────────────────────────────────────
print("\n── Coverage by TestYear ──")
for yr in [2020, 2022, 2024]:
    sub = panel[panel['TestYear'] == yr]
    dmf  = pd.to_numeric(sub['DMF (ng/DAY) Valisure'],  errors='coerce')
    ndma = pd.to_numeric(sub['NDMA (ng/DAY) Valisure'], errors='coerce')
    diff = pd.to_numeric(sub['Difference Factor'],        errors='coerce')
    ext  = pd.to_numeric(sub['iqvia_extended_units'],     errors='coerce')
    pr   = sub['prior_outcome'].notna()
    print(f"  {yr} ({len(sub)} NDC11s):")
    print(f"    DMF non-null:    {dmf.notna().sum():3d}  NDMA non-null: {ndma.notna().sum():3d}  DiffFactor: {diff.notna().sum():3d}")
    print(f"    IQVIA non-null:  {ext.notna().sum():3d}  Prior insp:   {pr.sum():3d}/{len(sub)}")

print("\n── Prior outcome by TestYear ──")
for yr in [2020, 2022, 2024]:
    sub = panel[panel['TestYear'] == yr]
    vc = sub['prior_outcome'].value_counts()
    print(f"  {yr}: NAI={vc.get('NAI',0)}  VAI={vc.get('VAI',0)}  OAI={vc.get('OAI',0)}  null={sub['prior_outcome'].isna().sum()}")

print("\n── Country distribution (unique NDC11s, TestYear=2022) ──")
sub22 = panel[panel['TestYear'] == 2022]
print(sub22['CountryCode'].value_counts().to_string())

print("\n── Multi-FEI NDCs ──")
print(f"  NDC11s with n_feis > 1: {(panel['n_feis'] > 1).sum() // 3}")
print(panel[panel['n_feis'] > 1][['NDC11','n_feis','fei_count']].drop_duplicates('NDC11').to_string(index=False))

print("\n── Sample rows ──")
sample = panel[panel['prior_outcome'].notna()].head(6)
print(sample[['NDC11','TestYear','DMF (ng/DAY) Valisure','prior_outcome','prior_score','prior_event_year','CountryCode']].to_string(index=False))
# %%
