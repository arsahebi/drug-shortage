# %%
"""
Build Valisure FEI Inspection History
======================================
Combines two Redica sources for the ~25 metformin Valisure panel FEIs and
produces one row per unique (FEI, inspection end date) event.

Sources
-------
  METFORMIN_old : METFORMIN_SITE_RED_FLAG_EVENTS.xlsx  (old metformin-specific export)
  Valisure14    : Valisure14_Sites_Red_Flag_Events.xlsx (current 14-drug export)

Filters applied
---------------
  Classification rows : FDA sole agency (x == ['US - FDA']) + DQA program or 483/No 483
  Coverage determination : FDA-any (US-FDA in agency list) for METFORMIN_old, so joint
    FDA+CA-HC inspections still count as "present in old data" even though the
    classified row comes from Valisure14.

Output columns
--------------
  FEI, Firm, CountryCode, FEI_in_old
  Site Display Name
  Event Start Date  — from METFORMIN_old (which has FDA OASIS start dates); null otherwise
  Event End Date, EventYear
  Classification, NAI, VAI, OAI, 483, No 483, Warning Letter
  483 critical, 483 major, 483 other   — from Valisure14 for "both" events
  Source         — "METFORMIN_old" | "Valisure14" (which row supplies the data)
  Insp_coverage  — "both" | "METFORMIN_old only" | "Valisure14 only"

Note on deduplication
---------------------
  For "both" events: METFORMIN_old row is kept (has start date); Site Display Name
  and 483 critical/major/other are enriched from Valisure14.
  For joint FDA+HC inspections in METFORMIN: only Valisure14 has a classified row
  (since METFORMIN_old requires FDA-sole for classification), but Insp_coverage
  is still set to "both" because the inspection is visible in the METFORMIN file.
"""

import ast
import pandas as pd
from pathlib import Path

BASE = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage"
)
METFORMIN_FILE = BASE / "Data/07 - Redica/raw/METFORMIN_SITE_RED_FLAG_EVENTS.xlsx"
VALISURE14_FILE = BASE / "Data/07 - Redica/raw/Valisure14_Sites_Red_Flag_Events.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
PANEL_CSV = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v1.csv"
OUT_FILE = BASE / "Data/07 - Redica/processed/valisure_fei_inspection_history.csv"

# =============================================================================
# HELPERS
# =============================================================================
NON_DQA_PROGRAMS = {
    'VAI: Generic Drug Evaluation', 'NAI: Generic Drug Evaluation', 'OAI: Generic Drug Evaluation',
    'VAI: Bioresearch Monitoring', 'NAI: Bioresearch Monitoring', 'OAI: Bioresearch Monitoring',
    'VAI: Postmarketing Surveillance and Epidemiology: Human Drugs',
    'NAI: Postmarketing Surveillance and Epidemiology: Human Drugs',
    'OAI: Postmarketing Surveillance and Epidemiology: Human Drugs',
    'VAI: New Drug Evaluation', 'NAI: New Drug Evaluation', 'OAI: New Drug Evaluation',
    'VAI: Compliance: Medical Devices',
}
DQA_CLASS_NAMES = {
    'VAI: Drug Quality Assurance', 'NAI: Drug Quality Assurance', 'OAI: Drug Quality Assurance'
}


def parse_list(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) and str(x).strip().startswith('[') else []
    except Exception:
        return []


def clean_fei(x):
    if pd.isna(x):
        return None
    return str(x).strip().replace('.0', '')


def dqa_priority_classification(names: set):
    """Extract NAI/VAI/OAI with Drug Quality Assurance taking priority."""
    cls_candidates = []
    for v in names:
        if 'NAI' in v:
            cls_candidates.append(('NAI', v.split(':', 1)[1].strip() if ':' in v else ''))
        elif 'VAI' in v:
            cls_candidates.append(('VAI', v.split(':', 1)[1].strip() if ':' in v else ''))
        elif 'OAI' in v:
            cls_candidates.append(('OAI', v.split(':', 1)[1].strip() if ':' in v else ''))
    dqa = next((cls for cls, prog in cls_candidates if prog == 'Drug Quality Assurance'), None)
    if dqa:
        return dqa
    for priority in ('OAI', 'VAI', 'NAI'):
        m = next((cls for cls, _ in cls_candidates if cls == priority), None)
        if m:
            return m
    return None


# =============================================================================
# 1. PANEL FEI UNIVERSE
# =============================================================================
panel = pd.read_csv(PANEL_CSV)
panel['fei_str'] = panel['FEI'].apply(clean_fei)
panel_feis = set(panel['fei_str'].dropna())
fei_in_old_map = dict(panel.dropna(subset=['fei_str']).groupby('fei_str')['FEI_in_old_Redica'].first())
firm_map = (
    panel.dropna(subset=['fei_str']).drop_duplicates('fei_str')
    .set_index('fei_str')[['Firm', 'CountryCode']].to_dict(orient='index')
)
print(f"Panel FEIs: {len(panel_feis)}")

# =============================================================================
# 2. METFORMIN FLAT FILE
# =============================================================================
df_met = pd.read_excel(METFORMIN_FILE)
df_met['agency_list'] = df_met['Agency Name'].apply(parse_list)
df_met['fda_sole'] = df_met['agency_list'].apply(lambda x: x == ['US - FDA'])
df_met['fda_any'] = df_met['agency_list'].apply(lambda x: 'US - FDA' in x)
df_met['fei_str'] = df_met['Fei'].apply(clean_fei)
df_met['end_dt'] = pd.to_datetime(df_met['Event End Date'], errors='coerce')
df_met['start_dt'] = pd.to_datetime(df_met['Event Start Date'], errors='coerce')

# Broad coverage set: any inspection where FDA appears in the agency list.
# Used for Insp_coverage determination — joint FDA+HC inspections still count
# as "present in old data."
met_broad_events = set(
    df_met[df_met['fda_any'] & df_met['end_dt'].notna() & df_met['fei_str'].notna()]
    .apply(lambda r: (r['fei_str'], r['end_dt'].strftime('%Y-%m-%d')), axis=1)
)

# Classified rows: FDA-sole only, DQA or 483/No 483 outcome required.
met_rows = []
for (fei, end), grp in df_met[df_met['end_dt'].notna()].groupby(['fei_str', 'end_dt']):
    if not grp['fda_sole'].any():
        continue
    names = set(grp['Red Flag Event Name'].tolist())
    has_dqa = bool(names & DQA_CLASS_NAMES)
    has_483 = '483' in names or 'No 483' in names or 'Warning Letter' in names
    has_not_provided = 'Not Provided' in names
    all_non_dqa = bool(names & NON_DQA_PROGRAMS) and not has_dqa
    if all_non_dqa and not has_483 and not has_not_provided:
        continue

    classification = dqa_priority_classification(names)
    start_dt = grp[grp['fda_sole']]['start_dt'].dropna().min()
    met_rows.append({
        'FEI': fei, 'Site Display Name': None,
        'Event Start Date': start_dt if pd.notna(start_dt) else None,
        'Event End Date': end, 'EventYear': end.year,
        'Classification': classification,
        'NAI': 1 if classification == 'NAI' else 0,
        'VAI': 1 if classification == 'VAI' else 0,
        'OAI': 1 if classification == 'OAI' else 0,
        '483': 1 if '483' in names else 0,
        'No 483': 1 if 'No 483' in names else 0,
        'Warning Letter': 1 if 'Warning Letter' in names else 0,
        '483 critical': 0, '483 major': 0, '483 other': 0,
        'Source': 'METFORMIN_old',
    })
df_met_out = pd.DataFrame(met_rows)
print(f"METFORMIN FDA-sole classified events: {len(df_met_out)}, "
      f"broad FDA events (coverage): {len(met_broad_events)}")

# =============================================================================
# 3. VALISURE14
# =============================================================================
df14 = pd.read_excel(VALISURE14_FILE)
df_site = pd.read_excel(SITE_LIST)
df14 = df14.merge(df_site[['Site Redica Id', 'FEI']], on='Site Redica Id', how='left')
df14['agency_list'] = df14['Agency List'].apply(parse_list)
df14['industry_list'] = df14['Industry List'].apply(parse_list)
df14['attr'] = df14['Risk Event Attribute'].apply(parse_list)
df14['vals'] = df14['Risk Event Attribute Value'].apply(parse_list)
df14['fei_str'] = df14['FEI'].apply(clean_fei)
df14['event_dt'] = pd.to_datetime(df14['Event Date'], errors='coerce')
df14 = df14[df14['agency_list'].apply(lambda x: x == ['US - FDA'])]
df14 = df14[df14['industry_list'].apply(lambda x: 'Human Drugs' in x or len(x) == 0)]

v14_rows = []
for (site_id, site_name, end), grp in df14[df14['event_dt'].notna()].groupby(
        ['Site Redica Id', 'Site Display Name', 'event_dt']):
    fei = grp['fei_str'].iloc[0]
    is_483, is_wl, classification = 0, 0, None
    crit, maj, oth = 0, 0, 0
    for _, row in grp.iterrows():
        attrs = row['attr']
        vals = row['vals']
        if 'Inspection Outcome' in attrs:
            if '483' in vals:
                is_483 = 1
            if 'Warning Letter' in vals:
                is_wl = 1
            cls_candidates = []
            for v in vals:
                if not isinstance(v, str):
                    continue
                if v == 'NA':
                    cls_candidates.append(('NA', ''))
                elif 'NAI' in v:
                    cls_candidates.append(('NAI', v.split(':', 1)[1].strip() if ':' in v else ''))
                elif 'VAI' in v:
                    cls_candidates.append(('VAI', v.split(':', 1)[1].strip() if ':' in v else ''))
                elif 'OAI' in v:
                    cls_candidates.append(('OAI', v.split(':', 1)[1].strip() if ':' in v else ''))
            dqa = next((cls for cls, prog in cls_candidates if prog == 'Drug Quality Assurance'), None)
            if dqa:
                classification = dqa
            elif cls_candidates:
                for priority in ('OAI', 'VAI', 'NAI', 'NA'):
                    m = next((cls for cls, _ in cls_candidates if cls == priority), None)
                    if m:
                        classification = m
                        break
        if 'Post Inspection Document: 483' in attrs:
            for v in vals:
                if isinstance(v, dict):
                    crit = v.get('critical', 0)
                    maj = v.get('major', 0)
                    oth = v.get('other', 0)
    v14_rows.append({
        'FEI': fei, 'Site Display Name': site_name,
        'Event Start Date': None, 'Event End Date': end, 'EventYear': end.year,
        'Classification': classification,
        'NAI': 1 if classification == 'NAI' else 0,
        'VAI': 1 if classification == 'VAI' else 0,
        'OAI': 1 if classification == 'OAI' else 0,
        '483': is_483, 'No 483': 1 if is_483 == 0 else 0,
        'Warning Letter': is_wl,
        '483 critical': crit, '483 major': maj, '483 other': oth,
        'Source': 'Valisure14',
    })
df_v14_out = pd.DataFrame(v14_rows)
print(f"Valisure14 classified events: {len(df_v14_out)}")

# =============================================================================
# 4. COMBINE, ASSIGN Insp_coverage, DEDUPLICATE
# =============================================================================
df_all = pd.concat([df_met_out, df_v14_out], ignore_index=True)
df_all['end_str'] = pd.to_datetime(df_all['Event End Date']).dt.strftime('%Y-%m-%d')
df_all = df_all[df_all['FEI'].isin(panel_feis)].copy()

# Valisure14 classified events (for coverage lookup)
v14_events = set(
    df_all.loc[df_all['Source'] == 'Valisure14', ['FEI', 'end_str']].apply(tuple, axis=1)
)
# METFORMIN broad events restricted to panel FEIs
met_broad_panel = {(fei, d) for (fei, d) in met_broad_events if fei in panel_feis}


def insp_cov(row):
    key = (row['FEI'], row['end_str'])
    in_old = key in met_broad_panel
    in_new = key in v14_events
    if in_old and in_new:
        return 'both'
    if in_old:
        return 'METFORMIN_old only'
    if in_new:
        return 'Valisure14 only'
    return None


df_all['Insp_coverage'] = df_all.apply(insp_cov, axis=1)

# For "both" events: prefer METFORMIN_old row (has start date); enrich with V14 detail.
met_sole_keys = set(
    df_all.loc[df_all['Source'] == 'METFORMIN_old', ['FEI', 'end_str']].apply(tuple, axis=1)
)
both_met = df_all[(df_all['Insp_coverage'] == 'both') & (df_all['Source'] == 'METFORMIN_old')].copy()
both_v14 = df_all[(df_all['Insp_coverage'] == 'both') & (df_all['Source'] == 'Valisure14')]

v14_site_map = both_v14.set_index(['FEI', 'end_str'])['Site Display Name'].to_dict()
v14_483_map = both_v14.set_index(['FEI', 'end_str'])[
    ['483 critical', '483 major', '483 other']].to_dict(orient='index')

both_met = both_met.set_index(['FEI', 'end_str'])
for col in ['483 critical', '483 major', '483 other']:
    both_met[col] = both_met.index.map(lambda k: v14_483_map.get(k, {}).get(col, 0))
both_met['Site Display Name'] = both_met.apply(
    lambda r: v14_site_map.get(r.name, r['Site Display Name']), axis=1)
both_met = both_met.reset_index()
both_met['Insp_coverage'] = 'both'

# "Both" events where the METFORMIN file had joint-agency only → no FDA-sole row
# → Valisure14 is the only classified row, but coverage is still "both"
both_v14_no_met = both_v14[
    ~both_v14.apply(lambda r: (r['FEI'], r['end_str']), axis=1).isin(met_sole_keys)
].copy()
both_v14_no_met['Insp_coverage'] = 'both'

v14_only = df_all[df_all['Insp_coverage'] == 'Valisure14 only'].copy()
met_only = df_all[df_all['Insp_coverage'] == 'METFORMIN_old only'].copy()

df_final = pd.concat([both_met, both_v14_no_met, v14_only, met_only], ignore_index=True)
df_final = df_final.drop(columns=['end_str'], errors='ignore')

df_final['Firm'] = df_final['FEI'].map(lambda f: firm_map.get(f, {}).get('Firm'))
df_final['CountryCode'] = df_final['FEI'].map(lambda f: firm_map.get(f, {}).get('CountryCode'))
df_final['FEI_in_old'] = df_final['FEI'].map(fei_in_old_map)

COLS = ['FEI', 'Firm', 'CountryCode', 'FEI_in_old', 'Site Display Name',
        'Event Start Date', 'Event End Date', 'EventYear', 'Classification',
        'NAI', 'VAI', 'OAI', '483', 'No 483', 'Warning Letter',
        '483 critical', '483 major', '483 other', 'Source', 'Insp_coverage']
COLS = [c for c in COLS if c in df_final.columns]
df_final = df_final[COLS].sort_values(['FEI', 'Event End Date'])

# =============================================================================
# 5. SAVE
# =============================================================================
df_final.to_csv(OUT_FILE, index=False)
print(f"\nSaved: {OUT_FILE}")
print(f"Shape: {df_final.shape}")
print(f"Insp_coverage: {df_final['Insp_coverage'].value_counts().to_dict()}")

# %%
