# %%
"""
═══════════════════════════════════════════════════════════════════════════════
  Metformin Quality Risk Index (MQRI) — Full Pipeline  v4.0
  Author: Amirreza Sahebi / Claude Code, April 2026
═══════════════════════════════════════════════════════════════════════════════

WHAT'S NEW IN v4.0
------------------
  1. TIME-VARYING PANEL: MQRI computed for each survey year (2020, 2022, 2024)
     using ONLY data available as of that year's cutoff date.
     Output: mqri_panel.csv (18 FEIs × 3 years = 54 rows).

  2. RAW INSPECTION SOURCE: OAI/VAI/NAI/483 counts now come from the raw
     "Inspections Details.xlsx" (FDA), filtered to Drug Project Area and each
     FEI, accumulated to each cutoff date.
     Includes diagnostic comparison vs. Q&A-derived counts.

  3. IQVIA PANEL: Market volume from monthly IQVIA+SDUD+NADAC panel (NDC11),
     linked via NDC→FEI crosswalk, summed per calendar year Y.
     Replaces Q&A-embedded IQVIA which was double-counted across inspection rows.

  4. TIME-VARYING FAERS: FAERS reports filtered to fda_date ≤ cutoff.

  5. YEAR-SPECIFIC VALIDATION: Spearman ρ run separately per survey year
     (MQRI_Y vs Valisure_Y). Measurement coverage:
       2020 & 2022: NDMA + DMF
       2024: DMF + Dissolution

DATA DEDUPLICATION (v3.1 fix, still applied for Valisure extraction)
--------------------------------------------------------------------
  Q&A row structure: (FEI × NDC × survey_year × inspection_event)
  Valisure → dedup on (FEI, NDC, Year), then max per (FEI, Year)
  Inspection counts → RAW FILE (no Q&A dedup issue)
  Market → IQVIA monthly panel (no Q&A dedup issue)

SCORING (3 domains, each 0–25 pts, normalized to 0–100)
  MQRI = (D_reg + D_saf + D_mkt) / 75 × 100
  HIGH ≥ 55  |  MODERATE 30–54  |  LOW < 30
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
from scipy import stats
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = ('/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu'
        '/My Drive/North Carolina State University/Project - Drug Shortage/Data')
OUT  = f'{BASE}/06_07_08_12_14_15_21_22_23 - MQRI'
os.makedirs(OUT, exist_ok=True)

SURVEY_YEARS = list(range(2017, 2025))   # 2017–2024 annual panel

# OAI-predictive CFR prefixes (from 483/Citation NLP analysis)
OAI_PRED_CFRS = {'211.188', '211.111', '211.56', '211.63', '211.113',
                 '211.192', '211.160', '211.68', '211.100', '211.22'}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ndc_to_11digit(s):
    """Convert hyphenated NDC (any format) to zero-padded 11-digit string (5-4-2)."""
    s = str(s).strip()
    parts = s.split('-')
    if len(parts) == 3:
        return parts[0].zfill(5) + parts[1].zfill(4) + parts[2].zfill(2)
    return s.replace('-', '').replace(' ', '').zfill(11)


def parse_excel_date(series):
    """Robustly parse Excel date column: handles serial ints AND datetime strings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() > 0.5:   # mostly numeric → Excel serial
        return pd.to_datetime(
            numeric.apply(
                lambda x: (pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(x)))
                           if pd.notna(x) else pd.NaT
            )
        )
    return pd.to_datetime(series, errors='coerce')


def snum(v, fill=0.0):
    v = pd.to_numeric(v, errors='coerce')
    return float(v) if pd.notna(v) else fill


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD ALL DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("STEP 1: Loading data sources")
print("=" * 68)

# ── Q&A base (spine: Valisure outcomes + facility metadata) ───────────────────
print("  Q&As1234_v8_v02.xlsx ...")
base = pd.read_excel(f'{BASE}/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx')
base['FEI'] = base['FEI'].astype(str).str.split('.').str[0].str.strip()
OUR_FEIS = base['FEI'].unique().tolist()
print(f"    {base.shape}  |  {len(OUR_FEIS)} unique FEIs")

# ── Raw FDA Inspections Details ───────────────────────────────────────────────
print("  Inspections Details.xlsx ...")
insp_raw = pd.read_excel(
    f'{BASE}/14 - FDA - Inspection/raw/Inspections Details.xlsx',
    engine='openpyxl'
)
insp_raw.columns = [c.strip() for c in insp_raw.columns]
insp_raw['FEI']         = insp_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
insp_raw['insp_end_dt'] = parse_excel_date(insp_raw['Inspection End Date'])
insp_raw['is_oai']      = insp_raw['Classification'].str.contains('Official Action',  case=False, na=False).astype(int)
insp_raw['is_vai']      = insp_raw['Classification'].str.contains('Voluntary Action', case=False, na=False).astype(int)
insp_raw['is_nai']      = insp_raw['Classification'].str.contains('No Action',        case=False, na=False).astype(int)
insp_raw['has_483']     = (pd.to_numeric(insp_raw['Posted Citations'], errors='coerce').fillna(0) > 0).astype(int)
insp_drug = insp_raw[insp_raw['Project Area'].str.contains('Drug', case=False, na=False)].copy()
insp_our  = insp_drug[insp_drug['FEI'].isin(OUR_FEIS)].copy()
print(f"    All rows: {len(insp_raw):,}  |  Drug: {len(insp_drug):,}  |  Our FEIs: {len(insp_our):,}  "
      f"|  FEIs matched: {insp_our['FEI'].nunique()}/{len(OUR_FEIS)}")

# ── Raw FDA Citations Details ─────────────────────────────────────────────────
print("  Inspections Citations Details.xlsx ...")
cite_raw = pd.read_excel(
    f'{BASE}/14 - FDA - Inspection/raw/Inspections Citations Details.xlsx',
    engine='openpyxl'
)
cite_raw.columns = [c.strip() for c in cite_raw.columns]
cite_raw['FEI']          = cite_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
cite_raw['cite_end_dt']  = parse_excel_date(cite_raw['Inspection End Date'])
cite_drug = cite_raw[cite_raw['Program Area'].str.contains('Drug', case=False, na=False)].copy()
cite_our  = cite_drug[cite_drug['FEI'].isin(OUR_FEIS)].copy()
cite_our['is_oai_pred']  = cite_our['Act/CFR Number'].apply(
    lambda x: any(str(x).strip().startswith(c) for c in OAI_PRED_CFRS)
    if pd.notna(x) else False
).astype(int)
print(f"    Citation rows (our FEIs): {len(cite_our):,}")

# ── IQVIA monthly panel ───────────────────────────────────────────────────────
print("  IQVIA monthly panel ...")
iqvia = pd.read_csv(
    f'{BASE}/04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)'
    f'/processed/2026-02-24-iqvia_with_sdud_nadac.cleaned.csv',
    low_memory=False
)
iqvia['date']     = pd.to_datetime(iqvia['date'], errors='coerce')
iqvia['cal_year'] = iqvia['date'].dt.year
# ndc11 stored as integer (leading zeros stripped) → zero-pad to 11
iqvia['ndc11_str'] = (iqvia['ndc11'].astype(str)
                       .str.replace(r'\.0$', '', regex=True)
                       .str.zfill(11))
print(f"    {len(iqvia):,} rows  |  {iqvia['date'].min().date()} – {iqvia['date'].max().date()}")

# ── NDC → FEI crosswalk ───────────────────────────────────────────────────────
print("  NDC–FEI crosswalk ...")
xwalk = pd.read_excel(
    f'{BASE}/07 - Redica/processed/ndc_fei_73_v4.xlsx',
    sheet_name='detailed with notes'
)
xwalk['FEI']      = xwalk['FEI'].astype(str).str.split('.').str[0].str.strip()
xwalk['ndc11_str'] = xwalk['NDC'].apply(ndc_to_11digit)
ndc_fei_map       = xwalk[['ndc11_str', 'FEI']].dropna().drop_duplicates()

# Merge IQVIA → FEI
iqvia_fei = iqvia.merge(ndc_fei_map, on='ndc11_str', how='inner')
iqvia_our = iqvia_fei[iqvia_fei['FEI'].isin(OUR_FEIS)].copy()
print(f"    IQVIA rows matched to our FEIs: {len(iqvia_our):,}  |  {iqvia_our['FEI'].nunique()} FEIs")

# ── FAERS ─────────────────────────────────────────────────────────────────────
print("  FAERS ...")
faers = pd.read_csv(
    f'{BASE}/15 - FDA - Adverse Event/processed'
    f'/faers_metformin_anda_linked_2015Q1_2025Q3.csv',
    low_memory=False
)
faers['fda_date']   = pd.to_datetime(faers['fda_date'], errors='coerce')
faers['appl_no_str'] = faers['appl_no'].astype(str).str.zfill(6)

# ANDA → FEI via crosswalk
anda_fei = (xwalk[['application_num', 'FEI']].dropna()
            .assign(appl_no_str=lambda d:
                d['application_num'].astype(str)
                .str.replace(r'\D', '', regex=True).str.zfill(6))
            [['appl_no_str', 'FEI']].drop_duplicates())
faers_our = faers.merge(anda_fei, on='appl_no_str', how='inner')
faers_our = faers_our[faers_our['FEI'].isin(OUR_FEIS)].copy()
print(f"    FAERS rows matched: {len(faers_our):,}  |  {faers_our['FEI'].nunique()} FEIs")

# ── Static feature files: only truly time-invariant (severity composite, Redica)
print("  Static feature files ...")
STATIC_SRCS = [
    (f'{BASE}/12-14-21-22-23 - FEI Network/fei_node_summary.csv', 'fei',
     ['fei','severity_score'], {}),
]
static_df = pd.DataFrame({'FEI': OUR_FEIS})
for src, fk, cols, renames in STATIC_SRCS:
    try:
        df = pd.read_csv(src) if src.endswith('.csv') else pd.read_excel(src)
        df[fk] = df[fk].astype(str).str.split('.').str[0].str.strip()
        df = df.rename(columns={fk: 'FEI', **renames})
        avail = ['FEI'] + [c for c in cols[1:] if c in df.columns]
        static_df = static_df.merge(df[avail], on='FEI', how='left')
    except Exception as e:
        print(f"    WARNING: could not load {src.split('/')[-1]}: {e}")

# ── Load raw time-varying source files (recalls, WLs, imports, 483 NLP) ───────
print("  Loading raw time-varying source files ...")
try:
    recall_raw = pd.read_csv(f'{BASE}/22 - FDA - Recall/processed/recall_filtered.csv')
    recall_raw['FEI'] = recall_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
    recall_raw['recall_dt'] = pd.to_datetime(recall_raw['Recall_Date'], errors='coerce')
    print(f"    Recalls: {len(recall_raw)} rows, {recall_raw['FEI'].nunique()} FEIs")
except Exception as e:
    print(f"    WARNING: recalls not loaded: {e}"); recall_raw = pd.DataFrame()

try:
    wl_raw = pd.read_csv(f'{BASE}/21 - FDA - Warning Letter/processed/warning_letter_records.csv')
    wl_raw['FEI'] = wl_raw['search_fei'].astype(str).str.split('.').str[0].str.strip()
    wl_raw['wl_dt'] = pd.to_datetime(wl_raw['wl_date'], errors='coerce')
    print(f"    Warning Letters: {len(wl_raw)} rows, {wl_raw['FEI'].nunique()} FEIs")
except Exception as e:
    print(f"    WARNING: WLs not loaded: {e}"); wl_raw = pd.DataFrame()

try:
    import_raw = pd.read_csv(f'{BASE}/23 - FDA - Import Refusal/processed/import_refusal_filtered.csv')
    import_raw['FEI'] = import_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
    import_raw['import_dt'] = pd.to_datetime(import_raw['Refused_Date'], errors='coerce')
    import_raw_drug = import_raw[import_raw.get('has_drug_charge', import_raw.get('is_drug_product', pd.Series(True, index=import_raw.index))).astype(bool)]
    print(f"    Import Refusals: {len(import_raw_drug)} drug rows, {import_raw_drug['FEI'].nunique()} FEIs")
except Exception as e:
    print(f"    WARNING: imports not loaded: {e}"); import_raw_drug = pd.DataFrame()

try:
    obs483_raw = pd.read_csv(f'{BASE}/12 - FDA - 483/processed/483_observations.csv')
    obs483_raw['FEI'] = obs483_raw['fei'].astype(str).str.split('.').str[0].str.strip()
    obs483_raw['obs_dt'] = pd.to_datetime(obs483_raw['insp_date'], errors='coerce')
    print(f"    483 observations: {len(obs483_raw)} rows, {obs483_raw['FEI'].nunique()} FEIs")
except Exception as e:
    print(f"    WARNING: 483 observations not loaded: {e}"); obs483_raw = pd.DataFrame()

redica = pd.read_excel(f'{BASE}/07 - Redica/processed/SITE_RED_FLAG_AGG_SCORE.xlsx')
redica['FEI'] = redica['Fei'].astype(str).str.split('.').str[0].str.strip()
redica = redica.rename(columns={'Total Score': 'redica_rf_total'})
static_df = static_df.merge(redica[['FEI','redica_rf_total']], on='FEI', how='left')
print(f"    Static features: {static_df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FACILITY IDENTITY + VALISURE BY YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Facility identity + Valisure outcomes by survey year")

fac_info = (base.drop_duplicates(subset=['FEI'])
            [['FEI','Firm','CountryName','CountryCode']]
            .reset_index(drop=True))
fac_info = fac_info.merge(
    base.groupby('FEI').agg(
        n_ndc=('NDC','nunique'),
        n_survey_years=('Year','nunique'),
    ).reset_index(), on='FEI', how='left'
)

# Valisure per (FEI, survey_year): dedup to unique (FEI, NDC, Year) first
val_dedup       = base.drop_duplicates(subset=['FEI','NDC','Year'])
valisure_by_year = (val_dedup
    .groupby(['FEI','Year'])
    .agg(
        ndma_max=('NDMA (ng/DAY) Valisure', 'max'),
        dmf_max=('DMF (ng/DAY) Valisure',   'max'),
        diss_max=('Difference Factor',       'max'),
        n_ndcs_measured=('NDC','nunique'),
    )
    .reset_index()
    .rename(columns={'Year':'survey_year'}))

# Valisure coverage summary per FEI
val_cov = (val_dedup.groupby('FEI')
    .apply(lambda g: pd.Series({
        'n_years_with_dmf':  g[g['DMF (ng/DAY) Valisure'].notna()]['Year'].nunique(),
        'n_years_with_ndma': g[g['NDMA (ng/DAY) Valisure'].notna()]['Year'].nunique(),
        'n_years_with_diss': g[g['Difference Factor'].notna()]['Year'].nunique(),
    })).reset_index())
fac_info = fac_info.merge(val_cov, on='FEI', how='left')
print(f"  fac_info: {fac_info.shape}  |  valisure_by_year: {valisure_by_year.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DIAGNOSTIC: Q&A vs RAW FDA INSPECTION COUNTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Diagnostic — Q&A-derived vs raw FDA inspection counts (all-time)")

qa_insp = (base
    .drop_duplicates(subset=['FEI','Event Start Date','Event End Date'])
    .groupby('FEI')
    .agg(qa_n_insp=('Event Start Date','count'),
         qa_n_oai=('OAI','sum'),
         qa_n_vai=('VAI','sum'),
         qa_n_483=('483','sum'))
    .reset_index())

raw_insp_alltime = (insp_our
    .groupby('FEI')
    .agg(raw_n_insp=('FEI','count'),
         raw_n_oai=('is_oai','sum'),
         raw_n_vai=('is_vai','sum'),
         raw_n_483=('has_483','sum'))
    .reset_index())

diag = (fac_info[['FEI','Firm']]
        .merge(qa_insp,          on='FEI', how='left')
        .merge(raw_insp_alltime, on='FEI', how='left'))

print(f"  {'Firm':38s}  QA_insp  Raw_insp  QA_OAI  Raw_OAI  QA_483  Raw_483")
print("  " + "-"*80)
for _, r in diag.iterrows():
    print(f"  {str(r['Firm'])[:38]:38s}  "
          f"{r.get('qa_n_insp',0) or 0:7.0f}  {r.get('raw_n_insp',0) or 0:8.0f}  "
          f"{r.get('qa_n_oai',0) or 0:6.0f}  {r.get('raw_n_oai',0) or 0:7.0f}  "
          f"{r.get('qa_n_483',0) or 0:6.0f}  {r.get('raw_n_483',0) or 0:7.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TIME-VARYING FEATURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def insp_asof(fei_list, cutoff_year):
    """Inspection counts from raw FDA file, accumulated through Dec 31, cutoff_year."""
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = insp_our[insp_our['insp_end_dt'].notna() &
                  (insp_our['insp_end_dt'] <= cutoff)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        n_insp=('FEI','count'),
        n_oai=('is_oai','sum'),
        n_vai=('is_vai','sum'),
        n_nai=('is_nai','sum'),
        n_483s=('has_483','sum'),
    ).reset_index()
    agg['OAI_rate'] = agg['n_oai'] / agg['n_insp'].clip(lower=1)
    # Average gap between consecutive inspection end-dates
    def avg_gap(g):
        dates = g['insp_end_dt'].sort_values().dropna()
        return dates.diff().dt.days.dropna().mean() if len(dates) >= 2 else np.nan
    gaps = (df.groupby('FEI').apply(avg_gap)
              .reset_index().rename(columns={0: 'avg_insp_gap_days'}))
    return agg.merge(gaps, on='FEI', how='left')


def cfr_asof(fei_list, cutoff_year):
    """Count inspections with ≥1 OAI-predictive CFR citation, through cutoff_year."""
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = cite_our[cite_our['cite_end_dt'].notna() &
                  (cite_our['cite_end_dt'] <= cutoff) &
                  (cite_our['is_oai_pred'] == 1)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    return (df.groupby('FEI')['Inspection ID']
              .nunique()
              .reset_index()
              .rename(columns={'Inspection ID': 'n_oai_pred_cfr_insp'}))


def market_asof(fei_list, cutoff_year):
    """IQVIA volume for calendar year = cutoff_year (current market share).
       Falls back to most-recent year ≤ cutoff if that year has no data."""
    df = iqvia_our[iqvia_our['cal_year'] == cutoff_year]
    if len(df) == 0:
        df = iqvia_our[iqvia_our['cal_year'] <= cutoff_year]
        if len(df) == 0:
            return pd.DataFrame({'FEI': fei_list})
        df = (df.sort_values('cal_year', ascending=False)
                .groupby('FEI', group_keys=False)
                .apply(lambda g: g[g['cal_year'] == g['cal_year'].max()])
                .reset_index(drop=True))
    return df.groupby('FEI').agg(
        iqvia_units=('iqvia_extended_units','sum'),
        iqvia_trx=('iqvia_trx','sum'),
        nadac_price=('nadac_price','mean'),
    ).reset_index()


def faers_asof(fei_list, cutoff_year):
    """FAERS AE reports accumulated through cutoff_year."""
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = faers_our[faers_our['fda_date'].notna() &
                   (faers_our['fda_date'] <= cutoff)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        faers_total=('primaryid','nunique'),
        faers_serious=('serious_flag','sum'),
    ).reset_index()
    agg['faers_serious_rate'] = agg['faers_serious'] / agg['faers_total'].clip(lower=1)
    return agg


def recalls_asof(fei_list, cutoff_year):
    """Drug recall counts accumulated through Dec 31, cutoff_year."""
    if recall_raw.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = recall_raw[recall_raw['recall_dt'].notna() &
                    (recall_raw['recall_dt'] <= cutoff) &
                    (recall_raw['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        n_recalls_drug=('FEI', 'count'),
        n_recall_class_I=('Event Classification', lambda x: (x == 'Class I').sum()),
    ).reset_index()
    return agg


def wl_asof(fei_list, cutoff_year):
    """Warning letter counts + NLP flags accumulated through Dec 31, cutoff_year."""
    if wl_raw.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = wl_raw[wl_raw['wl_dt'].notna() &
                (wl_raw['wl_dt'] <= cutoff) &
                (wl_raw['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        n_warning_letters=('FEI', 'count'),
        ever_management_oversight=('has_management_oversight', 'any'),
        ever_corporate_failure_lang=('has_corporate_failure_lang', 'any'),
    ).reset_index()
    agg['ever_management_oversight']  = agg['ever_management_oversight'].astype(float)
    agg['ever_corporate_failure_lang'] = agg['ever_corporate_failure_lang'].astype(float)
    return agg


def import_asof(fei_list, cutoff_year):
    """Drug import refusal counts accumulated through Dec 31, cutoff_year."""
    if import_raw_drug.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = import_raw_drug[import_raw_drug['import_dt'].notna() &
                         (import_raw_drug['import_dt'] <= cutoff) &
                         (import_raw_drug['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    return df.groupby('FEI').agg(n_import_refusals=('FEI','count')).reset_index()


def nlp483_asof(fei_list, cutoff_year):
    """483 NLP flags as-of Dec 31, cutoff_year (ever observed in any 483 through that year)."""
    if obs483_raw.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = obs483_raw[obs483_raw['obs_dt'].notna() &
                    (obs483_raw['obs_dt'] <= cutoff) &
                    (obs483_raw['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    flag_cols = ['has_contamination','has_oos_oot','has_systemic',
                 'has_data_integrity','has_oai_predictive','has_repeat']
    avail = [c for c in flag_cols if c in df.columns]
    agg = df.groupby('FEI')[avail].any().astype(float).reset_index()
    rename_map = {
        'has_contamination':   'ever_contamination',
        'has_oos_oot':         'ever_oos_oot',
        'has_systemic':        'ever_systemic',
        'has_data_integrity':  'ever_data_integrity',
        'has_oai_predictive':  'ever_oai_predictive',
        'has_repeat':          'ever_repeat',
    }
    return agg.rename(columns=rename_map)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — COMPUTE MQRI PANEL (FEI × YEAR)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Computing MQRI panel for years:", SURVEY_YEARS)

# Normalization anchors: computed across all-years for cross-year comparability
vol_max_all = float(
    iqvia_our.groupby('FEI')['iqvia_extended_units'].sum().max() or 1.0
)
sev_max_all = float(
    pd.to_numeric(static_df.get('severity_score', pd.Series([1])),
                  errors='coerce').max() or 1.0
)

panel_rows = []

for yr in SURVEY_YEARS:
    print(f"\n  ── Year {yr} ──────────────────────────────────────────────")
    insp_f   = insp_asof(OUR_FEIS,    yr)
    cfr_f    = cfr_asof(OUR_FEIS,     yr)
    mkt_f    = market_asof(OUR_FEIS,  yr)
    faers_f  = faers_asof(OUR_FEIS,   yr)
    rec_f    = recalls_asof(OUR_FEIS,  yr)
    wl_f     = wl_asof(OUR_FEIS,       yr)
    imp_f    = import_asof(OUR_FEIS,   yr)
    nlp_f    = nlp483_asof(OUR_FEIS,   yr)

    for fei in OUR_FEIS:
        def gf(df, col, default=0.0):
            """Get scalar for this FEI from a feature DataFrame."""
            if df is None or 'FEI' not in df.columns: return default
            row = df[df['FEI'] == fei]
            if len(row) == 0 or col not in row.columns: return default
            return snum(row.iloc[0][col], default)

        def gs(col, default=0.0):
            """Get scalar for this FEI from static_df."""
            row = static_df[static_df['FEI'] == fei]
            if len(row) == 0 or col not in static_df.columns: return default
            return snum(row.iloc[0][col], default)

        # ── D_reg: Regulatory (max 25 pts) ────────────────────────────────────
        # [A] Base enforcement (time-varying, from raw FDA inspections)
        oai_pts    = min(gf(insp_f,'n_oai'),    5)  * 2      # max 10
        vai_pts    = min(gf(insp_f,'n_vai'),   10)  * 0.5    # max  5
        f483_pts   = min(gf(insp_f,'n_483s')/15, 1) * 5     # max  5
        wl_pts     = min(gf(wl_f,'n_warning_letters'), 4) * 2    # max 8  (time-varying)
        import_pts = min(gf(imp_f,'n_import_refusals'), 5) * 0.5  # max 2.5 (time-varying)
        repeat_pts = gf(nlp_f,'ever_repeat', 0) * 2              # max 2  (time-varying NLP)

        # [B] 483 NLP signals (time-varying: only 483s issued through cutoff_year)
        contam_pts  = gf(nlp_f,'ever_contamination', 0)  * 3     # 4.0× OAI risk
        oos_pts     = gf(nlp_f,'ever_oos_oot', 0)        * 2.5   # 3.5× OAI risk
        systemic_pts= gf(nlp_f,'ever_systemic', 0)       * 2     # 2.7× OAI risk
        dataint_pts = gf(nlp_f,'ever_data_integrity', 0) * 1     # 1.6× OAI risk
        oaipred_pts = gf(nlp_f,'ever_oai_predictive', 0) * 2     # OAI-predictive

        # [C] Inspection history signals (time-varying)
        n_pred_cfr  = gf(cfr_f, 'n_oai_pred_cfr_insp', 0)
        cfr_pts     = min(n_pred_cfr / 10, 1) * 3            # max 3
        oai_rate_pts= min(gf(insp_f,'OAI_rate',0), 1) * 2   # max 2
        gap_days    = gf(insp_f,'avg_insp_gap_days', 0)
        gap_pts     = min(gap_days / 1500, 1) * 2            # max 2

        # [D] Warning letter NLP (time-varying: only WLs issued through cutoff_year)
        mgmt_pts    = gf(wl_f,'ever_management_oversight', 0) * 1.5
        corp_pts    = gf(wl_f,'ever_corporate_failure_lang', 0) * 1

        d_reg = min(
            oai_pts + vai_pts + f483_pts + wl_pts + import_pts + repeat_pts
            + contam_pts + oos_pts + systemic_pts + dataint_pts + oaipred_pts
            + cfr_pts + oai_rate_pts + gap_pts + mgmt_pts + corp_pts,
            25.0
        )

        # ── D_saf: Safety / FAERS (max 25 pts) ────────────────────────────────
        ae_vol  = gf(faers_f, 'faers_total', 0)
        ae_ser  = gf(faers_f, 'faers_serious_rate', 0)
        d_saf   = min(
            min(np.log1p(ae_vol) / np.log1p(7000), 1) * 15
            + ae_ser * 10,
            25.0
        )

        # ── D_mkt: Market / Structural (max 25 pts) ───────────────────────────
        iq_units = gf(mkt_f, 'iqvia_units', 0)
        vol_pts  = min(iq_units / max(vol_max_all, 1), 1) * 8   # max 8
        rec_pts  = min(gf(rec_f,'n_recalls_drug'), 5) * 2        # max 10 (time-varying)
        cls1_pts = min(gf(rec_f,'n_recall_class_I'), 3) * 2      # max  6 (time-varying)
        sev_pts  = min(gs('severity_score', 0) / max(sev_max_all, 1), 1) * 5
        d_mkt    = min(vol_pts + rec_pts + cls1_pts + sev_pts, 25.0)

        mqri = round((d_reg + d_saf + d_mkt) / 75 * 100, 1)
        tier = 'HIGH' if mqri >= 55 else 'MODERATE' if mqri >= 30 else 'LOW'

        panel_rows.append({
            'FEI': fei, 'survey_year': yr,
            'mqri_total':      mqri,
            'mqri_regulatory': round(d_reg, 1),
            'mqri_safety':     round(d_saf, 1),
            'mqri_market':     round(d_mkt, 1),
            'mqri_tier':       tier,
            # Time-varying inspection (raw FDA)
            'n_insp_cumul':    gf(insp_f,'n_insp',0),
            'n_oai_cumul':     gf(insp_f,'n_oai',0),
            'n_vai_cumul':     gf(insp_f,'n_vai',0),
            'n_483s_cumul':    gf(insp_f,'n_483s',0),
            'OAI_rate_cumul':  round(gf(insp_f,'OAI_rate',0), 3),
            'avg_insp_gap':    round(gap_days, 0),
            'n_pred_cfr_insp': n_pred_cfr,
            # Time-varying market (IQVIA panel)
            'iqvia_units':     round(iq_units, 0),
            'iqvia_trx':       round(gf(mkt_f,'iqvia_trx',0), 0),
            # Time-varying FAERS
            'faers_total':     round(ae_vol, 0),
            'faers_serious_rate': round(ae_ser, 4),
            # Time-varying recalls (accumulated through year-end)
            'n_recalls_drug':  gf(rec_f,'n_recalls_drug',0),
            'n_recall_class_I':gf(rec_f,'n_recall_class_I',0),
            # Time-varying warning letters (accumulated through year-end)
            'n_warning_letters':          gf(wl_f,'n_warning_letters',0),
            'ever_management_oversight':  gf(wl_f,'ever_management_oversight',0),
            'ever_corporate_failure_lang':gf(wl_f,'ever_corporate_failure_lang',0),
            # Time-varying import refusals (accumulated through year-end)
            'n_import_refusals':          gf(imp_f,'n_import_refusals',0),
            # Time-varying 483 NLP flags (accumulated through year-end)
            'ever_contamination':  gf(nlp_f,'ever_contamination',0),
            'ever_oos_oot':        gf(nlp_f,'ever_oos_oot',0),
            'ever_systemic':       gf(nlp_f,'ever_systemic',0),
            'ever_data_integrity': gf(nlp_f,'ever_data_integrity',0),
            'ever_oai_predictive': gf(nlp_f,'ever_oai_predictive',0),
            'ever_repeat':         gf(nlp_f,'ever_repeat',0),
        })

    # Print year leaderboard
    yr_df = (pd.DataFrame([r for r in panel_rows if r['survey_year'] == yr])
               .merge(fac_info[['FEI','Firm']], on='FEI', how='left')
               .sort_values('mqri_total', ascending=False))
    print(f"  {'Firm':40s}  MQRI  Tier       Reg   Saf   Mkt")
    for _, r in yr_df.iterrows():
        print(f"  {str(r['Firm'])[:40]:40s}  {r['mqri_total']:4.1f}  "
              f"{r['mqri_tier']:9s}  {r['mqri_regulatory']:4.1f}  "
              f"{r['mqri_safety']:4.1f}  {r['mqri_market']:4.1f}")

# Assemble panel
panel = pd.DataFrame(panel_rows)
panel = panel.merge(fac_info[['FEI','Firm','CountryName','CountryCode']], on='FEI', how='left')
panel = panel.merge(valisure_by_year, on=['FEI','survey_year'], how='left')

# Attach truly static features (no date information: severity composite, Redica)
redica = pd.read_excel(f'{BASE}/07 - Redica/processed/SITE_RED_FLAG_AGG_SCORE.xlsx')
redica['FEI'] = redica['Fei'].astype(str).str.split('.').str[0].str.strip()
redica = redica.rename(columns={'Total Score': 'redica_rf_total'})
static_df = static_df.merge(redica[['FEI','redica_rf_total']], on='FEI', how='left')

static_cols = ['FEI','severity_score','redica_rf_total']
avail_sc = [c for c in static_cols if c in static_df.columns]
panel = panel.merge(static_df[avail_sc], on='FEI', how='left')
print(f"\n  Panel: {panel.shape}  (expect {len(OUR_FEIS)*len(SURVEY_YEARS)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VALIDATION: SPEARMAN ρ BY YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Validation — MQRI vs Valisure outcomes, by survey year")

PREDICTORS = [
    ('mqri_total',      'MQRI Total'),
    ('mqri_regulatory', 'Regulatory'),
    ('mqri_safety',     'Safety'),
    ('mqri_market',     'Market'),
]
OUTCOMES_BY_YEAR = {
    2020: [('ndma_max','NDMA Max (ng/day)'), ('dmf_max','DMF Max (ng/day)')],
    2022: [('ndma_max','NDMA Max (ng/day)'), ('dmf_max','DMF Max (ng/day)')],
    2024: [('dmf_max','DMF Max (ng/day)'),   ('diss_max','Dissolution diff factor')],
}

val_rows = []
for yr in SURVEY_YEARS:
    sub = panel[panel['survey_year'] == yr]
    for gt_col, gt_label in OUTCOMES_BY_YEAR.get(yr, []):
        if gt_col not in sub.columns:
            continue
        for pred_col, pred_label in PREDICTORS:
            if pred_col not in sub.columns:
                continue
            x = pd.to_numeric(sub[gt_col],   errors='coerce')
            y = pd.to_numeric(sub[pred_col], errors='coerce')
            mask = x.notna() & y.notna() & (x > 0)
            n = mask.sum()
            if n < 4:
                continue
            r, p = stats.spearmanr(x[mask].values, y[mask].values)
            stars = ('***' if p<0.001 else '**' if p<0.01 else
                     '*'   if p<0.05  else '†'  if p<0.10  else '')
            val_rows.append({'Year': yr, 'Predictor': pred_col,
                             'Outcome': gt_label, 'n': n,
                             'Spearman_rho': round(r,3), 'p_value': round(p,4)})
            print(f"  {yr}  {gt_label:30s} vs {pred_col:20s}: "
                  f"ρ={r:+.3f}, p={p:.4f}{stars:3s}, n={n}")

val_df = pd.DataFrame(val_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Saving outputs ...")
panel.to_csv(f'{OUT}/mqri_panel.csv', index=False)
val_df.to_csv(f'{OUT}/mqri_validation_correlations.csv', index=False)
fac_master = panel[panel['survey_year'] == max(SURVEY_YEARS)].copy()
fac_master.to_csv(f'{OUT}/mqri_facility_master.csv', index=False)
print(f"  ✓ mqri_panel.csv               {panel.shape}")
print(f"  ✓ mqri_validation_correlations {val_df.shape}")
print(f"  ✓ mqri_facility_master.csv     {fac_master.shape}  (year={max(SURVEY_YEARS)})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — BUILD DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Building dashboard HTML ...")


def _sv(v):
    """Safe JSON-serialisable scalar."""
    if v is None: return None
    if isinstance(v, (bool, np.bool_)):   return bool(v)
    if isinstance(v, (int, np.integer)):  return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return str(v)


def build_dashboard(panel_df, fac_df, val_df, out_path):

    # Panel records for JS
    panel_js = json.dumps([{k: _sv(v) for k, v in r.items()}
                            for _, r in panel_df.iterrows()])

    # Validation records for JS
    val_js = json.dumps(val_df.to_dict('records'))

    # Trend records: one per FEI with array of yearly MQRI values
    trend_records = []
    for fei in OUR_FEIS:
        rows = panel_df[panel_df['FEI'] == fei].sort_values('survey_year')
        fi   = fac_df[fac_df['FEI'] == fei]
        trend_records.append({
            'FEI':  fei,
            'Firm': str(fi['Firm'].values[0]) if len(fi) else fei,
            'CountryName': str(fi['CountryName'].values[0]) if len(fi) else '',
            'points': [{'year': int(r['survey_year']),
                        'mqri_total':      _sv(r['mqri_total']),
                        'mqri_regulatory': _sv(r['mqri_regulatory']),
                        'mqri_safety':     _sv(r['mqri_safety']),
                        'mqri_market':     _sv(r['mqri_market']),
                        'tier':            str(r['mqri_tier'])}
                       for _, r in rows.iterrows()]
        })
    trends_js = json.dumps(trend_records)

    # Scatter data per outcome × year
    def make_scatter(outcome_col, year):
        sub = panel_df[(panel_df['survey_year'] == year) &
                       panel_df[outcome_col].notna() &
                       (pd.to_numeric(panel_df[outcome_col], errors='coerce') > 0)]
        return [{'x': _sv(r['mqri_total']),
                 'y': _sv(r[outcome_col]),
                 'firm': str(r['Firm']).strip(),
                 'tier': str(r['mqri_tier'])}
                for _, r in sub.iterrows()]

    scatter_js = json.dumps({
        f'{col}_{yr}': make_scatter(col, yr)
        for yr in SURVEY_YEARS
        for col in ['ndma_max', 'dmf_max', 'diss_max']
        if col in panel_df.columns
    })

    # Correlation summary helper
    def gc(pred, outcome, year=None):
        df = val_df[(val_df['Predictor'] == pred) & (val_df['Outcome'] == outcome)]
        if year is not None: df = df[df['Year'] == year]
        if len(df) == 0: return None, None, None
        r = df.iloc[0]
        return r['Spearman_rho'], r['p_value'], r['n']

    latest_yr  = max(SURVEY_YEARS)
    survey_yrs = json.dumps(SURVEY_YEARS)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MQRI Dashboard v4.0 — Metformin Quality Risk Index</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#1a1a2e}}
header{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
        color:#fff;padding:20px 28px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
header h1{{font-size:1.45rem;font-weight:700}}
header p{{font-size:.82rem;color:#a8b2d8;margin-top:3px}}
.badge{{background:#e94560;color:#fff;border-radius:6px;padding:3px 10px;
        font-size:.72rem;font-weight:600;letter-spacing:.4px;white-space:nowrap}}
.badge.v4{{background:#3949ab}}
.tabs{{background:#fff;border-bottom:2px solid #e0e4ec;display:flex;
       padding:0 20px;position:sticky;top:0;z-index:100;
       box-shadow:0 2px 8px rgba(0,0,0,.06);flex-wrap:wrap}}
.tab{{padding:13px 16px;cursor:pointer;font-size:.86rem;font-weight:600;
       color:#888;border-bottom:3px solid transparent;transition:all .2s;white-space:nowrap}}
.tab:hover{{color:#333}}.tab.active{{color:#0f3460;border-bottom-color:#0f3460}}
.main{{max-width:1440px;margin:0 auto;padding:18px 14px}}
.page{{display:none}}.page.active{{display:block}}
.card{{background:#fff;border-radius:12px;padding:18px;
       box-shadow:0 2px 10px rgba(0,0,0,.07);margin-bottom:16px}}
.card h3{{font-size:.9rem;color:#333;margin-bottom:12px;font-weight:600;
          border-bottom:2px solid #f0f2f5;padding-bottom:7px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}}
@media(max-width:1000px){{.g2,.g3{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-size:.81rem}}
th{{background:#f0f2f5;padding:8px 10px;text-align:left;font-weight:600;
    color:#444;cursor:pointer;white-space:nowrap;user-select:none}}
th:hover{{background:#e4e8f0}}
td{{padding:8px 10px;border-bottom:1px solid #f0f2f5;vertical-align:middle}}
tr:hover td{{background:#fafbff}}
.T-HIGH{{background:#ffe4e4;color:#c0392b;font-weight:700;border-radius:4px;padding:2px 7px;font-size:.73rem}}
.T-MODERATE{{background:#fff3e0;color:#d35400;font-weight:700;border-radius:4px;padding:2px 7px;font-size:.73rem}}
.T-LOW{{background:#e8f5e9;color:#27ae60;font-weight:700;border-radius:4px;padding:2px 7px;font-size:.73rem}}
.sbar{{height:7px;border-radius:3px;background:#eee;display:inline-block;
        vertical-align:middle;min-width:70px;margin-right:5px}}
.sfill{{height:100%;border-radius:3px}}
.mqn{{font-size:1.05rem;font-weight:800;vertical-align:middle}}
/* Validation */
.vhero{{background:linear-gradient(135deg,#0f3460,#16213e);border-radius:14px;
         padding:24px 28px;color:#fff;margin-bottom:20px}}
.vhero h2{{font-size:1.28rem;font-weight:700;margin-bottom:7px}}
.vhero p{{color:#a8b2d8;font-size:.86rem;line-height:1.6;max-width:800px}}
.ybtn-row{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center}}
.ybtn{{padding:7px 18px;border:2px solid #0f3460;border-radius:18px;cursor:pointer;
        font-size:.84rem;font-weight:600;color:#0f3460;background:#fff;transition:all .2s}}
.ybtn:hover,.ybtn.active{{background:#0f3460;color:#fff}}
.ccards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
          gap:12px;margin-bottom:20px}}
.ccard{{border-radius:10px;padding:16px;text-align:center;border:2px solid}}
.ccard.sig{{border-color:#27ae60;background:#f0fff4}}
.ccard.ns{{border-color:#e0e0e0;background:#fafafa}}
.crho{{font-size:2.1rem;font-weight:800}}
.crho.pos{{color:#0f3460}}.crho.weak{{color:#bbb}}
.clbl{{font-size:.75rem;font-weight:600;color:#555;margin-top:5px;
        text-transform:uppercase;letter-spacing:.4px}}
.spill{{display:inline-block;border-radius:3px;padding:1px 6px;font-size:.7rem;
         font-weight:700;color:#fff}}
.spill.sig{{background:#27ae60}}.spill.ns{{background:#bbb}}
.bspot{{background:#fff8e1;border:2px solid #f39c12;border-radius:10px;
         padding:13px 16px;margin-bottom:18px}}
.bspot h4{{color:#d35400;font-size:.88rem;font-weight:700;margin-bottom:4px}}
.bspot p{{font-size:.82rem;color:#555;line-height:1.6}}
.mbox{{background:#e8eaf6;border-left:4px solid #3949ab;border-radius:7px;
        padding:11px 14px;margin-bottom:16px;font-size:.83rem;color:#333;line-height:1.6}}
/* Trends */
.thero{{background:linear-gradient(135deg,#1a237e,#283593);border-radius:14px;
         padding:22px 26px;color:#fff;margin-bottom:20px}}
.thero h2{{font-size:1.25rem;font-weight:700;margin-bottom:6px}}
.thero p{{color:#c5cae9;font-size:.85rem;line-height:1.6;max-width:800px}}
/* Facility */
.det{{background:#fff;border-radius:12px;padding:18px;
       box-shadow:0 2px 10px rgba(0,0,0,.07);display:none;margin-top:12px}}
.det.on{{display:block}}
.dgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}
.dcard{{border-radius:9px;padding:12px;text-align:center}}
.dcard.r{{background:#e8eaf6}}.dcard.s{{background:#fce4ec}}.dcard.m{{background:#e8f5e9}}
.dscore{{font-size:1.9rem;font-weight:800}}
.dlbl{{font-size:.68rem;color:#666;margin-top:3px;font-weight:600;text-transform:uppercase}}
.igrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}
.iitem{{background:#f8f9ff;border-radius:7px;padding:9px 11px}}
.ival{{font-size:1.25rem;font-weight:700;color:#1a1a2e}}
.ilbl{{font-size:.7rem;color:#888;margin-top:2px}}
.fgrid{{display:flex;flex-wrap:wrap;gap:7px;margin-top:7px}}
.fchip{{border-radius:7px;padding:7px 11px;font-size:.77rem;min-width:145px;border:1px solid}}
.cbtn{{background:#f0f2f5;border:none;border-radius:7px;padding:7px 14px;
        cursor:pointer;font-size:.83rem;color:#444}}
.cbtn:hover{{background:#e0e4ec}}
select{{border:1px solid #dde;border-radius:6px;padding:5px 10px;
         font-size:.83rem;background:#f8f9ff;cursor:pointer;outline:none}}
/* Data source availability grid */
.dsrc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-top:8px}}
.dsrc-cell{{background:#f8f9ff;border-radius:7px;padding:8px 10px;font-size:.76rem}}
.dsrc-label{{font-weight:600;color:#444;margin-bottom:3px}}
.dsrc-val{{color:#1a1a2e;font-weight:700}}
.dsrc-val.has{{color:#27ae60}}.dsrc-val.none{{color:#bbb}}
</style>
</head>
<body>
<header>
  <div style="flex:1">
    <h1>Metformin Quality Risk Index (MQRI) <span class="badge v4">v4.0</span></h1>
    <p>Time-varying facility scores · 18 manufacturers · Survey years 2020 · 2022 · 2024 · Validated against Valisure independent lab testing</p>
  </div>
  <span class="badge">RESEARCH PREVIEW</span>
</header>

<div class="tabs">
  <div class="tab active" onclick="showPage('overview',this)">📊 Overview ({latest_yr})</div>
  <div class="tab"        onclick="showPage('trends',this)">📈 MQRI Over Time</div>
  <div class="tab"        onclick="showPage('validation',this)">🔬 Validation vs Ground Truth</div>
  <div class="tab"        onclick="showPage('facility',this)">🏭 Facility Profile</div>
</div>

<div class="main">

<!-- OVERVIEW -->
<div class="page active" id="page-overview">
  <div style="background:#e8f0fe;border:1px solid #4285f4;border-radius:9px;padding:12px 16px;
              margin-bottom:14px;font-size:.82rem;color:#1a237e;line-height:1.6">
    <strong>🔬 Proof-of-Concept Index (v4.0):</strong>
    Scoring weights are expert-calibrated based on empirical OAI-predictor analysis
    (e.g., CFR 211.188 appears in 67% of OAI inspections → higher weight).
    Future versions will replace additive weights with logistic regression coefficients
    (facility-level OAI prediction) and drug shortage events as ground truth.
    &nbsp;<strong>Ground truth:</strong> Valisure independent lab testing (2020/2022/2024) —
    limited coverage; MarketScan clinical outcomes and Utah shortage database planned.
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:16px">
    <label style="font-size:.81rem;font-weight:600;color:#555">Tier&nbsp;</label>
    <select id="fTier" onchange="applyF()">
      <option value="">All</option>
      <option>HIGH</option><option>MODERATE</option><option>LOW</option>
    </select>
    <label style="font-size:.81rem;font-weight:600;color:#555">Country&nbsp;</label>
    <select id="fCountry" onchange="applyF()">
      <option value="">All</option>
      <option>India</option><option>United States of America</option>
      <option>China</option><option>Canada</option><option>Bangladesh</option>
    </select>
    <span style="margin-left:auto;font-size:.8rem;color:#999">Scores as of {latest_yr} · click row for full profile</span>
  </div>
  <div class="g2">
    <div class="card"><h3>MQRI Total Score — {latest_yr}</h3><div id="cBar" style="height:330px"></div></div>
    <div class="card"><h3>Domain Breakdown (top 6 facilities)</h3><div id="cRadar" style="height:330px"></div></div>
  </div>
  <div class="card">
    <h3>Risk Leaderboard — {latest_yr}</h3>
    <div style="overflow-x:auto">
    <table><thead><tr>
      <th onclick="srt('Firm')">Manufacturer ↕</th>
      <th onclick="srt('CountryName')">Country ↕</th>
      <th onclick="srt('mqri_total')" title="Expert-calibrated weights: OAI×2 (based on 7% OAI base rate), contamination flag×3 (4× OAI risk), etc. Future: logistic regression coefficients">MQRI ↕</th><th>Tier</th>
      <th onclick="srt('mqri_regulatory')">Reg ↕</th>
      <th onclick="srt('mqri_safety')">Safety ↕</th>
      <th onclick="srt('mqri_market')">Market ↕</th>
      <th onclick="srt('n_oai_cumul')">OAI ↕</th>
      <th onclick="srt('faers_total')">FAERS ↕</th>
      <th onclick="srt('dmf_max')" title="Valisure — NOT in score">DMF Max†</th>
    </tr></thead><tbody id="lb-body"></tbody></table>
    <p style="font-size:.71rem;color:#999;margin-top:5px">†Valisure outcomes shown for reference — excluded from MQRI</p>
    </div>
  </div>
  <div class="bspot" style="margin-top:0">
    <h4>📋 Data Completeness Example: Alkem Laboratories (FEI 3006370533)</h4>
    <p>
      <strong>Public FDA inspections database:</strong> 8 inspections, 6 VAI, 0 posted citations (483s) recorded.<br>
      <strong>Commercial data (Redica/Q&amp;A):</strong> 8 inspections, 6 VAI, 4 Form 483s documented.<br>
      <strong>Impact:</strong> This facility's 483 NLP flags (contamination, OOS/OOT, etc.) are based solely
      on publicly available 483 PDFs — the 4 Redica-sourced 483s may contain additional violations not captured here.
      This gap applies to other facilities as well. The index uses public data only for NLP signals;
      commercial completeness would improve accuracy.
    </p>
  </div>
</div>

<!-- TRENDS -->
<div class="page" id="page-trends">
  <div class="thero">
    <h2>MQRI Evolution Over Time — 2020 → 2022 → 2024</h2>
    <p>Each year's score uses only data available by Dec 31 of that year:
       FDA inspection counts (cumulative), IQVIA volume for that calendar year,
       and FAERS reports filed to date.
       <strong>Time-invariant features (483 NLP flags, Warning Letter text signals) are set once
       based on all available public documents and do not change between years. They appear
       constant across 2020→2022→2024 because the underlying text analyses were not
       date-partitioned.</strong></p>
  </div>
  <div class="card"><h3>All-Facilities MQRI Trend (hover to identify)</h3>
    <div id="cAllTrends" style="height:400px"></div></div>
  <div class="g2">
    <div class="card"><h3>Average Domain Score by Year</h3>
      <div id="cDomainTrend" style="height:300px"></div></div>
    <div class="card"><h3>Score Distribution by Year</h3>
      <div id="cBoxplot" style="height:300px"></div></div>
  </div>
  <div class="card">
    <h3>Per-Facility MQRI Breakdown Over Time</h3>
    <div style="margin-bottom:11px">
      <select id="trendSel" onchange="renderFacilityTrend()">
        <option value="">— select facility —</option>
      </select>
    </div>
    <div id="cFacTrend" style="height:300px"></div>
  </div>
</div>

<!-- VALIDATION -->
<div class="page" id="page-validation">
  <div class="vhero">
    <h2>MQRI vs Valisure Ground Truth — Year-by-Year Validation</h2>
    <p>Valisure independently purchased and tested metformin tablets in 2020, 2022, and 2024.
       These NDMA, DMF, and dissolution measurements are <strong style="color:#fff">never used
       in any MQRI calculation</strong>. Significant correlations below show that regulatory,
       safety, and market signals can predict real contamination levels.</p>
  </div>
  <div class="mbox">
    <strong>Measurement availability:</strong>&nbsp;
    2020 &amp; 2022: NDMA + DMF &nbsp;|&nbsp; 2024: DMF + Dissolution.
    Each scatter uses the MQRI computed from data available in that same year.
  </div>
  <div class="ybtn-row" id="valYearRow">
    <div class="ybtn active" onclick="selValYear(2020,this)">2020</div>
    <div class="ybtn"        onclick="selValYear(2022,this)">2022</div>
    <div class="ybtn"        onclick="selValYear(2024,this)">2024</div>
  </div>
  <div id="valContent"></div>
  <div class="bspot">
    <h4>⚠ Regulatory Blindspot — Why NDMA Predictions Are Weak</h4>
    <p><strong>Marksans Pharma</strong> has the highest NDMA (396.8 ng/day, 4× FDA limit of 96 ng/day)
       yet scores LOW on MQRI every year — it has minimal FDA enforcement history.
       The MQRI measures <em>regulatory-visible</em> quality failure.
       Facilities that escape inspection or enforcement can have high contamination with no record.
       DMF and dissolution correlate with MQRI because their risk is broadly distributed across
       facilities with enforcement histories; NDMA was concentrated in a few "invisible" ones.</p>
  </div>
  <div class="card">
    <h3>Full Correlation Heatmap (best-powered year per outcome)</h3>
    <div id="cHeatmap" style="height:320px"></div>
  </div>
</div>

<!-- FACILITY -->
<div class="page" id="page-facility">
  <div class="card" style="margin-bottom:12px">
    <div style="overflow-x:auto">
    <table><thead><tr>
      <th onclick="srt2('Firm')">Manufacturer ↕</th>
      <th onclick="srt2('CountryName')">Country ↕</th>
      <th onclick="srt2('mqri_total')">MQRI ({latest_yr}) ↕</th><th>Tier</th>
      <th onclick="srt2('mqri_regulatory')">Reg ↕</th>
      <th onclick="srt2('mqri_safety')">Safety ↕</th>
      <th onclick="srt2('mqri_market')">Market ↕</th>
    </tr></thead><tbody id="lb2-body"></tbody></table>
    </div>
  </div>
  <div class="det" id="detPanel">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px">
      <div>
        <div style="font-size:1.15rem;font-weight:700" id="dTitle"></div>
        <div style="font-size:.83rem;color:#666;margin-top:2px" id="dSub"></div>
      </div>
      <button class="cbtn" onclick="closeD()">✕ Close</button>
    </div>
    <div class="mbox" style="margin-bottom:10px">
      <strong>Time-Invariant Features:</strong> Signals marked with 🔒 (483 NLP flags, Warning Letter NLP)
      are derived from publicly available regulatory documents processed without year-specific date attribution.
      They represent "ever observed" status and do not change between survey years. Completeness depends on
      public document availability — commercial data sources (e.g., Redica) may contain additional records.
    </div>
    <div class="ybtn-row" id="fac-year-row">
      <span style="font-size:.8rem;font-weight:600;color:#555;margin-right:6px">View year:</span>
      <div class="ybtn active" onclick="openFacYear(2020,this)">2020</div>
      <div class="ybtn" onclick="openFacYear(2022,this)">2022</div>
      <div class="ybtn active" onclick="openFacYear(2024,this)">2024</div>
    </div>
    <div class="dgrid" id="dDomains"></div>
    <div class="g2" style="margin-bottom:12px">
      <div class="card" style="padding:11px"><h3>Domain Radar</h3>
        <div id="dRadar" style="height:220px"></div></div>
      <div class="card" style="padding:11px"><h3>MQRI Trend 2020–2024</h3>
        <div id="dTrend" style="height:220px"></div></div>
    </div>
    <div class="g2" style="margin-bottom:12px">
      <div class="card" style="padding:11px"><h3>Key Metrics</h3>
        <div class="igrid" id="dMetrics"></div></div>
      <div class="card" style="padding:11px">
        <h3>Valisure Ground Truth <span style="font-size:.7rem;color:#888;font-weight:400">(NOT in score)</span></h3>
        <div class="igrid" id="dValisure"></div></div>
    </div>
    <div class="card" style="padding:11px;margin-bottom:12px">
      <h3>Signal Flags 🔒 (483 NLP + WL NLP — Time-Invariant (cumulative))</h3>
      <div class="fgrid" id="dFlags"></div></div>
    <div class="card" style="padding:11px">
      <h3>Data Source Availability</h3>
      <p style="font-size:.78rem;color:#555;margin-bottom:8px">
        Time-varying sources show per-year availability; time-invariant sources are cumulative.
      </p>
      <div id="dDataSrc"></div>
      <p style="font-size:.72rem;color:#888;margin-top:10px;border-top:1px solid #eee;padding-top:8px">
        ⚠ Time-Invariant (cumulative) signals are derived from publicly available 483 PDFs and Warning Letter text.
        They reflect "ever observed" across all inspections without year attribution.
        Some facilities may have undisclosed 483s not captured.
      </p>
    </div>
  </div>
</div>

</div><!-- /main -->

<script>
const PANEL = {panel_js};
const TRENDS = {trends_js};
const VAL = {val_js};
const SCATTER = {scatter_js};
const LATEST = {latest_yr};
const SY = {survey_yrs};

const fmt  = (v,d=1) => (v==null||isNaN(+v)) ? '–' : (+v).toFixed(d);
const tc   = t => t==='HIGH'?'#e74c3c':t==='MODERATE'?'#e67e22':'#27ae60';
const sf   = s => s.replace(/ (Pharmaceuticals?|Industries?|Limited|Laboratories?|Labs?|Corp\.?|Inc\.?|Ltd\.?)/gi,'').trim().slice(0,20);

const latest = () => PANEL.filter(d=>d.survey_year===LATEST);

function showPage(id,el){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  el.classList.add('active');
  if(id==='overview')  renderOverview();
  if(id==='trends')    renderTrends();
  if(id==='validation')renderValidation(2020);
  if(id==='facility')  renderFacList();
}}

// ── OVERVIEW ─────────────────────────────────────────────────────────────────
let sc='mqri_total',sa=false,filt=latest();
function applyF(){{
  const ct=document.getElementById('fCountry').value;
  const tr=document.getElementById('fTier').value;
  filt=latest().filter(d=>(!ct||d.CountryName===ct)&&(!tr||d.mqri_tier===tr));
  renderBar();renderLB();
}}
function srt(c){{if(sc===c)sa=!sa;else{{sc=c;sa=false;}}renderLB();}}
function renderBar(){{
  const s=[...filt].sort((a,b)=>(b.mqri_total||0)-(a.mqri_total||0));
  Plotly.react('cBar',[
    {{name:'Regulatory',x:s.map(d=>sf(d.Firm)),y:s.map(d=>d.mqri_regulatory||0),type:'bar',marker:{{color:'#3949ab'}}}},
    {{name:'Safety',    x:s.map(d=>sf(d.Firm)),y:s.map(d=>d.mqri_safety||0),    type:'bar',marker:{{color:'#e74c3c'}}}},
    {{name:'Market',    x:s.map(d=>sf(d.Firm)),y:s.map(d=>d.mqri_market||0),    type:'bar',marker:{{color:'#27ae60'}}}},
  ],{{barmode:'stack',margin:{{t:8,b:110,l:38,r:8}},
     yaxis:{{title:'MQRI',range:[0,100]}},xaxis:{{tickangle:-38,tickfont:{{size:9}}}},
     legend:{{orientation:'h',y:1.06}},
     shapes:[
       {{type:'line',x0:-0.5,x1:s.length-0.5,y0:55,y1:55,line:{{color:'#e74c3c',width:1,dash:'dot'}}}},
       {{type:'line',x0:-0.5,x1:s.length-0.5,y0:30,y1:30,line:{{color:'#e67e22',width:1,dash:'dot'}}}}]
  }},{{responsive:true}});
}}
function renderRadar(){{
  const top=[...filt].sort((a,b)=>(b.mqri_total||0)-(a.mqri_total||0)).slice(0,6);
  Plotly.react('cRadar',top.map(d=>{{
    return{{type:'scatterpolar',fill:'toself',name:sf(d.Firm),opacity:.6,
            r:[d.mqri_regulatory||0,d.mqri_safety||0,d.mqri_market||0,d.mqri_regulatory||0],
            theta:['Regulatory','Safety','Market','Regulatory']}};
  }}),{{polar:{{radialaxis:{{range:[0,25]}}}},margin:{{t:18,b:18,l:40,r:40}},
        legend:{{font:{{size:9}}}}}},{{responsive:true}});
}}
function renderLB(){{
  const data=[...filt].sort((a,b)=>{{
    const va=a[sc]??-Infinity,vb=b[sc]??-Infinity;
    return sa?(va>vb?1:-1):(va<vb?1:-1);
  }});
  document.getElementById('lb-body').innerHTML=data.map(d=>`
    <tr onclick="openFac('${{d.FEI}}')" style="cursor:pointer">
      <td><b>${{d.Firm}}</b></td><td>${{d.CountryName}}</td>
      <td><span class="sbar"><span class="sfill" style="width:${{d.mqri_total||0}}%;background:${{tc(d.mqri_tier)}}"></span></span>
          <span class="mqn" style="color:${{tc(d.mqri_tier)}}">${{fmt(d.mqri_total)}}</span></td>
      <td><span class="T-${{d.mqri_tier}}">${{d.mqri_tier}}</span></td>
      <td>${{fmt(d.mqri_regulatory)}}</td><td>${{fmt(d.mqri_safety)}}</td><td>${{fmt(d.mqri_market)}}</td>
      <td>${{fmt(d.n_oai_cumul,0)}}</td><td>${{fmt(d.faers_total,0)}}</td>
      <td style="color:${{(d.dmf_max||0)>8800?'#e74c3c':'#333'}}">${{fmt(d.dmf_max,0)}}</td>
    </tr>`).join('');
}}
function renderOverview(){{applyF();renderRadar();}}

// ── TRENDS ───────────────────────────────────────────────────────────────────
function renderTrends(){{
  Plotly.react('cAllTrends',TRENDS.map(t=>{{
    const pts=t.points.filter(p=>p.mqri_total!=null);
    return{{x:pts.map(p=>p.year),y:pts.map(p=>p.mqri_total),mode:'lines+markers',
            name:sf(t.Firm),line:{{width:2}},marker:{{size:6}},
            hovertemplate:`<b>${{t.Firm}}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>`}};
  }}),{{margin:{{t:8,b:40,l:48,r:16}},
        xaxis:{{tickvals:SY,title:'Survey Year',type:'category'}},
        yaxis:{{title:'MQRI Total (0–100)',range:[0,100]}},
        shapes:[
          {{type:'line',x0:0,x1:2,y0:55,y1:55,line:{{color:'#e74c3c',width:1,dash:'dot'}}}},
          {{type:'line',x0:0,x1:2,y0:30,y1:30,line:{{color:'#e67e22',width:1,dash:'dot'}}}}],
        legend:{{font:{{size:8}}}}}},{{responsive:true}});

  const avg=SY.map(yr=>{{
    const rows=PANEL.filter(d=>d.survey_year===yr);
    const mn=col=>rows.reduce((s,d)=>s+(d[col]||0),0)/Math.max(rows.length,1);
    return{{yr,r:mn('mqri_regulatory'),s:mn('mqri_safety'),m:mn('mqri_market')}};
  }});
  Plotly.react('cDomainTrend',[
    {{name:'Regulatory',x:avg.map(a=>String(a.yr)),y:avg.map(a=>a.r),type:'bar',marker:{{color:'#3949ab'}}}},
    {{name:'Safety',    x:avg.map(a=>String(a.yr)),y:avg.map(a=>a.s),type:'bar',marker:{{color:'#e74c3c'}}}},
    {{name:'Market',    x:avg.map(a=>String(a.yr)),y:avg.map(a=>a.m),type:'bar',marker:{{color:'#27ae60'}}}},
  ],{{barmode:'stack',margin:{{t:8,b:38,l:48,r:8}},
      xaxis:{{tickvals:SY.map(String),type:'category'}},
      yaxis:{{title:'Avg Score'}},legend:{{orientation:'h',y:1.06}}}},{{responsive:true}});

  Plotly.react('cBoxplot',SY.map(yr=>{{
    const vals=PANEL.filter(d=>d.survey_year===yr).map(d=>d.mqri_total).filter(v=>v!=null);
    return{{type:'box',y:vals,name:String(yr),marker:{{color:'#3949ab'}},boxmean:true}};
  }}),{{margin:{{t:8,b:38,l:48,r:8}},yaxis:{{title:'MQRI Total',range:[0,100]}}}},{{responsive:true}});

  const sel=document.getElementById('trendSel');
  if(sel.options.length<=1) TRENDS.forEach(t=>{{
    const o=document.createElement('option');o.value=t.FEI;o.textContent=t.Firm;sel.appendChild(o);
  }});
}}

function renderFacilityTrend(){{
  const fei=document.getElementById('trendSel').value; if(!fei) return;
  const t=TRENDS.find(t=>t.FEI===fei); if(!t) return;
  const pts=t.points.filter(p=>p.mqri_total!=null);
  const yrs=pts.map(p=>String(p.year));
  Plotly.react('cFacTrend',[
    {{name:'Total',      x:yrs,y:pts.map(p=>p.mqri_total),     mode:'lines+markers',line:{{color:'#1a1a2e',width:3}},marker:{{size:9}}}},
    {{name:'Regulatory', x:yrs,y:pts.map(p=>p.mqri_regulatory),mode:'lines+markers',line:{{color:'#3949ab'}}}},
    {{name:'Safety',     x:yrs,y:pts.map(p=>p.mqri_safety),    mode:'lines+markers',line:{{color:'#e74c3c'}}}},
    {{name:'Market',     x:yrs,y:pts.map(p=>p.mqri_market),    mode:'lines+markers',line:{{color:'#27ae60'}}}},
  ],{{title:{{text:t.Firm,font:{{size:11}}}},
      margin:{{t:36,b:38,l:48,r:16}},
      xaxis:{{type:'category',tickvals:SY.map(String),title:'Survey Year'}},
      yaxis:{{title:'Score',range:[0,100]}},
      shapes:[
        {{type:'line',x0:0,x1:SY.length-1,y0:55,y1:55,line:{{color:'#e74c3c',width:1,dash:'dot'}}}},
        {{type:'line',x0:0,x1:SY.length-1,y0:30,y1:30,line:{{color:'#e67e22',width:1,dash:'dot'}}}}],
      legend:{{font:{{size:9}}}}}},{{responsive:true}});
}}

// ── VALIDATION ───────────────────────────────────────────────────────────────
let curValYr=2020;
function selValYear(yr,el){{
  document.querySelectorAll('#valYearRow .ybtn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active'); curValYr=yr; renderValidation(yr);
}}
function renderValidation(yr){{
  const oMap={{2020:[['ndma_max','NDMA Max (ng/day)',96],['dmf_max','DMF Max (ng/day)',null]],
               2022:[['ndma_max','NDMA Max (ng/day)',96],['dmf_max','DMF Max (ng/day)',null]],
               2024:[['dmf_max','DMF Max (ng/day)',null], ['diss_max','Dissolution Diff Factor',null]]}};
  const outs=oMap[yr]||[];

  const cards=outs.map(([col,lbl])=>{{
    const row=VAL.find(r=>r.Predictor==='mqri_total'&&r.Outcome===lbl&&r.Year===yr);
    const rho=row?row.Spearman_rho:null, p=row?row.p_value:null, n=row?row.n:null;
    const sig=p!=null&&p<0.05;
    const badge=p==null?'n/a':p<0.001?'***':p<0.01?'**':p<0.05?'*':p<0.10?'†':'n.s.';
    return `<div class="ccard ${{sig?'sig':'ns'}}">
      <div class="crho ${{rho!=null&&rho>0?'pos':'weak'}}">${{rho!=null?((rho>0?'+':'')+rho.toFixed(3)):'N/A'}}</div>
      <div class="clbl">${{lbl}}</div>
      <div style="margin-top:6px"><span class="spill ${{sig?'sig':'ns'}}">${{badge}}</span>
        <span style="font-size:.76rem;color:#666;margin-left:5px">${{p!=null?'p='+p.toFixed(3):''}}</span>
      </div>
      <div style="font-size:.72rem;color:#888;margin-top:3px">n=${{n||'?'}} facilities</div>
    </div>`;
  }}).join('');

  const scatDivs=outs.map(([col,lbl])=>
    `<div class="card"><h3>MQRI vs ${{lbl}} (${{yr}})</h3>
       <div id="sc_${{col}}_${{yr}}" style="height:300px"></div></div>`
  ).join('');

  document.getElementById('valContent').innerHTML=
    `<div class="ccards">${{cards}}</div><div class="g2">${{scatDivs}}</div>`;

  setTimeout(()=>{{
    outs.forEach(([col,lbl,limit])=>{{
      const key=`${{col}}_${{yr}}`;
      const pts=(SCATTER[key]||[]).filter(p=>p.x!=null&&p.y!=null);
      if(!pts.length) return;
      Plotly.react(`sc_${{col}}_${{yr}}`,[{{
        x:pts.map(p=>p.x),y:pts.map(p=>p.y),
        mode:'markers+text',
        text:pts.map(p=>sf(p.firm)),textposition:'top center',textfont:{{size:8}},
        marker:{{color:pts.map(p=>tc(p.tier)),size:10,line:{{color:'#fff',width:1}}}},
        type:'scatter',
        hovertemplate:'<b>%{{text}}</b><br>MQRI: %{{x}}<br>'+lbl+': %{{y:.1f}}<extra></extra>'
      }}],{{
        margin:{{t:16,b:46,l:66,r:16}},
        xaxis:{{title:'MQRI Total Score ('+yr+')',range:[0,100]}},
        yaxis:{{title:lbl}},
        shapes:limit?[{{type:'line',x0:0,x1:100,y0:limit,y1:limit,
                        line:{{color:'#e74c3c',width:1.5,dash:'dot'}}}}]:[]
      }},{{responsive:true}});
    }});
  }},60);

  // Heatmap (all years, best-powered per outcome)
  const preds=['mqri_total','mqri_regulatory','mqri_safety','mqri_market'];
  const plbl=['MQRI Total','Regulatory','Safety','Market'];
  const outs2=['NDMA Max (ng/day)','DMF Max (ng/day)','Dissolution diff factor'];
  const yFor=o=>o.includes('NDMA')?2022:o.includes('Dissolution')?2024:LATEST;
  const z=preds.map(p=>outs2.map(o=>{{
    const r=VAL.find(r=>r.Predictor===p&&r.Outcome===o&&r.Year===yFor(o));
    return r?r.Spearman_rho:null;
  }}));
  const tx=preds.map(p=>outs2.map(o=>{{
    const r=VAL.find(r=>r.Predictor===p&&r.Outcome===o&&r.Year===yFor(o));
    if(!r) return '';
    const s=r.p_value<0.001?'***':r.p_value<0.01?'**':r.p_value<0.05?'*':r.p_value<0.10?'†':'';
    return `${{r.Spearman_rho>0?'+':''}}${{r.Spearman_rho.toFixed(2)}}${{s}}\\np=${{r.p_value.toFixed(3)}}`;
  }}));
  Plotly.react('cHeatmap',[{{
    type:'heatmap',z,x:outs2,y:plbl,text:tx,texttemplate:'%{{text}}',textfont:{{size:10}},
    colorscale:[[0,'#3949ab'],[0.5,'#f5f5f5'],[1,'#e74c3c']],
    zmid:0,zmin:-1,zmax:1,colorbar:{{title:'ρ',len:.8}}
  }}],{{margin:{{t:16,b:76,l:112,r:56}},
        xaxis:{{tickfont:{{size:10}}}},yaxis:{{tickfont:{{size:10}}}}}},{{responsive:true}});
}}

// ── FACILITY ─────────────────────────────────────────────────────────────────
let sc2='mqri_total',sa2=false,curFacFEI=null;
function srt2(c){{if(sc2===c)sa2=!sa2;else{{sc2=c;sa2=false;}}renderFacList();}}
function renderFacList(){{
  const data=[...latest()].sort((a,b)=>{{
    const va=a[sc2]??-Infinity,vb=b[sc2]??-Infinity;
    return sa2?(va>vb?1:-1):(va<vb?1:-1);
  }});
  document.getElementById('lb2-body').innerHTML=data.map(d=>`
    <tr onclick="openFac('${{d.FEI}}')" style="cursor:pointer">
      <td><b>${{d.Firm}}</b></td><td>${{d.CountryName}}</td>
      <td><span class="mqn" style="color:${{tc(d.mqri_tier)}}">${{fmt(d.mqri_total)}}</span></td>
      <td><span class="T-${{d.mqri_tier}}">${{d.mqri_tier}}</span></td>
      <td>${{fmt(d.mqri_regulatory)}}</td><td>${{fmt(d.mqri_safety)}}</td><td>${{fmt(d.mqri_market)}}</td>
    </tr>`).join('');
}}

function openFacYear(yr, el){{
  if(!curFacFEI) return;
  // Update active button state within fac-year-row only
  document.querySelectorAll('#fac-year-row .ybtn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  // Get the year-specific panel record
  const d=PANEL.find(r=>r.FEI===curFacFEI&&r.survey_year===yr);
  if(!d) return;
  // Update domain score cards
  document.getElementById('dDomains').innerHTML=[
    ['r','Regulatory',d.mqri_regulatory,'OAIs · 483s · WLs · NLP'],
    ['s','Safety',d.mqri_safety,'FAERS AE Reports'],
    ['m','Market / Structural',d.mqri_market,'Volume · Recalls'],
  ].map(([cls,lbl,val,sub])=>
    `<div class="dcard ${{cls}}"><div class="dscore">${{fmt(val)}}</div>
     <div class="dlbl">${{lbl}}</div>
     <div style="font-size:.66rem;color:#888;margin-top:2px">${{sub}}</div></div>`
  ).join('');
  // Update subtitle to reflect selected year
  document.getElementById('dSub').textContent=
    `${{d.CountryName}}  ·  FEI: ${{d.FEI}}  ·  MQRI ${{fmt(d.mqri_total)}} (${{d.mqri_tier}}) — ${{yr}}`;
  // Update key metrics for the selected year
  document.getElementById('dMetrics').innerHTML=[
    ['OAI (cumul)',fmt(d.n_oai_cumul,0)],
    ['VAI (cumul)',fmt(d.n_vai_cumul,0)],
    ['483s (cumul)',fmt(d.n_483s_cumul,0)],
    ['OAI Rate',fmt(d.OAI_rate_cumul,3)],
    ['FAERS Reports',fmt(d.faers_total,0)],
    ['Serious AE %',d.faers_serious_rate!=null?(+d.faers_serious_rate*100).toFixed(0)+'%':'–'],
    ['Drug Recalls',fmt(d.n_recalls_drug,0)],
    ['Warning Letters',fmt(d.n_warning_letters,0)],
    ['Redica Score',fmt(d.redica_rf_total,1)],
  ].map(([l,v])=>`<div class="iitem"><div class="ival">${{v}}</div><div class="ilbl">${{l}}</div></div>`).join('');
  // Update radar for selected year
  Plotly.react('dRadar',[{{
    type:'scatterpolar',fill:'toself',
    r:[d.mqri_regulatory||0,d.mqri_safety||0,d.mqri_market||0,d.mqri_regulatory||0],
    theta:['Regulatory','Safety','Market','Regulatory'],marker:{{color:tc(d.mqri_tier)}}
  }}],{{polar:{{radialaxis:{{range:[0,25]}}}},margin:{{t:8,b:8,l:28,r:28}},
       title:{{text:'Domain Radar ('+yr+')',font:{{size:10}}}}}},{{responsive:true}});
}}

function openFac(fei){{
  curFacFEI=fei;
  const d=latest().find(f=>f.FEI===fei); if(!d) return;
  const tr=TRENDS.find(t=>t.FEI===fei);
  document.getElementById('dTitle').textContent=d.Firm;
  document.getElementById('dSub').textContent=
    `${{d.CountryName}}  ·  FEI: ${{d.FEI}}  ·  MQRI ${{fmt(d.mqri_total)}} (${{d.mqri_tier}}) — ${{LATEST}}`;
  // Reset year buttons: default to latest year active
  document.querySelectorAll('#fac-year-row .ybtn').forEach((b,i)=>{{
    b.classList.remove('active');
    if(SY[i]===LATEST) b.classList.add('active');
  }});
  document.getElementById('dDomains').innerHTML=[
    ['r','Regulatory',d.mqri_regulatory,'OAIs · 483s · WLs · NLP'],
    ['s','Safety',d.mqri_safety,'FAERS AE Reports'],
    ['m','Market / Structural',d.mqri_market,'Volume · Recalls'],
  ].map(([cls,lbl,val,sub])=>
    `<div class="dcard ${{cls}}"><div class="dscore">${{fmt(val)}}</div>
     <div class="dlbl">${{lbl}}</div>
     <div style="font-size:.66rem;color:#888;margin-top:2px">${{sub}}</div></div>`
  ).join('');
  Plotly.react('dRadar',[{{
    type:'scatterpolar',fill:'toself',
    r:[d.mqri_regulatory||0,d.mqri_safety||0,d.mqri_market||0,d.mqri_regulatory||0],
    theta:['Regulatory','Safety','Market','Regulatory'],marker:{{color:tc(d.mqri_tier)}}
  }}],{{polar:{{radialaxis:{{range:[0,25]}}}},margin:{{t:8,b:8,l:28,r:28}}}},{{responsive:true}});
  if(tr){{
    const pts=tr.points.filter(p=>p.mqri_total!=null);
    const yrs=pts.map(p=>String(p.year));
    Plotly.react('dTrend',[
      {{name:'Total',     x:yrs,y:pts.map(p=>p.mqri_total),     mode:'lines+markers',line:{{color:'#1a1a2e',width:3}},marker:{{size:8}}}},
      {{name:'Regulatory',x:yrs,y:pts.map(p=>p.mqri_regulatory),mode:'lines+markers',line:{{color:'#3949ab'}}}},
      {{name:'Safety',    x:yrs,y:pts.map(p=>p.mqri_safety),    mode:'lines+markers',line:{{color:'#e74c3c'}}}},
      {{name:'Market',    x:yrs,y:pts.map(p=>p.mqri_market),    mode:'lines+markers',line:{{color:'#27ae60'}}}},
    ],{{margin:{{t:8,b:38,l:48,r:8}},
        xaxis:{{type:'category',tickvals:SY.map(String)}},
        yaxis:{{range:[0,100]}},legend:{{font:{{size:8}}}}}},{{responsive:true}});
  }}
  document.getElementById('dMetrics').innerHTML=[
    ['OAI (cumul)',fmt(d.n_oai_cumul,0)],
    ['VAI (cumul)',fmt(d.n_vai_cumul,0)],
    ['483s (cumul)',fmt(d.n_483s_cumul,0)],
    ['OAI Rate',fmt(d.OAI_rate_cumul,3)],
    ['FAERS Reports',fmt(d.faers_total,0)],
    ['Serious AE %',d.faers_serious_rate!=null?(+d.faers_serious_rate*100).toFixed(0)+'%':'–'],
    ['Drug Recalls',fmt(d.n_recalls_drug,0)],
    ['Warning Letters',fmt(d.n_warning_letters,0)],
    ['Redica Score',fmt(d.redica_rf_total,1)],
  ].map(([l,v])=>`<div class="iitem"><div class="ival">${{v}}</div><div class="ilbl">${{l}}</div></div>`).join('');

  const vRows=PANEL.filter(r=>r.FEI===fei).sort((a,b)=>a.survey_year-b.survey_year);
  document.getElementById('dValisure').innerHTML=SY.map(yr=>{{
    const vr=vRows.find(r=>r.survey_year===yr)||{{}};
    return `<div class="iitem">
      <div class="ival" style="font-size:.92rem">${{yr}}</div>
      <div class="ilbl">NDMA: ${{fmt(vr.ndma_max,1)}} · DMF: ${{fmt(vr.dmf_max,0)}} · Diss: ${{fmt(vr.diss_max,3)}}</div>
    </div>`;
  }}).join('');

  const flags=[
    ['Contamination (483)' ,d.ever_contamination,'4× OAI risk'],
    ['OOS/OOT (483)',        d.ever_oos_oot,      '3.5× OAI risk'],
    ['Systemic (483)',       d.ever_systemic,      '2.7× OAI risk'],
    ['Data Integrity (483)', d.ever_data_integrity,'1.6× OAI risk'],
    ['OAI-Predictive 483',   d.ever_oai_predictive,'OAI-tagged'],
    ['Mgmt Oversight (WL)',  d.ever_management_oversight,'WL flag'],
    ['Corp Failure (WL)',    d.ever_corporate_failure_lang,'WL flag'],
    ['Repeat 483s',          d.ever_repeat,        'Systemic'],
  ];
  document.getElementById('dFlags').innerHTML=flags.map(([nm,val,desc])=>{{
    const on=val===true||val===1||val==='True'||val==='1';
    return `<div class="fchip" style="background:${{on?'#ffe4e4':'#f0f2f5'}};
              border-color:${{on?'#e74c3c':'#dde'}};color:${{on?'#c0392b':'#888'}}">
      <strong>${{on?'⚠ YES':'✓ No'}}</strong> ${{nm}}
      <div style="font-size:.66rem;color:${{on?'#c0392b':'#aaa'}};margin-top:1px">${{desc}}</div>
    </div>`;
  }}).join('');

  // Data source availability section
  const tvHeader='<p style="font-size:.76rem;font-weight:700;color:#444;margin:8px 0 4px">Time-Varying Sources (public FDA / IQVIA / FAERS)</p>';
  const tvGrid=SY.map(yr=>{{
    const yr_row=PANEL.find(r=>r.FEI===fei&&r.survey_year===yr)||{{}};
    const n_insp=yr_row.n_insp_cumul!=null?yr_row.n_insp_cumul:null;
    const iq=yr_row.iqvia_units!=null?yr_row.iqvia_units:null;
    const fa=yr_row.faers_total!=null?yr_row.faers_total:null;
    const insp_html=n_insp!=null?`<span class="dsrc-val ${{n_insp>0?'has':'none'}}">${{n_insp>0?n_insp+' insp':'None'}}</span>`:`<span class="dsrc-val none">–</span>`;
    const iq_html=iq!=null?`<span class="dsrc-val ${{iq>0?'has':'none'}}">${{iq>0?'Yes':'No'}}</span>`:`<span class="dsrc-val none">–</span>`;
    const fa_html=fa!=null?`<span class="dsrc-val ${{fa>0?'has':'none'}}">${{fa>0?fa+' rpts':'None'}}</span>`:`<span class="dsrc-val none">–</span>`;
    return `<div class="dsrc-cell">
      <div class="dsrc-label">${{yr}}</div>
      <div>FDA Inspections: ${{insp_html}}</div>
      <div>IQVIA Market: ${{iq_html}}</div>
      <div>FAERS Reports: ${{fa_html}}</div>
    </div>`;
  }}).join('');
  const tiHeader='<p style="font-size:.76rem;font-weight:700;color:#444;margin:10px 0 4px">🔒 Time-Invariant (cumulative) Signals — No Year Attribution</p>';
  const nlp483=[
    ['ever_contamination','Contamination (483)'],
    ['ever_oos_oot','OOS/OOT (483)'],
    ['ever_systemic','Systemic (483)'],
    ['ever_data_integrity','Data Integrity (483)'],
    ['ever_oai_predictive','OAI-Predictive (483)'],
  ];
  const nlpwl=[
    ['ever_management_oversight','Mgmt Oversight (WL)'],
    ['ever_corporate_failure_lang','Corp Failure (WL)'],
  ];
  const tiGrid=[...nlp483,...nlpwl].map(([key,lbl])=>{{
    const val=d[key];
    const on=val===true||val===1||val==='True'||val==='1';
    return `<div class="dsrc-cell">
      <div class="dsrc-label">${{lbl}}</div>
      <span class="dsrc-val ${{on?'has':'none'}}">${{on?'YES — observed':'Not detected'}}</span>
    </div>`;
  }}).join('');
  document.getElementById('dDataSrc').innerHTML=
    tvHeader+'<div class="dsrc-grid">'+tvGrid+'</div>'+
    tiHeader+'<div class="dsrc-grid">'+tiGrid+'</div>';

  const pan=document.getElementById('detPanel');
  pan.classList.add('on');
  pan.scrollIntoView({{behavior:'smooth',block:'nearest'}});
  if(!document.getElementById('page-facility').classList.contains('active'))
    showPage('facility',document.querySelectorAll('.tab')[3]);
}}
function closeD(){{document.getElementById('detPanel').classList.remove('on');curFacFEI=null;}}

// INIT
renderOverview();
</script>
</body></html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓ Dashboard → {out_path}")


build_dashboard(panel, fac_info, val_df, f'{OUT}/MQRI_Dashboard.html')
print("\nAll done. ✓")
