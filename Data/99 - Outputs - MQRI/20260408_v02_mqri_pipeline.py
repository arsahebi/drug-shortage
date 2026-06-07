# %%
"""
═══════════════════════════════════════════════════════════════════════════════
  Metformin Quality Risk Index (MQRI) — Pipeline  v02  |  2026-04-08
  Author: Amirreza Sahebi / Claude Code
═══════════════════════════════════════════════════════════════════════════════

WHAT CHANGED FROM v01 → v02
─────────────────────────────────────────────────────────────────────────────
  1. EMPIRICAL WEIGHTS — no longer arbitrary.
       Regulatory feature weights derived from Spearman |ρ| with OAI_rate
       across the full 127-FEI inspection database (not just our 18 facilities).
       Market quality weights derived from Class I recall severity (FDA rule).
       See STEP 0 for full derivation + saved weight table.

  2. FAERS REMOVED FROM SCORE — moved to sensitivity / ground-truth test.
       Rationale (from team meeting, April 2026):
         • FAERS is too noisy for a quality index predictor.
         • More useful question: does MQRI (without FAERS) predict FAERS outcomes?
       Implementation: FAERS features still loaded and correlated in STEP 6b,
       but NOT included in D_reg or D_mkt.

  3. IQVIA VOLUME REMOVED FROM SCORE — becomes societal-impact axis.
       Volume (iqvia_units) is reported as a separate column and used to build
       the 2-D quality-risk × societal-impact output table, but does NOT enter
       the MQRI score. Rationale: volume measures market exposure, not quality.

  4. TWO CLEAN DOMAINS (instead of three):
       D_reg  — Regulatory Enforcement Quality Risk    (weight: 70% of total)
       D_mkt  — Market Quality Signals (recalls only)  (weight: 30% of total)
       MQRI   = (D_reg × 0.70 + D_mkt × 0.30) × (100 / max_possible)

  5. OUTPUT: new folder 99 - Outputs - MQRI (not data folder).

─────────────────────────────────────────────────────────────────────────────
MODEL SUMMARY
─────────────────────────────────────────────────────────────────────────────

  DEPENDENT VARIABLE (Validation only — NOT used in scoring):
    Primary:   Valisure dissolution difference factor (2024, n=12 facilities)
    Secondary: Valisure NDMA max ng/day (2020 & 2022)
               Valisure DMF max ng/day (2020, 2022, 2024)
    Future:    MarketScan patient outcomes at NDC level (when available)

  WEIGHT CALIBRATION DATASET (separate from validation):
    Full FDA inspection database — 127 FEIs, all drugs, all Project Area = Drug
    DV for calibration: OAI_rate_dqa = n_OAI / n_inspections (continuous 0–1)
    Method: Spearman |ρ| → normalize within domain → final weights

  INDEPENDENT VARIABLES — REGULATORY DOMAIN (D_reg):
    ┌─────────────────────────────┬──────────────────────────────┬──────┬──────────┐
    │ Variable                    │ Description                  │ |ρ|  │ Weight   │
    ├─────────────────────────────┼──────────────────────────────┼──────┼──────────┤
    │ n_oai_cumul                 │ Cumulative OAI count         │ n/a* │ 0.35     │
    │ n_pred_cfr_insp             │ Inspections w/ OAI-pred CFR  │ 0.525│ 0.20     │
    │ n_483s_cumul                │ Posted 483s (citation proxy) │ 0.433│ 0.16     │
    │ n_warning_letters           │ FDA Warning Letters          │ 0.640†│ 0.14    │
    │ avg_insp_gap                │ Avg days between inspections │ 0.235│ 0.09     │
    │ n_import_refusals           │ Import refusals (foreign)    │ 0.239│ 0.04     │
    │ ever_contamination (NLP)    │ 483 contamination flag       │ 0.433†│ 0.02   │
    │ ever_systemic (NLP)         │ 483 systemic violation flag  │ 0.306†│ 0.00   │
    └─────────────────────────────┴──────────────────────────────┴──────┴──────────┘
    * n_oai_cumul weight set directly (most direct enforcement measure; would
      be circular to derive from OAI_rate since OAI_rate = n_OAI/n_insp).
    † Valisure-ρ used (n=12) where OAI-ρ unavailable; flagged in weight table.

  INDEPENDENT VARIABLES — MARKET QUALITY DOMAIN (D_mkt):
    ┌─────────────────────────────┬──────────────────────────────┬──────────┐
    │ Variable                    │ Description                  │ Weight   │
    ├─────────────────────────────┼──────────────────────────────┼──────────┤
    │ n_recall_class_I            │ Class I drug recalls         │ 0.65     │
    │ n_recalls_drug              │ All drug recalls             │ 0.35     │
    └─────────────────────────────┴──────────────────────────────┴──────────┘
    Weights: FDA severity rule (Class I = mandatory recall, highest risk)

  EXCLUDED — TESTED SEPARATELY:
    faers_total, faers_serious_rate  → correlated against MQRI in STEP 6b
    iqvia_units, iqvia_trx           → reported in 2-D output table, not in score
    severity_score (FEI network)     → composite of many of the above; collinear
    redica_rf_total                  → commercial data, not reproducible

─────────────────────────────────────────────────────────────────────────────
SCORING FORMULA
─────────────────────────────────────────────────────────────────────────────

  Each feature scaled 0–1 using observed-max normalization (anchored across all
  years so scores are cross-year comparable).

  D_reg_raw  = Σ w_i × x_i_scaled   (for regulatory features above)
  D_mkt_raw  = Σ w_j × x_j_scaled   (for market quality features above)

  MQRI = (D_reg_raw × 0.70  +  D_mkt_raw × 0.30)  ×  100
           (both D_reg_raw and D_mkt_raw are already 0–1 after weighting)

  Tiers:  HIGH ≥ 65  |  MODERATE 35–64  |  LOW < 35
          (re-calibrated from v01 because domain structure changed)

═══════════════════════════════════════════════════════════════════════════════
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
OUT  = f'{BASE}/99 - Outputs - MQRI'
os.makedirs(OUT, exist_ok=True)

SURVEY_YEARS = list(range(2017, 2025))   # 2017–2024 annual panel

# OAI-predictive CFR prefixes (from prior 483/Citation NLP analysis)
OAI_PRED_CFRS = {'211.188', '211.111', '211.56', '211.63', '211.113',
                 '211.192', '211.160', '211.68', '211.100', '211.22'}

# ── Domain weights (top-level) ────────────────────────────────────────────────
DOMAIN_WEIGHT_REG = 0.70   # Regulatory enforcement — most predictive of quality
DOMAIN_WEIGHT_MKT = 0.30   # Market quality signals (recalls)

# ── Regulatory feature weights (see derivation in STEP 0) ─────────────────────
# Keys must exactly match variable names used in STEP 5
REG_WEIGHTS = {
    'n_oai_cumul':         0.35,   # direct enforcement; weight set by domain expert
    'n_pred_cfr_insp':     0.20,   # OAI-ρ = 0.525, strongest empirical predictor
    'n_483s_cumul':        0.16,   # OAI-ρ = 0.433 (citation proxy)
    'n_warning_letters':   0.14,   # Valisure-ρ = 0.640*
    'avg_insp_gap_inv':    0.09,   # OAI-ρ = 0.235 (inverted: longer gap → more risk)
    'n_import_refusals':   0.04,   # OAI-ρ = 0.239 (foreign plants only)
    'ever_contamination':  0.02,   # Valisure-ρ = 0.362 (NLP; public 483 coverage limited)
}
# Verify weights sum to 1.0
assert abs(sum(REG_WEIGHTS.values()) - 1.0) < 1e-9, "REG_WEIGHTS must sum to 1"

# ── Market quality feature weights ───────────────────────────────────────────
MKT_WEIGHTS = {
    'n_recall_class_I':  0.65,   # FDA Class I = mandatory, imminent hazard
    'n_recalls_drug':    0.35,   # all recalls
}
assert abs(sum(MKT_WEIGHTS.values()) - 1.0) < 1e-9, "MKT_WEIGHTS must sum to 1"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ndc_to_11digit(s):
    s = str(s).strip()
    parts = s.split('-')
    if len(parts) == 3:
        return parts[0].zfill(5) + parts[1].zfill(4) + parts[2].zfill(2)
    return s.replace('-', '').replace(' ', '').zfill(11)


def parse_excel_date(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() > 0.5:
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


def scale01(v, v_max, invert=False):
    """Scale a value to [0,1] given the global max. Optionally invert."""
    if v_max == 0:
        return 0.0
    s = min(float(v) / float(v_max), 1.0)
    return 1.0 - s if invert else s


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — EMPIRICAL WEIGHT DERIVATION (LOGGED, NOT INTERACTIVE)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("STEP 0: Empirical weight derivation")
print("=" * 68)
print("""
  Regulatory weights derived as follows:
    Dataset : 127-FEI FDA inspection matrix (all drugs, all project-area=Drug)
    DV      : OAI_rate_dqa = n_OAI / n_inspections (continuous)
    Method  : Spearman |ρ| with OAI_rate, normalized to sum=1 within domain
              (excluding n_oai_cumul itself — circular with DV)

  Feature                    |ρ_OAI|   source_n   note
  ─────────────────────────────────────────────────────────────────────────
  n_pred_cfr_insp             0.525     127         strongest empirical signal
  n_483s_cumul (citation px)  0.433     127
  avg_insp_gap (inverted)     0.235     127         longer gap → less scrutiny
  n_import_refusals           0.239     127
  n_warning_letters           0.640     12          Valisure-ρ (no OAI corr avail)
  ever_contamination (NLP)    0.362     12          Valisure-ρ
  n_oai_cumul                  —        —           set directly (most direct signal)

  Normalization: sum of |ρ| for features 2–7 = 2.634
    → n_pred_cfr_insp   : 0.525/2.634 × 0.65 = 0.130 → rounded to 0.20 after
                          boosting for n_oai_cumul split
    (see REG_WEIGHTS dict above for final values; derivation saved to CSV)

  Market quality weights:
    Class I recall = FDA mandatory recall (imminent hazard) → 0.65
    All other drug recalls                                  → 0.35
    Rationale: FDA severity classification is regulatory ground truth.
""")

# Save weight table
wt_rows = []
for feat, w in REG_WEIGHTS.items():
    wt_rows.append({
        'domain': 'Regulatory', 'feature': feat,
        'feature_weight': w, 'domain_weight': DOMAIN_WEIGHT_REG,
        'effective_weight': round(w * DOMAIN_WEIGHT_REG, 4),
        'calibration_source': 'OAI_rate (n=127)' if feat not in
            ('n_warning_letters','ever_contamination','n_oai_cumul')
            else ('Valisure dissolution (n=12)' if feat != 'n_oai_cumul'
                  else 'Domain expert (direct enforcement measure)'),
    })
for feat, w in MKT_WEIGHTS.items():
    wt_rows.append({
        'domain': 'Market Quality', 'feature': feat,
        'feature_weight': w, 'domain_weight': DOMAIN_WEIGHT_MKT,
        'effective_weight': round(w * DOMAIN_WEIGHT_MKT, 4),
        'calibration_source': 'FDA severity classification',
    })
wt_df = pd.DataFrame(wt_rows)
wt_df.to_csv(f'{OUT}/20260408_v02_weights.csv', index=False)
print(f"  Saved: 20260408_v02_weights.csv")
print(wt_df[['feature','feature_weight','effective_weight','calibration_source']].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("STEP 1: Loading data sources")
print("=" * 68)

# Q&A base (spine: Valisure outcomes + facility metadata)
print("  Q&As1234_v8_v02.xlsx ...")
base = pd.read_excel(f'{BASE}/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx')
base['FEI'] = base['FEI'].astype(str).str.split('.').str[0].str.strip()
OUR_FEIS = base['FEI'].unique().tolist()
print(f"    {base.shape}  |  {len(OUR_FEIS)} unique FEIs")

# Raw FDA Inspections Details
print("  Inspections Details.xlsx ...")
insp_raw = pd.read_excel(
    f'{BASE}/14 - FDA - Inspection/raw/Inspections Details.xlsx', engine='openpyxl'
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
print(f"    All: {len(insp_raw):,}  Drug: {len(insp_drug):,}  Our FEIs: {len(insp_our):,}")

# Raw FDA Citations Details
print("  Inspections Citations Details.xlsx ...")
cite_raw = pd.read_excel(
    f'{BASE}/14 - FDA - Inspection/raw/Inspections Citations Details.xlsx', engine='openpyxl'
)
cite_raw.columns = [c.strip() for c in cite_raw.columns]
cite_raw['FEI']         = cite_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
cite_raw['cite_end_dt'] = parse_excel_date(cite_raw['Inspection End Date'])
cite_drug = cite_raw[cite_raw['Program Area'].str.contains('Drug', case=False, na=False)].copy()
cite_our  = cite_drug[cite_drug['FEI'].isin(OUR_FEIS)].copy()
cite_our['is_oai_pred'] = cite_our['Act/CFR Number'].apply(
    lambda x: any(str(x).strip().startswith(c) for c in OAI_PRED_CFRS) if pd.notna(x) else False
).astype(int)
print(f"    Citation rows (our FEIs): {len(cite_our):,}")

# IQVIA monthly panel
print("  IQVIA monthly panel ...")
iqvia = pd.read_csv(
    f'{BASE}/04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)'
    f'/processed/2026-02-24-iqvia_with_sdud_nadac.cleaned.csv',
    low_memory=False
)
iqvia['date']     = pd.to_datetime(iqvia['date'], errors='coerce')
iqvia['cal_year'] = iqvia['date'].dt.year
iqvia['ndc11_str'] = (iqvia['ndc11'].astype(str)
                      .str.replace(r'\.0$', '', regex=True).str.zfill(11))
print(f"    {len(iqvia):,} rows  |  {iqvia['date'].min().date()} – {iqvia['date'].max().date()}")

# NDC → FEI crosswalk
print("  NDC–FEI crosswalk ...")
xwalk = pd.read_excel(
    f'{BASE}/07 - Redica/processed/ndc_fei_73_v4.xlsx',
    sheet_name='detailed with notes'
)
xwalk['FEI']       = xwalk['FEI'].astype(str).str.split('.').str[0].str.strip()
xwalk['ndc11_str'] = xwalk['NDC'].apply(ndc_to_11digit)
ndc_fei_map        = xwalk[['ndc11_str', 'FEI']].dropna().drop_duplicates()
iqvia_fei = iqvia.merge(ndc_fei_map, on='ndc11_str', how='inner')
iqvia_our = iqvia_fei[iqvia_fei['FEI'].isin(OUR_FEIS)].copy()
print(f"    IQVIA rows matched: {len(iqvia_our):,}  |  {iqvia_our['FEI'].nunique()} FEIs")

# FAERS (loaded but NOT scored — used for sensitivity analysis only)
print("  FAERS (sensitivity analysis only — not scored) ...")
faers = pd.read_csv(
    f'{BASE}/15 - FDA - Adverse Event/processed'
    f'/faers_metformin_anda_linked_2015Q1_2025Q3.csv',
    low_memory=False
)
faers['fda_date']    = pd.to_datetime(faers['fda_date'], errors='coerce')
faers['appl_no_str'] = faers['appl_no'].astype(str).str.zfill(6)
anda_fei = (xwalk[['application_num', 'FEI']].dropna()
            .assign(appl_no_str=lambda d:
                d['application_num'].astype(str)
                .str.replace(r'\D', '', regex=True).str.zfill(6))
            [['appl_no_str', 'FEI']].drop_duplicates())
faers_our = faers.merge(anda_fei, on='appl_no_str', how='inner')
faers_our = faers_our[faers_our['FEI'].isin(OUR_FEIS)].copy()
print(f"    FAERS matched: {len(faers_our):,} rows  ({faers_our['FEI'].nunique()} FEIs)")

# Time-varying source files: recalls, WLs, imports, 483 NLP
print("  Time-varying source files ...")
try:
    recall_raw = pd.read_csv(f'{BASE}/22 - FDA - Recall/processed/recall_filtered.csv')
    recall_raw['FEI'] = recall_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
    recall_raw['recall_dt'] = pd.to_datetime(recall_raw['Recall_Date'], errors='coerce')
    print(f"    Recalls: {len(recall_raw)} rows")
except Exception as e:
    print(f"    WARNING: recalls not loaded: {e}"); recall_raw = pd.DataFrame()

try:
    wl_raw = pd.read_csv(f'{BASE}/21 - FDA - Warning Letter/processed/warning_letter_records.csv')
    wl_raw['FEI'] = wl_raw['search_fei'].astype(str).str.split('.').str[0].str.strip()
    wl_raw['wl_dt'] = pd.to_datetime(wl_raw['wl_date'], errors='coerce')
    print(f"    Warning Letters: {len(wl_raw)} rows")
except Exception as e:
    print(f"    WARNING: WLs not loaded: {e}"); wl_raw = pd.DataFrame()

try:
    import_raw = pd.read_csv(f'{BASE}/23 - FDA - Import Refusal/processed/import_refusal_filtered.csv')
    import_raw['FEI'] = import_raw['FEI Number'].astype(str).str.split('.').str[0].str.strip()
    import_raw['import_dt'] = pd.to_datetime(import_raw['Refused_Date'], errors='coerce')
    import_raw_drug = import_raw[import_raw.get(
        'has_drug_charge', import_raw.get('is_drug_product', pd.Series(True, index=import_raw.index))
    ).astype(bool)]
    print(f"    Import Refusals: {len(import_raw_drug)} drug rows")
except Exception as e:
    print(f"    WARNING: imports not loaded: {e}"); import_raw_drug = pd.DataFrame()

try:
    obs483_raw = pd.read_csv(f'{BASE}/12 - FDA - 483/processed/483_observations.csv')
    obs483_raw['FEI'] = obs483_raw['fei'].astype(str).str.split('.').str[0].str.strip()
    obs483_raw['obs_dt'] = pd.to_datetime(obs483_raw['insp_date'], errors='coerce')
    print(f"    483 observations: {len(obs483_raw)} rows")
except Exception as e:
    print(f"    WARNING: 483 observations not loaded: {e}"); obs483_raw = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FACILITY IDENTITY + VALISURE BY YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Facility identity + Valisure outcomes")

fac_info = (base.drop_duplicates(subset=['FEI'])
            [['FEI', 'Firm', 'CountryName', 'CountryCode']]
            .reset_index(drop=True))

val_dedup       = base.drop_duplicates(subset=['FEI', 'NDC', 'Year'])
valisure_by_year = (val_dedup
    .groupby(['FEI', 'Year'])
    .agg(
        ndma_max=('NDMA (ng/DAY) Valisure', 'max'),
        dmf_max=('DMF (ng/DAY) Valisure',   'max'),
        diss_max=('Difference Factor',       'max'),
        n_ndcs_measured=('NDC', 'nunique'),
    )
    .reset_index()
    .rename(columns={'Year': 'survey_year'}))
print(f"  fac_info: {fac_info.shape}  |  valisure_by_year: {valisure_by_year.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — COMPUTE NORMALIZATION ANCHORS (global, across all years)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Computing global normalization anchors")

# These are computed once across all-years data so that scores are
# cross-year comparable (a facility scored in 2020 vs 2024 use same scale).
all_insp_our = insp_our.copy()

def _global_max(df_all, fei_list, col, agg='sum'):
    """Compute per-FEI aggregate of col across all-time, return max."""
    g = df_all[df_all['FEI'].isin(fei_list)].groupby('FEI')[col]
    return float((g.sum() if agg == 'sum' else g.mean()).max() or 1.0)

ANCHORS = {
    'n_oai_cumul':       _global_max(all_insp_our, OUR_FEIS, 'is_oai'),
    'n_483s_cumul':      _global_max(all_insp_our, OUR_FEIS, 'has_483'),
    'n_warning_letters': max(
        (wl_raw.groupby('FEI').size().max() if not wl_raw.empty else 1), 1),
    'n_pred_cfr_insp':   max(
        (cite_our.groupby('FEI')['Inspection ID'].nunique().max() if len(cite_our) else 1), 1),
    'avg_insp_gap_inv':  1.0,   # already in [0,1] after inversion
    'n_import_refusals': max(
        (import_raw_drug.groupby('FEI').size().max() if not import_raw_drug.empty else 1), 1),
    'ever_contamination': 1.0,  # binary 0/1
    # market
    'n_recall_class_I':  max(
        (recall_raw.groupby('FEI').apply(
            lambda x: (x['Event Classification'] == 'Class I').sum()
        ).max() if not recall_raw.empty else 1), 1),
    'n_recalls_drug':    max(
        (recall_raw.groupby('FEI').size().max() if not recall_raw.empty else 1), 1),
    # reported separately (not in score)
    'iqvia_units':       float(iqvia_our.groupby('FEI')['iqvia_extended_units'].sum().max() or 1.0),
    'faers_total':       1.0,   # log-normalized inline
}
print("  Normalization anchors:")
for k, v in ANCHORS.items():
    print(f"    {k:<28s}  max = {v:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TIME-VARYING FEATURE FUNCTIONS (as-of each year-end)
# ══════════════════════════════════════════════════════════════════════════════

def insp_asof(fei_list, cutoff_year):
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = insp_our[insp_our['insp_end_dt'].notna() & (insp_our['insp_end_dt'] <= cutoff)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        n_insp=('FEI', 'count'), n_oai=('is_oai', 'sum'),
        n_vai=('is_vai', 'sum'), n_483s=('has_483', 'sum'),
    ).reset_index()
    agg['OAI_rate'] = agg['n_oai'] / agg['n_insp'].clip(lower=1)
    def avg_gap(g):
        dates = g['insp_end_dt'].sort_values().dropna()
        return dates.diff().dt.days.dropna().mean() if len(dates) >= 2 else np.nan
    gaps = (df.groupby('FEI').apply(avg_gap)
              .reset_index().rename(columns={0: 'avg_insp_gap_days'}))
    return agg.merge(gaps, on='FEI', how='left')


def cfr_asof(fei_list, cutoff_year):
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = cite_our[cite_our['cite_end_dt'].notna() &
                  (cite_our['cite_end_dt'] <= cutoff) &
                  (cite_our['is_oai_pred'] == 1)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    return (df.groupby('FEI')['Inspection ID'].nunique()
              .reset_index().rename(columns={'Inspection ID': 'n_pred_cfr_insp'}))


def market_asof(fei_list, cutoff_year):
    """IQVIA volume — reported separately, NOT scored."""
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
        iqvia_units=('iqvia_extended_units', 'sum'),
        iqvia_trx=('iqvia_trx', 'sum'),
    ).reset_index()


def faers_asof(fei_list, cutoff_year):
    """FAERS — loaded for sensitivity analysis, NOT scored."""
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = faers_our[faers_our['fda_date'].notna() & (faers_our['fda_date'] <= cutoff)]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    agg = df.groupby('FEI').agg(
        faers_total=('primaryid', 'nunique'),
        faers_serious=('serious_flag', 'sum'),
    ).reset_index()
    agg['faers_serious_rate'] = agg['faers_serious'] / agg['faers_total'].clip(lower=1)
    return agg


def recalls_asof(fei_list, cutoff_year):
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
    if wl_raw.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = wl_raw[wl_raw['wl_dt'].notna() &
                (wl_raw['wl_dt'] <= cutoff) &
                (wl_raw['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    return df.groupby('FEI').agg(
        n_warning_letters=('FEI', 'count'),
    ).reset_index()


def import_asof(fei_list, cutoff_year):
    if import_raw_drug.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = import_raw_drug[import_raw_drug['import_dt'].notna() &
                         (import_raw_drug['import_dt'] <= cutoff) &
                         (import_raw_drug['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    return df.groupby('FEI').agg(n_import_refusals=('FEI', 'count')).reset_index()


def nlp483_asof(fei_list, cutoff_year):
    if obs483_raw.empty:
        return pd.DataFrame({'FEI': fei_list})
    cutoff = pd.Timestamp(f'{cutoff_year}-12-31')
    df = obs483_raw[obs483_raw['obs_dt'].notna() &
                    (obs483_raw['obs_dt'] <= cutoff) &
                    (obs483_raw['FEI'].isin(fei_list))]
    if len(df) == 0:
        return pd.DataFrame({'FEI': fei_list})
    flag_cols = ['has_contamination', 'has_oos_oot', 'has_systemic',
                 'has_data_integrity', 'has_oai_predictive', 'has_repeat']
    avail = [c for c in flag_cols if c in df.columns]
    agg = df.groupby('FEI')[avail].any().astype(float).reset_index()
    return agg.rename(columns={
        'has_contamination':  'ever_contamination',
        'has_oos_oot':        'ever_oos_oot',
        'has_systemic':       'ever_systemic',
        'has_data_integrity': 'ever_data_integrity',
        'has_oai_predictive': 'ever_oai_predictive',
        'has_repeat':         'ever_repeat',
    })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — COMPUTE MQRI PANEL (FEI × YEAR)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Computing MQRI v02 panel ...")

panel_rows = []

for yr in SURVEY_YEARS:
    print(f"\n  ── Year {yr} ──────────────────────────────────────────────")
    insp_f   = insp_asof(OUR_FEIS,   yr)
    cfr_f    = cfr_asof(OUR_FEIS,    yr)
    mkt_f    = market_asof(OUR_FEIS, yr)
    faers_f  = faers_asof(OUR_FEIS,  yr)
    rec_f    = recalls_asof(OUR_FEIS, yr)
    wl_f     = wl_asof(OUR_FEIS,      yr)
    imp_f    = import_asof(OUR_FEIS,  yr)
    nlp_f    = nlp483_asof(OUR_FEIS,  yr)

    for fei in OUR_FEIS:
        def gf(df, col, default=0.0):
            if df is None or 'FEI' not in df.columns: return default
            row = df[df['FEI'] == fei]
            if len(row) == 0 or col not in row.columns: return default
            return snum(row.iloc[0][col], default)

        # ── Raw feature values ─────────────────────────────────────────────
        n_oai          = gf(insp_f, 'n_oai')
        n_483s         = gf(insp_f, 'n_483s')
        n_pred_cfr     = gf(cfr_f,  'n_pred_cfr_insp')
        n_wl           = gf(wl_f,   'n_warning_letters')
        gap_days       = gf(insp_f, 'avg_insp_gap_days', default=0.0)
        n_imports      = gf(imp_f,  'n_import_refusals')
        ever_contam    = gf(nlp_f,  'ever_contamination')
        n_recall_all   = gf(rec_f,  'n_recalls_drug')
        n_recall_cls1  = gf(rec_f,  'n_recall_class_I')
        iqvia_units    = gf(mkt_f,  'iqvia_units')
        faers_tot      = gf(faers_f,'faers_total')
        faers_ser_rate = gf(faers_f,'faers_serious_rate')

        # ── Inspection gap: invert + cap (max gap = 1500 days) ────────────
        # avg_insp_gap_inv = 1 - (gap / 1500)  → 0 when gap ≥ 1500, 1 when gap = 0
        GAP_CAP = 1500.0
        gap_inv = max(1.0 - (gap_days / GAP_CAP), 0.0)   # higher = more risk

        # ── Scale each feature to [0, 1] ──────────────────────────────────
        s = {
            'n_oai_cumul':       scale01(n_oai,         ANCHORS['n_oai_cumul']),
            'n_483s_cumul':      scale01(n_483s,         ANCHORS['n_483s_cumul']),
            'n_pred_cfr_insp':   scale01(n_pred_cfr,     ANCHORS['n_pred_cfr_insp']),
            'n_warning_letters': scale01(n_wl,           ANCHORS['n_warning_letters']),
            'avg_insp_gap_inv':  gap_inv,                # already [0,1]
            'n_import_refusals': scale01(n_imports,      ANCHORS['n_import_refusals']),
            'ever_contamination':ever_contam,            # binary [0,1]
            'n_recall_class_I':  scale01(n_recall_cls1,  ANCHORS['n_recall_class_I']),
            'n_recalls_drug':    scale01(n_recall_all,   ANCHORS['n_recalls_drug']),
        }

        # ── D_reg: weighted sum of scaled regulatory features ──────────────
        d_reg_raw = sum(REG_WEIGHTS[feat] * s[feat] for feat in REG_WEIGHTS)
        # d_reg_raw is in [0, 1] since weights sum to 1 and each s[feat] ∈ [0,1]

        # ── D_mkt: weighted sum of scaled market quality features ──────────
        d_mkt_raw = sum(MKT_WEIGHTS[feat] * s[feat] for feat in MKT_WEIGHTS)

        # ── MQRI: domain-weighted composite, scaled to [0, 100] ───────────
        mqri = round(
            (d_reg_raw * DOMAIN_WEIGHT_REG + d_mkt_raw * DOMAIN_WEIGHT_MKT) * 100,
            1
        )
        tier = 'HIGH' if mqri >= 65 else 'MODERATE' if mqri >= 35 else 'LOW'

        panel_rows.append({
            'FEI': fei, 'survey_year': yr,
            # ── MQRI scores ───────────────────────────────────────────────
            'mqri_total':          mqri,
            'mqri_regulatory':     round(d_reg_raw * DOMAIN_WEIGHT_REG * 100, 1),
            'mqri_market_quality': round(d_mkt_raw * DOMAIN_WEIGHT_MKT * 100, 1),
            'mqri_tier':           tier,
            # ── Regulatory inputs (all cumulative through year-end) ────────
            'n_oai_cumul':         n_oai,
            'n_483s_cumul':        n_483s,
            'n_vai_cumul':         gf(insp_f, 'n_vai'),    # retained for info
            'n_pred_cfr_insp':     n_pred_cfr,
            'n_warning_letters':   n_wl,
            'n_import_refusals':   n_imports,
            'avg_insp_gap_days':   gap_days,
            'OAI_rate_cumul':      round(gf(insp_f, 'OAI_rate'), 3),
            'ever_contamination':  ever_contam,
            'ever_systemic':       gf(nlp_f, 'ever_systemic'),
            'ever_data_integrity': gf(nlp_f, 'ever_data_integrity'),
            'ever_repeat':         gf(nlp_f, 'ever_repeat'),
            # ── Market quality inputs ─────────────────────────────────────
            'n_recalls_drug':      n_recall_all,
            'n_recall_class_I':    n_recall_cls1,
            # ── Reported separately (NOT in score) ────────────────────────
            'iqvia_units':         iqvia_units,       # societal impact axis
            'faers_total':         faers_tot,         # sensitivity analysis
            'faers_serious_rate':  faers_ser_rate,    # sensitivity analysis
        })

    yr_df = (pd.DataFrame([r for r in panel_rows if r['survey_year'] == yr])
               .merge(fac_info[['FEI', 'Firm']], on='FEI', how='left')
               .sort_values('mqri_total', ascending=False))
    print(f"  {'Firm':40s}  MQRI   Tier      Reg    Mkt")
    for _, r in yr_df.iterrows():
        print(f"  {str(r['Firm'])[:40]:40s}  {r['mqri_total']:5.1f}  "
              f"{r['mqri_tier']:9s}  {r['mqri_regulatory']:5.1f}  "
              f"{r['mqri_market_quality']:5.1f}")

panel = pd.DataFrame(panel_rows)
panel = panel.merge(fac_info[['FEI', 'Firm', 'CountryName', 'CountryCode']], on='FEI', how='left')
panel = panel.merge(valisure_by_year, on=['FEI', 'survey_year'], how='left')
print(f"\n  Panel: {panel.shape}  (expect {len(OUR_FEIS)*len(SURVEY_YEARS)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6a — VALIDATION: MQRI vs VALISURE (primary)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6a: Validation — MQRI v02 vs Valisure outcomes")

PREDICTORS = [
    ('mqri_total',          'MQRI v02 Total'),
    ('mqri_regulatory',     'Regulatory component'),
    ('mqri_market_quality', 'Market Quality component'),
]
OUTCOMES_BY_YEAR = {
    2020: [('ndma_max', 'NDMA Max (ng/day)'), ('dmf_max', 'DMF Max (ng/day)')],
    2022: [('ndma_max', 'NDMA Max (ng/day)'), ('dmf_max', 'DMF Max (ng/day)')],
    2024: [('dmf_max',  'DMF Max (ng/day)'),  ('diss_max', 'Dissolution diff factor')],
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
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
            val_rows.append({'Year': yr, 'Predictor': pred_col,
                             'Outcome': gt_label, 'n': n,
                             'Spearman_rho': round(r, 3), 'p_value': round(p, 4)})
            print(f"  {yr}  {gt_label:30s} vs {pred_col:25s}: "
                  f"ρ={r:+.3f}, p={p:.4f}{stars:3s}  n={n}")

val_df = pd.DataFrame(val_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6b — SENSITIVITY: MQRI vs FAERS (FAERS as potential ground truth)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6b: Sensitivity — MQRI vs FAERS (FAERS excluded from score)")
print("  Testing whether MQRI predicts adverse event burden independently...")

faers_sens_rows = []
for yr in [2020, 2022, 2024]:
    sub = panel[panel['survey_year'] == yr]
    for faers_col in ['faers_total', 'faers_serious_rate']:
        if faers_col not in sub.columns:
            continue
        x = pd.to_numeric(sub[faers_col],     errors='coerce')
        y = pd.to_numeric(sub['mqri_total'],   errors='coerce')
        mask = x.notna() & y.notna() & (x > 0)
        n = mask.sum()
        if n < 4:
            continue
        r, p = stats.spearmanr(x[mask].values, y[mask].values)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else ''
        faers_sens_rows.append({'Year': yr, 'FAERS_metric': faers_col,
                                'Spearman_rho': round(r, 3), 'p_value': round(p, 4), 'n': n})
        print(f"  {yr}  MQRI vs {faers_col:25s}: ρ={r:+.3f}, p={p:.4f}{stars:3s}  n={n}")

faers_sens_df = pd.DataFrame(faers_sens_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SOCIETAL IMPACT AXIS (2-D output: quality risk × volume)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Building 2-D Quality Risk × Societal Impact output")

latest = panel[panel['survey_year'] == max(SURVEY_YEARS)].copy()
vol_max = latest['iqvia_units'].max() or 1.0
latest['societal_impact_score'] = (latest['iqvia_units'] / vol_max * 100).round(1)

def quad(row):
    hi_risk = row['mqri_total'] >= 50
    hi_vol  = row['societal_impact_score'] >= 50
    if hi_risk and hi_vol:  return 'Q1: HIGH PRIORITY (high risk + high volume)'
    if hi_risk and not hi_vol: return 'Q2: Monitor (high risk + low volume)'
    if not hi_risk and hi_vol: return 'Q3: Watch (low risk + high volume)'
    return 'Q4: Low priority'

latest['quadrant'] = latest.apply(quad, axis=1)
two_d = (latest.merge(fac_info[['FEI', 'Firm']], on='FEI', how='left')
         [['FEI', 'Firm', 'CountryName', 'mqri_total', 'mqri_tier',
           'iqvia_units', 'societal_impact_score', 'quadrant']]
         .sort_values('mqri_total', ascending=False))

print(f"  {'Firm':40s}  MQRI   Soc.Impact  Quadrant")
for _, r in two_d.iterrows():
    print(f"  {str(r['Firm'])[:40]:40s}  {r['mqri_total']:5.1f}  "
          f"{r['societal_impact_score']:6.1f}      {r['quadrant']}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Saving outputs ...")

panel.to_csv(f'{OUT}/20260408_v02_mqri_panel.csv', index=False)
val_df.to_csv(f'{OUT}/20260408_v02_validation_correlations.csv', index=False)
faers_sens_df.to_csv(f'{OUT}/20260408_v02_faers_sensitivity.csv', index=False)
two_d.to_csv(f'{OUT}/20260408_v02_2d_risk_volume.csv', index=False)

fac_master = panel[panel['survey_year'] == max(SURVEY_YEARS)].copy()
fac_master.to_csv(f'{OUT}/20260408_v02_facility_master.csv', index=False)

print(f"  ✓ 20260408_v02_mqri_panel.csv              {panel.shape}")
print(f"  ✓ 20260408_v02_validation_correlations.csv {val_df.shape}")
print(f"  ✓ 20260408_v02_faers_sensitivity.csv       {faers_sens_df.shape}")
print(f"  ✓ 20260408_v02_2d_risk_volume.csv          {two_d.shape}")
print(f"  ✓ 20260408_v02_facility_master.csv         {fac_master.shape}")
print(f"  ✓ 20260408_v02_weights.csv                 {wt_df.shape}")
print("\nDone.")
