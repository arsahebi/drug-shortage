"""
20260316_quality_signal_correlation.py

Tests whether FDA regulatory signals (inspection features, warning letter features,
483 text features, recall features, import refusal features) can predict drug quality
outcomes measured by Valisure/DoD testing.

Ground truth quality metrics (per FEI, Metformin):
  - NDMA (ng/DAY) Valisure  — 2022 data; impurity contamination
  - DMF (ng/DAY) Valisure   — 2020/2022/2024 data; impurity contamination
  - Difference Factor        — 2024 data; overall quality deviation score

Regulatory signal features (6 data sources):
  1. Inspection features:  OAI_rate_dqa, n_dqa_citations, n_oai_predictive_cfrs, etc.
  2. Warning letter feats: n_warning_letters, n_violations, wl_severity_score, etc.
  3. 483 text features:    n_observations_total, avg_obs_body_chars, ever_repeat, etc.
  4. Recall features:      n_recalls_drug, has_class_I_recall, recall_severity_max, etc.
  5. Import refusal feats: n_import_refusals_drug, import_refusal_severity, etc.

Outputs:
  - quality_fei_aggregated.csv       — quality scores aggregated to FEI level
  - quality_inspection_merged.csv    — merged quality + all signal features
  - quality_correlation_table.csv    — feature × quality metric Pearson r + p-value
  - quality_group_comparison.csv     — contaminated vs clean facility comparison
  - quality_wl_overlap.csv           — quality scores for WL facilities specifically
  - quality_data_coverage.csv        — per-FEI data source coverage summary
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parents[3]
QUAL     = BASE / "Data/06 - Metformin Data/Derived/Q&As134_v8_v02.xlsx"
FEAT     = BASE / "Data/14 - FDA - Inspection/processed/facility_feature_matrix.csv"
WL_FEI   = BASE / "Data/21 - FDA - Warning Letter/processed/warning_letter_fei_features.csv"
FEI483   = BASE / "Data/12 - FDA - 483/processed/483_fei_features.csv"
RECALL   = BASE / "Data/22 - FDA - Recall/processed/recall_fei_features.csv"
IMPREF   = BASE / "Data/23 - FDA - Import Refusal/processed/import_refusal_fei_features.csv"
OUT      = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & AGGREGATE QUALITY DATA TO FEI LEVEL
# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("QUALITY DATA — FEI LEVEL AGGREGATION")
print("="*70)

q = pd.read_excel(QUAL)
print(f"Quality rows: {len(q)}  |  FEIs: {q['FEI'].nunique()}  |  Years: {sorted(q['Year'].unique())}")
print(f"Quality columns: {list(q.columns)}\n")

# ── Per-year quality aggregation ────────────────────────────────────────────
# NDMA: 2022 only (2020/2024 have NaN)
ndma_2022 = (q[q['Year'] == 2022]
             .groupby('FEI')['NDMA (ng/DAY) Valisure']
             .agg(['max', 'mean'])
             .rename(columns={'max': 'ndma_2022_max', 'mean': 'ndma_2022_mean'}))

# DMF: available all years — report by year and overall max
dmf_by_year = {}
for yr in [2020, 2022, 2024]:
    sub = q[q['Year'] == yr].groupby('FEI')['DMF (ng/DAY) Valisure']
    dmf_by_year[yr] = sub.agg(['max', 'mean']).rename(
        columns={'max': f'dmf_{yr}_max', 'mean': f'dmf_{yr}_mean'})

dmf_all_max = (q.groupby('FEI')['DMF (ng/DAY) Valisure']
               .max()
               .rename('dmf_allyr_max'))
dmf_all_mean = (q.groupby('FEI')['DMF (ng/DAY) Valisure']
                .mean()
                .rename('dmf_allyr_mean'))

# Difference Factor: 2024 only
df_2024 = (q[q['Year'] == 2024]
           .groupby('FEI')['Difference Factor']
           .agg(['max', 'mean'])
           .rename(columns={'max': 'diff_factor_2024_max', 'mean': 'diff_factor_2024_mean'}))

# Firm name (most common)
firm    = q.groupby('FEI')['Firm'].first().rename('Firm')
country = q.groupby('FEI')['CountryName'].first().rename('Country')

# Merge all quality pieces
qual_fei = (firm
            .to_frame()
            .join(country)
            .join(ndma_2022, how='left')
            .join(dmf_by_year[2020], how='left')
            .join(dmf_by_year[2022], how='left')
            .join(dmf_by_year[2024], how='left')
            .join(dmf_all_max, how='left')
            .join(dmf_all_mean, how='left')
            .join(df_2024, how='left')
            .reset_index())

# Binary quality flags
FDA_NDMA_LIMIT = 96.0    # ng/day (FDA interim limit for metformin)
HIGH_DMF_THRESHOLD = 8_000.0  # conservative threshold (ng/day)

qual_fei['ndma_detected']   = (qual_fei['ndma_2022_max'].fillna(0) > 0).astype(int)
qual_fei['ndma_above_limit']= (qual_fei['ndma_2022_max'].fillna(0) > FDA_NDMA_LIMIT).astype(int)
qual_fei['dmf_high_2022']   = (qual_fei['dmf_2022_max'].fillna(0) > HIGH_DMF_THRESHOLD).astype(int)
qual_fei['dmf_high_2024']   = (qual_fei['dmf_2024_max'].fillna(0) > HIGH_DMF_THRESHOLD).astype(int)
qual_fei['any_quality_flag']= ((qual_fei['ndma_detected'] == 1) |
                                (qual_fei['dmf_high_2022'] == 1) |
                                (qual_fei['dmf_high_2024'] == 1)).astype(int)

print("FEI-level quality summary:")
print(qual_fei[['FEI','Firm','Country','ndma_2022_max','dmf_2022_max',
                 'dmf_2024_max','diff_factor_2024_max',
                 'ndma_detected','any_quality_flag']].to_string())
print(f"\nFEIs with any NDMA detected: {qual_fei['ndma_detected'].sum()}")
print(f"FEIs with NDMA > FDA limit: {qual_fei['ndma_above_limit'].sum()}")
print(f"FEIs with high DMF (any yr): {((qual_fei[['dmf_high_2022','dmf_high_2024']].max(axis=1))==1).sum()}")
print(f"FEIs with any quality flag: {qual_fei['any_quality_flag'].sum()}")

qual_fei.to_csv(OUT / "quality_fei_aggregated.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MERGE WITH ALL SIGNAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("MERGING WITH ALL SIGNAL FEATURES")
print("="*70)

feat = pd.read_csv(FEAT)
feat['FEI_NUMBER'] = feat['FEI_NUMBER'].astype(int)
qual_fei['FEI']    = qual_fei['FEI'].astype(int)

merged = qual_fei.merge(feat, left_on='FEI', right_on='FEI_NUMBER', how='left')
print(f"Quality FEIs: {len(qual_fei)}  |  Matched to feature matrix: {merged['FEI_NUMBER'].notna().sum()}")

# ── Warning letter features ─────────────────────────────────────────────────
if WL_FEI.exists():
    wl = pd.read_csv(WL_FEI)
    wl['primary_fei'] = wl['primary_fei'].astype(int)
    merged = merged.merge(wl[['primary_fei','n_warning_letters','n_violations',
                               'n_cfr_unique','n_repeat_sections',
                               'ever_repeat_at_facility','ever_repeat_multi_site',
                               'ever_management_oversight','ever_corporate_failure_lang',
                               'wl_severity_score']],
                          left_on='FEI', right_on='primary_fei', how='left')
    merged['has_warning_letter'] = merged['n_warning_letters'].notna().astype(int)
    merged['n_warning_letters']  = merged['n_warning_letters'].fillna(0).astype(int)
    merged['wl_severity_score']  = merged['wl_severity_score'].fillna(0)
    for col in ['ever_repeat_at_facility','ever_repeat_multi_site',
                'ever_management_oversight','ever_corporate_failure_lang']:
        merged[col] = merged[col].fillna(False).astype(int)
    print(f"Matched to WL features: {merged['has_warning_letter'].sum()}")
else:
    print("  WL feature file not found — skipping")
    merged['has_warning_letter'] = 0

# ── 483 text features ───────────────────────────────────────────────────────
if FEI483.exists():
    f483 = pd.read_csv(FEI483)
    f483['fei'] = f483['fei'].astype(int)
    COLS_483 = ['fei','n_483s_total','n_483s_extractable','n_observations_total',
                'avg_obs_per_483','avg_obs_body_chars','avg_n_examples',
                'n_obs_with_oai_predictive_cfr','pct_obs_oai_predictive',
                'n_unique_cfrs_in_483']
    # Add boolean severity signals (ever_* columns)
    ever_cols_483 = [c for c in f483.columns if c.startswith('ever_')]
    n_cols_483    = [c for c in f483.columns if c.startswith('n_obs_') and c != 'n_obs_with_oai_predictive_cfr']
    all_483_cols  = COLS_483 + ever_cols_483 + n_cols_483
    all_483_cols  = [c for c in all_483_cols if c in f483.columns]
    merged = merged.merge(f483[all_483_cols], left_on='FEI', right_on='fei', how='left')
    # Fill 0 for FEIs with no 483 in our corpus
    num_483_cols = [c for c in all_483_cols if c != 'fei']
    for col in num_483_cols:
        if merged[col].dtype in [float, 'float64']:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged[col].fillna(0)
    merged['has_483_in_corpus'] = (merged['n_483s_total'] > 0).astype(int)
    print(f"Matched to 483 features: {merged['has_483_in_corpus'].sum()}")
else:
    print("  483 feature file not found — skipping")
    merged['has_483_in_corpus'] = 0

# ── Recall features ─────────────────────────────────────────────────────────
if RECALL.exists():
    rec = pd.read_csv(RECALL)
    rec['fei'] = rec['fei'].astype(int)
    COLS_REC = ['fei','n_recalls_drug','has_drug_recall','n_recall_class_I',
                'n_recall_class_II','has_class_I_recall','recall_severity_max',
                'n_recalls_ongoing','n_recalls_terminated',
                'has_recall_contamination','has_recall_cgmp_failure',
                'has_recall_wrong_potency','has_recall_mislabeled',
                'has_recall_stability','has_recall_packaging_defect',
                'has_recall_data_integrity']
    COLS_REC = [c for c in COLS_REC if c in rec.columns]
    merged = merged.merge(rec[COLS_REC], left_on='FEI', right_on='fei', how='left')
    for col in [c for c in COLS_REC if c != 'fei']:
        merged[col] = merged[col].fillna(0)
    print(f"Matched to Recall features: {int(merged['has_drug_recall'].sum())}")
else:
    print("  Recall feature file not found — skipping")
    merged['has_drug_recall'] = 0

# ── Import Refusal features ──────────────────────────────────────────────────
if IMPREF.exists():
    imp = pd.read_csv(IMPREF)
    imp['fei'] = imp['fei'].astype(int)
    COLS_IMP = ['fei','n_import_refusals_total','n_import_refusals_drug',
                'has_import_refusal','has_drug_import_refusal','n_refusal_years',
                'has_drug_charge','n_lab_analysis_refusals','has_lab_analysis_flag',
                'import_refusal_severity',
                'has_charge_cgmp_violation','has_charge_adulterated_drug',
                'has_charge_misbranded_drug','has_charge_unregistered_facility',
                'has_charge_drug_adulteration_general','has_charge_labeling_deficiency',
                'has_charge_general_adulteration']
    COLS_IMP = [c for c in COLS_IMP if c in imp.columns]
    merged = merged.merge(imp[COLS_IMP], left_on='FEI', right_on='fei', how='left')
    for col in [c for c in COLS_IMP if c != 'fei']:
        merged[col] = merged[col].fillna(0)
    print(f"Matched to Import Refusal features: {int(merged['has_drug_import_refusal'].sum())}")
else:
    print("  Import Refusal feature file not found — skipping")
    merged['has_drug_import_refusal'] = 0

merged.to_csv(OUT / "quality_inspection_merged.csv", index=False)
print(f"\nSaved merged dataset: {len(merged)} rows × {len(merged.columns)} columns")


# ═══════════════════════════════════════════════════════════════════════════
# 3. DATA SOURCE COVERAGE TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DATA SOURCE COVERAGE (for quality-linked FEIs)")
print("="*70)

coverage_sources = {
    'Inspection DB':      merged['FEI_NUMBER'].notna(),
    'Warning Letter':     merged['has_warning_letter'] == 1,
    '483 in Corpus':      merged['has_483_in_corpus'] == 1,
    'Drug Recall':        merged['has_drug_recall'] == 1,
    'Import Refusal':     merged['has_drug_import_refusal'] == 1,
}
cov_rows = []
for src, mask in coverage_sources.items():
    cov_rows.append({'Data Source': src, 'FEIs with Data': int(mask.sum()),
                     'Total Quality FEIs': len(merged), 'Coverage %': f"{mask.mean()*100:.0f}%"})
cov_df = pd.DataFrame(cov_rows)
print(cov_df.to_string(index=False))
cov_df.to_csv(OUT / "quality_data_coverage.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FEATURE × QUALITY METRIC CORRELATIONS")
print("="*70)

INSPECTION_FEATS = [
    'OAI_rate_dqa', 'n_dqa_inspections', 'n_dqa_OAI',
    'n_dqa_citations', 'n_oai_predictive_cfrs', 'repeat_cfr_rate',
    'n_cit_buildings_equip', 'n_cit_lab_qc', 'n_cit_production',
    'n_unique_cfrs', 'avg_dqa_gap_days',
    'has_bioresearch_insp', 'n_bio_OAI',
]
WL_FEATS = [
    'has_warning_letter', 'n_warning_letters', 'n_violations',
    'n_cfr_unique', 'n_repeat_sections',
    'ever_repeat_at_facility', 'ever_repeat_multi_site',
    'ever_management_oversight', 'ever_corporate_failure_lang',
    'wl_severity_score',
]
FEATS_483 = [
    'has_483_in_corpus', 'n_483s_total', 'n_483s_extractable',
    'n_observations_total', 'avg_obs_per_483', 'avg_obs_body_chars',
    'n_obs_with_oai_predictive_cfr', 'pct_obs_oai_predictive',
    'n_unique_cfrs_in_483',
    # severity signals
    'n_obs_repeat_total', 'n_obs_systemic_total', 'n_obs_data_integrity_total',
    'n_obs_contamination_total', 'n_obs_oos_oot_total', 'n_obs_patient_risk_total',
    'n_obs_wl_ref_total',
    'ever_repeat', 'ever_systemic', 'ever_wl_ref',
    'ever_data_integrity', 'ever_contamination', 'ever_oos_oot', 'ever_patient_risk',
]
RECALL_FEATS = [
    'has_drug_recall', 'n_recalls_drug', 'n_recall_class_I',
    'has_class_I_recall', 'recall_severity_max',
    'has_recall_contamination', 'has_recall_cgmp_failure',
    'has_recall_wrong_potency', 'has_recall_stability',
]
IMPREF_FEATS = [
    'has_drug_import_refusal', 'n_import_refusals_drug',
    'n_refusal_years', 'has_drug_charge',
    'n_lab_analysis_refusals', 'has_lab_analysis_flag',
    'import_refusal_severity',
    'has_charge_cgmp_violation', 'has_charge_adulterated_drug',
    'has_charge_drug_adulteration_general',
]

FEAT_TYPE_MAP = (
    [(f, 'Inspection')      for f in INSPECTION_FEATS] +
    [(f, 'Warning Letter')  for f in WL_FEATS] +
    [(f, '483 Text')        for f in FEATS_483] +
    [(f, 'Recall')          for f in RECALL_FEATS] +
    [(f, 'Import Refusal')  for f in IMPREF_FEATS]
)
ALL_FEATS_TYPED = [(f, t) for f, t in FEAT_TYPE_MAP if f in merged.columns]

QUALITY_METRICS = {
    'ndma_2022_max':         "NDMA 2022 (ng/day, max)",
    'dmf_2022_max':          "DMF 2022 (ng/day, max)",
    'dmf_2024_max':          "DMF 2024 (ng/day, max)",
    'dmf_allyr_max':         "DMF All Years (ng/day, max)",
    'diff_factor_2024_max':  "Difference Factor 2024 (max)",
    'ndma_detected':         "NDMA Detected (binary, 2022)",
    'any_quality_flag':      "Any Quality Flag (binary)",
}

corr_rows = []
for feat_col, feat_type in ALL_FEATS_TYPED:
    row = {'Feature': feat_col, 'Feature_Type': feat_type}
    for qcol, qlabel in QUALITY_METRICS.items():
        sub = merged[[feat_col, qcol]].dropna()
        if len(sub) < 5 or sub[feat_col].std() == 0:
            row[f'r_{qcol}'] = None
            row[f'p_{qcol}'] = None
            row[f'n_{qcol}'] = len(sub)
        else:
            r, p = stats.pearsonr(sub[feat_col].astype(float), sub[qcol].astype(float))
            row[f'r_{qcol}'] = round(r, 4)
            row[f'p_{qcol}'] = round(p, 4)
            row[f'n_{qcol}'] = len(sub)
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT / "quality_correlation_table.csv", index=False)

# Print summary tables
for qcol in ['ndma_2022_max', 'dmf_2022_max', 'any_quality_flag']:
    r_col = f'r_{qcol}'
    if r_col not in corr_df.columns:
        continue
    sub = corr_df[corr_df[r_col].notna()].copy()
    sub['abs_r'] = sub[r_col].abs()
    sub = sub.sort_values('abs_r', ascending=False)
    print(f"\n--- Top correlations with {QUALITY_METRICS.get(qcol, qcol)} ---")
    print(sub[['Feature', 'Feature_Type', r_col, f'p_{qcol}', f'n_{qcol}']].head(12).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# 5. GROUP COMPARISON: CONTAMINATED vs CLEAN FACILITIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("GROUP COMPARISON: CONTAMINATED vs CLEAN")
print("="*70)

def group_compare(df, group_col, feat_col_pairs):
    g1 = df[df[group_col] == 1]
    g0 = df[df[group_col] == 0]
    rows = []
    for f, ftype in feat_col_pairs:
        if f not in df.columns: continue
        v1 = g1[f].dropna().astype(float)
        v0 = g0[f].dropna().astype(float)
        if len(v1) < 2 or len(v0) < 2: continue
        t, p = stats.ttest_ind(v1, v0, equal_var=False)
        rows.append({
            'Feature':          f,
            'Feature_Type':     ftype,
            'Contaminated_n':   len(v1),
            'Contaminated_mean':round(v1.mean(), 3),
            'Clean_n':          len(v0),
            'Clean_mean':       round(v0.mean(), 3),
            'Diff':             round(v1.mean() - v0.mean(), 3),
            'p_value':          round(p, 4),
            'Significant':      "**" if p < 0.05 else ("*" if p < 0.1 else ""),
        })
    return pd.DataFrame(rows).sort_values('p_value')

comp_df = group_compare(merged, 'any_quality_flag', ALL_FEATS_TYPED)
comp_df.to_csv(OUT / "quality_group_comparison.csv", index=False)
print(comp_df.head(15).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# 6. WARNING LETTER FACILITIES: QUALITY SPOTLIGHT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FACILITIES WITH WARNING LETTERS — QUALITY SCORES")
print("="*70)

wl_quality = merged[merged['has_warning_letter'] == 1][[
    'FEI', 'Firm', 'Country',
    'ndma_2022_max', 'dmf_2022_max', 'dmf_2024_max', 'diff_factor_2024_max',
    'any_quality_flag',
    'n_warning_letters', 'wl_severity_score',
    'OAI_rate_dqa', 'n_dqa_citations'
]].sort_values('wl_severity_score', ascending=False)

if len(wl_quality) > 0:
    print(f"\nFEIs with both Warning Letters AND quality data: {len(wl_quality)}")
    print(wl_quality.to_string(index=False))
else:
    print("No FEIs have both warning letters and quality data in this dataset.")

wl_quality.to_csv(OUT / "quality_wl_overlap.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MULTI-SOURCE RISK COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("MULTI-SOURCE COMPOSITE RISK SCORE")
print("="*70)

# Simple additive risk score combining signals from all 6 sources
# Each source normalized to 0-1 contribution
def safe_col(df, col, fill=0):
    return df[col].fillna(fill) if col in df.columns else pd.Series(fill, index=df.index)

merged['composite_risk_score'] = (
    # Inspection (weight 0.3)
    0.3 * safe_col(merged, 'OAI_rate_dqa').clip(0, 1)
    # Warning Letter (weight 0.2)
    + 0.2 * (safe_col(merged, 'wl_severity_score') / safe_col(merged, 'wl_severity_score').replace(0, 1).max()).clip(0, 1)
    # 483 text signals (weight 0.2)
    + 0.2 * (
        0.5 * safe_col(merged, 'ever_repeat').clip(0, 1)
        + 0.3 * safe_col(merged, 'ever_data_integrity').clip(0, 1)
        + 0.2 * safe_col(merged, 'ever_contamination').clip(0, 1)
    )
    # Recall (weight 0.15)
    + 0.15 * (
        0.6 * safe_col(merged, 'has_class_I_recall').clip(0, 1)
        + 0.4 * safe_col(merged, 'has_recall_contamination').clip(0, 1)
    )
    # Import Refusal (weight 0.15)
    + 0.15 * (safe_col(merged, 'import_refusal_severity') / 30).clip(0, 1)
).round(4)

print("\nTop facilities by composite risk score:")
top_risk = (merged[['FEI','Firm','composite_risk_score','any_quality_flag',
                     'OAI_rate_dqa','has_warning_letter','has_483_in_corpus',
                     'has_drug_recall','has_drug_import_refusal']]
            .sort_values('composite_risk_score', ascending=False))
print(top_risk.head(18).to_string(index=False))

# Correlation of composite score with quality
for qcol in ['any_quality_flag', 'dmf_allyr_max', 'ndma_2022_max']:
    sub = merged[['composite_risk_score', qcol]].dropna()
    sub_cs = sub['composite_risk_score'].astype(float)
    sub_q  = sub[qcol].astype(float)
    if len(sub) >= 5 and sub_cs.std() > 0:
        r, p = stats.pearsonr(sub_cs, sub_q)
        print(f"\nComposite risk score vs {qcol}: r={r:.4f}  p={p:.4f}  n={len(sub)}")

merged.to_csv(OUT / "quality_inspection_merged.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 8. KEY SUMMARY TABLE (for presentation)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

print(f"\nMetformin quality dataset: {len(qual_fei)} FEIs across 3 test years (2020/2022/2024)")
print(f"FEIs matched to inspection features: {merged['FEI_NUMBER'].notna().sum()}")
print(f"FEIs with warning letters: {merged['has_warning_letter'].sum()}")
print(f"FEIs with 483 in corpus:   {merged['has_483_in_corpus'].sum()}")
print(f"FEIs with drug recall:     {int(merged['has_drug_recall'].sum())}")
print(f"FEIs with import refusal:  {int(merged['has_drug_import_refusal'].sum())}")
print(f"FEIs with both WL + quality data: {len(wl_quality)}")

print(f"\nQuality outcomes (across {len(qual_fei)} Metformin FEIs):")
print(f"  NDMA contamination detected (2022):  {qual_fei['ndma_detected'].sum()} / {qual_fei['ndma_2022_max'].notna().sum()}")
print(f"  NDMA > FDA limit (96 ng/day):        {qual_fei['ndma_above_limit'].sum()} / {qual_fei['ndma_2022_max'].notna().sum()}")
print(f"  High DMF (any year, >{HIGH_DMF_THRESHOLD:,.0f} ng/day): {((qual_fei[['dmf_high_2022','dmf_high_2024']].max(axis=1))==1).sum()} / {len(qual_fei)}")
print(f"  Any quality flag:                    {qual_fei['any_quality_flag'].sum()} / {len(qual_fei)}")

# Top predictive features
best = corr_df.copy()
best['r_primary'] = best['r_any_quality_flag'].abs()
best = best.dropna(subset=['r_primary']).sort_values('r_primary', ascending=False)
print(f"\nTop 8 features by |r| with 'Any Quality Flag':")
for _, row in best.head(8).iterrows():
    r_val = row.get('r_any_quality_flag', 'N/A')
    p_val = row.get('p_any_quality_flag', 'N/A')
    print(f"  {row['Feature']:<40} r={r_val:>7}  p={p_val:>6}  [{row['Feature_Type']}]")

print(f"\nAll outputs saved to: {OUT}")
