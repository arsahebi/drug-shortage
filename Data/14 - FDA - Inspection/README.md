# FDA Inspection/Citation Pipeline

This folder contains FDA Dashboard inspection and citation inputs for the Drug
Shortage project. The active inspection/citation pipeline lives in
`processed/` and has two scripts:

1. `processed/20260315_cfr_cooccurrence_analysis.py`
   - Filters FDA inspection and citation rows to the project FEI universe from
     `08 - Valisure/raw/FEIs_March 2026.xlsx`.
   - Maps citations to project areas from CFR prefixes:
     - `21 CFR 211.*` -> Drug Quality Assurance
     - `21 CFR 312.*`, `21 CFR 314.*`, `21 CFR 320.*` -> Bioresearch Monitoring
   - Assigns each citation the inspection classification for its own mapped
     project area, with fallback to the worst available class for that
     inspection (`OAI > VAI > NAI`) only when the exact area row is missing.
   - Writes the cleaned mapped citation table and Drug QA CFR summaries.

2. `processed/20260315_facility_feature_matrix.py`
   - Consumes step 1 outputs.
   - Builds one row per FEI with inspection-history and citation features.
   - Derives OAI-predictive CFRs from `cfr_by_classification.csv`; it does not
     maintain a manual CFR list. The threshold is Drug QA only, `Total >= 5`,
     and `OAI_share >= 0.33`.

Active downstream outputs:

- `processed/citations_with_classification.csv`
  Used by warning-letter feature construction.
- `processed/facility_feature_matrix.csv`
  Used by metformin quality-signal analysis and MQRI materials.
- `processed/cfr_by_classification.csv`
  Source table for OAI-share CFR analysis and the OAI-predictive CFR feature.

Reference outputs from the active pipeline:

- `processed/inspection_details_filtered.csv`
- `processed/citation_details_filtered.csv`
- `processed/citations_bioresearch.csv`
- `processed/cfr_cooccurrence_pairs_with_labels.csv`
- `processed/api_classification_summary.csv`

Non-core or superseded artifacts are kept under `old_not_current_pipeline/`.
