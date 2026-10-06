# Fixed extracted text signals versus DoD scoring components

This revised analysis uses the same **17 existing text variables** as the FAERS/MarketScan models, read directly from the Claude v2 step02 feature file. It introduces no keyword variables, no new extraction, and no route/product filtering of observations. It supersedes the keyword-screen first pass, preserved in `../20261006_first_pass/`.

## Outcomes and scoring

The scoring workbook contains 302 rows across 13 API worksheets, including separate ampicillin–sulbactam labels. Every scoring row has an NDC represented in the testing workbook. Each original score reconciles exactly to:

`DoD score = 100 + sum(signed source components) = 100 - sum(positive penalty amounts)`.

Source entries such as -10, -30 and -61 are preserved. `--` is zero **recorded scoring penalty**, not a measured zero concentration. A component column absent from an API worksheet is missing, not zero. Scores below zero are retained. The score reconstruction includes dosage, benzene/EtOx and sterility where present, even though the five requested main outcomes are the total score, dissolution, DMF, nitrosamines and toxic elements.

The four component outcomes are positive penalty magnitudes: higher is worse. The overall outcome is the original score: higher is better. Coefficient signs therefore have opposite interpretations for total score versus penalties. The source labels the nitrosamine category as “Nitrosamines,” not an NDMA-specific concentration.

| outcome | source_rows_present | source_rows_with_penalty | distinct_values | minimum | maximum |
| --- | --- | --- | --- | --- | --- |
| DoD_score | 302 | 122 | 11 | -83.000 | 100.000 |
| D_dissolution | 136 | 64 | 4 | -0.000 | 61.000 |
| DMF | 302 | 12 | 4 | -0.000 | 61.000 |
| NDMA_nitrosamines | 302 | 0 | 1 | -0.000 | -0.000 |
| T_toxic | 302 | 47 | 10 | -0.000 | 183.000 |

Nitrosamine penalties are zero in all 302 scoring rows. Their relationship to text cannot be estimated from these files. Nonzero DMF penalties occur only in the metformin worksheet, and the main analysis includes just four connected clusters with a DMF penalty. Toxic-element penalties sum the source's toxic-element columns where several metals are scored separately; original columns remain in `component_raw` and `score_components.csv`. Raw laboratory dissolution values are retained in the sample file but are not dependent variables in this revision.

## Cohort

| cohort | source_samples | prior_text_samples | text_facilities | score_cells | shared_score_units |
| --- | --- | --- | --- | --- | --- |
| all_source_unique | 619 | 301 | 67 | 260 | 115 |
| march_only_unique | 619 | 293 | 77 | 255 | 115 |
| dated_metformin | 68 | 31 | 13 | 31 | 15 |

Facility mapping and time safeguards are carried forward from the first pass. Main linkage requires a singleton in the complete cross-source map before selecting text-covered plants. Latest features must precede known metformin intake dates; otherwise the cutoff is January 1, 2023, based on the documented collection period. This is a provisional facility-level quality association, not proven lot-specific attribution or patient harm.

## Existing fixed variables

- `severity_critmajor_share`
- `contamination_llm_share`
- `data_integrity_llm_share`
- `patient_risk_llm_share`
- `investigation_llm_share`
- `repeat_cross_insp_share`
- `scope_facilitywide_share`
- `cultural_root_cause_share`
- `vc_laboratorycontrolssystem_share`
- `vc_qualitysystem_share`
- `n_laboratorycontrolssystem_obs`
- `n_qualitysystem_obs`
- `joint_labcontrols_qualitysystem`
- `joint_labcontrols_dataintegrity`
- `joint_contamination_labcontrols`
- `joint_qualitysystem_production`
- `multi_domain_insp`

These definitions were not changed. The existing `joint_contamination_labcontrols` label is retained, but its implementation represents laboratory/facilities-system co-occurrence; it should not be reinterpreted as a newly confirmed contamination mechanism.

## Individual associations

Each of the 17 variables is tested separately against each of the five outcomes, adjusting within exact API/form/strength and for inspection age/observation count. Shared scoring units receive one total weight, and connected plant/score-unit clusters determine uncertainty. Effects are outcome points per one SD of the existing feature. No outcome-specific selection of text variables is performed.

All 85 feature/outcome combinations per cohort are reported in `fixed_signal_associations.csv`, including constant and unestimable cases. Holm correction covers the complete 85-test family for each cohort. Models with very sparse positive exposure remain descriptive. These exploratory models reuse previously examined data and are not confirmatory tests.

Main-cohort combinations surviving Holm correction: **0**.

| outcome | feature | status | n_facilities | effect_per_sd | ci_low | ci_high | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_toxic | vc_qualitysystem_share | estimated | 65 | 3.378 | 1.130 | 5.626 | 0.004 | 0.333 |
| DoD_score | vc_qualitysystem_share | estimated | 65 | -4.590 | -7.835 | -1.345 | 0.006 | 0.537 |
| DoD_score | repeat_cross_insp_share | estimated | 65 | -5.451 | -9.782 | -1.120 | 0.015 | 1.000 |
| D_dissolution | repeat_cross_insp_share | estimated | 37 | 6.541 | 1.236 | 11.846 | 0.017 | 1.000 |
| DoD_score | patient_risk_llm_share | estimated | 65 | -4.510 | -8.409 | -0.611 | 0.024 | 1.000 |
| T_toxic | n_qualitysystem_obs | estimated | 65 | 3.114 | 0.360 | 5.869 | 0.027 | 1.000 |
| DoD_score | vc_laboratorycontrolssystem_share | estimated | 65 | 3.909 | -0.384 | 8.202 | 0.073 | 1.000 |
| T_toxic | severity_critmajor_share | estimated | 65 | -2.672 | -5.611 | 0.268 | 0.074 | 1.000 |
| D_dissolution | vc_laboratorycontrolssystem_share | estimated | 37 | -4.265 | -8.981 | 0.451 | 0.075 | 1.000 |
| DMF | repeat_cross_insp_share | estimated | 65 | 0.860 | -0.095 | 1.814 | 0.077 | 1.000 |
| DMF | scope_facilitywide_share | estimated | 65 | 0.589 | -0.150 | 1.329 | 0.116 | 1.000 |
| DoD_score | scope_facilitywide_share | estimated | 65 | -2.785 | -6.315 | 0.745 | 0.120 | 1.000 |

The table shows the smallest nominal p-values for navigation; the full matrix includes every specified combination. Unadjusted findings should not be selected as a new primary hypothesis in this same dataset.

## Joint prediction with all 17 fixed signals

The outcomes retain their numeric values. Ridge regression (fixed alpha 10) and a constrained random forest (300 trees, depth 4, minimum leaf 5) compare the same baseline with and without all 17 existing signals. The baseline contains product, FDA classification, inspection age and observation count. Identical connected groups stay together in each five-fold comparison. Models are not tuned to select a winning endpoint or feature subset.

| outcome | model | status | n_cells | n_facilities | baseline_rmse | baseline_plus_fixed_text_rmse | rmse_difference | difference_ci_low | difference_ci_high | baseline_r2 | baseline_plus_fixed_text_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DoD_score | Ridge | estimated | 234 | 65 | 24.969 | 27.260 | 2.291 | 0.197 | 4.323 | -0.060 | -0.264 |
| DoD_score | RF | estimated | 234 | 65 | 26.644 | 25.605 | -1.039 | -2.568 | 0.530 | -0.207 | -0.115 |
| D_dissolution | Ridge | estimated | 131 | 37 | 22.137 | 26.117 | 3.980 | 1.170 | 6.671 | 0.034 | -0.345 |
| D_dissolution | RF | estimated | 131 | 37 | 25.295 | 26.407 | 1.112 | -2.620 | 4.497 | -0.261 | -0.375 |
| DMF | Ridge | estimated | 234 | 65 | 8.860 | 10.334 | 1.475 | 0.992 | 2.122 | -0.053 | -0.433 |
| DMF | RF | estimated | 234 | 65 | 9.258 | 9.268 | 0.011 | -0.526 | 0.566 | -0.150 | -0.153 |
| NDMA_nitrosamines |  | constant_outcome | 234 | 65 |  |  |  |  |  |  |  |
| T_toxic | Ridge | estimated | 234 | 65 | 16.038 | 16.894 | 0.855 | 0.136 | 1.718 | 0.051 | -0.053 |
| T_toxic | RF | estimated | 234 | 65 | 16.478 | 16.866 | 0.388 | -0.440 | 1.390 | -0.002 | -0.050 |

Lower RMSE is better. A negative difference means text improved prediction. Paired intervals bootstrap fixed out-of-fold predictions by connected group; they do not include all training/specification uncertainty. Negative R² means performance worse than the evaluated cohort's weighted overall-mean benchmark. All model predictions and sensitivities are exported.

The fixed signals do not show a reliable predictive gain in this dataset. For the total score, the random forest's RMSE falls from 26.64 to 25.60, but its paired interval includes no improvement; Ridge worsens from 24.97 to 27.26. Neither model improves the component outcomes reliably. Alongside the lack of multiplicity-adjusted associations, these results support reporting this as an exploratory laboratory-quality validation with limited evidence, rather than as demonstrated patient-outcome validation.

## Files and limits

`score_components.csv` preserves individual scoring columns and signs. `samples_fixed_all_source_unique.csv` preserves every sample, source score/component, linkage status and selected existing inspection features. `fixed_signal_cells_all_source_unique.csv` is the analysis panel. `fixed_signal_associations.csv` is the full association matrix. `fixed_signal_prediction.csv` and `fixed_signal_oof_predictions.csv` give paired continuous-outcome prediction results. `fixed_signal_manifest.json` records input hashes and settings.

Repeated NDCs/samples are not independent score measurements. Manufacturing links remain historical/role-sensitive. Dates are partly assumed collection boundaries. Generic facility signals need not concern the tested product. None of these results establishes patient risk or clinical harm.
