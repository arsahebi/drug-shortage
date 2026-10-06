# Valisure and inspection text: first exploratory results

This run uses all 13 API worksheets in both DoD workbooks, including the separate ampicillin–sulbactam labels. It evaluates recorded laboratory quality and scoring penalties, not patient outcomes. The analysis choices are in the adjacent README. Neither keyword screens nor facility attribution are yet expert-validated for the tested products.

## Interpretation

The main cohort does not establish a reproducible text/quality association: none of the six declared tests survives Holm correction, and adding the three existing LLM features does not improve the paired held-out prediction. The patient-risk flag has a positive nominal association with worse recorded quality and is a candidate for independent, mechanism-specific follow-up. The dissolution and chemical mention screens do not show a clear association in the main cohort. These results do not establish that the underlying defect mechanisms are absent or clinically irrelevant.

![Exploratory associations and cluster intervals](association_effects.png)

## Source reconciliation

- Testing workbook: 619 unique sample IDs, 13 worksheets, 14 distinct API labels.
- Scoring workbook: 302 source score rows and 227 conservatively shared score units. All source totals reconcile to 100 plus component penalties.
- Source scores range from -83 to 100; they are not clipped to 0–100.
- Scoring match statuses: `{'matched': 619}`. Score disagreements/ambiguous matches are excluded from score models, but retained in audit files.
- NDC parsing: `{'package_and_product_valid': 618, 'product_only_package_invalid': 1}`. One malformed package segment can retain an unambiguous product code; the raw identifier is preserved and the package code is not silently repaired.
- Assay status counts: `{'quantified': 458, 'below_unspecified_limit': 353, 'not_recorded': 100}`. Censored and absent values are distinct; neither is substituted with zero.
- Timing: 68 metformin samples have observed intake dates. Other samples use the conservative January 1, 2023 boundary. This cannot establish lot manufacture time or future prediction.

## Cohort flow

| cohort | source_samples | unique_site_samples | prior_compatible_text_samples | text_facilities | scored_cells | dissolution_cells |
| --- | --- | --- | --- | --- | --- | --- |
| all_source_unique | 619 | 411 | 293 | 66 | 252 | 75 |
| march_only_unique | 619 | 436 | 285 | 76 | 247 | 79 |
| dated_metformin | 68 | 43 | 31 | 13 | 31 | 13 |

Main linkage requires one candidate FEI in the full cross-source union before selecting text-covered plants. DailyMed manufacture/FDF roles are included; ProPublica explicitly API-only links are excluded. Other links can still have uncertain roles and dates. The March-only analysis deliberately admits cross-source disagreements and is a sensitivity analysis.

## Coverage by API

| api | samples | unique_site_samples | prior_text_samples | prior_text_facilities | raw_dissolution_samples | chemical_measured_samples |
| --- | --- | --- | --- | --- | --- | --- |
| ampicillin | 19 | 12 | 9 | 2 | 0 | 16 |
| ampicillin; sulbactam | 18 | 11 | 5 | 2 | 0 | 14 |
| atorvastatin | 63 | 38 | 16 | 5 | 0 | 63 |
| bupropion | 66 | 43 | 30 | 10 | 52 | 0 |
| calcium gluconate | 15 | 12 | 5 | 2 | 0 | 15 |
| lisinopril | 40 | 15 | 1 | 1 | 0 | 40 |
| magnesium sulfate | 23 | 16 | 16 | 6 | 0 | 0 |
| metformin | 71 | 46 | 33 | 15 | 34 | 70 |
| metoprolol | 58 | 47 | 32 | 6 | 58 | 58 |
| metronidazole | 41 | 26 | 16 | 6 | 0 | 40 |
| pantoprazole | 33 | 25 | 23 | 11 | 24 | 0 |
| potassium chloride | 41 | 33 | 24 | 13 | 0 | 41 |
| tacrolimus | 37 | 26 | 25 | 5 | 26 | 0 |
| vancomycin | 94 | 61 | 58 | 13 | 0 | 76 |

## All six declared associations in the main cohort

Effects are outcome units per weighted between-cell SD of the exposure, estimated within exact product and adjusted for inspection age/observation count. Scores have one total weight per shared score unit. Connected FEI/score-unit clusters account for both shared plants and repeated scores. Raw dissolution uses within-product percentiles and FEI clustering. The numeric raw value is never given an unverified clinical/regulatory cutoff.

| analysis | status | n_cells | n_facilities | n_clusters | n_exposed_clusters | effect_per_sd | ci_low | ci_high | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dissolution screen and dissolution penalty | estimated | 97 | 27 | 24 | 5 | -1.471 | -14.098 | 11.157 | 0.812 | 1.000 |
| Chemical screen and chemical penalty | estimated | 155 | 52 | 45 | 12 | -1.616 | -6.747 | 3.515 | 0.529 | 1.000 |
| LLM patient-risk flag and score loss | estimated | 230 | 64 | 56 | 26 | 4.398 | 0.863 | 7.933 | 0.016 | 0.094 |
| LLM data-integrity flag and score loss | estimated | 230 | 64 | 56 | 24 | -1.839 | -8.091 | 4.412 | 0.558 | 1.000 |
| LLM laboratory system and score loss | estimated | 230 | 64 | 56 | 32 | -5.004 | -9.409 | -0.599 | 0.027 | 0.134 |
| Dissolution screen and raw difference factor | estimated | 74 | 27 | 27 | 5 | 0.053 | -0.096 | 0.202 | 0.473 | 1.000 |

All sensitivities and skipped models are in `association_results.csv`. A small p-value does not validate attribution or the keyword mechanism. Lack of an association does not establish clinical equivalence. Multiple products and cohorts have already been examined, so these are exploratory results even with multiplicity correction.

The dated metformin raw-dissolution analysis has only one facility with a positive dissolution mention screen. Its nominal association is not replicated across exposed facilities. Such sparse-exposure models retain descriptive coefficients and nominal outputs in the CSV, but their inferential p-values/intervals are withheld from the interpreted results. The main raw-dissolution analysis has only five exposed facilities, which is also a substantial precision and generalizability limit.

Reading the candidate text confirms why semantic adjudication matters. Observation `3005406526:2019-06-07:6` discusses dissolution-bath qualification/calibration, not a confirmed product dissolution failure. Observation `3004554612:2022-12-09:3` discusses product dissolution OOS results and market batches, but the product identity is redacted. Observation `3002809586:2022-05-09:9` mixes several products and mechanisms and includes a challenged dissolution-result invalidation. A mention count cannot distinguish these situations or establish that the tested API was affected. These examples are source-review notes, not expert-validated new labels, and no additional association was selected after reading them.

## Paired held-out prediction comparison

The endpoint is any recorded scoring penalty, not patient harm. Both models use identical observations and folds. The baseline contains exact product, FDA class (unknown retained), inspection age and observation count; the extension adds the three existing LLM patient-risk, data-integrity and laboratory-system shares. Every connected score/site group stays in one fold.

| cohort | status | n_cells | n_facilities | n_clusters | baseline_auc | baseline_plus_text_auc | auc_difference | delta_ci_low | delta_ci_high | baseline_brier | baseline_plus_text_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_source_unique | estimated | 230 | 64 | 56 | 0.585 | 0.578 | -0.007 | -0.084 | 0.054 | 0.252 | 0.259 |
| march_only_unique | estimated | 228 | 74 | 66 | 0.568 | 0.543 | -0.025 | -0.112 | 0.051 | 0.255 | 0.262 |
| dated_metformin | estimated | 31 | 13 | 12 | 0.401 | 0.275 | -0.127 | -0.339 | 0.000 | 0.293 | 0.384 |

Bootstrap intervals resample groups of fixed out-of-fold predictions. They quantify conditional test-sample uncertainty and do not include all model-training or specification-selection uncertainty. Results need an untouched validation cohort before a predictive claim.

## What is ready for review

`samples_all_source_unique.csv` retains every sample, score match, candidate sites, dates and inclusion reasons. `assays_long.csv` preserves raw values, units in assay names, censoring, and source cells. `scorecards.csv` and `scorecard_ndcs.csv` reconcile scoring provenance. `facility_link_audit.csv` lists the contributing source/role for each product/site link. `annotation_blind.csv` provides the full observation text, candidate mechanism/route screens, and empty expert-review fields without laboratory outcomes. `annotation_priority_blind.csv` restricts this to inspections actually linked in the main cohort and places candidate dissolution/chemical mentions first; its ordering uses no assay outcomes.

Next steps are expert adjudication of those mechanisms, historical finished-dose/product linkage and assay documentation. The broad facility-text association is not the same as demonstrating a matching defect in the tested product. The original manuscripts, workbooks and patient-outcome scripts were not modified.
