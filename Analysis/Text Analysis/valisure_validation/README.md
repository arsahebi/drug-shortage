# Existing text signals versus DoD quality components

The current analysis follows the user's instruction to use the fixed extracted text variables and separate DoD scoring components. It replaces the earlier keyword-screen experiment; that experiment and its specification are preserved in `outputs/20261006_first_pass/`. The source workbooks are read only.

## Current specification

These are exploratory analyses of previously examined data, not preregistered confirmatory tests.

- Read the exact 17-column `TEXT_FEATURES` list from the existing `vai_signal_validation/02_vai_signal_model.py`. Values come directly from `step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv`. No feature redefinition, new extraction, keyword exposure, route screen or product-name screen is used.
- Use all 13 worksheets of the DoD scoring and testing workbooks, retaining separate ampicillin–sulbactam labels. All 302 scoring rows are represented in the testing source. Keep source sample/NDC/score provenance.
- Outcomes: original overall DoD score; dissolution, DMF, nitrosamine and toxic-element penalty magnitudes. Retain signed source values as well. `--` is no recorded scoring penalty; an absent component column is missing. Toxic-element columns are summed within their source row. Reconcile the original score against all components, including dosage, benzene/EtOx and sterility. Do not clip negative total scores.
- The source labels the nitrosamine family rather than an NDMA-specific concentration. This family is constant at zero, so no association can be estimated. DMF variation is confined to metformin. Raw laboratory dissolution is not a dependent variable in this revision.
- Retain the full-source singleton facility map and timing safeguards. Match the latest existing inspection snapshot strictly before observed metformin intake when available; otherwise use January 1, 2023 as the conservative collection-period boundary. Historical manufacturing attribution and manufacture time remain unresolved.
- Unit: facility × exact product (API/form/strength) × shared source score unit. Average existing snapshot values only when several samples in the same cell have different eligible snapshots. Do not treat repeated packages or copied scores as independent measurements. Require product strata with at least two facilities for within-product comparisons.
- Estimate each of the 17 features separately against each of the five outcomes, adjusting for exact product, log inspection age and log observation count. Weight shared score units equally and cluster connected facilities/score units. Apply Holm correction to the full 85-test family within each cohort, including constant/unestimable combinations. Preserve the previous sparse-exposure inference safeguard.
- Joint models use all 17 signals together. Compare product/FDA-class/inspection-age/observation-count baseline with the same baseline plus text, using paired five-fold connected-group holdouts and numeric outcomes. Report RMSE/MAE/R². Fit Ridge with alpha 10 and a constrained random forest with 300 trees, maximum depth 4 and minimum leaf 5. No outcome-guided parameter or feature selection.
- Bootstrap fixed out-of-fold predictions by connected group for paired RMSE differences. These intervals quantify conditional test-sample uncertainty, not complete model-training uncertainty. Lower RMSE and negative text-minus-baseline differences indicate better prediction.
- Sensitivities remain March-only singleton mapping and the observed-intake-date metformin subset. Do not choose the strongest sensitivity as the primary result.

The initial experiment's sparse-exposure rule withholds inferential intervals and p-values when fewer than four clusters carry nonzero exposure and zero exposure is also present. Coefficients and nominal outputs remain available as descriptive diagnostics. This was a post-diagnostic safeguard, not a preregistered rule.

## Run

From the project root:

```bash
.venv/bin/python 'Analysis/Text Analysis/valisure_validation/20261006_fixed_signals_dod_components.py'
```

Updated files and the report are in `outputs/20261006/`. The original script now defaults to the archived directory so it does not overwrite current results. Source workbooks, extraction outputs, patient-outcome scripts and manuscript are unchanged.
