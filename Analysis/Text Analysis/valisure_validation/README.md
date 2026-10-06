# Valisure validation of inspection text

This first pass uses all 13 worksheets in both DoD source workbooks. Ampicillin and ampicillin–sulbactam remain separate API labels. It tests laboratory quality associations, not patient outcomes. The source workbooks are read only.

## Analysis choices, October 6, 2026

These choices were recorded before computing the new associations. The data and earlier outcome analyses have already been reviewed, so this is an exploratory specification, not a preregistered confirmatory protocol.

- Unit: a tested sample for source reconciliation; a facility × API × formulation × strength cell for raw dissolution analyses; a facility × exact product × shared score unit for scorecard analyses. Repeated samples and NDC packages do not become independent score outcomes.
- Source scores are used on their recorded scale, including negative totals. Component penalties are source scoring decisions, not regulatory failure determinations. `--` means no recorded penalty. Blank assay entries and censored assay results are not zeroes.
- Main linkage: union of full ProPublica links, March Valisure links, and DailyMed manufacture/FDF-manufacture links before restricting to text coverage. ProPublica links explicitly marked API-only are excluded from the candidate union but retained in the audit. Admit only one candidate FEI. Historical finished-dose attribution remains provisional.
- Link latest text strictly before observed metformin intake when available. For other samples use January 1, 2023 as a conservative collection-period boundary from the supplied documentation. This is cross-sectional validation with temporal safeguards, not established prediction of future manufactured lots. Never use expiration minus assumed shelf life.
- Restrict observation applicability by a transparent route screen: exclude sterile-only observations for oral solids, and oral-only observations for injectables; keep observations without an explicit route. Exclude observations explicitly naming other covered APIs without naming the sampled API. This is a candidate screen, not expert-confirmed product matching. Export a laboratory-blinded annotation table.
- Six declared exploratory associations: dissolution mention share versus dissolution score penalty; chemical/impurity mention share versus chemical score penalties; existing LLM patient-risk, data-integrity, and laboratory-system shares individually versus overall score loss; dissolution mention share versus within-exact-product raw dissolution percentile.
- Higher outcomes mean worse recorded quality. The raw difference factor is ranked within exact product to avoid pooling incompatible scales or importing an unverified failure threshold.
- Estimate within-product effects, adjusting for log inspection age and log observation count. Require product strata with at least two facilities. Use one total weight per shared score unit for score outcomes and one weight per facility/product cell for raw dissolution. Cluster score analyses on connected components of shared FEIs and shared score units; cluster raw dissolution on FEI. Report all six analyses, with Holm correction within each cohort. Very small cluster counts remain descriptive.
- Compare product/inspection-age/observation-count/FDA-class baseline with that baseline plus the three existing LLM shares, using the identical scored cohort and held-out connected groups. Report pooled out-of-fold AUC and Brier score. A paired group bootstrap of fixed out-of-fold predictions is conditional uncertainty, not complete training uncertainty.
- Sensitivities: March-only singleton mapping, acknowledging disagreements; and the observed-intake-date metformin subset. Do not pick the strongest sensitivity as the main result.
- No new LLM API calls, trained extraction model, patient-data refresh, manuscript claim, or clinically validated threshold is introduced by this first pass.

After the initial run, an exposure-count diagnostic found that the dated metformin dissolution signal came from just one exposed facility. Inferential intervals and p-values are therefore withheld when fewer than four clusters carry a nonzero exposure (with zero exposure also present). The coefficient and nominal model output remain in the CSV as descriptive diagnostics. This safeguard was added after inspecting the first results, is applied to every association, and must not be treated as a preregistered rule.

## Run

From the project root:

```bash
.venv/bin/python 'Analysis/Text Analysis/valisure_validation/20261006_validate_text_against_valisure.py'
```

Outputs go to `outputs/20261006/`: sample and assay records, scorecard/NDC reconciliation, facility-link audit, cohort flow, all association results, paired out-of-fold predictions, a summary report and source hashes. The blind annotation file has no laboratory outcomes; its candidate screens require expert review before interpreting them as validated defect mechanisms.
