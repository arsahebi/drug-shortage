# Revision prompt: 483 LLM extraction manuscript, October 19 2026

Paste everything below this line into Codex.

---

You are revising an existing research manuscript for **npj Digital Medicine**. Work in:

```
/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/
```

The manuscript is `Paper/Text Analysis/manuscript_483_text_patient_harm.tex`. It was restructured on
October 19 (commit `85e87eb`) to follow the story below. Your job is to turn that draft into a
complete, polished manuscript **without changing the story or any number**. Earlier briefs
(`20260930_codex_drafting_prompt.md`, `20261006_codex_revision_prompt.md`) are background only;
**this prompt supersedes them wherever they disagree.**

## 1. The story (do not change it)

The paper makes three claims, in this order and with this strength:

1. **Measurement (solid).** A language model (Claude Sonnet 5, v2 schema) codes Form 483
   observations close to a trained annotator's own consistency. Factual flags transfer across
   models; inferred fields (root cause) do not, and the annotator finds them hard too.
2. **The grade carries little of the text (solid, descriptive).** FDA's NAI/VAI/OAI grade orders the
   text in the expected direction but explains at most 8.7% (median 4.2%) of the variation in any
   of the 17 signals. Many VAI inspections read like OAI ones. **Never say or imply that FDA graded
   anything wrongly.** The grade rests on the establishment inspection report, the firm's response
   and agency policy, which we do not observe (Pazhayattil et al. 2020 make the same point).
3. **Outcomes (exploratory).**
   - General serious FAERS reports: neither text nor grade anticipates a rise. No support.
   - Product-defect FAERS reports: text does better than the grade (0.54 vs 0.43; product-system
     findings 0.61 vs 0.46). **This is a lead, not a finding.** Say plainly that the outcome was
     chosen after the general analyses and needs a pre-specified test on new inspections. Report
     the robustness check (without "switch to another maker" reports) as a check, not as a second
     win.
   - Laboratory-control lag correlations reflect facility size: none survives once each facility
     is compared with its own prior year.
   - Time to first defect report (Cox, Li et al.'s design): null for text and grade.
   - MarketScan and Valisure: no support once counting choices and multiple testing are handled.

All outcome results are associations. Say so where they are reported, not only in limitations.

## 2. What to do

1. **Tighten and complete each section** of the existing `.tex`. Keep its order: Introduction,
   Related Work, Results (corpus; extraction validation; what observations describe; keyword
   rules; how much the grade carries; exploratory outcomes), Discussion, Methods, back matter.
2. **Add a short Figure 1 caption placeholder** describing the pipeline (483 observation ->
   11 fields -> 17 inspection signals -> grade comparison and outcomes). Use a `\todo` for the
   figure itself; do not draw it.
3. **Add Table: grade vs text.** From `within_grade_spread.csv`, a table of the 17 signals with
   mean by NAI/VAI/OAI, variance share explained by grade, and Kruskal-Wallis p. Reference it in
   the "How much of the text the inspection grade carries" subsection.
4. **Add a Supplement** (after the appendix heading) with:
   - S1: inspection exclusion counts (246 -> 213; categories below).
   - S2: product-defect term rule and the category summary (kept and removed groups with report
     counts), pointing to the full term list file.
   - S3: product-defect robustness table (main vs without substitution; known grade, product
     systems, all inspections).
   - S4: lag correlations unadjusted vs the two partial-correlation controls.
   - S5: product-by-product (per-ANDA) serious-report results.
   - S6: MarketScan product-cell and quality-score details; Valisure details.
5. **Keep every `\todo` and `[PENDING]`** that is still unresolved; add new `\todo`s rather than
   inventing anything. Remove a `\todo` only if this prompt or a result file resolves it.
6. **Do not touch** `references.bib` entries except to add a missing key you actually cite, and
   flag any new entry with `\todo{verify}`.

## 3. Writing rules

- Every number must come from a file in section 4. If you cannot find it, write `[PENDING: what]`.
- No em dashes anywhere. Use commas, colons, parentheses or separate sentences. Numeric ranges in
  LaTeX keep `--`.
- Write like a researcher, not like an AI: short declarative sentences, no "notably", "crucially",
  "importantly", "delve", "underscore", "landscape", no rhetorical questions, no lists of three
  adjectives.
- Report n with every percentage and interval with every estimate where the source has one.
- AUCs are medians over 100 facility-grouped partitions with 2.5th to 97.5th percentiles; the
  share of partitions where text beats a baseline is a robustness summary, not a test.
- Use "product-defect reports" consistently (not "quality reports"), "drug-quality inspections",
  "single-site ANDA".

## 4. Sources of numbers

Root for analysis outputs: `Analysis/Text Analysis/vai_signal_validation/outputs/`.

| Content | File |
|---|---|
| Round-2 held-out agreement, Claude | `Analysis/Text Analysis/eval/results_and_notes/round2_agreement.csv` |
| Round-2 held-out agreement, Claude vs GPT-5-mini | `.../eval/results_and_notes/round2_agreement_claude_vs_gpt.csv` |
| Annotator test-retest | `.../eval/results_and_notes/round2_test_retest.csv` |
| Cross-model agreement (1,067 obs) | `.../eval/results_and_notes/claude_vs_gpt_agreement.csv` |
| Keyword baseline | `.../eval/results_and_notes/semantic_lift_vs_regex.csv` |
| Inspection list, grades, exclusions | `Analysis/Text Analysis/step00_redica_inspections.csv` (columns `keep`, `drop_reason`, `dqa_grade`, `other_programs`) |
| Grade vs text | `tables/within_grade_spread.csv` (script `12_within_grade_spread.py`) |
| Serious-report AUCs (factory total) | `tables/repeated_cv_summary.csv`, rows F1-F3 |
| Serious-report AUCs (per ANDA) | `_per_anda_outcome_dqa/tables/repeated_cv_summary.csv` |
| Product-defect AUCs, main and variants | `tables/quality_variant_robustness.csv` (variant `quality` is the main definition; `quality_nosub` drops substitution; `quality_nopatch` and `quality_core` are identical to those two) |
| Paired product-system filter | `_factory_outcome_dqa/tables/paired_text_filter.csv` (serious), `_quality_factory_dqa/tables/paired_text_filter.csv` (defect) |
| Rise rates by grade, within-VAI probe | `_factory_outcome_dqa/tables/rise_by_fda_class.csv`, `_factory_outcome_dqa/tables/vai_within_group_probe.csv`; the permutation AUC is printed in `_saved_before_dqa/run07.log` (despite the folder name, that log is the drug-quality run) |
| Lag correlations | `tables/lag_correlation_table_anda_pp.csv`, `..._adj_n_andas.csv`, `..._adj_n_ae_pre4q.csv` |
| Time to first defect report | `tables/time_to_quality_report.csv` (script `13_time_to_quality_report.py`) |
| Product-defect term list | `Data/15 - FDA - Adverse Event/processed/faers_product_quality_terms.csv` |
| FAERS attribution flow | docstring and printout of `Data/08 - Valisure/processed/20260930_build_propublica_anda_faers.py` |
| MarketScan product cells | `tables/product_cell_results.csv`, `product_cell_all17.csv`, `product_cell_numerator_sensitivity.csv`, `product_cell_robustness.csv` |
| MarketScan quality score | `tables/quality_score_results.csv`, `quality_score_flow.csv` |
| Valisure | `Analysis/Text Analysis/valisure_validation/outputs/20261019/` (`fixed_signal_associations.csv`, `fixed_signal_prediction.csv`, `fixed_signal_cohort_flow.csv`, `RESULTS.md`) |

Key numbers already in the draft (verify against the files, do not change them):

- Corpus: 1,067 observations, 246 inspections, 98 facilities (2018-01-16 to 2026-04-21).
  Analysis set: 213 drug-quality inspections, 985 observations, 93 facilities. Excluded 33:
  pre-approval 15, clinical-study or adverse-event-reporting audit 7, devices 6, blood / vaccine /
  animal 4, no human-drug tag 1. Grades: VAI 125, OAI 24, NAI 11, none posted 53.
- Validation: Claude 79.5% (74.7-83.8) held-out; weighted 81.2%; GPT-5-mini 76.2% (71.1-80.9);
  annotator test-retest 86.4% (79.5-91.8); severity 52.0%, critical-or-major split 82.0%; root
  cause 68% Claude, 56% GPT, 75% annotator self.
- Grade: 160 graded inspections at 87 facilities; variance share max 8.7%, median 4.2%; 39% of VAI
  at or above the OAI median critical-or-major share.
- FAERS linkage: ProPublica 37,584 NDCs, 7,507 ANDAs, 723 facilities; 6,522 single-facility ANDAs;
  2,958 at 90 corpus facilities; 2,262 at 88 with a serious case; dedup 65,781 of 554,509 (11.9%).
  Product-defect reports: 88,019 nationwide; 85 of 186 candidate terms kept.
- Outcome samples: serious 213 -> 198 -> 178 -> 163 (129 graded); product-defect 131 (102 graded),
  product systems 112.

## 5. Things that are settled (do not reopen)

- The production model is Claude Sonnet 5 with the v2 schema, chosen before the held-out round. One
  schema for both models; development-set agreement is reported only as fit to the data used to
  build the instrument.
- There is no comparison of our labels with Redica's codes in this paper. Do not add one.
- FAERS scope is every single-site ANDA in ProPublica, not Valisure's list. MarketScan and Valisure
  keep their own product scopes.
- The inspection grade comes from Redica's event file (Drug Quality Assurance label only), not the
  FDA Inspections Details file.
- Pazhayattil et al. (2020) counted cited CFR subparts across five yearly totals; we extend from
  yearly counts to observation content, inspection grade and outcomes. Do not describe them as
  relating citations to individual inspection grades; they did not.

## 6. Deliverables

1. The revised `manuscript_483_text_patient_harm.tex`.
2. Any new table `.tex` files under `Paper/Text Analysis/generated_tables/`, built by extending
   `generate_manuscript_tables.py` (read paths from section 4; keep the script simple).
3. A short `CHANGES.md` next to the manuscript listing what you changed, every new `\todo`, and any
   number you could not verify.

Do not run any LLM API calls. Do not modify analysis code or result files.
