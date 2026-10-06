# Revision prompt: 483 LLM extraction manuscript, October 6 2026

Paste everything below this line into Codex.

---

You are revising an existing research manuscript for **npj Digital Medicine**. Work in:

```
/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/
```

The manuscript is `Paper/Text Analysis/manuscript_483_text_patient_harm.tex`. Its tables are
built by `Paper/Text Analysis/generate_manuscript_tables.py` into
`Paper/Text Analysis/generated_tables/*.tex`. The earlier drafting brief is
`Paper/Text Analysis/20260930_codex_drafting_prompt.md`. Read it for background, but **this
prompt supersedes it wherever they disagree** (listed in section 1).

## 1. What changed since the September 30 brief

1. **The framing is reversed.** The paper is now about the LLM extraction. The extraction method,
   its validation (human labels, self-consistency, cross-model agreement, keyword baseline) and
   what it reveals about 483 text are the contribution. The outcome analyses become one
   exploratory section that tests the extracted signals against three independent outcomes.
   Rule 8 of the old brief ("do not describe the LLM pipeline as the contribution") and its
   "headline claim" section no longer apply.
2. **The single-split AUCs in the draft are not stable and must be replaced.** The models in
   `Analysis/Text Analysis/vai_signal_validation/02_vai_signal_model.py` and
   `07_three_class_and_vai_probe.py` use one GroupKFold split. That split differs between
   scikit-learn 1.9.0 and 1.6.1, and on identical data it moves the results (07 text-only LR:
   0.592, p = 0.007 under 1.9.0; 0.540, p = 0.23 under 1.6.1). Every AUC in the paper must now
   come from `09_repeated_cv_robustness.py`: 100 random 5-fold partitions of facilities, text
   and FDA baselines scored on identical partitions. Report the median and the 2.5 to 97.5
   percentile range across partitions, and the share of partitions where text beats the FDA
   baseline. Describe the single-split sensitivity briefly in Methods or the supplement as the
   reason for this choice.
3. **MarketScan is no longer future work.** The data arrived and has been analysed. Old section
   5b is superseded. MarketScan results go in the exploratory section (section 3 below).
4. **Valisure lab results have been analysed** against the same 17 text features (no support).
   They go in the exploratory section too.
5. **All outcome findings are associations, not causal effects.** Say so where they are reported,
   not only in limitations.

## 2. Read these before writing

Analysis code and notes (read the docstrings; they record design decisions and results):
- `Analysis/Text Analysis/README.md`
- `Analysis/Text Analysis/vai_signal_validation/01b_build_marketscan_outcomes.py`
- `Analysis/Text Analysis/vai_signal_validation/02_vai_signal_model.py`
- `Analysis/Text Analysis/vai_signal_validation/08_product_cell_analysis.py`
- `Analysis/Text Analysis/vai_signal_validation/09_repeated_cv_robustness.py`
- `Analysis/Text Analysis/valisure_validation/README.md` and
  `Analysis/Text Analysis/valisure_validation/outputs/20261006/RESULTS.md`
- `Data/20 - Market Scan/raw/Description.docx` (how MarketScan measures were built),
  `Data/20 - Market Scan/docs/Qual_Score_Explained.pdf` (the data team's own validation)
- The group slides `Presentation/2026-10-06-Update_Journal_LLM_NDC_Mapping.html`, Part 5,
  summarise the current results in plain language. Use them to check your reading, but take
  every number from the result files below, not from the slides.

Result files (the only sources of numbers):

| Content | File (under `Analysis/Text Analysis/vai_signal_validation/outputs/`) |
|---|---|
| Repeated-partition AUCs, FAERS and MarketScan pre/post | `tables/repeated_cv_summary.csv` |
| FAERS feature-lag correlations (64 tests, Bonferroni) | `tables/lag_correlation_table_anda_pp.csv` |
| FDA class rise rates, within-VAI probe | `tables/vai_within_group_probe.csv`; rise rates by class are printed by `07` |
| MarketScan product-cell, pre-specified H1 to H4 + replication | `tables/product_cell_results.csv` |
| MarketScan product-cell robustness (permutation, per cohort, leave-one-API-out) | `tables/product_cell_robustness.csv` |
| MarketScan product-cell, all 17 features, 6 windows, Holm over 17 | `tables/product_cell_all17.csv` |
| MarketScan diagnostics (cohort agreement, period stability) | `tables/product_cell_diagnostics.csv` |
| CCAE quarterly abandonment (seasonality) | `tables/marketscan_ccae_quarterly_abandonment.csv` |
| MarketScan plant bridge coverage | `tables/marketscan_bridge_coverage.csv` |
| Valisure association matrix (17 features x 5 outcomes x 3 cohorts) | `../valisure_validation/outputs/20261006/fixed_signal_associations.csv` |
| Valisure prediction comparison | `../valisure_validation/outputs/20261006/fixed_signal_prediction.csv` |
| Valisure cohort flow | `../valisure_validation/outputs/20261006/fixed_signal_cohort_flow.csv` |

Extraction validation files are unchanged and are already wired into
`generate_manuscript_tables.py` (human agreement, Claude vs GPT agreement, regex comparison).

## 3. Tasks, in order

### 3.1 Fix the table generator first

`generate_manuscript_tables.py` still points at `Data/99 - Outputs - Text Analysis/`, which no
longer exists. The folder moved to `Analysis/Text Analysis/`. Fix the paths. Then:
- Replace `models()` and `robustness()` so they read `repeated_cv_summary.csv`, not
  `three_class_baseline.csv` or the `ablation_metrics_*.csv` files.
- Add generators for: the MarketScan product-cell hypotheses (H1 to H4 and comparators, main
  window), the all-17 matrix for the main window, the 2023Q1 to 2024Q3 replication, and a
  compact Valisure table (the 17 features against the DoD total score, plus counts of tests and
  of Holm survivors per cohort).
- Each generated table starts with a comment naming its source file, as the existing ones do.
- Run it with `/Users/asahebi/Projects/drug-shortage-research/venv/bin/python`. Do not
  hand-type a number that a generator can produce.

### 3.2 Title and abstract

Retitle around the extraction. The current title ("FDA Inspection Classifications and Form 483
Text as Signals of Subsequent Serious Adverse Events") promises an outcome result the data does
not deliver. Offer two or three options in a comment and use your best one.

Rewrite the abstract in this order: the problem (483 text is unstructured and unused at scale);
what was built (1,067 observations, 246 inspections, 98 facilities, FDA six-system schema);
validation (human, self-consistency, cross-model: agreement is high on factual flags and low on
judgments such as root cause); keyword comparison; then one or two sentences on the exploratory
outcome tests: text beats FDA's classification modestly and consistently on FAERS; data
integrity is the one signal that recurs across FAERS and MarketScan; Valisure shows no
association; all associations, none causal. Keep `[PENDING: round-2 blinded labels and
test-retest scoring]` where held-out human validation goes.

### 3.3 Results: keep the extraction sections, rebuild the exploratory section

Keep the current order: corpus, extraction validation, what the observations describe, LLM
versus keyword rules. Update them only for flow and for the new framing.

Replace "Exploratory application: inspection text and subsequent reports" with one exploratory
section of four short subsections:

1. **FAERS adverse events.** Inspection-centred design, ANDA-attributed reports, outcome = reports
   rose in the 4 quarters after vs before. Use the repeated-partition results: on the 123
   inspections with a known FDA class, text-only median AUC about 0.55 against about 0.52 for the
   FDA three-class dummies, text higher on about 80% of identical partitions; on the broader
   143-inspection panel, random-forest text about 0.66, beating the OAI flag on every partition.
   Product-system restriction (119 inspections, 501 of 1,067 observations) scores about the same
   as all text; do not claim it improves stability. Keep the FDA class rise rates (VAI 59.8%, OAI
   and NAI 50.0%) and the within-VAI null. Keep the lag table: data integrity at Q0 and Q+1, lab
   controls at Q+1, surviving Bonferroni over 64 tests, unadjusted for repeated facilities.
2. **MarketScan, facility level.** Explain the switch design briefly (a settled patient handed
   another labeler's version of the identical product; abandonment = no refill within days
   supply + 60, net of dose changes and coverage loss). Report that the inspection-centred
   pre/post design was near chance for abandonment (repeated-partition medians in
   `repeated_cv_summary.csv`, M1), that net ER visits looked positive but were one of four
   outcomes tried (M2), and why the facility level fails: facility-pooled abandonment barely
   agrees between the commercial and Medicare cohorts, while labeler x exact-product percentiles
   agree moderately (`product_cell_diagnostics.csv`).
3. **MarketScan, product level.** Unit = labeler x exact product (oral solids), within-product
   percentile of abandonment, 2016Q3 to 2022Q4, both cohorts. Report H1 (lab controls) and H2
   (data integrity) with Holm over H1 to H4, the robustness checks, the all-17 matrix (data
   integrity survives Holm over 17; three quality-system features survive with the opposite
   sign; lab controls just misses), the two halves, and the 2023Q1 to 2024Q3 replication
   failure. State plainly that H1 and H2 were chosen after a first look at this data. Explain
   that same-length 7-quarter windows inside 2019 to 2022 also show roughly half-size effects, so
   the replication window has limited power, and that product percentiles are only weakly stable
   across periods.
4. **Valisure laboratory results.** 17 features x 5 DoD outcomes, within exact product: 0 of 85
   survive Holm (68 estimable in the main cohort); adding the text to a prediction model gives no
   reliable gain. Note that for lab controls and quality-system share, the Valisure and MarketScan
   signs point in opposite directions.

Close the section with a short synthesis: data integrity is the only signal that appears in two
independent outcomes and survives correction in both; it is a candidate, not a confirmed
predictor; everything here is association.

### 3.4 Methods

Add: MarketScan data (CCAE and MDCR delivered separately, 2016Q1 to 2024Q4, MDCR 2023 excluded
by the provider, 13 of 14 APIs present, five short-course or hospital APIs set aside); the
labeler-to-plant bridge (ProPublica single plant, then the Valisure and new crosswalks, then
DailyMed manufacture operations; a row counts only if all its NDC9s resolve to one plant; 54.5%
of non-acute commercial switches linked vs 48.3% with ProPublica alone); the product-cell model
(OLS, product fixed effects, plant-clustered SEs, Holm); the repeated-partition CV; the Valisure
design (point to its README for detail). Move Methods after Discussion per journal style if it is
not there already.

### 3.5 Discussion and limitations

Make these points, briefly:
- The unit of analysis matters: facility-level outcomes are too blunt, product-level comparisons
  are where any signal appears. This is a practical lesson for anyone using this kind of data.
- Data integrity recurs across two outcomes built from different data and different methods.
- Association, not causation; facilities with weak data integrity may differ in many ways.
- Limitations: single-split AUC sensitivity (now handled by repeated partitions); signals chosen
  after looking in MarketScan; no out-of-time replication; MarketScan attributes labelers, not
  plants, for about half of switches; the provider's plant-level file reports far fewer switches
  than the labeler rows mapped to the same plant, and its definition is unconfirmed; FAERS
  reporting bias; single human annotator (test-retest, not interrater).
- Future work: pre-registered test of the data-integrity signal on the expanded Redica facility
  set; MarketScan Lab Database (HbA1c, LDL, tacrolimus trough) as a physiological outcome.

### 3.6 Supplement

Add: CCAE vs MDCR agreement and period stability (`product_cell_diagnostics.csv`); the CCAE
quarterly abandonment table, noting that abandonment is lower in Q3 and Q4 every year because
coverage-ended stops rise near plan-year end, so this seasonality is not specific to 2023; the
full all-17 matrix for every window; the single-split vs repeated-partition comparison.

## 4. Hard rules

1. **Never invent a number.** Every number comes from a file in section 2, through the table
   generator where possible. If something is missing write `[PENDING: <what is needed>]`.
2. **Edit the .tex in place.** Keep the preamble, the author list, the `\todo{}` command and any
   wording the author has written that is still accurate. Do not regenerate the manuscript from
   scratch.
3. **No em dashes anywhere.** Commas, colons, or a new sentence instead.
4. **Write like a researcher.** No "delve", "leverage", "underscore", "it is important to note",
   "in the realm of", "robust" as filler. Plain declarative sentences, short paragraphs.
5. **n beside every statistic**, including cells and facilities for MarketScan and partitions for
   AUCs.
6. **Uncertainty on every estimate**: Wilson intervals for proportions, partition percentile
   ranges for AUCs, CIs for regression coefficients.
7. **Do not overclaim.** Say "associated with", never "predicts harm" or "causes". Where a result
   depends on classifier, window or cohort, say so in the same sentence.
8. **Do not rerun the LLM extraction, call any LLM API, or change the v2 prompt or schema.**
   Round-2 human labels are being scored against that schema.
9. **Do not report single-split AUCs as results.** They may appear only in the supplement, to
   show the sensitivity.
10. **Do not commit to git.** The author reviews the diff first.
11. `pdflatex` is not installed on this machine. Do not try to compile; check LaTeX syntax by
    reading.

## 5. When you finish

Output a short list of: every `[PENDING]` and `\todo{}` left; every place the data contradicted
the existing draft (there are several in the current abstract and in the model and robustness
tables); every table you generated with its source file; and any analysis you think the paper
needs that has not been run.
