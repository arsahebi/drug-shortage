# Drafting prompt: 483 text → patient harm manuscript (npj Digital Medicine)

Paste everything below this line into Codex.

---

You are drafting a research manuscript for submission to **npj Digital Medicine**. Work in the
repository at:

```
/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/
```

## 1. What the paper is

We use a large language model to convert the free text of FDA Form 483 inspectional observations
into structured quality signals, then test whether those text-derived signals predict downstream
patient harm (FAERS adverse events) better than FDA's own categorical inspection outcome (the
OAI/VAI/NAI classification).

The LLM pipeline is the instrument that makes 483 text measurable and comparable across
facilities. **It is not the contribution by itself.** The contribution is what the measured text
tells us about regulatory signal quality. Keep that framing throughout.

### The headline claim (decided; do not substitute your own)

Lead with: **FDA's categorical inspection outcome carries little predictive signal about
downstream patient harm, while the underlying inspection text carries more.**

Do NOT lead with "483 text predicts patient harm better than FDA's grade." That stronger claim
rests on a single classifier in a single subsample (random forest, ANDA-restricted, AUC 0.626)
and does not survive a switch to logistic regression or to the drug-level sample. A reviewer will
find that in our own table.

The claim above is chosen because the OAI-only model never exceeds chance in ANY specification we
ran (AUC 0.457, p=0.99). That is the most robust result in the study. The text advantage then
supports the claim rather than carrying it alone, and the predictive fragility becomes a stated
caveat instead of a hole in the main argument.

## 2. The exemplar to follow

`Paper/Text Analysis/s41746-026-02353-7.pdf` — Li et al., *npj Digital Medicine* 2026;9:221,
"Scaling medical device surveillance with LLMs." Read it first. It is the structural model,
published in the target journal, doing the same class of work (LLM extraction from regulatory
free text, validated against human annotation, then applied at scale).

Emulate specifically:

- **One pipeline, several applications.** They build one extraction pipeline and show three case
  studies. We have an analogous shape: descriptive characterization, a rule-based comparison,
  and a predictive application.
- **Validate before you scale.** Every application is preceded by accuracy on an annotated
  subset.
- **95% confidence intervals on every reported number.** They do this without exception. We
  currently do not. Add them.
- **Report agreement beside accuracy.** They print interrater agreement next to every LLM
  accuracy figure so the reader can judge the ceiling. Our analogue is model self-agreement and
  cross-model agreement — see §5.
- **Let the rule-based baseline win where it wins.** Their Table 7 shows a regex baseline beating
  the LLM on one field and they report it plainly. We have the same pattern and must do the same.
- **Label exploratory work as exploratory.** Their third case study says outright that it "is
  exploratory and is intended to demonstrate what is possible." Our predictive modelling needs
  the same register.
- **Honest accuracy framing.** They describe roughly 80% accuracy as "sufficient for many but not
  necessarily all use cases." Match that tone.

Do **not** copy their sentences, section titles, or table captions. Match the structure and the
epistemic standard, not the prose.

## 3. Start from the existing draft

`Paper/Text Analysis/manuscript_483_text_patient_harm.tex` (LaTeX, ~1,090 lines).

- **Sections 1–4 (Introduction, Related Work, Data, Methodology) are current and good.** Revise
  for flow and for the journal, but do not rebuild them. They already describe the correct
  pipeline: Redica-sourced corpus, 98 facilities, 1,067 observations, Claude v2 prompt, FDA
  six-system violation categories, mutually exclusive contamination flags.
- **Sections 5–7 (Results, Discussion, Conclusion) are stale placeholders** from a superseded run
  (622 observations, 38 facilities, GPT-5-mini, an old 8-class taxonomy). They sit behind a box
  saying nothing below should be cited. **Replace them entirely** using the numbers in §5 below.
  Delete the box.
- `Paper/Text Analysis/references.bib` holds the bibliography. Four entries have unverified
  placeholder author/year fields; find and flag them, do not invent citations.

## 4. Read the full analytical history before writing

Do not write from the numbers alone. The reasoning matters and much of it is only in these files.

**Pipeline code**
- `Data/99 - Outputs - Text Analysis/01_extract_observation_signals.py` — extraction. Contains
  both v1 and v2 prompts for both providers, plus the tool schemas. The v2 Anthropic prompt
  (`_ANTHROPIC_PROMPT_FIXED_V2`) is what produced the reported data. Read the field rules; the
  Methods section must describe them accurately.
- `Data/99 - Outputs - Text Analysis/02_aggregate_fei_features.py` — aggregation to
  facility × inspection-date snapshots.
- `Data/99 - Outputs - Text Analysis/vai_signal_validation/*.py` — panel construction, the
  ablation models, lag correlations, gap trajectory, the silent-problem analysis.
- `Data/99 - Outputs - Text Analysis/eval/code/03_score_cross_model_agreement.py`,
  `05_semantic_lift_vs_regex.py`, `07_extraction_reproducibility.py`.

**Analytical narrative and decisions**
- `Data/99 - Outputs - Text Analysis/eval/results_and_notes/20260902_human_eval_round1_findings_and_fixes.md`
  — the single most important document. Round-1 human validation, the bugs it found, the prompt
  fixes, the self-agreement discovery, and why scaling up human labeling was rejected.
- `Data/99 - Outputs - Text Analysis/eval/results_and_notes/20260916_LLM_Extraction_Validation_Report.docx`
- `Data/99 - Outputs - Text Analysis/eval/results_and_notes/20260917_AE_Outcome_Confound_and_Fix.docx`
  — why the outcome is a per-facility relative change rather than a raw count. 88.6% of AE
  variance is between-facility; this fix is essential to the Methods.
- `Data/99 - Outputs - Text Analysis/eval/results_and_notes/20260909_session_handoff.md`
- `Paper/Text Analysis/20260929_Exemplar_Paper_Analysis.docx` — structural comparison against
  the exemplar, including where we match and where we fall short.
- `Paper/Text Analysis/20260928_Draft_Completion_Plan.docx`
- `Paper/Text Analysis/20260928_Potential_Journals.docx` — journal fit reasoning.
- `Paper/Text Analysis/20260805_v2_prompt_calibration_and_model_comparison.docx`
- `Paper/Text Analysis/20260629_483_LLM_Prompts_Expert_Review_YI.docx` — pharmacist expert review
  that drove the v2 prompt. Cite this as expert validation of the schema.
- `Data/99 - Outputs - Text Analysis/eval/sent_to_abdul/483_Labeling_Rules_v2.docx` — the human
  annotation protocol, identical in content to the LLM prompt. Appendix material.

**Result tables (authoritative, regenerated 2026-09-30)**
- `.../eval/results_and_notes/claude_vs_gpt_agreement.csv`
- `.../eval/results_and_notes/semantic_lift_vs_regex.csv`
- `.../eval/results_and_notes/extraction_reproducibility_anthropic.csv`
- `.../eval/results_and_notes/models_vs_human_agreement.csv`
- `.../vai_signal_validation/outputs/models/ablation_metrics_anda.csv` and
  `ablation_metrics.csv`
- `.../vai_signal_validation/outputs/tables/gap_trajectory.csv`,
  `silent_problem_groups.csv`, `lag_correlation_table.csv`

**Prompt development history (how the signals were derived)**
- `Paper/Text Analysis/20260629_483_LLM_Prompts_Expert_Review_YI.docx` and
  `20260629_483_LLM_Prompts_Expert_Review.docx` — **the pharmacist expert review by Yelena
  Ionova that produced the v2 schema.** This is where the field definitions come from. The v1
  blanket exclusion of oral solid dose was found to miss real risk for narrow-therapeutic-index
  drugs, oral oncology, and nitrosamine products, which is why patient-risk scenario (a2) exists.
  Cite this as expert validation of the schema and describe it in Methods. Do not present the
  categories as if we invented them unaided.
- `Paper/Text Analysis/20260805_v2_prompt_calibration_and_model_comparison.docx` — v2 calibration
  and the first Claude/GPT comparison.
- `Meeting/ALison & Yelena - Redica - 07-07-2025.docx` — Redica data provenance.
- `Data/99 - Outputs - Text Analysis/eval/prompt_debug_reruns/*.csv` — eleven re-run files
  documenting each prompt iteration on the same 50 observations (original, SEVFIX, SEVFIX2,
  FINAL, and repeated runs of each). These are the evidence that prompt changes, not chance,
  produced the improvements. Use them if you need to show the development trajectory.
- `Data/99 - Outputs - Text Analysis/eval/results_and_notes/human_eval_metrics_v2.md` — round-1
  per-field metrics.
- `Data/99 - Outputs - Text Analysis/eval/sent_to_abdul/483_Background_Reference_Guide.docx` and
  `483_Worked_Example_FEI3003342394_obs3.docx` — annotator onboarding materials.
- `Data/99 - Outputs - Text Analysis/eval/sent_to_abdul/20260903_Question_for_Abdul_data_integrity_rule.docx`
  — the escalation that produced the testing-into-compliance rule.
- `Data/99 - Outputs - Text Analysis/README.md` — pipeline overview.

**How the signals connect to patient harm (read before writing Methods)**
- `Data/99 - Outputs - Text Analysis/vai_signal_validation/01_build_inspection_panel.py` — the
  linkage. Read this closely; see §5a below on what it does and does not establish.
- `Data/15 - FDA - Adverse Event/processed/code/*.py` — FAERS preparation, including
  `2026-05-12-faers_valisure_filter_and_eda.py`.
- `Data/15 - FDA - Adverse Event/raw/FEARS Summary.docx` — FAERS structure and caveats.
- `Data/08 - Valisure/raw/FEIs_March 2026.xlsx`, sheet `API Only_FEI Mapping` — the
  facility-to-ingredient map, 129 FEIs.
- `Data/17 - NDC, FEI Mapping/ndc_fei_from_labels.csv` — NDC to FEI crosswalk.
- `Data/14 - FDA - Inspection/raw/Inspections Details.xlsx` — primary OAI/VAI/NAI source.
- `Data/07 - Redica/processed/redica_all_drugs_combined.csv` — fallback classification source.

**Figures already generated** (regenerate from code, do not re-plot by hand)
- `.../vai_signal_validation/outputs/figures/ablation_auc_bar_anda.png` (headline),
  `ablation_auc_bar.png`, plus `_global` variants, `lag_correlation_heatmap.png`, and 14
  per-facility `trend_<FEI>.png` files.

Git history on `Data/99 - Outputs - Text Analysis/` records why each prompt rule exists. Use
`git log` on the extraction script when a Methods claim needs justification.

## 5. The numbers (use these; verify against the CSVs; do not invent)

**Corpus.** 1,067 observations, 98 facilities (FEIs), 246 inspection events, 2018-01-16 to
2026-04-21. 36 of 246 inspections (14.6%) classified OAI. 14 drug ingredients.

**Extraction distributions (Claude Sonnet 5, v2 prompt).** Violation category:
FacilitiesEquipment 28.2%, LaboratoryControls 26.1%, QualitySystem 24.5%, Production 14.3%,
Materials 5.5%, PackagingLabeling 0.9%, Other 0.4%. Severity: Moderate 43.3%, Major 37.2%,
Critical 17.8%, Minor 1.7%. Scope: FacilityWide 48.8%, MultipleProducts 34.0%, SingleBatch
14.2%, Unclear 2.9%. Root cause: Capital 52.1%, Mixed 24.0%, Cultural 22.6%, Unclear 1.3%.
Flags: patient risk 20.1%, data integrity 19.5%, contamination confirmed 8.7%, contamination
risk 43.3%, investigation 32.0%, repeat 6.1%.

**Reproducibility (same prompt, same input, 3 passes, Claude, n=50).** Perfect (100%):
violation_category, severity_tier, repeat, patient_risk, investigation. Then data_integrity
98%, contamination_flag 96%, contamination_risk 94%, remediation 94%, root_cause 93%, scope 90%.
Compare against the August baseline in the round-1 findings doc (72–96%) to show the prompt
revisions improved stability. **Temperature cannot be set**: claude-sonnet-5 returns HTTP 400
"`temperature` is deprecated for this model", and gpt-5-mini is a reasoning model whose
Responses API also rejects it. State this as a methods constraint; it is why the exemplar's
temperature=0 is not available to us.

**Cross-model agreement (Claude vs GPT-5-mini, identical text and prompt, n=1,067).**
repeat 99.4% (κ=0.95), contamination_flag 95.4% (0.76), severity collapsed 94.2% (0.81),
investigation 93.3% (0.85), data_integrity 92.8% (0.76), patient_risk 86.0% (0.62),
violation_category 83.4% (0.79), contamination_risk 81.7% (0.64), severity 4-class 76.4% (0.63),
scope 70.7% (0.54), **root_cause 55.5% (0.37)**, remediation 48.2% (0.16, n=141).

This is a genuine finding and should be prominent. Root cause is the field the qualitative
literature cares most about, and two competent models disagree on nearly half of observations.
It is not sampling noise: Claude's self-agreement on that same field is 93%, far above the 55.5%
cross-model figure. Frame it as a caution for any single-model LLM coding study.

**Semantic lift vs regex (n=1,067).** Data integrity: regex 5.4%, LLM 19.5%, LLM-only 15.0pp,
regex-only 0.9%. Contamination confirmed-or-risk: regex 30.4%, LLM 52.0%, LLM-only 24.1pp.
Contamination confirmed only: regex 30.4%, LLM 8.7%, regex-only 23.8pp — the keyword baseline
massively over-flags, because contamination vocabulary appears in observations describing
control gaps with no confirmed event. Patient risk: LLM-only 10.3pp. **But** investigation 90.3%
agreement and repeat 98.3% agreement — for lexically marked concepts the regex baseline is
nearly as good, and the LLM's advantage is concentrated in concepts requiring judgment about
whether something was *confirmed* versus merely *at risk*. Report this split honestly; it is the
analogue of the exemplar's Table 7.

**Human validation.** Round 1: 50 observations, blind, single annotator, per-field accuracy in
`models_vs_human_agreement.csv` and the round-1 findings doc. Round 2 is in the field: 70
observations (50 new, stratified to oversample rare categories, plus 20 embedded repeats of
round-1 rows for test–retest reliability), issued 2026-09-30 under rules frozen beforehand.
**Round 1 is a development set, not a validation set** — it drove the prompt fixes and the rule
amendments, so scoring the final pipeline against it is circular. Say this explicitly. Round 2
is the held-out set. If its results are not in the repository when you draft, mark those cells
clearly as pending rather than guessing.

**Predictive models.** Outcome is a per-facility relative change in FAERS adverse event counts
around each inspection, not a raw count (see the AE confound document). Five-fold CV, AUC.

ANDA-specific AE counts, n=156 facility-inspections:

| Config | LR AUC | p | RF AUC | p |
|---|---|---|---|---|
| A: Text only | 0.520 | 0.306 | **0.626** | **0.015** |
| B: Text + OAI | 0.509 | 0.400 | 0.609 | 0.028 |
| C: OAI flag only | 0.457 | 0.985 | 0.457 | 0.985 |
| D: VAI-only facilities (n=93) | 0.533 | 0.212 | 0.502 | 0.478 |
| E: OAI-ever facilities (n=63) | 0.466 | 0.682 | 0.570 | 0.244 |

Drug-level AE counts, n=238: text AUC 0.514–0.549, none significant; OAI-only 0.521, p=0.030.

**Report these conditionalities plainly, in the Results, not buried in limitations.** The text
advantage holds for random forest on the ANDA-restricted sample and does not hold for logistic
regression, nor at drug level. What is robust across every specification is that the OAI flag
alone never exceeds chance — 0.457 in the ANDA sample. That is the defensible claim: FDA's
categorical outcome carries little predictive signal about downstream adverse events, and the
text carries more, though how much more depends on specification.

**Gap trajectory (n=237 inspections with a matched gap).** Ratio of adverse events at the
inspection quarter to four quarters prior: gap under 1 year 1.145, 1–2 years 1.266, 2–3.5 years
1.090, over 3.5 years 1.031. Adverse events rise *before* FDA arrives. Per the project's agreed
framing, this reflects FDA's Site Selection Model working as designed — risk signals draw the
inspection — and is not evidence FDA is late. The 483 text contribution is the content of the
*prior* visit, which the SSM does not capture.

**Silent problem groups.** High-signal VAI facilities (17 FEIs) show the largest pre-inspection
rise at 1.211, above Low-signal VAI (55 FEIs, 1.175) and above OAI-ever facilities (26 FEIs,
1.129). Facilities FDA graded VAI but whose 483 text is severe behave worse than facilities FDA
graded OAI. The documented OAI→VAI downgrade mechanism is the proposed explanation; treat it as
a hypothesis, and note the supporting work is unpublished.

**Lag correlations.** Laboratory controls is the only feature with consistent significant
association with subsequent adverse events: Spearman r 0.15–0.18, p<0.02, sustained from the
inspection quarter through four quarters after. No other feature reaches significance. Do not
present the correlation heatmap as if many cells were meaningful.

## 5a. How adverse events attach to facilities, and what that limits (audited 2026-09-30)

Read `vai_signal_validation/01_build_inspection_panel.py` before writing any Methods sentence
about the outcome. The linkage was audited and these are its actual properties. They must be
stated in Methods and Limitations, not discovered by a reviewer.

**There are two outcome constructions, and they are not equally attributable.**

*Drug-level* (`_load_faers_raw`, default run). Serious FAERS reports are matched to a facility by
active ingredient alone, via the Valisure `API Only_FEI Mapping` sheet. Every facility making
metformin is assigned every serious metformin adverse event nationally. The median facility
quarter carries 413 serious events, which is not a plausible facility-attributable harm count.
Across facilities sharing an ingredient and inspected in the same year, the counts are frequently
identical. **This construction cannot support a causal or facility-specific claim.** Report it as
a sensitivity analysis only, and say plainly that it is ingredient-level exposure, not facility
attribution.

*ANDA-specific* (`_load_anda_ae_quarterly`, the `--anda-ae` run; **this is the headline**). Events
are matched through ANDA numbers to specific products, from
`Data/08 - Valisure/processed/valisure_anda_faers_ae_counts_quarterly.csv`. The median facility
quarter carries 15 serious events, which is consistent with genuine product-level attribution.
This is why the headline model uses it.

**The cost of that specificity is coverage.** Of 246 inspection events, 95 have no ANDA-linked AE
count at the inspection quarter and 101 have none four quarters prior; only 131 have both. The
modelling sample of n=156 is therefore a subset of inspections with usable ANDA-linked outcome
data, not the full corpus. **State the 156-of-246 figure and its cause explicitly**, and address
whether facilities with ANDA linkage differ systematically from those without. That is a
selection concern a reviewer will raise.

**Two minor issues found in the audit. Mention the first, fix or flag the second.**

1. Ingredient keys are matched on the first whitespace-delimited token, so `Ampicillin` and
   `Ampicillin; Sulbactam` collapse to one key. This is arguably correct for a facility that
   makes ampicillin products, but it is an unintended merge and should be noted.
2. In the drug-level path, `_faers_quarterly` counts `primaryid` rather than counting distinct
   `primaryid`. A single report matching two ingredients made by the same facility is counted
   twice. Measured inflation is 24,479 of 2,848,865 joined rows, **0.9%**. It does not change any
   conclusion, but the correct aggregation is `nunique`. Fix it and note the size, or flag it.

**Other linkage facts worth stating in Methods.** Serious outcomes only (Death, Hospitalization,
Life-threatening, Disability, Congenital anomaly, Required intervention, Other serious); non-
serious reports are excluded. OAI/VAI/NAI comes from FDA `Inspections Details.xlsx` matched
exact-date, with Redica as first fallback and an FDA plus-or-minus-30-day match as second; 40 of
246 inspections resolved to neither and defaulted to 0, which should be reported. Text features
use the Major/Moderate severity collapse rather than the raw four tiers, because human-eval
accuracy is 90-94% for that collapse against 66-68% for four tiers; this choice is deliberate and
justified in the validation report.

## 5b. MarketScan: future work, not a pending result

The author is awaiting Truven MarketScan commercial claims data, intended as a second and more
direct outcome variable than FAERS. Timing is unknown and it will not be in this paper.

Treat this as a Discussion and future-work item, framed around why it matters: FAERS is a
voluntary, passive surveillance system with well-known reporting bias and no denominator, whereas
claims data would give observed utilization and outcomes with a population at risk. Say that the
present adverse-event outcome is a proxy whose limitations are inherent to spontaneous reporting,
and that a claims-based replication is the natural next step. See
`Data/20 - Market Scan/20260917_MarketScan_Data_Request.docx` for scope.

Do not speculate about what MarketScan would show, and do not state or imply that it would
improve model performance. Write it as a design improvement, not a predicted result.

## 6. Structure to produce

Follow npj Digital Medicine conventions: Abstract, Introduction, Results, Discussion, Methods
(Methods *after* Discussion), then Data availability, Code availability, References, and
supplementary material. Note this differs from the current draft's ordering, which puts
Methodology before Results — restructure accordingly.

Results should run: corpus description → extraction validation (human, reproducibility,
cross-model) → what the text shows descriptively → LLM versus rule-based baseline → the
predictive application, labeled exploratory.

## 7. Hard rules

1. **Never invent a number.** If a value is not in the files, write `[PENDING: <what is needed>]`.
2. **95% CIs on every accuracy, agreement, and AUC.** Wilson intervals for proportions,
   bootstrap or the fold standard deviations already in the ablation CSVs for AUC.
3. **No em dashes anywhere.** Use commas, colons, or restructure the sentence. This is a firm
   preference of the author.
4. **Write like a researcher, not like a language model.** No "delve", "leverage", "underscore",
   "it is important to note", "in the realm of". Plain declarative sentences. Short paragraphs.
5. **State n beside every statistic**, including per-panel n where subsamples differ.
6. **Do not overclaim.** Where a result is conditional on classifier, sample, or model, say so in
   the sentence that reports it, not only in the limitations.
7. Single annotator is a real limitation. We have test–retest reliability from the embedded
   repeats, not interrater reliability. The exemplar had four annotators. Be explicit that
   test–retest measures consistency, not correctness, and that a consistently mistaken annotator
   would look reliable by this measure.
8. Do not describe the LLM pipeline as the contribution.
9. Data files stay in Google Drive and are not public. Code is at
   https://github.com/arsahebi/drug-shortage. Write the availability statements accordingly, and
   note FDA 483s, FAERS, and the FDA inspection database are publicly available while the Redica
   Systems corpus is licensed.

## 8. Deliverable

Update `Paper/Text Analysis/manuscript_483_text_patient_harm.tex` in place. Keep the existing
LaTeX preamble and the `\todo{}` command. Use `\todo{}` for anything you cannot resolve from the
files. Produce every table from the CSVs rather than from this prompt, so the manuscript stays
reproducible, and say in a comment which file each table came from.

When you finish, output a short list of: every `[PENDING]` and `\todo{}` you left, every place
the data contradicted something in the old draft, and any analysis you think the paper needs but
that has not been run.
