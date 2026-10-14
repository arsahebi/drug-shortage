# Text analysis, next session brief (October 14, 2026)

Paste the section below into a new Claude Code thread.

---

Continuing the 483 text analysis. Two datasets arrived today and they change what to work on. Read `CLAUDE.md`, `Analysis/Text Analysis/README.md`, and `Analysis/Text Analysis/eval/results_and_notes/20261006_patient_outcome_research_review.md` before touching anything. The last three commits on this workstream are `6bcd1e0`, `478da36` and the ones they reference.

## New data 1: Abdul's round-2 labels

`Analysis/Text Analysis/eval/sent_to_abdul/labeling_template_round2_done.xlsx`, three sheets: Instructions, Labeling, AI_Revision_Log.

- Labeling: 70 observations, all 11 fields filled.
- Key: `eval/validation_data/DO_NOT_SHARE_round2_key.csv`, 70 rows. `source` is 50 new and 20 repeat, `round1_row_id` identifies the repeats, `stratum` gives the sampling cell (20 random_natural, 8 remediation_nonblank, 8 violation_rare, 8 severity_Minor, 3 scope_Unclear, 3 rootcause_Unclear, 20 repeat).
- **AI_Revision_Log records 33 cells across 14 of the 70 rows that an AI model revised after Abdul submitted**, with original_value, revised_value and a reason for each. Most affected: contamination_why (6), contamination_risk_flag (4), severity_tier (4), data_integrity_flag (3).

This matters because round 2 is the paper's held-out human validation. Labels an AI rewrote are not independent human judgment, and scoring the LLM against them is circular.

**Required approach:** reconstruct the as-submitted labels by reverting the 33 logged cells, and make that the primary scored set. Report the AI-revised version only as a secondary sensitivity, labelled as such. If the two disagree materially, the as-submitted numbers are the ones that go in the paper. Do not quietly use the file as delivered.

**What to produce:**
1. Agreement of the Claude v2 extraction against the as-submitted round-2 labels, per field, with Wilson 95% intervals. Reuse the round-1 scorer at `eval/code/04_score_models_vs_human.py` or `eval/code/human_eval_02_score.py` rather than writing a new one; check which one produced `results_and_notes/models_vs_human_agreement.csv`.
2. Test-retest reliability from the 20 repeats: Abdul against his own round-1 labels on the same observations, per field. This is the human ceiling, and the paper needs it beside every LLM accuracy figure. Say plainly that it measures consistency, not correctness.
3. The same comparison restricted to the 50 genuinely new observations, since the repeats were chosen for prior coverage and are not a random sample.
4. A short note in `eval/results_and_notes/` recording the AI-revision problem, how many cells, and which fields.

Round 1 is scored in `results_and_notes/human_eval_metrics_v2.md` and the prompt fixes from it are in v2, so round 1 cannot be the final validation. Round 2 can be, once the above is done.

## New data 2: MarketScan refresh with Nima's quality score

`Data/20 - Market Scan/raw/`, all four CSVs replaced today.

- The **only** change is one new column: `quality_score_v2` on the manufacturer files, `plant_quality_score` on the FEI files. Row counts and switch totals are byte-identical to the previous delivery (CCAE manufacturer file 21,821 rows, 14,790,526 switches).
- The score is a 0.03 to 1.0 percentile, higher means worse, constant within manufacturer by exact product (718 scored CCAE cells), non-null on 78% of CCAE and 84% of MDCR manufacturer rows. It correlates +0.39 with dose-change-only abandonment, which is the expected direction.
- `Description.docx` was not updated and does not document the score. The component ranks, evidence level and uncertainty range that were requested in `Data/20 - Market Scan/20261007_MarketScan_Followup_Request.docx` were not delivered.
- **None of the three open questions were answered.** `n_coverage_ended` still exceeds `n_stops` in 6.9% of CCAE manufacturer rows (the identical 1,511 rows), and the plant-file switch discrepancy is unchanged (Solco lisinopril: 605,434 switches in the manufacturer rows linked to plant 1038197, 2,079 in the plant file).

**Why this is good news anyway.** The abandonment numerator is still unusable, and `08_product_cell_analysis.py` now proves the old headline depended on it: data integrity is +11.0 points per SD (p = 0.0001) with the coverage subtraction, +10.5 clipped at zero, but **+6.3 (p = 0.12) with dose-change adjustment only**, and the quality-system reversals disappear entirely. See the numerator sensitivity block and `outputs/tables/product_cell_numerator_sensitivity.csv`.

The score sidesteps that. It is externally validated against Valisure and Department of Defense laboratory results (87% concordance, see `Data/20 - Market Scan/docs/Qual_Score_Explained.pdf`), and it sits at the labeler by exact product unit where the text signal appeared. So use the score as the outcome instead of rebuilding abandonment ourselves.

**What to build:** a new script in `vai_signal_validation/`, numbered after `10`, that tests the 483 text features against `quality_score_v2`.

- Unit: labeler by exact product (ingredient, route, form, strength), bridged to one finished-dose plant with the same bridge as `01b_build_marketscan_outcomes.py`.
- Outcome: the score. It is already a within-product percentile, so check whether it is ranked within product or across all products before deciding whether product fixed effects are still needed. Verify this empirically rather than assuming.
- Model: mirror `08_product_cell_analysis.py`. OLS with product fixed effects where appropriate, standard errors clustered by plant, Holm correction over the 17 fixed features, plus the plant-level Spearman as a robustness check.
- Report the 17 features, not a selected pair. Data integrity and lab controls were chosen after looking at the abandonment data, so on this outcome they are no longer privileged.
- Restrict to oral solids and drop the five acute APIs, as in `08`.
- Also report how many scored cells survive the plant bridge and the text-coverage restriction, since that will be the binding constraint.

Treat this as the cleanest available test of whether the text carries product-quality signal. If data integrity holds against an independently validated score, that is a real result. If it does not, that is the answer and the paper says so.

## Then, if there is time

The Oct 6 review lists five corrections. Two are tractable and affect the FAERS results that are currently in the manuscript:

- **FAERS counts report versions as separate events.** 61,838 distinct primary IDs but 55,225 distinct case IDs, about 6,613 excess versions. Deduplicating within a quarter misses a case updated in a later quarter. Build a case history across the quarterly files and count each case once.
- **The FAERS outcome is defined two ways.** The main model sums with missing quarters filled as zero; the fair FDA comparison uses means over available quarters. Eight of 143 labels flip between them, and the two scripts use 17 versus 14 features. Settle on one definition and one feature list, and state which.

The other three (circular single-facility attribution, inspection date versus defect onset, the MarketScan risk set) are larger and should not be started in the same session.

## Hard rules

1. Python is `/Users/asahebi/Projects/drug-shortage-research/venv/bin/python`, called by absolute path. Note that the project `.venv` has a different scikit-learn and gives different GroupKFold splits, so any AUC that gets reported comes from `09_repeated_cv_robustness.py` (100 random facility partitions), never a single split.
2. Do not rerun the LLM extraction, call any LLM API, or change the v2 prompt or schema. Abdul's labels are scored against that schema.
3. Commit and push after every code change, staging only the files for the task, no Co-Authored-By trailer.
4. Never delete files. Anything that should go moves to `_TO_DELETE_review/<date>_<topic>/` at the project root with a MANIFEST.md row.
5. Never invent a number. Every figure comes from a file in the repo, and scripts write their results to `outputs/tables/` so they are reproducible.
6. Write-ups meant for a person go in .docx, not markdown. No em dashes anywhere.
7. State n beside every statistic, and put uncertainty on every estimate.

## Where the paper stands

`Paper/Text Analysis/manuscript_483_text_patient_harm.tex` has not been revised since October 6 and still carries single-split AUCs and the old outcome-led title. The revision brief is `Paper/Text Analysis/20261006_codex_revision_prompt.md`, but **do not run it as written**: it instructs a full MarketScan abandonment subsection built on the broken numerator. It needs updating for the score-based analysis and the round-2 results first. `Paper/Text Analysis/generate_manuscript_tables.py` and its `generated_tables/` are still untracked and point at the retired `Data/99 - Outputs - Text Analysis` path.
