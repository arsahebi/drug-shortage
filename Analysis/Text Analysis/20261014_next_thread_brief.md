# Text analysis, next session brief (October 14, 2026)

Paste the section below into a new Claude Code thread.

---

Continuing the 483 text analysis. Two datasets arrived today and they change what to work on. Read `CLAUDE.md`, `Analysis/Text Analysis/README.md`, and `Analysis/Text Analysis/eval/results_and_notes/20261006_patient_outcome_research_review.md` before touching anything. The last three commits on this workstream are `6bcd1e0`, `478da36` and the ones they reference.

## New data 1: Abdel's round-2 labels (resolved, two files)

Abdel sent two workbooks, both in `Analysis/Text Analysis/eval/sent_to_abdel/`:

- **`labeling_template_round2_done_v02.xlsx` is his own labels, unassisted. Use this one.** Sheets: Instructions, Labeling.
- `labeling_template_round2_done.xlsx` is the same labels after he ran them through Claude as a self-check. Sheets: Instructions, Labeling, AI_Revision_Log.

He explained it himself: he labeled first, then ran a Claude check and kept both versions so we could see the difference and confirm his reasoning was not far off. **Verified, and his account reconciles exactly:**

- All 33 cells in the AI_Revision_Log hold the logged `original_value` in v02, 33 of 33, no mismatches.
- A full cell-by-cell diff of the two Labeling tabs shows exactly 33 differing cells, precisely the 33 in the log. Nothing changed that is not logged, and nothing logged failed to change.
- `fei`, `obs_num` and `observation_text` are identical across both files on all 70 rows.

So there is a clean human set and a genuine AI-assisted sensitivity version, and the log reconciles them. No reconstruction is needed and there is no need to ask Abdel to relabel anything.

**Key details:**
- 70 observations. The key at `eval/validation_data/DO_NOT_SHARE_round2_key.csv` gives `source` (50 new, 20 repeat), `round1_row_id` for the repeats, and `stratum` (20 random_natural, 8 remediation_nonblank, 8 violation_rare, 8 severity_Minor, 3 scope_Unclear, 3 rootcause_Unclear, 20 repeat).
- `human_remediation_signal` is filled on 2 of 70 rows, both "Partial". Abdel confirmed blank means he observed nothing about remediation. That is the intended convention and matches the LLM, which leaves the field blank on 912 of 1,067 observations. Treat blank as a value, not as missing.
- Direction of the AI edits, for the sensitivity write-up: of the 24 edits to categorical or flag fields, 18 moved his label to exactly the Claude v2 value, 2 moved away, 4 landed on neither. Pooled agreement rises from 79.9% with his own labels to about 82% with the assisted ones. Report the clean number as primary and this as the sensitivity, and note that an edit made by the model family under evaluation cannot count as independent human judgment even where the rule reading is correct.

**What to produce:**
1. Per-field agreement of the Claude v2 extraction against v02, with Wilson 95% intervals, using the existing scorer (`eval/code/04_score_models_vs_human.py` or `eval/code/human_eval_02_score.py`; check which produced `results_and_notes/models_vs_human_agreement.csv`) rather than a new one. An ad-hoc check on 2026-10-14 gave pooled 79.9% (77 to 83) over 11 fields and 79.5% (76 to 83) on the 50 new observations alone, with severity_tier the weak field at 60.0% (48 to 71) and repeat_flag the strongest at 95.7% (88 to 99). Treat those as a sanity target, not as the result: the real scorer may map field values differently.
2. The same comparison on the 50 new observations separately, since the repeats were chosen for prior coverage and are not a random sample.
3. Test-retest reliability from the 20 repeats, Abdel against his own round-1 labels, per field. This is the human ceiling and the paper needs it beside every accuracy figure. **First locate the round-1 human labels**: `labels_from_abdel/20260902_round1_labels_abdel.xlsx` may be the blank template, and the round-1 results in `results_and_notes/human_eval_metrics_v2.md` and `models_vs_human_agreement.csv` may be the only surviving record. If the per-observation round-1 labels cannot be found, say so rather than improvising, and ask before proceeding.
4. The AI-assisted sensitivity from the v01 file, clearly labelled.
5. A short note in `eval/results_and_notes/` recording the two-file arrangement and the verification above, so the provenance is documented for the paper.

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
2. Do not rerun the LLM extraction, call any LLM API, or change the v2 prompt or schema. Abdel's labels are scored against that schema.
3. Commit and push after every code change, staging only the files for the task, no Co-Authored-By trailer.
4. Never delete files. Anything that should go moves to `_TO_DELETE_review/<date>_<topic>/` at the project root with a MANIFEST.md row.
5. Never invent a number. Every figure comes from a file in the repo, and scripts write their results to `outputs/tables/` so they are reproducible.
6. Write-ups meant for a person go in .docx, not markdown. No em dashes anywhere.
7. State n beside every statistic, and put uncertainty on every estimate.

## Where the paper stands

`Paper/Text Analysis/manuscript_483_text_patient_harm.tex` has not been revised since October 6 and still carries single-split AUCs and the old outcome-led title. The revision brief is `Paper/Text Analysis/20261006_codex_revision_prompt.md`, but **do not run it as written**: it instructs a full MarketScan abandonment subsection built on the broken numerator. It needs updating for the score-based analysis and the round-2 results first. `Paper/Text Analysis/generate_manuscript_tables.py` and its `generated_tables/` are still untracked and point at the retired `Data/99 - Outputs - Text Analysis` path.
