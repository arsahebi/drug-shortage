# Session Handoff — 483 Text Analysis / AE Prediction, 2026-09-09

> Purpose: paste this into a new thread to continue with full context. Written by
> Claude at the end of a multi-day session (2026-09-04 through 2026-09-09).

## Project structure (Text Analysis piece of the Drug Shortage project)

Base path: `Data/99 - Outputs - Text Analysis/`

```
01_extract_observation_signals.py   LLM extraction (v1 legacy + v2 active prompts,
                                     OpenAI + Anthropic variants). CLI: --source
                                     {pdf,redica} --provider {anthropic,openai}
                                     --model <id> --prompt-version {v1,v2}
02_aggregate_fei_features.py        Observation-level -> FEI-level time-series
                                     features (step01_*.csv -> step02_*.csv).
                                     Now has --model flag matching 01's naming.
07_merge_text_signals.py            Merges text signals with other feature sources
eval/                                Human-eval harness + all round-1 artifacts
  human_eval_01_generate_template.py
  human_eval_02_score.py
  labeling_template_v2.xlsx          Abdul's round-1 labels (50 rows, ground truth)
  483_Labeling_Rules_v2.docx         Labeling rules doc (kept in sync with prompt)
  20260902_human_eval_round1_findings_and_fixes.md   Group-facing findings doc
```

Output file naming (important, has bitten us twice): `01_extract...` tags outputs
with `_<modelslug>` only when `--model` overrides that provider's own default
(anthropic default = claude-haiku-4-5; openai default = gpt-5-mini). We always ran
the validated pipeline with `--model claude-sonnet-5`, so Claude files carry
`_claudesonnet5` in the name; GPT files (default model) don't carry a model tag,
which is a latent collision risk if anthropic is ever run with its own default —
not yet fixed, flagged as tech debt.

**Downstream consumer:** `Data/99 - Outputs - Shortage Prediction/code/`
- `config.py` — `TEXT_TIMESERIES_REDICA_CSV` must point at the current validated
  step02 file. It was stale for a long time (pointed at a file that didn't exist),
  silently starving the models of text features — fixed 2026-09-08.
- `m14_recall_fei_model.py`, `m17_faers_fei_model.py` — FEI x year panels, AE / recall
  prediction models (L2 logistic + Random Forest, GroupKFold by FEI). `TEXT_FEATURES`
  list must match the current step02 schema's column names exactly (v2 renamed
  violation-category columns to the FDA six-system scheme, e.g.
  `vc_labcontrols_share` -> `vc_laboratorycontrolssystem_share`).

**Manuscript:** `Paper/Text Analysis/manuscript_483_text_patient_harm.tex` — the
Results section is explicitly marked "PENDING, NOT CURRENT" (placeholder numbers
from an older pipeline). Not yet updated with any of this session's numbers.

**INFORMS slides (July 2026):** `Paper/Text Analysis/20260714_informs_slides.tex`.
Flagship claim: text-only AUC = 0.656 (p=0.046) separating VAI-only facilities by
future AE risk, vs. 0.585 full-sample / 0.545 for the FDA outcome flag alone. At the
time, extraction quality was flagged as "human validation in progress — treat as
preliminary." This session's work was largely that promised validation.

## What's been done, in order

1. **API key maintenance.** Environment's Anthropic key ran out of credit; replaced
   in `~/.zshenv` with a personal key (backed up the file first). Same for OpenAI
   later. Both verified live before use.

2. **Human eval round 1 validation** (Abdul blind-labeled n=50 against v2 prompt):
   - Found + fixed a real bug: contamination_flag / contamination_risk_flag were
     supposed to be mutually exclusive, weren't — fixed in `_validate()`, code-level
     guarantee now, verified 0/11 co-occurrence.
   - Found + fixed real prompt gaps: patient_risk (PPQ/pre-commercial batches,
     market-complaint-as-distribution), scope (template-level failure calibration).
   - Introduced + fixed a regression: the scope fix's reasoning pattern bled into
     root_cause_type ("Mixed" over-triggering) in the same combined tool call.
     Diagnosed via flip analysis, patched by tightening the Mixed definition.
   - Confirmed severity_tier's moderate->major bias is a genuine, pre-existing LLM
     tendency (present at equal magnitude in a rerun of the *unedited* original
     prompt) — not fixable by two different prompt-wording attempts, documented as
     a known limitation rather than silently dropped.
   - Established methodology: single n=50/single-run accuracy isn't precise enough
     (model self-agreement only 72-96% run-to-run on identical input) — switched to
     3x majority-vote as the reporting standard.
   - Two items required an actual regulatory judgment call, not just code: escalated
     to Abdul with FDA-guidance-grounded reasoning (FDA's 2018 Data Integrity
     guidance; FDA's 2004 aseptic processing guidance), resolved as decisions, then
     implemented in the live v2 prompt (both providers), the tool-schema field
     descriptions, and `483_Labeling_Rules_v2.docx`:
       - `data_integrity_flag`: "testing into compliance" (retesting/resampling or
         invalidating an unfavorable result without a documented justified
         investigation) now counts.
       - `patient_risk_flag` scenario (a): a confirmed EM excursion inside a
         classified Grade A/B aseptic area now counts on its own.
     Verified against the exact rows that surfaced each gap (rows 7, 29, 49 for DI;
     row 30 for patient risk) — all 4 flip correctly under the fixed prompt.

3. **Full-scale v2 extraction — complete, 0 errors.**
   - Claude Sonnet 5: 622 obs (pdf source, 38 FEIs) + 1,067 obs (redica source, 98
     FEIs).
   - GPT-5-mini: same two sources, same counts. Hit an OpenAI rate-limit wall
     running both GPT jobs in parallel once (killed by something external after a
     string of consecutive rate-limit errors); resumed successfully running them
     sequentially instead of in parallel.
   - `02_aggregate_fei_features.py` run on both Claude sources ->
     `step02_483_fei_text_features_timeseries_{fdapdf,redica}_claudesonnet5_v2.csv`
     (79 snapshots/37 FEIs; 246 snapshots/98 FEIs).

4. **Found + fixed a real downstream bug:** `m14`/`m17` had been silently running
   with **zero text features** because `config.py`'s path pointed at a file that no
   longer existed. Fixed the path, fixed a stale v1 column name
   (`vc_labcontrols_share` -> `vc_laboratorycontrolssystem_share` under v2's FDA
   six-system violation scheme), verified both models now load all 12 TEXT_FEATURES
   populated across 98 FEIs.

5. **Model results — each one checked, and corrected after user pushback:**
   - AE prediction lift: first pass compared against the *full* 125-FEI panel
     (non-Redica facilities zero-filled), diluting the result to +0.077 AUC. User
     caught this ("why include facilities not in Redica"). Redone restricted to the
     96 Redica-covered FEIs only: **0.564 -> 0.677 AUC** (L2), **0.547 -> 0.681**
     (RF) — the real, undiluted lift.
   - INFORMS flagship claim (text separates VAI-only facilities by future AE risk):
     first attempt tested this with univariate Spearman correlations per feature,
     found nothing significant, reported "claim hasn't reproduced." **This was
     methodologically wrong** — INFORMS's actual test was a combined multivariate
     logistic regression AUC via GroupKFold CV, not per-feature correlation. Redone
     to match exactly, restricted to the 51 "pure VAI" (never-OAI) Redica-covered
     FEIs: **AUC = 0.666, p = 0.0001** (vs. INFORMS's 0.656, p=0.046) — the claim
     *replicates*, more strongly than before. Verified stable across 10 random
     fold-shuffles (SD = 0.002, not a lucky split).
   - Recall model (`m14`): only 20 positive events across 1,250 FEI-years — too
     sparse to be reliable, explicitly not presented as a result.
   - "Two Failure Modes" correlation table (governance vs. technical failure,
     from the manuscript's placeholder table) recomputed on current data, n=96
     FEIs. After Bonferroni correction for 36 comparisons, only 2 of 12
     relationships are robust: `contamination_llm_share` and `remediation_none_share`
     both track OAI strongly. The rest (lab-controls/data-integrity -> AE) are
     directionally consistent with the paper's theory but only nominally
     significant (p 0.04-0.09), not yet confirmed. One real divergence from the old
     pipeline: contamination now correlates *positively* with OAI (+0.37, was -0.05
     in the old pipeline) — checked, not an inspection-frequency confound (partial
     correlation barely moves), most likely explained by v2's stricter
     confirmed-only contamination definition.

6. **Deliverables produced:**
   - `eval/20260902_human_eval_round1_findings_and_fixes.md` — updated 3x across
     the session, is the canonical write-up of the validation work.
   - Plain-text email to Abdul (delivered in chat, not saved as a file) — thanks
     him first, covers every flagged item with row IDs, states decisions (not
     open questions) for items 1-2 with FDA-guidance rationale.
   - `Presentation/2026-09-08-Update_Since_INFORMS.html` — 7-slide deck, matches
     INFORMS beamer color scheme (self-contained HTML, no LaTeX/pdflatex needed,
     since pdflatex isn't installed on this machine). Tells the "what changed
     since July" story: recap of INFORMS claims -> validation work done ->
     human-eval accuracy table -> AE lift (corrected, stronger) -> flagship VAI
     claim (corrected: replicated, not weakened) -> bottom line.
   - 7 commits pushed to `main` (see below).

## Commits this session (chronological)
```
41eacfc  Fix v2 483 extraction: contamination exclusivity + patient-risk/scope gaps
93ca0f3  Tighten v2 root_cause Mixed definition to guard against cross-field bleed
16798c2  Add human-eval round 1 findings and fixes summary for group update
493fafb  Add stabilized 3x-majority-vote metrics and severity_tier finding
a74c8cd  Add data_integrity testing-into-compliance rule and EM Grade A/B patient-risk rule
77b6d58  Add targeted verification results for data_integrity and patient_risk rule fixes
bd594a2  Wire validated v2 483-text features into AE/recall FEI models
```
(Slide deck and this handoff doc are not yet committed — .html/.md status TBD,
check `git status` in the new thread.)

## What's still open / next steps

1. **Cross-model agreement check** (Claude Sonnet 5 vs. GPT-5-mini, both fully
   extracted now) — not yet run. This was the planned "free, at-scale robustness
   signal": where both models agree, treat as higher confidence; disagreement rows
   become a smaller targeted human-review pass instead of a large blind one.
2. **Manuscript's placeholder Results section** — still says "PENDING, NOT
   CURRENT." None of this session's numbers (accuracy table, AE lift, VAI-AUC
   replication, two-failure-modes table) have been written in yet. Waiting on
   go-ahead before editing a multi-author paper draft.
3. **Two Failure Modes table** — only 2/12 relationships are Bonferroni-robust
   right now; the paper's core "technical failures -> harm, governance failures ->
   escalation" story is directionally present but underpowered at n=96 FEIs. Worth
   revisiting once/if facility coverage grows.
4. **The 483_Labeling_Rules_v2.docx and prompt changes** are live for *future*
   extraction and labeling; nothing retroactively re-labels Abdul's existing round-1
   answer key — that's fine, it was accounted for in the verification step, just
   worth remembering if a round 2 human-eval is planned.
5. **GPT pdf output filename collision risk** (see "Project structure" above,
   `01_extract_observation_signals.py`'s model-tagging logic) is unresolved tech
   debt, low urgency.

## Methodology lessons worth remembering (things that bit us this session)

- **Never mix Redica-covered and non-Redica facilities** in a "with text vs.
  without text" comparison without restricting to the covered subset first — the
  zero-fill dilutes real lift substantially (0.077 diluted vs. 0.113 real, in one
  case).
- **When trying to reproduce an old result, match the exact original test
  statistic**, not something that sounds equivalent. Univariate correlation and a
  multivariate CV-AUC test can give opposite-looking conclusions from the same
  data; assumed equivalence there led to a wrong "the claim didn't reproduce"
  conclusion that had to be walked back.
- **Apply multiple-testing correction before calling a correlation table
  "replicated."** Uncorrected p<0.05 across a dozen-plus comparisons produces
  spurious hits.
- **`claude-sonnet-5` must be passed explicitly** (`--model claude-sonnet-5`) for
  every run in this pipeline — the anthropic provider default is claude-haiku,
  which is not the validated model.
- Background long-running extraction jobs occasionally get killed by something
  external (never fully diagnosed — possibly sandbox/session lifecycle related);
  they're safe to just rerun since the script only processes rows not already in
  the output file.
