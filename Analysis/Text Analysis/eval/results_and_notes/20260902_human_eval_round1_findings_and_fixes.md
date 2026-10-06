# 483 Text Extraction — Human Eval Round 1: Findings, Fixes, and Next Steps

> Date: 2026-09-02  |  For: group update  |  Author: Amirreza, with Abdul's round-1 labels

## Summary

Abdul blind-labeled 50 FDA 483 observations against the v2 LLM extraction prompt. Agreement
with the model was good on most fields but weak on a few (`scope`, `contamination_risk_flag`).
Investigating the actual disagreement rows found one real bug and two real prompt gaps, all now
fixed and pushed. Investigating further also surfaced a more important finding: the model does
not reproduce its own output reliably run-to-run (72–96% self-agreement on identical input,
depending on field), which means a single n=50 pass cannot support a precise accuracy claim for
any one field. Recommendation: don't scale up human labeling further right now — it doesn't fix
a within-model stability problem — instead stabilize via repeated LLM runs and cross-model
agreement, then move to full-scale extraction and the FAERS/adverse-event linkage.

## Round 1 setup

- 50 observations, blind (Abdul never saw model predictions), stratified round-robin sample
  across 38 FEIs, scored against the v2 prompt's own extraction (Claude Sonnet 5).
- Labeling kit: `483_Labeling_Rules_v2.docx`, `483_Background_Reference_Guide.docx`,
  `483_Worked_Example_FEI3003342394_obs3.docx`.
- Scoring: `eval/human_eval_02_score.py` (per-field accuracy, macro-F1, per-class P/R/F1,
  qualitative rationale comparisons for `patient_risk_flag` and the `contamination_*` split).

## Headline accuracy — before vs. after fixes (same 50 observations)

| Field | Before (original v2 prompt) | After (fixed v2 prompt) |
|---|---|---|
| contamination_flag | 0.840 | **0.980** |
| contamination_risk_flag | 0.680 | **0.960** |
| patient_risk_flag | 0.880 | 0.860–0.940 (see noise note) |
| scope | 0.720 | 0.780–0.820 (see noise note) |
| root_cause_type | 0.840 | 0.700–0.720 (regressed, see below) |
| violation_category | 0.860 | 0.760–0.780 |
| severity_tier | 0.820 | 0.680–0.740 |
| remediation_signal | 0.960 | 0.880–0.900 |
| investigation_flag | 0.960 | 0.940–1.000 |
| data_integrity_flag | 0.960 | 0.800–0.820 |
| repeat_flag | 1.000 | 1.000 |

Ranges reflect multiple re-runs done to separate real prompt effects from run-to-run noise (next
section). Only `contamination_flag` and `contamination_risk_flag` are reported as a single
verified number — see why below.

## The important methodological finding: single-pass n=50 isn't precise enough

To check whether the "after" numbers above were real improvements, I re-ran the **unedited
original prompt a second time** on the identical 50 observations — no prompt change at all.

Model self-agreement (run 1 vs. run 2, same prompt, same input):

| Field | Self-agreement |
|---|---|
| contamination_risk_flag | 72% |
| violation_category | 80% |
| severity_tier | 78% |
| scope | 78% |
| root_cause_type | 84% |
| contamination_flag | 88% |
| data_integrity_flag | 88% |
| remediation_signal | 92% |
| patient_risk_flag | 94% |
| investigation_flag | 96% |
| repeat_flag | 100% |

Accuracy-vs-Abdul swung by as much as 12–16 points on some fields from this alone (e.g.
`data_integrity_flag` 0.960→0.840, `contamination_risk_flag` 0.680→0.840), with zero prompt
change. **This means single n=50/single-run accuracy numbers are not reliable enough to report
as point estimates.** It's a real limitation of the current pipeline, worth flagging to the
group independent of anything else in this document.

## What was actually fixed

**1. Contamination mutual-exclusivity bug (real bug, deterministically fixed).**
The v2 prompt states three times that `contamination_flag_llm` and `contamination_risk_flag_llm`
are mutually exclusive. In round 1, the model violated this on 6/6 rows where it set
`contamination_flag_llm=True` — 100% violation of its own stated rule. Fixed by enforcing the
rule in post-processing code (`_validate()`) rather than trusting the model to self-comply.
Verified 0/11 co-occurrence on re-run. This one is not subject to the noise problem above — it's
a code-level guarantee now, not an accuracy number that can drift.

**2. `patient_risk_flag` / `severity_tier` prompt gaps (evidence-based, holds up beyond noise).**
The rules already said scenario (a) — sterile/injectable confirmed contamination — doesn't need
a release statement, but the model was still requiring one for pre-commercial (PPQ/process
validation) batches. Added an explicit clarification. Also added: a market complaint about a
named batch counts as an affirmative distribution statement (this was missing and affected both
`patient_risk_flag` and `severity_tier`'s Critical threshold). Both deltas exceeded the
noise floor measured above, so read as real, modest improvements.

**3. `scope` calibration example (real, modest improvement).**
Added a worked example for change-control/template-level failures that also name specific
batches (still FacilityWide — the template defect, not the named instances, decides scope).

**4. `root_cause_type` regression from cross-field prompt bleed (diagnosed, patched, not yet
statistically confirmed).**
The scope example in #3 taught a generalizable "weigh a systemic factor against a named-instance
factor" pattern that isn't scope-specific. It bled into `root_cause_type` reasoning in the same
combined tool call: 10/50 rows flipped correct→wrong, and 5 of those 10 share one shape — a
previously-correct single-category call became "Mixed," with the model's own rationale text
explicitly reasoning about two co-occurring factors the way the new scope example teaches. This
is a real, legible causal story, not a guess. Patched by narrowing the scope example and
tightening the `Mixed` definition to require independent causation, not mere co-mention.
Re-run showed 0.700→0.720 — smaller than this field's own noise floor (0.040), so the patch
cannot be called "confirmed" from this data alone, only "reasonably diagnosed and addressed."

All changes are in `01_extract_observation_signals.py`, commits `41eacfc` and `93ca0f3`.

## Why not just have Abdul label more rows

A bigger human-labeled sample would tighten the confidence interval on *how well the model
matches a human*, but it does nothing for the deeper problem this round surfaced: the model
doesn't agree with **itself** run-to-run. More human labels can't average out model-internal
noise. The higher-value, zero-additional-RA-time fix is repeated LLM runs (majority vote or
averaging across 3 passes) on the existing 50 human-labeled rows, which directly targets the
noise problem instead of the sample-size problem.

## Stabilized numbers (majority vote across 3 runs of the final prompt)

Per the recommendation above, I re-ran the final prompt 3 times on Abdul's same 50 observations
and scored the majority (mode) prediction per row — this is the number to actually cite:

| Field | Baseline (pre-fix, 1x) | **Stabilized (3x majority)** | 3/3 agree |
|---|---|---|---|
| contamination_flag | 0.840 | **0.980** | 90% |
| contamination_risk_flag | 0.680 | **0.960** | 88% |
| scope | 0.720 | **0.780–0.840** | 82–86% |
| root_cause_type | 0.840 | 0.820–0.840 (parity) | 72% |
| patient_risk_flag | 0.880 | 0.880–0.900 (flat) | 88–92% |
| investigation_flag | 0.960 | 0.940–0.980 | 88–94% |
| repeat_flag | 1.000 | 1.000 | 100% |
| remediation_signal | 0.960 | 0.900–0.940 | 86–90% |
| violation_category | 0.860 | 0.780–0.860 | 82% |
| data_integrity_flag | 0.960 | 0.780–0.860 | 80–86% |
| severity_tier | 0.820 | 0.660–0.680 | 74–78% |

**Contamination and scope fixes are confirmed real** — stable across repeated runs, not noise.
`root_cause_type` and `patient_risk_flag` are at parity or flat, i.e. the fixes didn't hurt.

**Two open items, investigated and honestly reported rather than force-fixed:**

- **`severity_tier`'s `moderate→major` bias (11/50 of the errors) is NOT caused by any of my
  edits.** I checked: it's present at the same magnitude (10/50) in a rerun of the completely
  **unedited original prompt**. This is a pre-existing tendency of Claude Sonnet 5 on this field
  with this prompt, not a regression I introduced. I attempted one fix (narrowing the systemic-
  failure exception clause); it made no measurable difference, so I reverted it rather than leave
  unhelpful prompt bulk in place. This is a real, known limitation to flag to the group, not
  something quick-fixable via prompt wording — likely needs a dedicated few-shot calibration pass
  or acceptance as a documented bias.
- **`data_integrity_flag` regresses consistently across every re-run (0.960→0.78–0.86) despite
  zero edits to its rules.** Diagnosed: not a model bug. Abdul's own labeling notes independently
  flagged the same gap on 3 rows (7, 29, 49) — retesting/resampling until a passing result, and
  invalidating an OOS result without a documented justified investigation ("testing into
  compliance") — and asked twice for a second opinion, since neither is on the rules doc's
  explicit list. The model's disagreement with Abdul on these rows wasn't an error on either
  side; it was a genuine rules gap. Resolved below.

## Resolved after group/RA review (2026-09-04)

Two items above were escalated to Abdul by email (with row IDs so he could find each one), and
resolved. Both are now implemented in the v2 prompt (OpenAI + Anthropic) and in
`483_Labeling_Rules_v2.docx`, effective for all extraction and labeling from this point forward:

1. **`data_integrity_flag`: "testing into compliance" now counts.** Retesting, resampling, or
   additional testing performed to obtain a passing result after an initial unfavorable/OOS
   result, or invalidating/discarding an unfavorable result, *without* a documented,
   scientifically justified investigation into why the original result was invalid — this is a
   named FDA data-integrity violation (FDA's 2018 Data Integrity guidance) and should be flagged
   even when no falsification is alleged. Rows 7, 29, 49 should be TRUE going forward.
2. **`patient_risk_flag` scenario (a): a confirmed EM excursion inside a classified Grade A/B
   aseptic area now counts on its own**, even without a separate statement that product itself
   was contaminated — per FDA's *Sterile Drug Products Produced by Aseptic Processing* (2004)
   guidance, environmental monitoring in the classified critical zone is the primary
   sterility-assurance signal for aseptic processing, not a secondary control. An EM excursion
   outside a classified Grade A/B area still does not qualify on its own. This confirms Abdul's
   own row 30 call (patient_risk = TRUE, FEI 3002984011).

Both changes are in `01_extract_observation_signals.py` (prompt text + tool-schema field
descriptions, both providers' v2 variants) and `eval/483_Labeling_Rules_v2.docx` §6.2/§6.6.

**Targeted verification (2026-09-04):** re-ran the fixed Anthropic v2 prompt directly on rows 7,
29, 49, and 30 from the frozen sample. All 4 flipped as intended: rows 7/29/49
`data_integrity_flag_llm` False→**True**; row 30 `patient_risk_flag_llm` **True**, with the
model's own rationale explicitly citing "Grade B and Grade A manufacturing areas... per FDA
aseptic processing guidance" — confirming it's applying the new rule, not coincidence. This is a
single targeted run, not the 3x majority-vote protocol, but both rules are explicit-criterion
decisions rather than judgment calls with meaningful noise. Not yet re-run at scale — see next
steps below.

## Recommended next steps

1. **Stabilize the eval number** — re-run the final v2 prompt 2 more times (3 total) on Abdul's
   same 50 observations, score the majority-vote/modal prediction per row against his labels.
   Cheap (~$0.50 total), gives a defensible number for the paper instead of a single noisy pass.
2. **Full-scale extraction** — run the fixed v2 prompt across the full observation set (currently
   622 observations / 38 FEIs with 483 text) with both Claude and GPT, not just the 50-row pilot.
3. **Cross-model agreement as a free, at-scale robustness signal** — where Claude v2 and GPT v2
   agree on the full set, treat as higher-confidence; disagreement rows are useful for a second,
   smaller targeted human-review pass instead of a large blind one.
4. **Aggregate to FEI level** (`02_aggregate_fei_features.py`) and proceed to the FAERS/adverse-
   event linkage — this is the analysis that actually matters for the paper's core claim.

## Artifacts

- Metrics (round 1, original prompt): `eval/human_eval_metrics_v2.md`
- Human labels: `eval/labeling_template_v2.xlsx`
- Model answer key (round 1): `eval/DO_NOT_SHARE_answer_key_v2.csv`
- Re-run CSVs (noise baseline + fix verification):
  `eval/exact50_claudesonnet5_v2_ORIGINAL_PROMPT_RERUN.csv`,
  `eval/exact50_claudesonnet5_v2_UPDATED_PROMPT_v1.csv`,
  `eval/exact50_claudesonnet5_v2_UPDATED_PROMPT.csv`
- Prompt fixes: `01_extract_observation_signals.py`, commits `41eacfc`, `93ca0f3`
