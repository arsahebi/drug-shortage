# Facility-Year Model Results

**What this document is:** a plain explanation of what `m14`/`m17` actually found, for
someone who wants the outcome without reading the modeling code. See `README.md` for
pipeline/folder structure. Last run: 2026-09-15, after fixing the methodology gap
described below.

## The question

For each FDA-registered facility (FEI) manufacturing one of the 14 Valisure-tested
APIs, in each year from 2015–2024: do this year's regulatory/quality signals
(Redica inspection history + 483-text LLM signals) predict whether that facility has
a recall (`m14`) or above-median serious adverse-event volume (`m17`) *next* year?

## The fix made today: no zero-filling

Earlier versions of this model ran on all 125 FEIs with Redica inspection history,
filling in `0` for the 27 FEIs (and for facility-years before a given FEI's first
scored inspection) that don't actually have 483-text signals yet. A `0` in every
text-feature column reads as "confirmed clean" to the model — but for those rows it
actually means "we don't know." That's not a neutral default; it dilutes whatever
real signal the text features carry.

Fixed today, in both `m14_recall_fei_model.py` and `m17_faers_fei_model.py`:
1. **Facility universe restricted** to the 98 FEIs that have Redica 483-text
   coverage at all (not just Redica inspection history, which covers 125).
2. **No zero-fill for text features.** A facility-year is only modeled if it has an
   actual "as-of-year" text snapshot (a scored inspection on or before Dec 31 of that
   year). Rows without one are dropped, not zeroed. Non-text features (inspection
   counts, structural flags) keep their zero-fill, since a facility genuinely having
   zero inspections in a given year is a real zero, not missing data.

This is a materially stricter requirement than "the facility has ever been scored" —
most of a facility's early panel-years still get dropped if its first Redica-scored
inspection came late in the 2015–2024 window. That's why the modeled sample shrinks
so much below.

## Recall model (`m14`): not currently viable

| | Full panel (pre-fix, for reference) | After fix |
|---|---|---|
| Facilities | 125 | 98 |
| Rows with a valid outcome | 1,250 | 980 |
| Rows with an actual text snapshot | — | **177** (51 FEIs) |
| Recall events in that modeled set | 20 | **2** |

Two positive events cannot support cross-validated modeling of anything — the script
now detects this and exits without producing metrics, ROC curves, or a feature-
importance ranking (previously it silently ran on the diluted 1,250-row panel and
reported AUC ≈ 0.53, which should not be read as a real result either way). This
isn't a bug to fix; it's what honestly restricting to real data looks like when the
outcome is this rare and text coverage this recent. The recall dashboard
(`outputs/figures/recall_fei_dashboard.html`) now shows this status directly instead
of stale numbers.

**What would fix this:** more calendar time (more facilities cross into having an
early-enough Redica snapshot), or extending Redica's scored-inspection history
further back for already-covered facilities.

## Adverse-event model (`m17`): runs, but text shows no measurable lift here

| | Full panel (pre-fix, for reference) | After fix |
|---|---|---|
| Facilities | 125 | 98 |
| Rows modeled | 1,113 | **148** (46 FEIs) |
| AE-high events | 684 (61.5%) | **119** (80.4% of the modeled rows) |

| Model | AUC (pre-fix) | AUC (after fix) |
|---|---|---|
| L2 Logit | 0.647 | **0.550** |
| Random Forest | 0.650 | **0.582** |

**Text-feature ablation (L2 logit, after fix):**

| | AUC |
|---|---|
| Without text (inspection + structural only) | 0.568 |
| With text | 0.550 |

Under the honest restriction, adding 483-text features does not improve AUC over
inspection/structural features alone — if anything it's very slightly lower. Read
this as **"no measurable lift in this cut," not "text signals don't work."** n=148
across 46 facilities is a small, noisy sample for 5-fold GroupKFold CV; a few
mis-ranked facilities swing AUC substantially at this size. This also should **not**
be read as contradicting the separate VAI-classification finding from the Text
Analysis validation session (AUC 0.666, p=0.0001) — that's a different target
(OAI/VAI regulatory classification, not next-year AE volume) on a different subset
(pure-VAI facilities), not the same claim.

**Random Forest feature importance (top 5, full-feature model):**

| Feature | Importance |
|---|---|
| `n_feis_drug` (supply concentration — structural) | 0.117 |
| `investigation_llm_share` (text) | 0.106 |
| `data_integrity_llm_share` (text) | 0.084 |
| `severity_critmajor_share` (text) | 0.078 |
| `remediation_none_share` (text) | 0.070 |

Text features still dominate the RF importance ranking even though the ablation AUC
didn't move — RF importance reflects how much a feature is used to split, not
necessarily how much it improves held-out ranking, especially at this sample size.
Full ranking: `outputs/models/rf_importance_faers_fei.csv`.

## Bottom line

1. The corrected, honest version of this analysis is a smaller, harder result than
   what was reported before today: recall can't currently be modeled at all, and the
   AE model shows no clear text-feature lift once zero-filling is removed.
2. This isn't a reason to distrust the underlying 483-text signals generally — it's
   specific to this facility-year, next-year-AE framing, at current data volume.
   Sample size is the binding constraint on both models, not signal quality.
3. Revisit both once Redica's scored-inspection history either extends further back
   or accumulates more calendar time going forward.

## Where the underlying numbers live

- `outputs/tables/faers_fei_panel_summary.md`, `outputs/tables/recall_fei_panel_summary.md`
- `outputs/models/metrics_faers_fei.csv`, `rf_importance_faers_fei.csv`, `text_ablation_faers_fei.csv`
- `outputs/figures/faers_fei_dashboard.html`, `recall_fei_dashboard.html` — interactive versions of the above, plus supply-chain/case-study context
