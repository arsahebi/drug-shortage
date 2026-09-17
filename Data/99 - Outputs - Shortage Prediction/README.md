# Drug Shortage Prediction / Quality-to-Impact Modeling

**Drug Shortage Prediction Project**
Last updated: 2026-09-16

## What this folder does

Takes the quality/regulatory signals built elsewhere in the project (Redica inspection
events, FDA recalls, FAERS adverse events, the 483-text LLM signals from
`99 - Outputs - Text Analysis/`) and models their **aggregated impact**: do these
signals predict recall, adverse-event, or shortage risk at the facility level. This is
the "impact" layer — `fei_inspection_explorer/` (in `99 - Outputs - Dashboards/`) is the
*history/exploration* layer for the same underlying facilities and drugs; this folder
answers "does it predict anything," not "what happened."

## Current pipeline (facility-year, current)

| Script | Purpose |
|---|---|
| `config.py` | Paths and panel parameters — single source of truth |
| `utils.py` | Shared helpers |
| `m14_recall_fei_model.py` | FEI × year panel (2015–2024); predicts whether a recall occurs at that facility in year t+1, from Redica inspection features + 483 text/LLM features + structural features |
| `m17_faers_fei_model.py` | Same panel/feature design; predicts above-median serious-AE volume in year t+1 |
| `m19_shortage_fei_model.py` | Same panel/feature design (FDA Inspection Details as the OAI/VAI source instead of Redica); predicts shortage exposure in year t+1, plus an OAI-vs-VAI descriptive check and a VAI-only text-signal subgroup model |
| `m15_recall_dashboard.py` | Interactive HTML story dashboard for the recall model (`outputs/figures/recall_fei_dashboard.html`) |
| `m18_faers_dashboard.py` | Interactive HTML story dashboard for the AE model (`outputs/figures/faers_fei_dashboard.html`) |

Run: `cd code && python3 m14_recall_fei_model.py && python3 m17_faers_fei_model.py && python3 m19_shortage_fei_model.py`,
then `python3 m15_recall_dashboard.py && python3 m18_faers_dashboard.py` to refresh
the dashboards from the model output. (No dashboard exists yet for m19.)

Each of the three models now checks and runs **two populations independently**: a
**baseline** (inspection + structural features only, every facility with a valid
outcome, not restricted to text coverage) and a **with-text** model (the full feature
set, naturally restricted to facilities with a real as-of-year 483-text snapshot).
Earlier versions restricted the whole panel to text-covered facilities even for the
baseline, which discarded real inspection-based signal for no reason related to text
availability — fixed 2026-09-16, see `RESULTS.docx`.

### Current results

**See `RESULTS.docx` for the full explanation.** Short version, as of 2026-09-16:
`m14`'s with-text model still can't run (2 events), but its baseline can now be
attempted for the first time (1,250 rows / 125 facilities / 20 events) and scores
below random (AUC 0.26–0.34) — a real, if discouraging, result. `m17`'s baseline
(1,113 rows / 125 facilities) lands almost exactly where its with-text model does
(AUC ~0.56–0.58 either way), confirming "no measurable text lift in this cut" on a
much larger sample than before. `m19` (new) is the one clear positive: its baseline
reaches AUC 0.54–0.61 on the full 127-facility universe, well above its own
with-text (0.49–0.50) and VAI-only (0.51–0.59) models — real signal that was
invisible in every restricted-population version of this pipeline. `RESULTS.docx`
has the full tables and reasoning.

## Folder structure

```
99 - Outputs - Shortage Prediction/
├── README.md     ← pipeline / folder structure (you are here)
├── RESULTS.docx  ← what the models found, in plain language
├── code/
│   ├── config.py, utils.py, m14, m15, m17, m18, m19   ← current
│   └── old_not_current_pipeline/
│       ├── gen1_drug_year_pipeline/   ← see "Archived: Generation 1" below
│       └── gen2_fei_exploration_abandoned/
├── data/            ← current: faers_fei_panel.csv, recall_fei_panel.{csv,parquet},
│                        shortage_fei_panel.{csv,parquet}, redica_fei_year.{csv,parquet};
│                        + old_not_current_pipeline/
├── outputs/
│   ├── figures/     ← current: recall_fei_dashboard.html, faers_fei_dashboard.html,
│   │                    roc/feature-importance/lift PNGs (no dashboard or figures for
│   │                    m19 yet); + old_not_current_pipeline/
│   ├── models/      ← current: metrics_*_baseline.csv (all 3 models), metrics_faers_fei.csv,
│   │                    metrics_shortage_fei.csv, metrics_shortage_vai_only.csv,
│   │                    rf_importance/text_ablation *_faers_fei.csv,
│   │                    rf_importance_recall_fei.csv; + old_not_current_pipeline/
│   └── tables/      ← current: *_fei_panel_summary.md, oai_vs_vai_shortage_rates.csv;
│                        + old_not_current_pipeline/
└── logs/            ← current: m14/m15/m17/m18/m19 logs; + old_not_current_pipeline/
```

## Archived: Generation 1 — drug-year model (`code/old_not_current_pipeline/gen1_drug_year_pipeline/`)

`m01`–`m10`, `mm01`–`mm07`, `main.py`, `main_monthly.py`. The original approach:
predict UUtah shortage onset per *drug*-year (1,988 drugs) from FAERS/recall/
Valisure/Redica signals joined through a drug-name map. Last run May 13 – June 10,
2026 — before any of the September text-signal validation work existed. Superseded
by the facility-year approach above, largely because the drug-level Redica join was
too thin to use text signals well (its own analysis found it resolved to only 126
facilities / 14 drugs). Its results (AUC ≈0.695 broader-panel L2 logit) are a
historical artifact, not current.

## Archived: Generation 2 — abandoned FEI exploration (`code/old_not_current_pipeline/gen2_fei_exploration_abandoned/`)

`m11_fei_timeline`, `m12_text_signal_grid`, `m13_case_study_timeline`,
`m16_fei_dashboard` — an earlier attempt at facility-level modeling from June 2026,
self-archived the same month it was built. Its orphaned outputs (per-FEI timeline
PNGs, `text_signal_grid.csv`, etc.) are archived alongside it in the matching
`old_not_current_pipeline/` subfolder under `data/`, `outputs/*/`, and `logs/`.

## Limitations

1. **All three with-text models are sample-size constrained** now that zero-filling is
   gone — see `RESULTS.docx`. Revisit as scored-inspection history extends further back
   or accumulates more calendar time. The baseline models added 2026-09-16 (inspection
   + structural only, full facility universe) sidestep this for the features that don't
   need text, but m14's baseline itself scores below random on the current sample.
2. **m19's baseline finding (AUC up to 0.61) now has a significance test** (one-tailed
   t-test of fold-level AUCs vs. 0.5, same convention as the VAI-only text-signal work)
   and clears it: p=0.003 (logistic), p=0.011 (Random Forest). It still hasn't had the
   repeated-cohort, multiple-rerun scrutiny the VAI-only comparison has had over several
   sessions — treat it as a real first signal, not yet as battle-tested as that one.
3. **SDUD volume weighting not incorporated** — see the TODO block in `config.py`.
