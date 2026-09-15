# Drug Shortage Prediction / Quality-to-Impact Modeling

**Drug Shortage Prediction Project**
Last updated: 2026-09-15

## What this folder does

Takes the quality/regulatory signals built elsewhere in the project (Redica inspection
events, FDA recalls, FAERS adverse events, the 483-text LLM signals from
`99 - Outputs - Text Analysis/`) and models their **aggregated impact**: do these
signals predict recall or adverse-event risk at the facility level. This is the
"impact" layer — `fei_inspection_explorer/` (in `99 - Outputs - Dashboards/`) is the
*history/exploration* layer for the same underlying facilities and drugs; this folder
answers "does it predict anything," not "what happened."

## Current pipeline (facility-year, current)

| Script | Purpose |
|---|---|
| `config.py` | Paths and panel parameters — single source of truth |
| `utils.py` | Shared helpers |
| `m14_recall_fei_model.py` | FEI × year panel (2015–2024); predicts whether a recall occurs at that facility in year t+1, from Redica inspection features + 483 text/LLM features + structural features |
| `m17_faers_fei_model.py` | Same panel/feature design; predicts above-median serious-AE volume in year t+1 |
| `m15_recall_dashboard.py` | Interactive HTML story dashboard for the recall model (`outputs/figures/recall_fei_dashboard.html`) |
| `m18_faers_dashboard.py` | Interactive HTML story dashboard for the AE model (`outputs/figures/faers_fei_dashboard.html`) |

Run: `cd code && python3 m14_recall_fei_model.py && python3 m17_faers_fei_model.py`,
then `python3 m15_recall_dashboard.py && python3 m18_faers_dashboard.py` to refresh
the dashboards from the model output.

### Current results

**See `RESULTS.md` for the full explanation.** Short version: as of 2026-09-15,
`m14`/`m17` restrict modeling to facilities with actual Redica 483-text coverage and
no longer zero-fill missing text features (previously they ran on the full 125-FEI
universe and zero-filled the 27 FEIs without text coverage, which diluted the real
signal — see `RESULTS.md` for how that was found and fixed). Under the honest
restriction: the recall model (`m14`) no longer has enough events to model at all
(2, down from 20 once non-snapshot rows are dropped); the AE model (`m17`) runs on
148 rows / 46 FEIs and shows no measurable AUC lift from text features in this cut
(0.550–0.582 with text vs. 0.568 without, L2/RF) — smaller and more honest numbers
than what was reported before this fix, not a reason to distrust the underlying text
signals generally. `RESULTS.md` has the full tables and the reasoning for why.

## Folder structure

```
99 - Outputs - Shortage Prediction/
├── README.md        ← pipeline / folder structure (you are here)
├── RESULTS.md        ← what the models found, in plain language
├── code/
│   ├── config.py, utils.py, m14, m15, m17, m18   ← current
│   └── old_not_current_pipeline/
│       ├── gen1_drug_year_pipeline/   ← see "Archived: Generation 1" below
│       └── gen2_fei_exploration_abandoned/
├── data/            ← current: faers_fei_panel.csv, recall_fei_panel.{csv,parquet},
│                        redica_fei_year.{csv,parquet}; + old_not_current_pipeline/
├── outputs/
│   ├── figures/     ← current: recall_fei_dashboard.html, faers_fei_dashboard.html,
│   │                    roc/feature-importance/lift PNGs (AE model only — recall's
│   │                    aren't produced, see RESULTS.md); + old_not_current_pipeline/
│   ├── models/      ← current: metrics/rf_importance/text_ablation *_faers_fei.csv
│   │                    (AE model only); + old_not_current_pipeline/
│   └── tables/      ← current: *_fei_panel_summary.md; + old_not_current_pipeline/
└── logs/            ← current: m14/m15/m17/m18 logs; + old_not_current_pipeline/
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

1. **Both models are sample-size constrained** now that zero-filling is gone — see
   `RESULTS.md`. Revisit as Redica's scored-inspection history extends further back
   or accumulates more calendar time.
2. **SDUD volume weighting not incorporated** — see the TODO block in `config.py`.
