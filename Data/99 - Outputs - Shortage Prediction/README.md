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

### Current results (as currently coded — see caveat below)

| Target | Model | AUC | n (modeled) | Events |
|---|---|---|---|---|
| Recall next-year | L2 Logit | 0.465 | 1,250 | 20 |
| Recall next-year | RandomForest | 0.351 | 1,250 | 20 |
| AE-high next-year | L2 Logit | 0.647 | 1,113 | 684 |
| AE-high next-year | RandomForest | 0.650 | 1,113 | 684 |

(Rerun 2026-09-15 to confirm reproducibility. Recall's AUC swings noticeably run to
run — expected with only 20 positive events across `GroupKFold` splits, not a code
issue; it's exactly the sparsity problem noted below.)

**⚠️ Known methodology gap, not yet fixed in code:** these numbers run the model
across the full 125-FEI universe, zero-filling text/LLM features for the 27 FEIs
that Redica doesn't cover. The Text Analysis validation session
(`../99 - Outputs - Text Analysis/eval/results_and_notes/20260909_session_handoff.md`)
found this dilutes the real signal — restricting to the 96 Redica-covered FEIs only
gave a materially different result for the AE model (AUC 0.564→0.677 L2,
0.547→0.681 RF). That restriction was done as an ad-hoc analysis in that session and
was **never implemented back into `m14`/`m17`'s code** — the numbers above are what
the code on disk actually produces today, not the corrected ones. Recall's n=20
events is also too sparse to trust either way (flagged, not presented as a result, in
the same handoff).

## Folder structure

```
99 - Outputs - Shortage Prediction/
├── README.md
├── code/
│   ├── config.py, utils.py, m14, m15, m17, m18   ← current
│   └── old_not_current_pipeline/
│       ├── gen1_drug_year_pipeline/   ← see "Archived: Generation 1" below
│       └── gen2_fei_exploration_abandoned/
├── data/            ← current: faers_fei_panel.csv, recall_fei_panel.{csv,parquet},
│                        redica_fei_year.{csv,parquet}; + old_not_current_pipeline/
├── outputs/
│   ├── figures/     ← current: recall_fei_dashboard.html, faers_fei_dashboard.html,
│   │                    roc/feature-importance/lift PNGs; + old_not_current_pipeline/
│   ├── models/      ← current: metrics/rf_importance/text_ablation *_fei.csv;
│   │                    + old_not_current_pipeline/
│   └── tables/      ← current: *_fei_panel_summary.md, fei_risk_ranking.csv;
│                        + old_not_current_pipeline/
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

1. **See the methodology gap above** — highest-priority fix.
2. **Recall events are sparse** (20 across 1,250 FEI-years) — not presented as a
   reliable result regardless of the zero-fill issue.
3. **SDUD volume weighting not incorporated** — see the TODO block in `config.py`.
