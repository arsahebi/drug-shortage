# 99 — Outputs: Dashboards

Home for EDA and facility-history dashboards — visualization/exploration outputs that
*consume* data built elsewhere, rather than pipelines that build data themselves.

## `fei_inspection_explorer/` (active)

Interactive facility-history dashboard: a vis.js network of all 129 reference FEIs
(colored by regulatory severity), with a per-facility detail panel — event timeline,
full event table, CFR citation breakdown, and an LLM Risk Signals tab.

Moved here 2026-09-15 from `Data/99 - Outputs - Text Analysis/` — it's a downstream
consumer of that folder's LLM text signals (plus structured data from folders 14, 21,
22, 23), not part of text extraction itself. **All 483 data (Overview-tab regex
badges, the event timeline's "483" events, and the Risk Signals tab) now comes solely
from Redica** (`step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv` /
`step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv` in
`../99 - Outputs - Text Analysis/`, 98/129 FEIs) — folder 12's PDF-only files
(38/129 FEIs) and the pre-stepXX-prefix filenames the dashboard used to read (which no
longer existed, so the Risk Signals tab had been silently empty) are no longer used
anywhere in this dashboard. Inspections/warning letters/recalls/import refusals still
come from their own structured FDA sources (folders 14/21/22/23), unaffected.

Run order: `01_build_combined_dataset.py` → `02_build_interactive_network.py`
(optional, standalone) → `03_build_interactive_dashboard.py` (main output,
`fei_dashboard.html` — open directly in a browser, no server needed).

## `processed/code/2025-12-18-eda_redica_iqvia_dashboard.py` (dormant)

A December 2025 Metformin-specific EDA dashboard (IQVIA volumes vs. Redica inspection
outcomes) from before the current 14-drug Valisure-focused analysis. Not part of the
active pipeline; kept for reference.
