# 99 — Outputs: Text Analysis
**Drug Shortage Prediction Project**
Last updated: 2026-05-27

---

## What this folder does

This folder builds two things on top of the structured FDA regulatory data in
folders 12–23:

1. **Combined dataset + visualizations** (scripts 01–03): merges five FDA sources
   into a unified FEI-level event table, network, and interactive dashboard.

2. **Optional LLM observation signals** (scripts 04–05): sends each cleaned 483
   observation (`obs_text_clean`) to Claude, extracts context-aware structured
   signals (severity tier, root-cause type, etc.), and aggregates them to a
   Text Risk Index (TRI) per facility.

These two pipelines are **independent**. The combined dataset (01–03) runs
without any API key and is currently active. The LLM pipeline (04–05) requires
an Anthropic API key.

---

## Execution order

### Core pipeline (required — run these first)

```bash
python 01_build_combined_dataset.py       # ~1–2 min  — no API key needed
python 03_build_interactive_dashboard.py  # ~1 min    — no API key needed
```

Script 02 is optional (standalone network visualization, separate from dashboard):

```bash
python 02_build_interactive_network.py    # optional, ~30 sec
```

### Optional LLM pipeline (run after core pipeline)

Requires `ANTHROPIC_API_KEY` to be set in the environment.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Preview what will be processed and estimate API cost — no calls made
python 04_extract_observation_signals.py --dry-run

# Test on a single FEI before running the full corpus
python 04_extract_observation_signals.py --fei 3002808406 --limit 5

# Full extraction (~30–90 min, all 347 observations)
python 04_extract_observation_signals.py

# Aggregate to FEI level and compute Text Risk Index (<1 min)
python 05_aggregate_fei_features.py

# Optional: merge TRI onto fei_node_summary.csv (<1 min)
python 07_merge_text_signals.py

# Re-run dashboard to add the Risk Signals tab (<1 min)
python 03_build_interactive_dashboard.py
```

---

## Active scripts

| Script | Purpose | Required? | API key? | Run time |
|---|---|---|---|---|
| `01_build_combined_dataset.py` | Merge 5 FDA sources → 4 output files | YES | No | ~1–2 min |
| `02_build_interactive_network.py` | pyvis standalone network HTML | Optional | No | ~30 sec |
| `03_build_interactive_dashboard.py` | Main dashboard HTML | YES | No | ~1 min |
| `04_extract_observation_signals.py` | Claude API → per-observation signals | Optional | **YES** | ~30–90 min |
| `05_aggregate_fei_features.py` | Aggregate obs signals → FEI-level TRI | Optional | No | <1 min |
| `07_merge_text_signals.py` | Join TRI onto node summary | Optional | No | <1 min |

Scripts `06_aggregate_score.py`, `04_ingest_build_vectorstore.py`, and
`05_extract_signals_langgraph.py` (the old ChromaDB/LangGraph pipeline) are
archived in `old_not_current_pipeline/`.

---

## Active outputs

### Produced by 01 (updated whenever source data changes)

| File | Description |
|---|---|
| `fei_events_timeline.csv` | One row per regulatory event per FEI (all 5 sources) |
| `fei_node_summary.csv` | One row per FEI: counts, severity score, worst outcome |
| `fei_edge_list.csv` | Cross-FEI edges: WL cross-site + same-company pairs |
| `fei_cfr_data.json` | Per-FEI CFR frequency, domain breakdown, co-occurrence |

### Produced by 02 (optional)

| File | Description |
|---|---|
| `fei_network.html` | Standalone pyvis network (open in browser) |

### Produced by 03 (depends on 01)

| File | Description |
|---|---|
| `fei_dashboard.html` | Main interactive dashboard (open in browser) |

### Produced by LLM pipeline 04–05 (optional)

| File | Description |
|---|---|
| `483_observation_context_signals.csv` | One row per observation. Join key: `(fei, filename, obs_num)`. Carries renamed regex flags + all LLM fields. |
| `483_fei_context_features.csv` | One row per FEI. Aggregate shares + TRI. |
| `fei_node_summary_enriched.csv` | node_summary + TRI columns (produced by 07, optional) |

---

## LLM pipeline design

### Input

Script 04 reads directly from the pre-cleaned observation file:

```
Data/12 - FDA - 483/processed/483_observations.csv
  347 rows (one per observation)
  Stable join key: (fei, filename, obs_num)
  obs_text_clean — OCR-cleaned, page-header-stripped observation text
  cfr_codes      — CFR sections cited (passed to prompt for context)
  13 regex flags — has_repeat, has_systemic, … (carried through as _regex columns)
```

No PDF re-parsing. No vector store. No embedding model.

### Per-observation prompt

For each row, script 04 sends `obs_text_clean` to `claude-sonnet-4-6` with a
structured JSON prompt that:

- Identifies this as FDA Form 483 observation text
- Passes the `cfr_codes` for context
- Requests exactly these fields:

| Field | Type | Values |
|---|---|---|
| `violation_category` | categorical | LabControls, ProductionControls, BuildingsEquipment, OrgPersonnel, PackagingLabeling, RecordsReports, QualitySystem, Other |
| `severity_tier` | categorical | Low, Moderate, High |
| `severity_rationale` | text | 1–2 sentence explanation |
| `root_cause_type` | categorical | Capital, Cultural, Mixed, Unclear |
| `root_cause_rationale` | text | 1–2 sentence explanation |
| `remediation_signal` | categorical | Strong, Partial, Weak, None |
| `repeat_flag_llm` | bool | repeat finding? |
| `systemic_flag_llm` | bool | facility-wide failure? |
| `patient_risk_flag_llm` | bool | direct patient harm risk? |
| `data_integrity_flag_llm` | bool | data integrity failure? |
| `contamination_flag_llm` | bool | contamination/sterility issue? |
| `documentation_flag_llm` | bool | documentation as primary finding? |
| `investigation_flag_llm` | bool | failure to investigate? |
| `evidence_quote` | text | verbatim substring (evidence guard enforced) |
| `confidence` | float 0–1 | overall extraction confidence |

**Evidence guard**: `evidence_quote` must appear verbatim in `obs_text_clean`.
If it fails the check, it is cleared to `""` rather than accepting a paraphrase.

### Output schema for `483_observation_context_signals.csv`

Columns in order:

```
# Stable join keys
fei, filename, insp_date, obs_num

# Source text + metadata
obs_text_clean, cfr_codes, n_cfrs, n_examples

# Regex baseline flags (renamed for clarity)
has_repeat_regex, has_systemic_regex, has_wl_ref_regex,
has_data_integrity_regex, has_contamination_regex, has_oos_oot_regex,
has_patient_risk_regex, has_quality_unit_regex, has_investigation_regex,
has_documentation_regex, has_laboratory_regex, has_equipment_facility_regex,
has_process_control_regex

# LLM fields
violation_category, severity_tier, severity_rationale,
root_cause_type, root_cause_rationale, remediation_signal,
repeat_flag_llm, systemic_flag_llm, patient_risk_flag_llm,
data_integrity_flag_llm, contamination_flag_llm,
documentation_flag_llm, investigation_flag_llm,
evidence_quote, confidence

# Provenance
model_name, extraction_status, extraction_error
```

### Idempotency and checkpointing

- Re-running script 04 skips already-scored rows (matched by `fei + filename + obs_num`).
- Results are written to disk every 50 observations so a crash does not lose progress.
- Use `--force` to re-score everything from scratch.

### Text Risk Index (TRI) formula

Computed by script 05 and bounded [0, 100]:

```
TRI = (
    0.35 × severity_high_share
  + 0.20 × severity_mod_share
  + 0.20 × (1 − remediation_strong_share)
  + 0.15 × repeat_llm_share
  + 0.10 × systemic_llm_share
) × 100
```

### Two-layer comparison

The regex flags (Layer 1) and LLM flags (Layer 2) now share the same stable
observation key `(fei, filename, obs_num)`, enabling direct per-observation
agreement analysis:

```python
import pandas as pd
df = pd.read_csv("483_observation_context_signals.csv")
# Compare regex vs. LLM on repeat signal
agree_repeat = (df["has_repeat_regex"] == df["repeat_flag_llm"]).mean()
```

---

## Quick test commands

```bash
# Test 1: dry run — no API calls
python 04_extract_observation_signals.py --dry-run

# Test 2: first 5 observations of one FEI
python 04_extract_observation_signals.py --fei 3002808406 --limit 5

# Test 3: first 5 observations across all FEIs
python 04_extract_observation_signals.py --limit 5

# After a small test run, check the output
python -c "
import pandas as pd
df = pd.read_csv('483_observation_context_signals.csv')
print(df[['fei','obs_num','severity_tier','root_cause_type','confidence','extraction_status']].to_string())
"

# Then aggregate to FEI level
python 05_aggregate_fei_features.py
```

---

## Key data relationships

```
Data/08 - Valisure (reference FEI list, 129 FEIs)
         ↓
Data/12 - FDA - 483/processed/
  483_observations.csv ──────────────────────────────────────────┐
    347 rows, obs_text_clean already cleaned                      │
    stable key: (fei, filename, obs_num)                          │
    13 regex flags included                                       │
         │                                                        │
         ↓                                                  04 reads this
Data/14 - FDA - Inspection ──→ 01_build_combined_dataset.py      │
Data/21 - FDA - Warning Letter → 01                              │
Data/22 - FDA - Recall ──────→ 01                                │
Data/23 - FDA - Import Refusal → 01                              │
                                                                  │
01 → fei_events_timeline.csv                                      │
   → fei_node_summary.csv ──────────────→ 07 → enriched          │
   → fei_edge_list.csv                                            │
   → fei_cfr_data.json                                            │
      ↓                                                           │
     03 → fei_dashboard.html (Risk Signals tab if 04+05 run)      │
      ↑                                                           │
     05 → 483_fei_context_features.csv                            │
      ↑                                                           │
     04 → 483_observation_context_signals.csv ←─── 483_observations.csv
          (Claude API, no PDF re-parsing, no vector store)
```

---

## Dependencies

```bash
# Core pipeline (01–03)
pip install pandas openpyxl pyvis networkx

# LLM pipeline (04–05)
pip install -r requirements_text_pipeline.txt
# i.e.: pip install anthropic pandas openpyxl
```

---

## Archived files

Old versions of scripts and documentation that are no longer active are in
`old_not_current_pipeline/`. Nothing has been deleted.

| Archived file | Reason |
|---|---|
| `old_not_current_pipeline/04_ingest_build_vectorstore.py` | Old LangChain + ChromaDB PDF ingestion — replaced by reading `483_observations.csv` directly |
| `old_not_current_pipeline/05_extract_signals_langgraph.py` | Old LangGraph extraction — replaced by direct Anthropic SDK calls in new script 04 |
| `old_not_current_pipeline/06_aggregate_score.py` | Aggregation for old pipeline — replaced by new `05_aggregate_fei_features.py` |
| `old_not_current_pipeline/README_text_pipeline.md` | LangGraph pipeline walkthrough — superseded by this README |
| `old_not_current_pipeline/requirements_text_pipeline.txt` | Old LangChain/ChromaDB requirements — replaced by simplified version |
| `old_not_current_pipeline/pipeline_upgrade_prompt.md` | Internal development prompt |

---

## Reference

`UMich Ross Paper Pipeline.pdf` — original architecture diagram showing the full
pipeline from raw regulatory text to shortage risk prediction.
