# 99 — Outputs: Text Analysis
**Drug Shortage Prediction Project**
Last updated: 2026-06-08

---

## What this folder does

Extracts structured risk signals from FDA Form 483 observation text using an LLM,
aggregates them to FEI-level features, and merges them with the structured regulatory
event summary for use in MQRI and shortage prediction models.

**Two independent sub-pipelines:**

| Sub-pipeline | Scripts | Purpose |
|---|---|---|
| **LLM pipeline** (primary) | `01` → `02` → `03` | 483 text → structured features → FEI-level risk indices |
| **Analytics** (optional) | `analytics/01` → `analytics/02` → `analytics/03` | Structured data → CFR network + interactive dashboard |

---

## Execution order

### LLM pipeline (primary)

```bash
# 1. LLM extraction — requires OPENAI_API_KEY
export OPENAI_API_KEY="sk-..."

python 01_extract_observation_signals.py --dry-run   # preview cost, no API calls
python 01_extract_observation_signals.py --limit 10  # test run
python 01_extract_observation_signals.py             # full run (~622 observations)

# 2. Aggregate to FEI level — no API key needed
python 02_aggregate_fei_features.py

# 3. Merge onto node summary — optional
python 03_merge_text_signals.py
```

### Analytics pipeline (optional, for dashboard/visualization)

```bash
cd analytics/
python 01_build_combined_dataset.py
python 03_build_interactive_dashboard.py   # open fei_dashboard.html in browser
python 02_build_interactive_network.py     # optional standalone network
```

---

## Outputs

| File | Produced by | Description |
|---|---|---|
| `483_observation_context_signals.csv` | `01` | One row per observation. Join key: `(fei, filename, obs_num)`. Regex flags + LLM fields. |
| `483_fei_context_features.csv` | `02` | One row per FEI. Layered features + composite indices (TRI, SCRI, IRWI, QCI). |
| `fei_node_summary_enriched.csv` | `03` | 129-FEI node summary joined with LLM features (NaN where unscored). |
| `analytics/fei_node_summary.csv` | `analytics/01` | One row per FEI: structured regulatory event counts. |
| `analytics/fei_dashboard.html` | `analytics/03` | Interactive dashboard (open in browser). |
| `analytics/fei_network.html` | `analytics/02` | Standalone CFR co-occurrence network. |

---

## Current run statistics (2026-06-08)

- **622 observations** across **38 FEIs** (all status `ok`)
- Mean confidence: **0.84**
- 6 low-confidence rows (<0.70), 34 blank evidence quotes (OCR artifacts)
- **Coverage gap**: 38 / 129 FEIs scored (29.5%) — 91 FEIs have no 483 PDFs in dataset

### Key aggregate signals (FEI means)

| Signal | Mean |
|---|---|
| Severity High share | 69.1% |
| Severity Moderate share | 28.7% |
| Systemic flag (LLM) | 95.1% |
| No remediation mentioned | 65.7% |
| Repeat flag (LLM) | 9.4% |
| TRI range | 47.5 – 80.0 (mean 60.8) |

### Semantic lift: what LLM detects that regex misses

| Flag | LLM-only share (LLM=True, regex=False) |
|---|---|
| `patient_risk` | 71.1% |
| `systemic` | 68.9% |
| `contamination` | 27.1% |
| `data_integrity` | 13.1% |

The `systemic` and `patient_risk` lifts are large because the regex rules are conservative
(keyword-based). The LLM infers systemic scope from context ("your firm", multi-line findings).

---

## Feature layers (script 02 output)

Script `02` produces 61 columns grouped into five layers:

| Layer | Columns | Description |
|---|---|---|
| 1 — Quality | `n_obs_total`, `mean_confidence`, … | Extraction quality per FEI |
| 2 — Regex | `*_regex_share` (13 flags) | Deterministic baseline (all rows) |
| 3 — LLM semantic | `severity_*_share`, `*_llm_share`, `dominant_*`, `remediation_*` | LLM-derived, scored rows only |
| 4 — Agreement | `*_regex_llm_agreement`, `*_llm_only_share` | Regex vs LLM comparison |
| 5 — Composite | `TRI`, `SCRI`, `IRWI`, `QCI` | Weighted risk indices [0–100] |

### Composite index formulas

```
TRI  (Text Risk Index):
  0.35 × severity_high_share
+ 0.20 × severity_moderate_share
+ 0.20 × (1 − remediation_strong_share)
+ 0.15 × repeat_llm_share
+ 0.10 × systemic_llm_share

SCRI (Sterility/Contamination Risk):
  0.50 × contamination_llm_share
+ 0.30 × contamination_regex_share
+ 0.20 × severity_high_share

IRWI (Investigation/Remediation Weakness):
  0.40 × remediation_none_share
+ 0.35 × investigation_llm_share
+ 0.25 × remediation_weak_share

QCI  (Quality Culture):
  0.40 × systemic_llm_share
+ 0.35 × repeat_llm_share
+ 0.25 × cultural_root_cause_share
```

---

## Flag definitions

### LLM categorical fields

| Field | Values | Definition |
|---|---|---|
| `violation_category` | LabControls, ProductionControls, BuildingsEquipment, OrgPersonnel, PackagingLabeling, RecordsReports, QualitySystem, Other | Primary regulatory domain of the observation |
| `severity_tier` | High, Moderate, Low | High = direct patient-harm risk or gross deviation; Moderate = significant process deviation; Low = documentation/administrative gap |
| `root_cause_type` | Capital, Cultural, Mixed, Unclear | Capital = equipment/SOP design gap; Cultural = training/management/data-integrity failure; Mixed = clear evidence of both |
| `remediation_signal` | Strong, Partial, Weak, None | Strength of corrective action mentioned in the observation text |

### LLM binary flags

**`repeat_flag_llm`**
True only when the observation explicitly states this is a repeat observation or
finding, previously observed/cited, or recurring from a prior inspection.
Not true merely because multiple examples appear within the same observation.

**`systemic_flag_llm`**
True when the observation describes facility-wide, multi-process, multi-product,
or quality-system-level failure. Can include repeated failures within a current
inspection if the text supports a broader system breakdown.

**`patient_risk_flag_llm`**
True when the violation could directly affect patient safety: risk of non-sterile
product, contaminated product, sub/super-potent drug, or release without adequate
QA. Does not require confirmed patient harm.

**`data_integrity_flag_llm`**
True only for explicit data trustworthiness failures: falsification, backdating,
deleted/altered records, missing raw data, audit-trail problems, unreported results,
or records that cannot be trusted.
Not true for ordinary missing SOPs, incomplete documentation, or weak recordkeeping
unless data reliability is directly at issue.

**`contamination_flag_llm`**
True for actual contamination or clear contamination-control risk: sterility
assurance failures, aseptic processing deficiencies, environmental monitoring
failures, microbial/particulate contamination, inadequate cleaning/sterilization,
or cross-contamination control failures.
Does not necessarily mean confirmed contaminated product.

**`documentation_flag_llm`**
True when inadequate documentation is a central finding: missing/inadequate SOPs,
missing required records, incomplete records, or procedures that do not reflect
actual practice. Not true when documentation is only incidental.

**`investigation_flag_llm`**
True only for an explicit failed, missing, delayed, or inadequate investigation
of a concrete event: deviation, complaint, batch failure, OOS/OOT result, positive
unit, contamination event. Includes missing root cause, missing CAPA, or failure
to assess product impact.
Not true for general missing evaluation/rationale or procedure-only investigation
requirements.

### Regex baseline flags (from 483 extraction script)

Columns ending in `_regex` come from deterministic keyword rules in
`Data/12 - FDA - 483/processed/483_observations.csv`. Transparent and reproducible
but context-unaware; use alongside LLM flags for Layer 4 agreement analysis.

---

## Downstream use

These features are intended to be merged into the MQRI and shortage prediction
pipelines via the FEI identifier. The TRI and SCRI indices are the most informative
single-number summaries per facility.

```
483_fei_context_features.csv  (38 FEIs, 61 columns)
  → join on fei
  → MQRI pipeline:              add TRI as regulatory domain feature
  → Shortage prediction (m07):  bridge FEI→drug via NDC-FEI crosswalk,
                                 then add drug-year aggregated TRI/SCRI
```

---

## Archived scripts

Old ChromaDB/LangGraph pipeline and prior versions are in `old_not_current_pipeline/`.
