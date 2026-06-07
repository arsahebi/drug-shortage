# Claude Code Prompt — FDA 483 Text-Analysis Pipeline Upgrade

> Paste everything below the line into Claude Code, run from the repo root of the
> Drug Shortage project. It assumes Claude Code has read access to the whole repo.

---

## Context

I am building a research pipeline that turns unstructured FDA regulatory text into
structured, facility-level quality-risk signals for a paper (UMich Ross + INFORMS
Healthcare). The behavioral question is whether manufacturing-violation text predicts
downstream supply-chain risk (recalls, import refusals, drug shortages).

**What already exists** — in `Data/99 - Outputs - Text Analysis/`:

- `01_build_combined_dataset.py` — merges five FDA sources (Inspections, Form 483s,
  Warning Letters, Recalls, Import Refusals) into:
  `fei_events_timeline.csv`, `fei_node_summary.csv`, `fei_edge_list.csv`,
  `fei_cfr_data.json` — covering 129 reference FEIs.
- `02_build_interactive_network.py` — pyvis cross-site network → `fei_network.html`.
- `03_build_interactive_dashboard.py` — vis.js dashboard with three right-panel tabs
  (Overview, Events, CFR Analysis) → `fei_dashboard.html`.

**Upstream source data the pipeline can read** (relative to repo root):

- 483 PDFs + processed features: `Data/12 - FDA - 483/` — note especially
  `processed/483_pdf_inventory.csv` and `processed/483_fei_features.csv`.
- Warning Letters: `Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv`.
- Reference FEI list: `Data/08 - Valisure/raw/FEIs_March 2026.xlsx`
  (sheet `API Only_FEI Mapping`).

**Current state of the text analysis** — so far only rule-based regex flags
(`ever_repeat`, `ever_data_integrity`, `ever_contamination`, `ever_systemic`, etc.)
have been extracted. There is **no LLM extraction layer yet.**

## Goal

Implement the LLM extraction architecture shown in the attached pipeline diagram
("From Regulatory Text to Shortage Risk"). Build it end-to-end so it actually runs
on the existing 483 + Warning Letter corpus, produces structured per-observation
risk signals, aggregates them to the facility level, and surfaces them in the
dashboard as a new **Risk Signals** tab.

The diagram has three named layers — build all three:

1. **LangChain — Ingestion & Chunking:** Loader → Chunker → Embedder → Vector Store.
2. **LangGraph — Stateful Extraction Graph:** Retrieve → Extract → Classify →
   Validate, with a confidence-threshold re-extraction loop.
3. **LangSmith — Tracing, Evaluation & Iteration:** every LLM call traced, a small
   human-labeled eval set, and precision/recall reporting for the paper.

Then: Aggregate → Score → Persist, and finally wire the result into the dashboard.

## Hard constraints

- **Do NOT rename the folder** `Data/99 - Outputs - Text Analysis/`. Keep all new
  scripts there, following the existing `NN_name.py` numbering convention.
- **Do NOT break existing outputs.** `01`–`03` and their CSV/JSON/HTML outputs must
  still run and produce identical results. New scripts are additive.
- **LLM = Claude via the Anthropic API.** Use `langchain-anthropic` with the
  `ChatAnthropic` class and a current Claude model. The model name must be a single
  config constant at the top of the file so it is trivial to change.
- **API key from environment only** (`ANTHROPIC_API_KEY`). Never hardcode a key.
  If the key is missing, fail with a clear message — do not silently skip.
- **Embeddings run locally** (`sentence-transformers`, `all-MiniLM-L6-v2`) so there
  is no second API dependency. Vector store = local **ChromaDB** (persisted to disk).
- Every LLM call must be **idempotent and cached** — if an observation has already
  been scored, skip it on re-run. Store partial progress so a crash is recoverable.
- Keep cost visible: print a running token/observation count.

## Build it in phases — commit after each phase

### Phase 0 — Setup
- Read `01`, `02`, `03` fully so the new code matches their style and paths.
- Create `requirements_text_pipeline.txt` (langchain, langchain-anthropic,
  langgraph, langsmith, chromadb, sentence-transformers, pydantic, pandas).
- Add a short `README_text_pipeline.md` describing the run order.

### Phase 1 — `04_ingest_build_vectorstore.py` (LangChain ingestion)
- Load the extractable 483 PDFs and the Warning Letter text for the 129 reference
  FEIs. Use the existing `483_pdf_inventory.csv` to locate PDFs. OCR only if a PDF
  has no text layer.
- Chunk on **observation boundaries** (each numbered 483 observation = one logical
  unit), not blind character splits — preserve `(fei, doc_id, observation_id)`.
- Embed with `all-MiniLM-L6-v2`; store in a persisted ChromaDB collection keyed by
  `(doc_id, chunk_id)` with `fei` as metadata.
- Output: a populated `./chroma_text_store/` directory + an ingest manifest CSV.

### Phase 2 — `05_extract_signals_langgraph.py` (LangGraph extraction graph)
Build a `StateGraph` with four nodes:
- **Retrieve** — pull the observation chunk plus top-k related context for the FEI.
- **Extract** — one Claude call returning structured JSON validated against the
  Pydantic schema below.
- **Classify** — resolve `root_cause_type` (the theory-laden field) with a focused
  Claude call; keep a deterministic keyword baseline for `severity_tier` running in
  parallel as an eval comparator.
- **Validate** — schema-integrity + hallucination guard (every flag must be backed
  by an `evidence_quote` that exists in the source text). If `confidence` is below a
  threshold, loop back to Retrieve with broader context (cap at 2 retries).

**Pydantic output schema (per observation):**

```
fei: int
doc_id: str
observation_id: str
violation_category: Literal["LabControls","ProductionControls",
    "BuildingsEquipment","OrgPersonnel","PackagingLabeling",
    "RecordsReports","QualitySystem","Other"]
severity_tier: Literal["Low","Moderate","High"]
severity_rationale: str
root_cause_type: Literal["Capital","Cultural","Mixed","Unclear"]
root_cause_rationale: str
remediation_signal: Literal["Strong","Partial","Weak","None"]
repeat_flag: bool
systemic_flag: bool
patient_risk_flag: bool
evidence_quote: str          # verbatim span from source text
confidence: float            # 0.0 - 1.0
```

**Root-cause definitions to put in the prompt** (this is the paper's theoretical core):
- **Capital** = equipment, facility, or SOP/process-design gap.
- **Cultural** = training, management oversight, or data-integrity failure.
- **Mixed** = clear evidence of both.
- **Unclear** = text insufficient to decide.

- Enable **LangSmith tracing** (`LANGCHAIN_TRACING_V2`) so every call is logged.
- Output: `fei_observation_signals.csv` (one row per observation) and the raw JSON.

### Phase 3 — `06_aggregate_score.py` (Aggregate → Score → Persist)
- Roll observation-level signals up to per-FEI:
  `n_obs_scored`, `severity_high_share / mod / low`, `dominant_root_cause`,
  `capital_share`, `cultural_share`, `remediation_weak_share`, `mean_confidence`.
- Build a transparent composite **`text_risk_index`** (document the weights;
  keep it interpretable — no black box).
- Output: `fei_risk_signals.csv`.

### Phase 4 — Integrate into the combined dataset
- Add a step (extend `01` or a new `07_merge_text_signals.py`) that left-joins
  `fei_risk_signals.csv` onto `fei_node_summary.csv` so every FEI carries its
  text-derived signals. Keep the original columns intact.

### Phase 5 — Dashboard: new "Risk Signals" tab
- Modify `03_build_interactive_dashboard.py`. Add a **4th tab** "Risk Signals"
  alongside Overview / Events / CFR (the tab system uses `data-tab` attributes and
  a `showTab()` function — follow that exact pattern).
- The tab, per selected FEI, should show: severity-tier distribution, root-cause
  breakdown (Capital / Cultural / Mixed / Unclear), remediation-signal mix, the
  composite `text_risk_index`, mean confidence, and a scrollable list of
  per-observation cards (category, severity, root cause, evidence quote).
- Match the existing visual style (navy `#1F3564`, the current color palette,
  fonts, card styling). Gracefully show "no data" for FEIs with no scored text.

### Phase 6 — Evaluation (LangSmith)
- Create `eval/` with a small human-labeled sample (~30–50 observations) — generate
  a blank labeling template CSV for me to fill, and an eval script that, once the
  labels exist, reports precision / recall / F1 per field and compares the Claude
  output against the rule-based keyword baseline.
- Write the metrics to `eval/extraction_metrics.md` for the paper's audit trail.

## What to do / not do regarding scope

- This pipeline covers the **Inputs → LangChain → LangGraph → Aggregate/Score →
  LangSmith** portion of the diagram. The downstream "Research Outputs"
  (predictive model, empirical validation) are **out of scope** for this task —
  do not build the shortage-prediction model; just make sure `fei_risk_signals.csv`
  is clean and ready to feed it.
- Do not invent data. If the 483 corpus is thin for some FEIs, report coverage
  honestly (how many FEIs / observations were actually scored).

## Definition of done

- All six phases run from the repo with documented commands.
- `01`–`03` still produce their original outputs unchanged.
- `fei_observation_signals.csv`, `fei_risk_signals.csv`, and the updated
  `fei_dashboard.html` with a working Risk Signals tab all exist.
- A re-run skips already-scored observations (idempotency confirmed).
- `eval/extraction_metrics.md` exists (metrics filled once I provide labels).
- Start by exploring the repo and showing me a short written plan before writing
  code. Pause for my OK before Phase 1.
