# FDA Text Analysis Pipeline — Run Order

## Overview
This pipeline adds an LLM extraction layer on top of the existing rule-based signals
(scripts 01–03). It reads 483 PDFs and Warning Letter text, extracts structured
per-observation risk signals via Claude, aggregates them to the facility level, and
surfaces a new "Risk Signals" tab in the dashboard.

## Prerequisites

### 1. Install dependencies
```bash
pip install -r requirements_text_pipeline.txt
```

### 2. Set environment variables
```bash
# Required — Claude API access
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional — LangSmith tracing (recommended for the paper's audit trail)
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_PROJECT="fda-extraction"
```

If `ANTHROPIC_API_KEY` is missing, script `05` will fail with a clear error message.
Scripts `04`, `06`, `07`, and `03` do **not** require an API key.

## Run Order

Run from the `Data/99 - Outputs - Text Analysis/` directory, or use absolute paths.

```
# Must exist first (run once if not already done):
python 01_build_combined_dataset.py

# New pipeline — run in order:
python 04_ingest_build_vectorstore.py     # ~5–15 min (PDF extraction + embedding)
python 05_extract_signals_langgraph.py    # ~30–120 min (LLM calls, cost depends on corpus size)
python 06_aggregate_score.py              # <1 min
python 07_merge_text_signals.py           # <1 min
python 03_build_interactive_dashboard.py  # ~1 min (regenerates dashboard with Risk Signals tab)
```

## Outputs

| File | Description |
|---|---|
| `chroma_text_store/` | Persisted ChromaDB vector store |
| `ingest_manifest.csv` | One row per chunk: fei, doc_id, observation_id, status |
| `fei_observation_signals.csv` | One row per observation: all Pydantic schema fields |
| `fei_observation_signals_raw.json` | Same data as JSON for downstream use |
| `fei_risk_signals.csv` | Per-FEI aggregated scores + `text_risk_index` |
| `fei_node_summary_enriched.csv` | `fei_node_summary.csv` left-joined with risk signals |
| `fei_dashboard.html` | Regenerated dashboard with 4th "Risk Signals" tab |

## Idempotency

Scripts `04` and `05` are safe to re-run after a crash:
- `04` skips chunks already in the ChromaDB collection (checks by chunk ID).
- `05` skips observations already in `fei_observation_signals.csv` (checks by `doc_id` + `observation_id`).

## Evaluation

After running `05` and filling in `eval/labeling_template.csv`:
```bash
python eval/evaluate_extraction.py
```
Results are written to `eval/extraction_metrics.md`.

## Cost Estimate

Two Claude calls per observation (Extract + Classify).
Typical 483 observation: ~400 tokens in, ~200 tokens out × 2 calls ≈ 1 200 tokens/observation.
With ~500 observations across the 129-FEI corpus: ~600 K tokens total.
At claude-sonnet pricing this is modest; run `05` with `--dry-run` to count observations first.
