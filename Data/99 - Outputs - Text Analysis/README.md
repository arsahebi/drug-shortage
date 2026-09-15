# 99 — Outputs: Text Analysis
**Drug Shortage Prediction Project**
Last updated: 2026-09-15

---

## What this folder does

Extracts structured risk signals from FDA Form 483 observation text using an LLM,
aggregates them to FEI-level features, and merges them with the structured regulatory
event summary for use in MQRI and shortage prediction models.

**Redica is the primary and only actively maintained text source.** The pipeline can
also run against raw 483 PDF text (`--source pdf`), but that only covers 38/129 FEIs
(the ones with a scanned PDF on file) vs. Redica's 98/129, so pdf-source output is not
part of the current analysis — see `old_not_current_pipeline/pdf_source_and_superseded/`.

## Pipeline

```bash
# 1. LLM extraction (redica source, both providers run for cross-model comparison)
python 01_extract_observation_signals.py --source redica --provider anthropic \
    --model claude-sonnet-5 --prompt-version v2
python 01_extract_observation_signals.py --source redica --provider openai \
    --prompt-version v2

# 2. Aggregate to FEI-level features
python 02_aggregate_fei_features.py --model claudesonnet5 --prompt-version v2
```

`claude-sonnet-5` must be passed explicitly with `--model` — the anthropic provider's
own default is claude-haiku, which is not the validated model for this pipeline.

## Current validated outputs (redica source, v2 prompt)

| File | Model | Rows | FEIs |
|---|---|---|---|
| `step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv` | Claude Sonnet 5 | 1,067 | 98 |
| `redica_483_obs_llm_signals_v2.csv` | GPT-5-mini | 1,067 | 98 |
| `step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv` | Claude Sonnet 5 | 246 snapshots | 98 |

`step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv` is the file
`Data/99 - Outputs - Shortage Prediction/code/config.py`'s `TEXT_TIMESERIES_REDICA_CSV`
must point to — this is what feeds `m14_recall_fei_model.py` / `m17_faers_fei_model.py`.

## Subfolders

| Folder | Purpose |
|---|---|
| `eval/` | Human-eval harness (Abdul's round-1 labels), prompt v2 validation, labeling rules doc — the canonical record of extraction-quality validation work. |
| `fei_inspection_explorer/` | Structured-data (non-LLM) CFR co-occurrence network + interactive dashboard, independent of the pdf/redica text-extraction question above. |
| `old_not_current_pipeline/` | Archived scripts/outputs, including `pdf_source_and_superseded/` (pdf-source runs, pre-v2 redica files, the old `ae_validation/` and `signal_verification/` pipelines — both superseded by `eval/` and the Shortage Prediction `m14`/`m17` models). |

## Output file naming

`01_extract_observation_signals.py` tags output filenames with `_<modelslug>` only
when `--model` overrides that provider's own default (anthropic default =
claude-haiku-4-5; openai default = gpt-5-mini). The validated pipeline always runs
Claude with `--model claude-sonnet-5`, so Claude files carry `_claudesonnet5`; GPT
files (its own default) don't carry a model tag. The pdf-source and redica-source
branches now both key their output filename on `--provider`, fixed 2026-09-15 after
a run left GPT-5-mini pdf-source data in a file misnamed
`step01_fdapdf_483_obs_llm_signals_anthropic_v2.csv` (now archived).

## Downstream use

```
step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv
  → join on fei
  → MQRI pipeline:              add TRI as regulatory domain feature
  → Shortage prediction (m14/m17): AE / recall prediction, GroupKFold by FEI
```
