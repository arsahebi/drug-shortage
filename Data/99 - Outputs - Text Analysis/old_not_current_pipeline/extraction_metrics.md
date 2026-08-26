# LLM Extraction Metrics

> **Status:** Template not yet filled in.
> Run `eval/generate_labeling_template.py`, complete the `human_*` columns,
> then re-run `eval/evaluate_extraction.py` to populate this file.

## Summary Table

| Field | Accuracy | Macro F1 | Baseline Acc | Baseline F1 | N |
|---|---|---|---|---|---|
| violation_category | — | — | — | — | — |
| severity_tier | — | — | — | — | — |
| root_cause_type | — | — | — | — | — |
| remediation_signal | — | — | — | — | — |
| repeat_flag | — | — | — | — | — |
| systemic_flag | — | — | — | — | — |
| patient_risk_flag | — | — | — | — | — |

## Per-Class Breakdown

*(populated by evaluate_extraction.py after labels are provided)*

## Notes for Paper

- Evaluation set: ~40 stratified observations (High / Moderate / Low severity).
- Baseline (`severity_tier_baseline`) is a deterministic keyword classifier
  (see `05_extract_signals_langgraph.py`, `keyword_severity_baseline()`).
- Macro F1 is computed across all classes including rare ones; interpret with
  caution for classes with < 5 examples.
- Root cause type (`Capital / Cultural / Mixed / Unclear`) is the paper's
  theoretical core variable — treat its F1 as the primary metric.
