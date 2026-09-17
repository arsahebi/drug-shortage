# Recall FEI Panel Summary

- **FEI × year rows (full panel):** 1,250
- **Unique FEIs:** 125
- **Years:** 2015–2024
- **Rows with a valid outcome (baseline model population):** 1,250 (125 FEIs, 20 events)
- **Rows with an as-of-year text snapshot (with-text model population, no zero-fill):** 177 (51 FEIs, 2 events)
- **Recall events (y=1), full outcome set:** 20 (1.6%)
- **FEIs with ≥1 recall event in panel:** 13

Two models are attempted independently (see RESULTS.docx): a baseline using inspection + structural features on the full panel above, and a with-text model restricted to the as-of-year-snapshot population. Each is skipped on its own if it doesn't clear the minimum-events threshold -- one running does not require the other to.

## Feature coverage
- Inspection features: 4/4 present
- Text/LLM features:   12/12 present
- Structural features: 2/2 present