# Shortage FEI Panel Summary

- **FEI x year rows (full panel, baseline model population):** 1,270
- **Unique FEIs:** 127
- **Rows with a valid outcome (baseline model population):** 1,270 (127 FEIs)
- **Rows with an as-of-year text snapshot (with-text model population, no zero-fill):** 177 (51 FEIs, 36 shortage-exposed events)
- **Shortage-exposed FEI-years, full outcome set:** 243 (19.1%)

Three models are attempted independently (see RESULTS.docx): a baseline using inspection + structural features on the full panel above, a with-text model restricted to the as-of-year-snapshot population, and a VAI-only text-signal subgroup model within that.

Caveat: shortage exposure is bridged from drug-level UUtah events through the FEI-API map. Every facility manufacturing a shortaged drug is marked exposed; this does not mean that facility caused the shortage.