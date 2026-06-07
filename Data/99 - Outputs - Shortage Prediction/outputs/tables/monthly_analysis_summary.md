# Monthly Lead-Lag Analysis — Exploratory Summary
**Data:** 14 Valisure-tested APIs × 120 months (Jan 2015–Dec 2024), 26 shortage-onset months.
> **Framing note:** With only ~21 shortage-onset events the estimates below are highly uncertain. All findings are exploratory and suggestive; no causal inference is intended.
---
## 1. Signals that precede shortage onset (month offsets −12 to 0)
| Signal | Avg elevation (−12 to −4m) | Peak offset | Peak vs baseline |
|---|---|---|---|
| recall_cgmp | ↑ +0.067 | -11m | +0.517 |
| recall_total | ↑ +0.064 | -11m | +0.500 |
| redica_n_483_critical | ↑ +0.021 | -1m | +0.220 |
| recall_contam | ↑ +0.009 | -7m | +0.038 |
| recall_class_I | — -0.002 | -3m | +0.037 |
| recall_potency | — -0.002 | -12m | -0.002 |
| redica_n_warning_letters | — -0.004 | +0m | +0.060 |
| redica_n_oai | — -0.016 | -12m | +0.044 |
| redica_n_inspections | — -0.077 | -1m | +0.314 |
| faers_severity_score_w3m | — -1.108 | -8m | +17.398 |
| faers_n_serious_w3m | — -3.002 | -12m | +10.103 |
| faers_n_reports_w3m | — -10.762 | -7m | +0.767 |

**Suggestive leading signals:** `recall_cgmp` shows the largest average elevation before shortage onset (+0.067 vs baseline in months −12 to −4). This is exploratory; the wide error bars (small n=26 events) prevent firm conclusions.

---
## 2. Recall circularity
Total matched recalls: **45** (21 CGMP, 24 non-CGMP).

**Timing breakdown (all recalls):**

- Pre-shortage (gap < −3m): 28 recalls (62%)
- Coincident (gap −3 to 0m): 2 recalls (4%)
- Post-onset (gap > 0m): 6 recalls (13%)
- No shortage for this drug: 9 recalls

**CGMP recalls pre-shortage:** 20 of 21 CGMP recalls (95%) fall >3 months before the nearest shortage onset, suggesting a possible upstream manufacturing signal. However the absolute counts are very small; this is suggestive only.

**Circularity concern:** Recalls that fall during an active shortage (8 of 45) are likely mechanically circular — the shortage may have prompted or coincided with the recall rather than caused it. These are flagged as `recall_during_shortage = 1` in master_panel_monthly.csv and should be excluded from causal analysis.

---
## 3. Data quality notes
- **FAERS resolution:** Quarterly (non-zero in months 1, 4, 7, 10 only). Lead-lag analysis uses 3-month rolling sums (*_w3m). Interpret with care.
- **Valisure scores:** 2024 snapshot only — NOT time-varying. Excluded from all lead-lag analysis.
- **Recall sparsity:** Very few matched recall events across 14 drugs. CGMP/other breakdowns are based on small counts.
- **Sample size:** ~21 shortage-onset months drives all event-study estimates. Standard errors are large; all signals should be treated as hypotheses for future validation, not confirmed findings.
