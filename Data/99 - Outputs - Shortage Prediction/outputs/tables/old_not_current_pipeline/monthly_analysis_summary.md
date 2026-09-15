# Monthly Lead-Lag Analysis — Exploratory Summary
**Data:** 14 Valisure-tested APIs × 120 months (Jan 2015–Dec 2024), 26 shortage-onset months.
> **Framing note:** With only ~21 shortage-onset events the estimates below are highly uncertain. All findings are exploratory and suggestive; no causal inference is intended.
---
## 1. Signals that precede shortage onset (month offsets −12 to 0)
| Signal | Avg elevation (−12 to −4m) | Peak offset | Peak vs baseline |
|---|---|---|---|
| redica_n_inspections | ↑ +0.035 | -23m | +0.443 |
| recall_cgmp | ↑ +0.027 | -11m | +0.517 |
| recall_total | ↑ +0.020 | -11m | +0.500 |
| recall_contam | ↑ +0.004 | -7m | +0.038 |
| redica_n_oai | — -0.002 | -21m | +0.052 |
| recall_class_I | — -0.002 | -3m | +0.037 |
| recall_potency | — -0.002 | -24m | -0.002 |
| redica_n_warning_letters | — -0.006 | +0m | +0.060 |
| redica_n_483_critical | — -0.009 | -1m | +0.220 |
| faers_n_serious_w3m | — -12.675 | -16m | +11.414 |
| faers_severity_score_w3m | — -18.462 | -16m | +30.939 |
| faers_n_reports_w3m | — -21.817 | -17m | +3.358 |

**Suggestive leading signals:** `redica_n_inspections` shows the largest average elevation before shortage onset (+0.035 vs baseline in months −12 to −4). This is exploratory; the wide error bars (small n=26 events) prevent firm conclusions.

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
