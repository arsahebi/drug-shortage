# Pre-Revision vs July 2026 Refresh — Statistics Comparison

*Figures and stats produced from the new automated pipeline (steps 1–6, July 2026 Redica refresh)
compared to the pre-revision paper (Health Affairs Scholars 2026-05-29).*

*Prior inspection rule: **EventYear strictly < TestYear** (year before the test year only; same-year inspections excluded).*

Old figures: `~/Desktop/MetforminFigures/`
New figures: `Data/99 - Outputs - Metformin Analysis/processed/outputs/`

---

## Universe counts

| | Pre-revision (Q&A file) | July 2026 |
|--|--|--|
| Total Metformin NDC11s (Valisure file) | **112** | **112** |
| NDC11s with FEI found | **88** | **89** (23 without FEI) |
| NDC11s excluded (CAN / BGD) | **6** (CAN=4, BGD=2) | **5** (CAN=3, BGD=2) |
| NDC11s in analysis (IND/CHN/USA) | **82** | **84** |
| Unique FEIs (IND/CHN/USA) | **15** | **28** |
| Redica: classified FDA inspections | **82** (18 FEIs, through 2025) | **195** (29 FEIs, through May 2026) |

**Country breakdown of FEI-mapped NDC11s (pre-revision Q&A):**
IND = 54 · USA = 16 · CHN = 12 · CAN = 4 · BGD = 2

**Country breakdown of FEI-mapped NDC11s (July 2026):**
IND = 55 · USA = 17 · CHN = 12 · CAN = 3 · BGD = 2

**Why counts differ from pre-revision:**
Two reasons. First, the NDC→FEI linking was done manually in both versions but the new version uses DailyMed drug labels and ProPublica facility data for higher accuracy, and explicitly handles NDCs linked to multiple manufacturing sites (multi-FEI NDCs) — this adds FEIs that were missed when only a single FEI per NDC was recorded. Second, the Redica July 2026 refresh provides inspection data through May 2026 (vs. through 2025 in the pre-revision data), adds 11 additional FEIs not previously captured, and substantially increases the total number of classified FDA inspection events (82 → 195).

---
---

## Figure 1 — Relationship between Market Outcomes and Prior FDA Inspection Outcome

**Old figure:** `Figure2_Price_Volume_by_Inspection.pdf`
**New figure:** `Figure1_Market_by_Outcome.pdf`
*(Left panel = NADAC price is blank in new figure — price data not yet integrated)*

### NDC-year counts

| | Pre-revision | July 2026 (strict rule) |
|--|--|--|
| Total NDC-year obs (volume panel) | **110** | **221** |
| Unique NDC11s | 82 (IND/CHN/USA) | **81** |
| NAI obs (unique NDC11s) | 64 (—) | 26 (25) |
| VAI obs (unique NDC11s) | 33 (—) | 155 (77) |
| OAI obs (unique NDC11s) | 13 (—) | 40 (29) |

**Distribution by year (new, strict rule):**

| Outcome | 2020 | 2022 | 2024 |
|---------|------|------|------|
| NAI | 20 | 6 | 4 |
| VAI | 45 | 63 | 64 |
| OAI | 16 | 12 | 13 |

### Breakdown by FEI (new, strict rule — for verification)

**NAI facilities (6 FEIs):**

| FEI | Country | n_obs | n_ndc | Facility |
|-----|---------|-------|-------|---------|
| 3004097901 | IND | 17 | 17 | Granules India (Qutubullapur) — prior NAI inspection 2018 |
| 3006346108 | CHN | 4 | 2 | Novast Laboratories (Nantong) |
| 3008565058 | IND | 3 | 1 | Glenmark Pharmaceuticals (Dhar) |
| 3005263655 | USA | 2 | 2 | Amneal Pharmaceuticals of New York (Centereach) |
| 3008223599 | IND | 2 | 1 | Amneal Pharmaceuticals (Bavla) |
| 3010254278 | IND | 2 | 2 | Amneal Pharmaceuticals (Sanand) |

*Note: Granules India (3004097901) is NAI here because its most recent classified inspection strictly before 2020 was a 2018 NAI. For TestYear=2022 and 2024, Granules received a 2020 VAI inspection and so appears in VAI for those years.*

**OAI facilities (4 FEIs, spans 2020–2024):**

| FEI | Country | prior_year | n_obs | n_ndc | Facility | TestYears |
|-----|---------|-----------|-------|-------|---------|---------|
| 3002984011 | IND | 2019 | 16 | 8 | Zydus Lifesciences (Sanand) | 2020, 2022 |
| 3007373532 | IND | 2019 | 8 | 4 | Aurobindo Pharma (Jadcherla) | 2020, 2022 |
| 3004819820 | IND | 2019 | 4 | 4 | Lupin Limited (Mormugao) | 2020 |
| 3008298016 | USA | 2023 | 13 | 13 | ScieGen Pharmaceuticals (Hauppauge) | 2024 |

*Zydus, Aurobindo, Lupin: all had OAI inspections in 2019 (the most recent classified prior inspection for 2020). Aurobindo and Zydus had no subsequent classified inspection before 2022, so their 2019 OAI is also the prior for TestYear=2022. ScieGen received an OAI inspection in 2023; its 2024 NAI is excluded by strict rule, making 2023 OAI the prior for TestYear=2024.*

**VAI top facilities (21 FEIs total):**

| FEI | Country | n_obs | n_ndc | Facility |
|-----|---------|-------|-------|---------|
| 3004097901 | IND | 34 | 17 | Granules India (Qutubullapur) — 2022+2024 only |
| 3008298016 | USA | 26 | 13 | ScieGen Pharmaceuticals (Hauppauge) — 2020+2022 |
| 2000021110 | CHN | 21 | 7 | CSPC Ouyi Pharmaceutical (Shijiazhuang) |
| 3011922870 | CHN | 9 | 3 | Qingdao BAHEAL (Jimo) |
| 3011538548 | IND | 9 | 3 | Laurus Labs (Rambilli) |
| 3006370533 | IND | 9 | 3 | Alkem Laboratories (Baddi) |
| 3006230648 | IND | 9 | 3 | Marksans Pharma (Mormugao) |
| 3007938603 | IND | 7 | 7 | Zydus Lifesciences (Sanand — different unit) |
| 3008232264 | IND | 6 | 2 | Inventia Healthcare (Ambernath) |
| 3006785788 | IND | 6 | 2 | Ajanta Pharma (Paithan) |
| 1930436    | USA | 6 | 2 | MPP Pharma (Kansas City) |
| 3004819820 | IND | 6 | 4 | Lupin Limited (Mormugao) — 2022+2024 |
| (9 more FEIs, 1–4 obs each) | | | | |

### Statistics — Volume panel (IQVIA extended units)

**Primary model in paper (Model B: MixedLM random NDC intercept + CGM two-way clustered SE, reference = NAI)**

| Coefficient | Pre-revision | July 2026 (strict) |
|-------------|-------------|-----------|
| VAI vs NAI | β=**−1.820**, SE=0.801, 95% CI [−3.389, −0.250], **p=0.025** | β=+2.311, SE=1.421, 95% CI [−0.474, +5.096], p=0.105 |
| OAI vs NAI | β=+1.747, SE=0.952, 95% CI [−0.120, +3.613], p=0.069 | β=+2.182, SE=1.895, 95% CI [−1.533, +5.897], p=0.251 |
| OAI vs VAI | β=+3.566, SE=0.782, 95% CI [+2.033, +5.100], **p<0.001** | (implied ≈−0.129, ns) |

n_obs: old=110 → new=221 (n_NDC=80, n_FEI=23, ICC=0.667)

**Descriptive (new, IQVIA extended units):**

| Outcome | n | Mean | Median | P25 | P75 |
|---------|---|------|--------|-----|-----|
| NAI | 26 | 24,745,069 | 862,990 | 47,187 | 6,535,536 |
| VAI | 155 | 38,744,119 | 2,483,318 | 363,740 | 16,434,334 |
| OAI | 40 | 46,534,770 | 1,453,286 | 212,008 | 8,824,912 |

### Non-primary methods (not used in pre-revision paper)

| Method | NAI vs VAI | NAI vs OAI | VAI vs OAI |
|--------|-----------|-----------|-----------|
| Approach 1 (KW+Dunn) | p_adj=0.179 | p_adj=1.000 | p_adj=1.000 |
| Approach 2 (NDC-cluster bootstrap) | p_boot=0.089 | p_boot=0.383 | p_boot=0.440 |
| Approach 3 (FEI-cluster bootstrap) | p_boot=0.187 | p_boot=0.656 | p_boot=0.707 |

KW overall: p=0.143 (not significant)

### Conclusion

**Old Observation 1 (volume part): NOT SUPPORTED.**

With the primary model the result is not significant (VAI p=0.105, OAI p=0.251). The direction also reversed from the old claim: new data shows NAI < VAI < OAI (old paper claimed NAI > VAI). The old paper text — *"VAI inspection outcomes associated with significantly lower market volume than NAI (β=−1.820, p=0.025)"* — does not replicate.

**Old Observation 1 (price part):** Cannot be evaluated — NADAC not yet in pipeline.

**Potential defense or path forward:**

Non-parametric and bootstrap approaches show no significant pairwise differences after correcting for clustering (all p>0.08). There is no defensible statistical path to reproduce the old finding. The paper text for this observation needs to be updated.

The new direction (NAI < VAI < OAI) has a plausible substantive interpretation: FDA's risk-based inspection program allocates more scrutiny to higher-volume facilities, which tend to receive VAI or OAI outcomes. This would be a revised Observation 1 — but it is not statistically supported by the primary model (p=0.105 for VAI, p=0.251 for OAI).

---
---

## Figure 2 — Market Volume vs Tested Drug Quality

**Old figure:** `Figure3_Quality_vs_Volume.pdf`
**New figure:** `Figure2_Volume_vs_Quality.pdf`
*(All years pooled in both old and new)*

### NDC-year counts

| Panel | Pre-revision (Q&A) | July 2026 |
|-------|-------------|-----------|
| DMF (2020+2022+2024) | n=**111** (82 NDC11s) | n=**126** (94 NDC11s) |
| NDMA (2020+2022) | n=**63** (54 NDC11s) | n=**71** (62 NDC11s) |
| Difference Factor (2024) | n=**48** (48 NDC11s) | n=**25** (25 NDC11s) |

*DiffFactor count is larger in pre-revision because the old Q&A had more 2024 observations (2024 data was more complete in the pre-revision dataset).*

**DMF panel country breakdown (new):** IND=87 · CHN=18 · USA=21

### Statistics (NDC-cluster block bootstrap Spearman ρ)

| Association | Pre-revision | July 2026 |
|-------------|-------------|-----------|
| DMF vs Volume | ρ=+0.279, p=0.004, 95% CI [+0.064, +0.454] | ρ=+0.302, p_boot=0.002, 95% CI [+0.112, +0.467], n=126 |
| NDMA vs Volume | not significant (p>0.10) | ρ=−0.064, p_boot=0.635, 95% CI [−0.312, +0.181], n=71 |
| Diff Factor vs Volume | not significant (p>0.10) | ρ=−0.162, p_boot=0.454, 95% CI [−0.566, +0.246], n=25 |

### Conclusion

**Old Observation 2 (volume, DMF): STILL SUPPORTED — strengthened.**

Higher DMF contamination is positively associated with higher market volume, and remains significant with more data (ρ=+0.30, p=0.002). Direction and approximate magnitude are consistent with old finding.

NDMA vs volume remains non-significant. Difference Factor vs volume remains non-significant.

---
---

## Figure 3 — Price vs Tested Drug Quality

**Old figure:** `Figure4_Quality_vs_Price.pdf`
**New figure:** `Figure3_Price_vs_Quality.pdf` *(blank — NADAC not in pipeline)*

### Statistics

| Association | Pre-revision | July 2026 |
|-------------|-------------|-----------|
| DMF vs Price | not significant (p>0.10) | *pending NADAC integration* |
| NDMA vs Price | ρ=+0.282, p=0.013, 95% CI [+0.056, +0.490] | *pending NADAC integration* |
| Diff Factor vs Price | not significant (p>0.10) | *pending NADAC integration* |

*Note: Q&A file has NADAC coverage for 107/111 pre-revision NDC-years. NADAC was available for 2020 (25/27), 2022 (38/38), 2024 (44/46) within the IND/CHN/USA subset.*

### Conclusion

**Cannot evaluate.** Old Observation 2 price finding (NDMA vs price ρ=+0.282, p=0.013) cannot be reproduced until NADAC is added to step5 pipeline. This remains an open item.

---
---

## Figure 4 — Drug Quality by Country of Manufacture

**Old figure:** `Figure1_Quality_by_Country.pdf`
**New figure:** `Figure4_Quality_by_Country.pdf`

### NDC-year counts

| Panel | Pre-revision (Q&A) | July 2026 |
|-------|-------------|-----------|
| DMF (2020+2022+2024) | n=**111** obs (IND=79, CHN=18, USA=14); n_ndc: IND=54, CHN=12, USA=16 | n=**127** obs (IND=87, CHN=18, USA=22); n_ndc: IND=66, CHN=12, USA=17 |
| NDMA (2020+2022) | n=**63** obs (IND=44, CHN=13, USA=6); n_ndc: IND=40, CHN=10, USA=12 | n=**71** obs (IND=46, CHN=13, USA=12); n_ndc: IND=40, CHN=10, USA=12 |
| Difference Factor (2024) | n=**48** obs (IND=30, CHN=10, USA=8) | n=**25** obs (IND=16, CHN=5, USA=4); n_ndc: IND=16, CHN=5, USA=4 |

### Descriptive means (new)

| Metric | IND | CHN | USA |
|--------|-----|-----|-----|
| DMF mean (ng/day) | 28,607 | 3,355 | 4,696 |
| NDMA mean (ng/day) | 65.2 | 2.0 | 0.0 |
| Diff Factor mean | 0.261 | 0.226 | 0.153 |

### Statistics (primary model: MixedLM random NDC + CGM two-way SE, reference = USA)

**DMF:**

| Coefficient | Pre-revision | July 2026 |
|-------------|-------------|-----------|
| IND vs USA | not significant (p>0.10) | β=+2.650, SE=1.765, p=0.136 (not significant) |
| CHN vs USA | — | β=+0.277, SE=1.648, p=0.867 |
| CHN vs IND | — | β=−2.374, SE=1.394, p=0.092 (marginal) |

**NDMA:**

| Coefficient | Pre-revision | July 2026 |
|-------------|-------------|-----------|
| IND vs USA | β=+1.345, SE=0.377, 95% CI [+0.605, +2.084], **p<0.001** | β=+1.603, SE=0.630, 95% CI [+0.369, +2.837], **p=0.014** |
| CHN vs USA | not significant | β=+0.310, SE=0.286, p=0.284 (not significant) |
| CHN vs IND | β=−1.090, SE=0.464, 95% CI [−1.999, −0.180], **p=0.022** | β=−1.293, SE=0.692, 95% CI [−2.649, +0.063], p=0.067 (marginal) |

**Difference Factor:**

| Coefficient | Pre-revision | July 2026 |
|-------------|-------------|-----------|
| IND vs USA | β=+0.117, SE=0.041, 95% CI [+0.036, +0.198], **p=0.011** | β=+0.074, SE=0.037, 95% CI [+0.002, +0.147], p=0.061 (marginal) |
| CHN vs USA | not significant | β=+0.060, SE=0.043, p=0.185 (not significant) |
| CHN vs IND | not significant | β=−0.015, SE=0.057, p=0.802 |

### Conclusion

**Old Observation 3 (NDMA, India vs USA): STILL SUPPORTED.**
India significantly higher NDMA than USA in both old and new (old p<0.001, new p=0.014). Coefficient magnitude similar (old β=+1.345, new β=+1.603).

**Old Observation 3 (NDMA, China vs India): WEAKENED — marginal.**
Old: significant (p=0.022). New: marginal (p=0.067, 95% CI barely includes zero). Direction unchanged. Claim should be softened to "marginally significant" or "suggestive."

**Old Observation 3 (Dissolution/Diff Factor, India vs USA): WEAKENED — marginal.**
Old: significant (p=0.011). New: marginal (p=0.061). Direction unchanged (India > USA). Fewer NDC11s have 2024 DiffFactor data (only 25 obs across all countries). Claim should be softened.

**Old Observation 3 (DMF, all country pairs): CONSISTENT.**
Not significant in either old or new. No change needed.

**China vs USA for all metrics:** Consistently not significant in old and new. No change needed.

**Potential defense for weakened findings:** The NDMA China-vs-India and DiffFactor India-vs-USA results are directionally consistent with the old paper and remain at p<0.10. The weakening reflects fewer 2024 observations (only 25 obs for DiffFactor) and a smaller NDMA dataset (only 2020+2022). Reporting as "marginally significant" or "directionally consistent" is reasonable.

---
---

## Additional finding (new, not in pre-revision paper)

**DMF vs Inspection Outcome (from quality ~ outcome model, strict rule):**

| Coefficient | July 2026 |
|-------------|-----------|
| VAI vs NAI (DMF) | β=−2.328, SE=1.096, 95% CI [−4.475, −0.181], **p=0.036** |
| OAI vs NAI (DMF) | β=−0.926, SE=1.700, p=0.587 |

VAI facilities have significantly lower DMF contamination than NAI facilities (p=0.036). Interpretation: facilities under more active FDA scrutiny (VAI outcome) may have lower DMF levels, while NAI facilities (less scrutinized) have higher DMF. This is a new finding not in the old paper.

**Difference Factor vs Inspection Outcome (cross-section 2024):**

| Coefficient | July 2026 |
|-------------|-----------|
| VAI vs NAI (DiffFactor) | β=+0.067, SE=0.029, 95% CI [+0.010, +0.124], **p=0.033** |

VAI facilities have significantly higher Difference Factor (worse dissolution) than NAI in 2024. Note: the OAI coefficient is degenerate (only ScieGen in OAI for 2024; all products have same DiffFactor value).
