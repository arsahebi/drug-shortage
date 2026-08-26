# Figure 1 Comparison Notes — July 2026 Refresh

**Figure files**
- Old: `~/Desktop/MetforminFigures/Figure2_Price_Volume_by_Inspection.pdf`
- New: `processed/outputs/Figure1_Market_by_Outcome.pdf`

---

## 1. Sample size shift (why counts changed dramatically)

| Group | Old paper | New (July 2026) | Change |
|-------|-----------|-----------------|--------|
| NAI   | 64        | 35              | −29    |
| VAI   | 33        | 173             | +140   |
| OAI   | 13        | 16              | +3     |
| **Total** | **110** | **224**       | +114   |

**Three reasons for the shift:**

1. **More NDC-years overall (110 → 224).**
   New pipeline automated from raw FDA NDC / DailyMed / Redica / IQVIA data → 112 NDC11s, all with IQVIA volume. Old Q&A Sheet1 was manually curated with a smaller NDC set and required `Event Start Date` (often null in Redica) to assign prior score → many NDC-years fell through with null score.

2. **Redica July 2026 refresh adds 5+ years of inspections.**
   Facilities that had NAI as their most recent prior inspection in the old data have since received VAI inspections. The "most recent prior" for many NDC-years is now VAI instead of NAI. This is the dominant driver of NAI↓ / VAI↑.

3. **Bug fix in prior score assignment (sort-order mismatch).**
   Old `build_ndc_year_table` computed `agg` with `sort=True` but assigned prior scores via `groupby(sort=False)` — scores were silently assigned to wrong NDC-Year rows. This inflated NAI counts artificially. New pipeline computes prior inspection within a single correctly-ordered operation.

**OAI note:** All 16 OAI observations are from TestYear=2020 only. For TestYear=2022 and 2024, the same OAI facilities (Zydus, Lupin, Aurobindo) received subsequent VAI inspections — so their most recent prior inspection as of 2022/2024 is VAI, not OAI.

---

## 2. New group breakdown (for verification)

### NAI (n=35, 28 NDC11s, 9 FEIs)
Concentrated in **TestYear=2024** (25 of 35 obs). Most are facilities that received a clean inspection after 2022.

| FEI | Country | n_obs | n_ndc | Facility |
|-----|---------|-------|-------|---------|
| 3008298016 | USA | 12 | 12 | ScieGen Pharmaceuticals (Hauppauge, NY) |
| 3006346108 | CHN | 4  | 2  | Novast Laboratories (Nantong) |
| 3008223599 | IND | 3  | 1  | Amneal Pharmaceuticals (Bavla) |
| 3008565058 | IND | 3  | 1  | Glenmark Pharmaceuticals (Dhar) |
| 3010254278 | IND | 3  | 2  | Amneal Pharmaceuticals (Sanand) |
| 3011922870 | CHN | 3  | 3  | Qingdao BAHEAL Pharmaceutical (Jimo) |
| 3015838038 | IND | 3  | 3  | Harman Finochem (Chhatrapati Sambhaji) |
| 3005263655 | USA | 2  | 2  | Amneal Pharmaceuticals of New York (Central Islip) |
| 3006785788 | IND | 2  | 2  | Ajanta Pharma (Paithan) |

**Note:** ScieGen (3008298016) also appears in VAI (n=24) for TestYears 2020 and 2022 — consistent: ScieGen's most recent prior inspection was VAI before 2024, then NAI for 2024.

### VAI (n=173, 77 NDC11s, 21 FEIs)
Spread across all three test years. Dominated by India (115 of 173 obs).

| FEI | Country | n_obs | n_ndc | Facility |
|-----|---------|-------|-------|---------|
| 3004097901 | IND | 47 | 17 | Granules India (Qutubullapur) |
| 3008298016 | USA | 24 | 12 | ScieGen Pharmaceuticals (Hauppauge, NY) |
| 2000021110 | CHN | 19 | 7  | CSPC Ouyi Pharmaceutical (Shijiazhuang) |
| 3006230648 | IND | 9  | 3  | Marksans Pharma (Mormugao) |
| 3002984011 | IND | 9  | 8  | Zydus Lifesciences (Sanand) |
| 3006370533 | IND | 9  | 3  | Alkem Laboratories (Baddi) |
| 3007938603 | IND | 7  | 7  | Zydus Lifesciences (Sanand) – different unit |
| 3004819820 | IND | 6  | 4  | Lupin Limited (Mormugao) |
| 3011922870 | CHN | 6  | 3  | Qingdao BAHEAL (Jimo) |
| 3008232264 | IND | 6  | 2  | Inventia Healthcare (Ambernath) |
| 1930436    | USA | 6  | 2  | MPP Pharma (Kansas City) |
| (10 more FEIs with 1–4 obs each) | | | | |

### OAI (n=16, 16 NDC11s, 3 FEIs — TestYear=2020 ONLY)
All three facilities received VAI outcomes in subsequent inspections, so they move to VAI for 2022/2024.

| FEI | Country | n_obs | n_ndc | Facility |
|-----|---------|-------|-------|---------|
| 3002984011 | IND | 8 | 8 | Zydus Lifesciences (Sanand) |
| 3004819820 | IND | 4 | 4 | Lupin Limited (Mormugao) |
| 3007373532 | IND | 4 | 4 | Aurobindo Pharma (Jadcherla) |

---

## 3. Statistical comparison

### Primary model used in paper (Model B: RE + two-way clustered SE, reference = NAI)

|  | Old result | New result |
|--|-----------|-----------|
| VAI vs NAI | β=**−1.820**, SE=0.801, 95% CI [−3.389, −0.250], **p=0.025** | β=+0.970, SE=0.882, 95% CI [−0.759, +2.699], p=0.273 |
| OAI vs NAI | β=+1.747, SE=0.952, 95% CI [−0.120, +3.613], p=0.069 | β=+2.067, SE=1.255, 95% CI [−0.393, +4.527], p=0.101 |
| OAI vs VAI | β=+3.566, SE=0.782, 95% CI [+2.033, +5.100], **p<0.001** | (not separately shown; implied β≈+1.097, ns) |

**With the primary model: no inspection outcome significantly predicts volume in new data.**

### Descriptive (new data)

| Group | n | Mean volume | Median volume |
|-------|---|-------------|---------------|
| NAI   | 35  | 7,704,983   | 376,394 |
| VAI   | 173 | 42,777,514  | 2,671,204 |
| OAI   | 16  | 57,085,635  | 3,938,817 |

Direction reversed vs. old: new data shows NAI < VAI < OAI (old paper claimed NAI > VAI). But this direction reversal is NOT supported by the primary model (all p > 0.10).

---

## 4. Potential defense options and caveats

### Approach 1 — Kruskal-Wallis + Dunn (Bonferroni), independent obs
- KW: p=0.00084 (overall significant)
- NAI vs VAI: p_adj=0.002 **; NAI vs OAI: p_adj=0.007 **; VAI vs OAI: p_adj=0.882
- **Problem:** Assumes independence. Same NDC appears across 3 years. Not used in pre-revision paper.

### Approach 2 — NDC-cluster bootstrap pairwise
- NAI vs VAI: p_boot=0.0005 **; NAI vs OAI: p_boot=0.0005 **; VAI vs OAI: p_boot=0.232
- **Problem:** Clusters only at NDC level, ignores FEI-level correlation. Not used in pre-revision paper.

### Approach 3 — FEI-cluster bootstrap pairwise
- NAI vs VAI: p_boot=0.013 *; NAI vs OAI: p_boot=0.003 **; VAI vs OAI: p_boot=0.444
- **Problem:** Clusters only at FEI level, ignores NDC-level repeated obs. Not used in pre-revision paper.

### The user's correct observation
None of these approaches cluster at BOTH FEI and NDC levels simultaneously. Model B (RE + two-way clustered SE) is the only method that does both — and it shows no significance. There is no clean way to defend the old volume finding with the new data using the existing methods.

**However:** The direction of the descriptive finding also reversed (old: NAI > VAI > OAI; new: NAI < VAI < OAI), which itself is substantively important — larger-volume products appear to attract more scrutiny (VAI/OAI outcomes), consistent with FDA's risk-based inspection allocation. The old finding may have been an artifact of the prior-score sorting bug.

---

## 5. Recommended framing for paper

The old Observation 1 (volume part) should be updated: inspection outcome does not predict volume in either direction with the corrected data and updated Redica inspections. The direction reversal is informative but non-significant. The price finding (Observation 1 price part) remains pending NADAC integration.
