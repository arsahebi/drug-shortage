# Metformin Analysis Pipeline — Design Notes
*Updated: July 2026*

---

## Overview

We build a panel linking every Metformin NDC ever independently tested by Valisure (3 test years: 2020, 2022, 2024) to (a) the FDA inspection history of the manufacturing facility and (b) market volume. The unit of analysis for estimation is one row per **(NDC11 × TestYear)**.

---

## Step 0 — Data sources

| Source | Role | Key file |
|--------|------|----------|
| FDA NDC Directory (`product.csv`, `package.csv`) | Drug universe for Metformin | `Data/03 - FDA - NDC/` |
| DailyMed SPL labels (XML) | NDC → DUNS → FEI mapping | `Data/02 - DailyMed - Labels/` |
| ProPublica rx-inspector | Secondary NDC → FEI source | `Data/19 - ProPublica/` |
| Redica Systems (July 2026 refresh) | Per-FEI inspection history | `Data/07 - Redica/` |
| Valisure raw testing data | Quality outcome (DMF, NDMA, Difference Factor) | `Data/08 - Valisure/raw/` |
| IQVIA NDC-level (Jul 2019–Jun 2025) | Commercial volume & prescriptions | `Data/06 - IQVIA/raw/` |
| CMS SDUD (2020, 2022, 2024 annual CSVs) | Medicaid volume & prescriptions | `Data/04 - Medicaid - SDUD/raw/` |

---

## Step 1 — NDC → FEI mapping (`step1_ndc_fei_map.csv`)

**Goal:** For every Metformin NDC11, identify the manufacturing facility (FEI).

**Sources used (in priority order):**
1. DailyMed SPL label XML: extract DUNS numbers (OID `1.3.6.1.4.1.519.1`) → look up FEI in FDA DRLS registration file
2. ProPublica rx-inspector: secondary facility lookup
3. Redica Site List: confirmatory (29 FEIs with known Metformin NDCs)
4. Amir & Amirreza manual review tab in `Q&As1234_v8_v02.xlsx` ("Amir and Amirreza Review all ND" tab, col D): authoritative override for ambiguous/conflicting cases

**NDC normalization:** All NDCs standardized to 11-digit bare format (no hyphens, leading zeros padded). 10-digit NDCs: pad with leading zero.

**fei_count categories** (recorded for transparency):
- `Single - Both`: one FEI found in both DailyMed + ProPublica
- `Single - DailyMed`: one FEI found in DailyMed only
- `Single - Propublica`: one FEI found in ProPublica only
- `Multi FEI - Both`: multiple FEIs in both sources (manual review used)
- `Multi FEI - DailyMed`: multiple FEIs in DailyMed only
- `Multi FEI - Propublica`: multiple FEIs in ProPublica only
- `Not Applicable`: no FEI found in any source

**Outputs:** 112 NDC11s → 28 unique FEIs; 23 NDC11s have null FEI (13 "Not Applicable", 10 multi-source with no resolution).

`facility_distance_km`: Euclidean distance from facility to nearest US port of entry (for import analysis; not used in primary models).

---

## Step 2 — Build inspection panel (`step2_panel_july26.csv`)

**Goal:** One row per (NDC11 × FEI × InspectionEvent). Each row represents one annual inspection of one facility that makes this NDC.

**Join:** step1 NDC→FEI map ✕ Redica inspection history.

**Inspection data (Redica, July 2026 refresh):**
- Each row = one inspection event at one FEI in one year
- `EventYear`: calendar year of inspection
- `Event Start Date` / `Event End Date`: exact dates (often missing in Redica; EventYear is reliable)
- `NAI` / `VAI` / `OAI`: binary (0/1), exactly one = 1 per classified inspection row
- `483` / `No 483`: binary, whether a Form 483 was issued
- `Inspections per Year`: Redica's annual inspection frequency metric for the facility (constant for all rows of the same FEI)
- `Site Display Name`: facility name and location as reported by Redica

**NDC11s with null FEI:** kept in the panel with all inspection columns null. They appear in Valisure quality analysis but are excluded from inspection-outcome analysis.

**NDC11s with multiple FEIs (25 NDC11s):** one row per FEI × InspectionEvent, so these NDCs have more rows than single-FEI NDCs. Handled in step 5 (see below).

**Recent/unclassified inspections:** Some rows have EventYear > 2024 or have NAI=VAI=OAI=0 (inspection event recorded but outcome not yet classified by Redica). These are retained in the panel but filtered in analysis.

**Output:** 1,131 rows; 112 NDC11s; 28 FEIs.

---

## Step 3 — Add Valisure quality data (`step3_panel_july26.csv`)

**Goal:** Expand panel × 3 test years (2020, 2022, 2024) and attach Valisure quality measurements.

**Expansion:** step2 × {2020, 2022, 2024} via cross-join → 3,393 rows.

**Valisure raw sources — all sheets come from a single file:**
- File: `Data/08 - Valisure/raw/Valisure_2024_raw.xlsx`
  - Sheet `"2020 Testing Data"` → 2020 NDMA + DMF (multiple lots → MAX)
  - Sheet `"2022 Testing Data - Actual"` → 2022 NDMA + DMF (multiple lots → MAX)
  - Sheet `"2024 Testing Data"` → 2024 DMF only (no NDMA in 2024)
- Difference Factor: `Data/08 - Valisure/raw/Testing Data_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx`, sheet `"Metformin"`

**NDC matching:** NDC11 bare (11-digit, no hyphens) join between step2 NDCs and Valisure NDCs.

**Multi-lot aggregation:** when multiple lots of the same NDC were tested in the same year, take the **maximum** (worst-case) DMF/NDMA across lots.

**Missing-value substitutions:**
| Raw value | Column | Substitution | Rationale |
|-----------|--------|--------------|-----------|
| `ND`, `N/D` | DMF, NDMA | **0.0** | Not detected = zero contamination |
| `<LOQ`, `LOQ`, `<LOD`, `BLOQ` | DMF, NDMA | **151.54 ng/day** | Below limit of quantification; substituted with LOQ/2 = 303.08/2. LOQ = minimum detected DMF value (303.08 ng/day) |
| `--`, `-`, `NA`, `N/A`, `` | DMF, NDMA, Difference Factor | **NaN** | Genuinely missing |

**Year-specific measurement availability:**
- NDMA: measured in 2020 and 2022; **not measured in 2024** (NDMA column = NaN for all 2024 rows)
- Difference Factor (dissolution proxy): measured in **2024 only** (null for 2020 and 2022)
- DMF: measured in all three years

**`valisure_tested_years`:** NDC-level string summarizing which years a given NDC was tested (e.g., `"2020+2022+2024"`, `"2022"`, `"Not tested"`). Built from the union of test years across all lots for that NDC.

**`n_lots`:** count of distinct lots tested per (NDC11, TestYear). Null for untested (NDC, year) combinations.

**Firm columns:**
- `valisure_firm`: manufacturer name as reported by Valisure (NDC-level; prefer 2024 → 2022 → 2020 when multiple years available)
- `valisure_labeler`: distributor/labeler name (Distributor for 2020; Labeler for 2022/2024)
- `redica_firm`: text before `[` in Redica's `Site Display Name`, title-cased

**Output:** 3,393 rows; 112 NDC11s; 28 FEIs; 28 quality-metric columns.

---

## Step 4 — Add volume data (`step4_panel_july26.csv`)

**Goal:** Attach annual commercial and Medicaid volume for the test year.

### IQVIA (commercial prescriptions + extended units)
- Source: `Metformin Jul 2019 - Jun 2025 NDC Level.xlsx`
- Sheets: `TRx` (total prescriptions) and `Extended Units` (primary volume measure used in paper)
- Aggregation: sum of monthly values for **January through December** of the test year
- Coverage: **100%** (112/112 NDC11s for all three test years)
- **Volume used in paper:** `iqvia_extended_units` (extended units = pill-equivalent units, not prescription count). Confirmed from previous JAMA graphs code.

### SDUD (Medicaid)
- Source: `SDUD_{year}.csv` raw CMS annual files (2020, 2022, 2024)
- Utilization types summed: **FFSU** (fee-for-service) + **MCOU** (managed care) — represents total Medicaid
- State filter: **50 states + DC only** (exclude state code `XX` and territories PR, VI, GU, AS, MP)
  - `XX` = national aggregate duplicate row in SDUD; including it would double every sum
- Aggregation: sum across all states and all quarters of the test year per NDC11
- Coverage: 92–98/112 NDC11s per year (some NDCs have zero Medicaid utilization; coverage increases over years as more products enter Medicaid formularies)
- `sdud_num_prescriptions`: Medicaid total prescriptions (FFSU + MCOU)
- `sdud_units_reimbursed`: Medicaid total units reimbursed (FFSU + MCOU)

**Output:** 3,393 rows; 4 new volume columns; validated against prior Q&A file (all values match).

---

## Step 5 — Analysis-ready panel (`step5_analysis_panel_july26.csv`)

**Goal:** Collapse to one row per **(NDC11 × TestYear)** for estimation. Attach the "prior inspection" outcome for each NDC as of each test year.

### Prior inspection logic

**For each (NDC11, TestYear):**
1. Take all inspection rows for this NDC11 where:
   - FEI is not null
   - EventYear **strictly < TestYear** (inspection happened before the test year; same-year inspections are excluded)
   - NAI + VAI + OAI == 1 (classified inspection — excludes unclassified/future rows)
2. Among qualifying rows, select the row with the **maximum EventYear** (most recent prior inspection)
3. Tie-break (same EventYear across FEIs): take the **worst outcome** (OAI > VAI > NAI)
4. Result columns: `prior_outcome` (NAI/VAI/OAI), `prior_score` (0.0/1.5/3.5), `prior_event_year`, `prior_fei`, `prior_site`

**For NDC11s with no FEI:** `prior_*` columns = NaN. These NDCs are included in quality and volume analysis but excluded from inspection-outcome comparisons.

**For NDC11s with multiple FEIs (25 NDC11s):** the prior inspection is selected across all their FEIs combined (most recent classified inspection of any facility making that NDC).

**CountryCode / CountryName:** taken from the FEI that provided the prior inspection. If no prior inspection, taken from any non-null FEI associated with this NDC.

**`prior_score` mapping:** NAI → 0.0, VAI → 1.5, OAI → 3.5 (same as old paper convention).

### Country assignment
Country (IND / CHN / USA / other) is taken from the Redica `CountryCode` field, which reflects the manufacturing facility location, not the labeler. This is the authoritative classification.

For multi-FEI NDCs where the FEIs are in different countries: CountryCode is assigned from the FEI whose inspection was most recently prior to the test year (the `prior_fei`). If no prior inspection exists, the first non-null CountryCode is used.

### What is excluded from the (NDC11 × TestYear) table
- Inspection rows where EventYear ≥ TestYear (current-year or future inspections — still in step4 for completeness)
- Inspection rows with NAI = VAI = OAI = 0 (unclassified; typically recent Redica entries pending outcome assignment)

---

## Step 6 — Analysis graphs + statistical models (`step6_graphs_july26.py`)

**Input:** `step5_analysis_panel_july26.csv` (336 rows: 112 NDC11s × 3 TestYears).

**Figures produced** (saved as PDF + PNG to `processed/outputs/`):

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `Figure1_Market_by_Outcome` | Left: NADAC price by inspection outcome (blank — NADAC not in pipeline). Right: IQVIA annual volume box + jitter, colored by country (log scale). |
| Figure 2 | `Figure2_Volume_vs_Quality` | Scatter of IQVIA volume vs DMF / NDMA / Difference Factor (all available years pooled per metric). Red dashed trend line. NDC-cluster bootstrap Spearman ρ annotated. |
| Figure 3 | `Figure3_Price_vs_Quality` | Scatter of NADAC price vs DMF / NDMA / Difference Factor (all three panels blank — NADAC pending). |
| Figure 4 | `Figure4_Quality_by_Country` | Bar chart of mean DMF / NDMA / Difference Factor by manufacturing country (IND / CHN / USA). |

**Statistical approach:**

All primary regression models use **Model B: MixedLM random NDC intercept + Cameron-Gelbach-Miller (2011) two-way clustered SE (NDC × prior_fei)**. This accounts for repeated observations of the same NDC across test years (random intercept) and correlation within NDC and within FEI clusters simultaneously (two-way CGM SE).

| Analysis | Outcome | Predictors | Reference |
|----------|---------|------------|-----------|
| Figure 4 — quality by country | log1p(DMF / NDMA / Diff Factor) | IND, CHN dummies | USA |
| Figure 4 — CHN vs IND | same | USA, CHN dummies | IND |
| Figure 1 (right) — volume by outcome | log(IQVIA extended units) | VAI, OAI dummies | NAI |
| Quality by outcome (supplemental) | log1p(DMF / NDMA / Diff Factor) | VAI, OAI dummies | NAI |

*Special case:* Difference Factor (2024 only, cross-section) uses FEI-only clustered SE (NDC clustering not applicable when n_years = 1).

**Sensitivity / alternative approaches** (printed to console alongside Model B):
- Approach 1: Kruskal-Wallis + Dunn post-hoc (Bonferroni) — assumes independence across observations
- Approach 2: NDC-cluster block bootstrap pairwise — clusters at NDC level only
- Approach 3: FEI-cluster block bootstrap pairwise — clusters at FEI level only

These are diagnostic outputs; Model B is the primary model reported in the paper.

---

## Analysis design

### Unit of analysis
One row per (NDC11 × TestYear). The July 2026 Valisure file covers all three test years for all 112 NDC11s, giving N = 336 rows. Of these, 252 are in the IND/CHN/USA analysis set (84 NDC11s × 3 years), and 243 have a prior classified Redica inspection. The Figure 1 regression panel uses 221 rows after requiring non-null IQVIA volume.

### Primary outcome
`DMF (ng/DAY) Valisure` — DMF contamination level at the NDC level. Log-transformed (log1p) for regression.

Secondary outcomes: `NDMA (ng/DAY) Valisure` (2020 and 2022 only), `Difference Factor` (2024 only).

### Primary predictor
`prior_outcome` (NAI / VAI / OAI) — most recent classified FDA inspection outcome for the NDC's manufacturing facility, strictly before the test year.

### Covariates
- `iqvia_extended_units` — market volume (log-transformed)
- `CountryCode` — manufacturing country (IND / CHN / USA / other)
- `TestYear` — fixed effect for temporal trends

### Statistical approach
**Primary:** Model B — MixedLM random NDC intercept + CGM (2011) two-way clustered SE on NDC × FEI. Implemented in step6; accounts for repeated NDC observations across years and FEI-level correlation simultaneously.

**Scatter correlations:** Spearman ρ with NDC-cluster block bootstrap (2,000 resamples). Resamples whole NDC clusters with replacement to obtain cluster-robust p-values and 95% CI.

---

## Notes on "Not tested" NDCs

Two NDC11s (`60505-0260-01`, `60505-1329-01`) have `valisure_tested_years = "Not tested"`. These NDCs were in the Redica FEI list but were never tested by Valisure. They are included in step4 (have IQVIA/SDUD volume) but contribute no quality outcome rows to the analysis.
