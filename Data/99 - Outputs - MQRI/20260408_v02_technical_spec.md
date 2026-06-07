# MQRI v02 — Technical Specification
**Date:** 2026-04-08 | **Author:** Amirreza Sahebi Fakhrabad

---

## 1. What is the Model?

**MQRI (Manufacturing Quality Risk Index)** is a composite score — a weighted linear
combination of observable regulatory and market-quality signals — that estimates the
accumulated manufacturing quality risk for a generic drug facility at a given point in time.

- It is **not** a machine-learning model (no training set, no prediction in the statistical sense).
- It is a **transparent scoring rule** with empirically calibrated weights.
- The score is designed to be **non-circular**: the validation data (Valisure chemical
  testing) is never used in computing the score itself.

---

## 2. Dependent Variable (DV) — What We Are Trying to Predict

We have **two DVs** used at different stages:

| Role | DV | Source | n | When Used |
|------|----|--------|---|-----------|
| **Weight calibration** | OAI rate (n_OAI ÷ n_inspections) | FDA Inspection DB | 127 FEIs | Derives feature weights |
| **External validation** | Valisure dissolution difference factor | Valisure (2024) | 12 facilities | Tests whether the score works |
| **External validation** | Valisure NDMA / DMF ng/day | Valisure (2020, 2022) | 14–18 facilities | Tests whether the score works |
| **Sensitivity test** | FAERS adverse event count | FDA FAERS | 18 facilities | Not used in score — tested separately |
| **Future DV** | Patient readmission / therapeutic substitution | MarketScan | TBD | When data arrives |

**Key principle:** Valisure data and FAERS data are **never entered into the score**.
They are held out and used only after the score is computed.

---

## 3. Independent Variables (IVs) — What Goes Into the Score

### Domain 1: Regulatory Enforcement (70% of total score)

| Variable | What It Measures | Data Source | Time-Varying? |
|----------|------------------|-------------|---------------|
| `n_oai_cumul` | Cumulative OAI (Official Action Indicated) inspections | FDA Inspections | Yes — accumulated through year-end |
| `n_pred_cfr_insp` | # inspections citing ≥1 OAI-predictive CFR regulation | FDA Citations DB | Yes |
| `n_483s_cumul` | Cumulative posted Form 483s (written inspection citations) | FDA Inspections | Yes |
| `n_warning_letters` | Cumulative FDA Warning Letters issued | FDA Warning Letters | Yes |
| `avg_insp_gap_inv` | Inverted average days between inspections (longer gap = more risk) | FDA Inspections | Yes |
| `n_import_refusals` | Cumulative drug import refusals (foreign plants only) | FDA Import Refusals | Yes |
| `ever_contamination` | Has any 483 observation flagged contamination language (NLP) | 483 PDFs (NLP) | Yes |

**OAI-predictive CFRs** (from prior co-occurrence analysis):
211.188, 211.111, 211.56, 211.63, 211.113, 211.192, 211.160, 211.68, 211.100, 211.22

### Domain 2: Market Quality Signals (30% of total score)

| Variable | What It Measures | Data Source | Time-Varying? |
|----------|------------------|-------------|---------------|
| `n_recall_class_I` | Cumulative Class I drug recalls (imminent health hazard) | FDA Recalls | Yes |
| `n_recalls_drug` | All cumulative drug recalls | FDA Recalls | Yes |

### Reported Separately — NOT in the Score

| Variable | Why Excluded | How Used Instead |
|----------|--------------|-----------------|
| `faers_total`, `faers_serious_rate` | Too noisy as predictor; tested as outcome | STEP 6b sensitivity: does MQRI predict FAERS? |
| `iqvia_units` | Measures market exposure, not quality | Societal impact axis in 2-D output table |

---

## 4. How Weights Are Calculated

### Step 1 — Regulatory domain: Spearman correlation with OAI rate
Using the **full 127-FEI FDA inspection database** (all drugs, not just metformin):

| Feature | Spearman |ρ| vs OAI_rate | n | Source |
|---------|--------------------------|---|--------|
| `n_oai_cumul` | — (circular with DV) | — | Set directly |
| `n_pred_cfr_insp` | **0.525** | 127 | FDA Citations |
| `n_483s_cumul` | 0.433 | 127 | FDA Inspections |
| `avg_insp_gap` (inverted) | 0.235 | 127 | FDA Inspections |
| `n_import_refusals` | 0.239 | 127 | FDA Import |
| `n_warning_letters` | 0.640 (vs Valisure) | 12 | Valisure† |
| `ever_contamination` | 0.362 (vs Valisure) | 12 | Valisure† |

† Where OAI-ρ was unavailable, Valisure dissolution correlation used as proxy.

### Step 2 — Normalize within domain

Raw correlations → normalize to sum to 1 within domain, then
redistribute a fixed share (35%) to `n_oai_cumul` (most direct enforcement signal,
weight set by expert judgment rather than correlation since it would be circular).

**Final regulatory weights:**

| Feature | Weight |
|---------|--------|
| n_oai_cumul | **0.35** |
| n_pred_cfr_insp | 0.20 |
| n_483s_cumul | 0.16 |
| n_warning_letters | 0.14 |
| avg_insp_gap_inv | 0.09 |
| n_import_refusals | 0.04 |
| ever_contamination | 0.02 |
| **Total** | **1.00** |

### Step 3 — Market quality domain: FDA severity rule

Class I recall = FDA's own mandatory classification for imminent hazard → **0.65**
All other drug recalls → **0.35**

### Step 4 — Domain-level weights

Regulatory domain = **70%** of final score (enforcement signals most directly reflect quality)
Market quality domain = **30%** (recalls are infrequent and involve managerial discretion)

These domain weights are expert-assigned and reported transparently. Future improvement:
replace with logistic regression coefficients once MarketScan patient outcomes are available.

---

## 5. Scoring Formula

```
For each feature x_i:
    x_i_scaled = min(x_i / global_max_i, 1.0)        ← normalized to [0,1]
    global_max anchored across all years (cross-year comparable)

    Special case avg_insp_gap:
    x_gap_inv = max(1 - gap_days / 1500, 0)            ← inverted, capped at 1500 days

D_reg = Σ (w_i_reg × x_i_scaled)   for all regulatory features  ∈ [0, 1]
D_mkt = Σ (w_j_mkt × x_j_scaled)   for all market quality features  ∈ [0, 1]

MQRI = (D_reg × 0.70  +  D_mkt × 0.30)  ×  100
```

**Risk tiers:** HIGH ≥ 65 | MODERATE 35–64 | LOW < 35

---

## 6. Key Design Choices and Justifications

| Choice | Rationale |
|--------|-----------|
| **Time-varying annual panel** | Captures trajectory; avoids penalizing facilities permanently for old violations |
| **Two domains (not three)** | FAERS removed; volume removed. Cleaner and better-justified |
| **Empirical weights from 127-FEI DB** | n=127 provides stable correlations vs n=18 |
| **Non-circular validation** | Valisure never enters score; used only post-hoc |
| **Global max normalization** | Allows meaningful comparison across years |
| **FAERS excluded from score** | Tested as potential outcome (sensitivity) — team recommendation April 2026 |
| **Volume on separate axis** | Team recommendation: volume ≠ quality; 2-D matrix preferred |

---

## 7. What v02 Does NOT Yet Do (future work)

| Limitation | Proposed Fix |
|------------|-------------|
| Domain weights (70/30) are still expert-assigned | Replace with logistic regression once MarketScan data arrives |
| n=12–18 for Valisure validation is very small | Expand to more drugs when Valisure data available |
| Does not model post-OAI improvement dynamics | Add decay function or year-since-last-OAI feature |
| Import refusals = 0 for all U.S. plants | Add seizure / injunction / criminal action as equivalent |
| NLP coverage incomplete (public 483 portal gaps) | Supplement with Redica or commercial 483 source |

---

## 8. Outputs Produced

| File | Contents |
|------|----------|
| `20260408_v02_mqri_panel.csv` | 18 FEIs × 8 years = 144 rows with all features + scores |
| `20260408_v02_facility_master.csv` | Latest-year snapshot (2024) per facility |
| `20260408_v02_weights.csv` | Full weight table with calibration source for each feature |
| `20260408_v02_validation_correlations.csv` | Spearman ρ vs Valisure by year and outcome |
| `20260408_v02_faers_sensitivity.csv` | Spearman ρ of MQRI vs FAERS (excluded-from-score test) |
| `20260408_v02_2d_risk_volume.csv` | Quality risk × societal impact 2-D table |

---

## 9. Folder Structure

```
99 - Outputs - MQRI/
├── 20260408_v01/           ← archived v01 (all original files)
│   ├── mqri_pipeline.py
│   ├── mqri_panel.csv
│   ├── MQRI_Dashboard.html
│   ├── MQRI_Team_Presentation.pptx
│   └── ...
├── 20260408_v02_mqri_pipeline.py       ← this pipeline
├── 20260408_v02_technical_spec.md      ← this document
├── 20260408_v02_weights.csv            ← weight table (saved at runtime)
├── 20260408_v02_mqri_panel.csv         ← main output (saved at runtime)
├── 20260408_v02_validation_correlations.csv
├── 20260408_v02_faers_sensitivity.csv
├── 20260408_v02_2d_risk_volume.csv
└── 20260408_v02_facility_master.csv
```
