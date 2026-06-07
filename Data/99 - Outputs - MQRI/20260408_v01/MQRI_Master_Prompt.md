# MQRI Project Master Prompt
## Metformin Quality Risk Index — Full Context for Continuation

> **Purpose**: This document is a complete handoff prompt. It describes everything built so far in the MQRI project — data sources with exact paths, scoring methodology, validation results, pipeline code location, outputs, known limitations, and open decisions. Use it to continue development without needing the full prior conversation.

---

## 1. Project Goal

Build a **facility-level quality risk index** for generic metformin manufacturers called the **Metformin Quality Risk Index (MQRI)**. The index scores manufacturing facilities (identified by FDA FEI number) on a 0–100 scale based on regulatory enforcement history, adverse event signals, and market/structural factors. It is designed like a credit score — transparent, additive, expert-weighted, interpretable to regulators and procurement decision-makers.

**Research context**: Generic drug quality is unobservable to pharmacists and patients. FDA enforcement records, adverse event data, and market structure variables are publicly available proxies for latent quality risk. The MQRI aggregates these into a single actionable score.

**Unit of analysis**: One row per manufacturing facility, identified by **FEI number** (FDA Establishment Identifier, stored as string). n = 18 facilities for metformin.

**Key methodological constraint**: This is n=18. No regression or ML models. The index is an expert-designed additive weighted sum across 3 domains, each capped at 25 points, normalized to 0–100.

---

## 2. Critical Design Principle — Valisure Is Ground Truth Only

**Valisure data (NDMA contamination, DMF, dissolution results) is completely excluded from MQRI scoring.**

Valisure is an independent pharmacy lab that measured NDMA contamination (ng/day), DMF levels, and dissolution rates in commercially purchased metformin tablets. These measurements are used **only as external validation** — i.e., to test whether the MQRI predicts contamination after the fact. They are never used as inputs to the score.

This separation is non-negotiable. Using Valisure in the score and then validating against Valisure would be circular. The Valisure columns (`ndma_max`, `dmf_max`, `diss_max`) exist in the master CSV as outcome-only columns, not scored.

**Validation logic**: After MQRI is computed using only regulatory/FAERS/market data, run Spearman correlations (non-parametric, appropriate for n=18) between MQRI domains and Valisure outcomes. Statistically significant correlations demonstrate predictive validity.

**Current validation results** (Spearman ρ):
- MQRI_total vs NDMA: ρ = 0.142, p = 0.574 (n=18, not significant)
- MQRI_total vs DMF: ρ = 0.525, p = 0.025* (n=18, **significant**)
- MQRI_total vs Dissolution: ρ = 0.580, p = 0.048* (n=12, **significant**)
- MQRI_regulatory vs DMF: ρ = 0.519, p = 0.027* (**significant**)
- MQRI_safety vs DMF: ρ = 0.487, p = 0.040* (**significant**)

The MQRI predicts DMF and dissolution outcomes well. It does not predict NDMA specifically, likely because NDMA contamination was concentrated in a few facilities with no prior FDA enforcement (regulatory blindspot — see Section 7).

---

## 3. Data Sources & Exact File Paths

All paths relative to Google Drive base:
```
BASE = '/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data'
```

When running in the sandbox (Cowork/Claude), paths are under:
```
/sessions/youthful-stoic-fermi/mnt/Project - Drug Shortage/Data/
```

### 3a. Base Dataset (Q&As)
| Path | File |
|------|------|
| `06 - Metformin Data/Derived/` | `Q&As1234_v8_v02.xlsx` |

- 1,001 rows (facility × survey year combinations for 2020, 2022, 2024)
- 18 unique FEI numbers for metformin
- Contains: facility name, FEI, IQVIA market volume (`iqvia_units_total`), Valisure columns (`ndma_max`, `dmf_max`, `diss_max`) stored as outcome-only
- **This is the spine** — all other files are left-joined onto this using FEI as the key

### 3b. FEI Network Summary
| Path | File |
|------|------|
| `12-14-21-22-23 - FEI Network/` | `fei_node_summary.csv` |

Key variables: `n_oai`, `n_vai`, `n_483s`, `n_warning_letters`, `n_recalls`, `severity_score`

### 3c. 483 Observations (NLP-derived)
| Path | File |
|------|------|
| `12 - FDA - 483/processed/` | `483_fei_features.csv` |

Key variables (binary flags from NLP text analysis):
- `ever_repeat` — facility received repeat 483 observations
- `ever_data_integrity` — data integrity issues flagged
- `ever_contamination` — contamination issues flagged (4× OAI risk per presentation)
- `ever_oos_oot` — OOS/OOT (out-of-spec/out-of-trend) issues flagged (3.5× OAI risk)
- `ever_systemic` — systemic quality failures flagged (2.7× OAI risk)
- `ever_oai_predictive` — observation tagged as highly predictive of OAI escalation

### 3d. Inspection Features
| Path | File |
|------|------|
| `14 - FDA - Inspection/processed/` | `facility_feature_matrix.csv` |

Key variables:
- `OAI_rate_dqa` — OAI rate per DQA (Drug Quality Audit) inspection
- `avg_dqa_gap_days` — average days between DQA inspections (gaps = reduced oversight)
- `n_oai_predictive_cfrs` — count of CFR citations historically associated with OAI outcomes

### 3e. Warning Letters (NLP-derived)
| Path | File |
|------|------|
| `21 - FDA - Warning Letter/processed/` | `warning_letter_fei_features.csv` |

Key variables:
- `wl_count` — total warning letters received
- `ever_management_oversight` — warning letter cited management/leadership failures
- `ever_corporate_failure_lang` — corporate-level failure language detected in WL text

### 3f. Recall Data
| Path | File |
|------|------|
| `22 - FDA - Recall/processed/` | `recall_fei_features.csv` |

Key variables: `n_recalls_drug`, `n_recall_class_I`

### 3g. Import Refusals
| Path | File |
|------|------|
| `23 - FDA - Import Refusal/processed/` | `import_refusal_fei_features.csv` |

Key variables: `n_import_refusals_drug`

### 3h. FAERS Adverse Events
| Path | File |
|------|------|
| `15 - FDA - Adverse Event/processed/` | `faers_metformin_anda_linked_2015Q1_2025Q3.csv` |

- Contains: `primaryid`, `serious_flag`, `appl_no` (ANDA number)
- **Not FEI-keyed directly.** Linked via ANDA→FEI crosswalk (see 3i)
- Covers 2015 Q1 through 2025 Q3

### 3i. ANDA→FEI Crosswalk (Critical for FAERS linkage)
| Path | File |
|------|------|
| `07 - Redica/processed/` | `ndc_fei_73_v4.xlsx` |

- Columns: `application_num` (ANDA number), `FEI`
- FEI column has float format in raw data (e.g., `3.008298e+09`) — must strip decimal: `.astype(str).str.split('.').str[0].str.strip()`
- ANDA normalization: strip non-digits, zero-pad to 6 chars
- 16,770 of 40,684 FAERS rows matched to an FEI

### 3j. Redica Red Flags (External Benchmark Only)
| Path | File |
|------|------|
| `07 - Redica/processed/` | `SITE_RED_FLAG_AGG_SCORE.xlsx` |

- `redica_rf_total` — commercial third-party quality risk score
- Used only for benchmark comparison, **not included in MQRI score**

---

## 4. MQRI Scoring Methodology

### Framework
Three domains, each 0–25 points. Sum divided by 75, multiplied by 100 → final score 0–100.

```
MQRI = (D_reg + D_saf + D_mkt) / 75 × 100
```

**Risk tiers:**
- HIGH: ≥ 55
- MODERATE: 30–54
- LOW: < 30

---

### Domain 1: Regulatory (D_reg, 0–25 pts)

```python
oai_pts    = s('n_oai').clip(0,5) * 2          # OAI actions, max 5 events × 2 pts = 10 pts
vai_pts    = s('n_vai').clip(0,10) * 0.5        # VAI actions, max 10 events × 0.5 = 5 pts
f483_pts   = (s('n_483s')/15).clip(0,1) * 5    # 483 volume, normalized to 15 max = 5 pts
wl_pts     = s('n_warning_letters').clip(0,4) * 2  # Warning letters, max 4 × 2 = 8 pts
import_pts = s('n_import_refusals').clip(0,5) * 0.5  # Import refusals, max = 2.5 pts
repeat_pts = s('ever_repeat', 0) * 2            # Binary: repeat 483 observations = 2 pts
d_reg = (oai_pts + vai_pts + f483_pts + wl_pts + import_pts + repeat_pts).clip(0, 25)
```

**Rationale**: Regulatory enforcement actions (OAI, warning letters) are direct evidence of FDA-identified quality failures. OAIs are weighted highest because they represent the most serious enforcement. Warning letters are binary escalations. Import refusals indicate systemic quality issues at international facilities. Repeat 483s indicate failure to correct problems. 483 volume (normalized) measures inspection burden.

---

### Domain 2: Safety / FAERS (D_saf, 0–25 pts)

```python
ae_vol_pts = (np.log1p(s('faers_total_reports')) / np.log1p(7000)).clip(0,1) * 15
ae_ser_pts = s('faers_serious_rate') * 10
d_saf = (ae_vol_pts + ae_ser_pts).clip(0, 25)
```

- `faers_total_reports` = total FAERS reports for facility (via ANDA→FEI crosswalk)
- `faers_serious_rate` = proportion of reports marked serious
- Log1p transform on volume (right-skewed distribution); normalized to log1p(7000) as practical max
- Volume contributes up to 15 pts, serious rate contributes up to 10 pts

**Rationale**: FAERS is the FDA's post-market safety surveillance system. High adverse event volume signals real-world patient harm signals. Serious rate (hospitalizations, deaths) is a quality-of-harm indicator. FAERS is consumer-reported and physician-reported, independent of FDA inspections — provides a complementary signal.

---

### Domain 3: Market / Structural (D_mkt, 0–25 pts)

```python
vol_max = float(fac['iqvia_units_total'].max()) or 1.0
sev_max = float(pd.to_numeric(fac['severity_score'], errors='coerce').max() or 1.0)

vol_pts    = (s('iqvia_units_total') / vol_max).clip(0,1) * 8     # Market share, max 8 pts
rec_pts    = s('n_recalls_drug').clip(0,5) * 2                    # Recall count, max 10 pts
class1_pts = s('n_recall_class_I').clip(0,3) * 2                  # Class I recalls, max 6 pts
sev_pts    = (s('severity_score',0) / max(sev_max,1)).clip(0,1) * 5  # Severity score, max 5 pts
d_mkt = (vol_pts + rec_pts + class1_pts + sev_pts).clip(0, 25)
```

- `iqvia_units_total` = IQVIA prescription volume (market exposure / systemic risk)
- `n_recalls_drug` = number of drug recalls issued
- `n_recall_class_I` = number of Class I recalls (most serious — health hazard)
- `severity_score` = aggregate severity from FEI network analysis

**Rationale**: Market volume determines exposure magnitude (a high-volume facility failing causes more harm). Recalls are direct evidence of post-market quality failures (distinct from pre-market inspection findings). Class I recalls are the highest FDA severity tier. Severity score is a composite of recall and network risk signals.

---

### Helper function

```python
def s(col, default=0.0):
    v = pd.to_numeric(fac[col], errors='coerce').fillna(default)
    return v
```

---

## 5. Master Pipeline Location

```
/sessions/youthful-stoic-fermi/mnt/Project - Drug Shortage/Data/
  06_07_08_12_14_15_21_22_23 - MQRI/mqri_pipeline.py
```

The pipeline:
1. Loads Q&As base (spine)
2. Aggregates to facility level (one row per FEI, taking max/sum across survey years)
3. Left-merges all 7 FEI-keyed sources above
4. Loads FAERS, normalizes ANDA numbers, merges with crosswalk to get FEI, aggregates to facility level
5. Left-merges FAERS features onto facility table
6. Computes 3-domain MQRI
7. Runs external validation (Spearman correlations vs. Valisure outcomes)
8. Saves master CSV

**To run (from sandbox)**:
```bash
cd /sessions/youthful-stoic-fermi/mnt/Project\ -\ Drug\ Shortage/Data/06_07_08_12_14_15_21_22_23\ -\ MQRI/
python mqri_pipeline.py
```

---

## 6. Master Output CSV

```
.../06_07_08_12_14_15_21_22_23 - MQRI/mqri_facility_master.csv
```

- 18 rows × 89 columns
- One row per facility (FEI)
- Contains all merged source variables + computed MQRI domain scores
- Valisure columns (`ndma_max`, `dmf_max`, `diss_max`) present as outcome-only, not scored

**Current MQRI leaderboard** (as of last pipeline run):
| Rank | Facility | MQRI | Tier |
|------|----------|------|------|
| 1 | Sun Pharma | 52.4 | MODERATE |
| 2 | Aurobindo | 44.5 | MODERATE |
| 3 | Nostrum Labs | 43.4 | MODERATE |
| 4 | Lupin Ltd. | 40.1 | MODERATE |
| 5 | Amneal Pharma | 38.4 | MODERATE |
| ... | ... | ... | ... |
| 13 | Marksans Pharma | 16.5 | LOW |

*Marksans: NDMA = 396.8 ng/day but MQRI = 16.5 — see Known Limitation below*

---

## 7. Known Limitation: Regulatory Blindspot

**Marksans Pharma** is the key example. It has the highest NDMA contamination (396.8 ng/day, far above FDA limit of 96 ng/day) but scores LOW on MQRI (16.5) because it has **no FDA enforcement history** — no OAIs, no warning letters, no import refusals.

This is a structural limitation of any enforcement-record-based index: facilities that have never been inspected or never had enforcement action taken against them score near zero regardless of their actual quality. The MQRI measures **regulatory visibility of quality failure**, not quality itself.

**Documented in**:
- Pipeline output (printed warning when gap between MQRI and NDMA is identified)
- Slide 8 of `MQRI_Team_Presentation.pptx` (Regulatory Blindspot case study)
- Dashboard (flagged in facility detail for Marksans)

**Potential mitigations** (not yet implemented):
- Country-of-manufacture flag (India/China flagging latent oversight gaps)
- Inspection gap variable (`avg_dqa_gap_days` — long gaps = less oversight)
- Incorporate inspection frequency as a risk signal

---

## 8. Outputs

### PowerPoint Deck
```
.../06_07_08_12_14_15_21_22_23 - MQRI/MQRI_Team_Presentation.pptx
```
10 slides, generated via `pptxgenjs` Node.js script (`build_deck_v3.js`):
- Slide 1: Title
- Slide 2: Design principle — Valisure as ground truth, not input
- Slide 3: 3-domain framework
- Slide 4–6: Data sources, methodology, scoring details
- Slide 7: External validation (4 correlation cards with Spearman ρ values)
- Slide 8: Regulatory Blindspot (Marksans case study)
- Slide 9: MQRI leaderboard
- Slide 10: Next steps

Rebuild deck:
```bash
node /sessions/youthful-stoic-fermi/build_deck_v3.js
```

### Interactive Dashboard
```
.../06_07_08_12_14_15_21_22_23 - MQRI/MQRI_Dashboard.html
```
Plotly.js interactive dashboard with:
- Facility rankings bar chart
- Radar chart (3 domains: Regulatory, Safety, Market)
- Scatter plots: MQRI vs Valisure outcomes
- Per-facility detail panel with domain breakdown

---

## 9. Candidate Variables to Add (Decisions Pending)

These variables were identified from presentations and papers but **not yet incorporated into the score**. The user needs to decide which to add.

### From 483 NLP Analysis (`483_fei_features.csv` — already merged but not scored)
Already present in master CSV, just not used in formula:

| Variable | Evidence | Proposed addition |
|----------|----------|-------------------|
| `ever_contamination` | 4× elevated OAI risk in 483 text analysis | Add to D_reg |
| `ever_data_integrity` | 1.6× elevated OAI risk | Add to D_reg |
| `ever_oos_oot` | 3.5× elevated OAI risk | Add to D_reg |
| `ever_systemic` | 2.7× elevated OAI risk | Add to D_reg |
| `ever_oai_predictive` | Explicitly tagged as OAI-predictive | Add to D_reg |

### From Inspection Features (`facility_feature_matrix.csv` — already merged but not scored)

| Variable | Evidence | Proposed addition |
|----------|----------|-------------------|
| `n_oai_predictive_cfrs` | Count of CFR citations historically associated with OAI | Add to D_reg |
| `avg_dqa_gap_days` | Long inspection gaps = less oversight = higher risk | Add to D_reg or D_mkt |
| `OAI_rate_dqa` | OAI rate per DQA inspection | Add to D_reg |

### From Warning Letter NLP (`warning_letter_fei_features.csv` — already merged but not scored)

| Variable | Evidence | Proposed addition |
|----------|----------|-------------------|
| `ever_management_oversight` | Management failure language in WL text | Add to D_reg |
| `ever_corporate_failure_lang` | Corporate-level failure language | Add to D_reg |

### From Presentations and Papers (require new data extraction)

| Variable | Source | Notes |
|----------|--------|-------|
| Country of manufacture (India/China binary flag) | `Statins SC Analysis v2.pptx`, `Extended Abstract MSOM 2026.pdf` | API/FDF sourcing from high-risk origins; needs geocoding or manual tagging |
| API vs. FDF supplier count | `Statins SC Analysis v2.pptx` | Supply chain depth — more suppliers = more risk touchpoints |
| Vertical integration flag | `Statins SC Analysis v2.pptx` | Integrated manufacturers have more internal control |
| GDUFA facility fees paid (binary) | `Statins SC Analysis v2.pptx` | Delinquency signals financial/compliance issues |
| Price trend slope | `Extended Abstract MSOM 2026.pdf` | Steep price decline → cost pressure → quality shortcuts |
| Herfindahl competition index | `Extended Abstract MSOM 2026.pdf` | Low competition → less incentive for quality investment |
| Shortage history | `Statins SC Analysis v2.pptx` | Past shortages signal fragile supply/quality |

---

## 10. Presentations and Papers Read

All in Google Drive under `Project - Drug Shortage`:

| File | Key findings used |
|------|------------------|
| `Presentation/tmp-Drug_Quality_Risk_Index.pptx` | CFR citation analysis, OAI-predictive CFR counts, index design framework |
| `Presentation/tmp-FDA_483_Analysis.pptx` | 483 NLP: contamination (4× OAI), OOS/OOT (3.5×), systemic (2.7×), data integrity (1.6×) |
| `Presentation/2026-03-19-FDA-Text_Sources_Analysis.pptx` | WL NLP: management oversight, corporate failure language, WL severity scoring |
| `Presentation/Statins SC Analysis v2.pptx` | Supply chain structure: FDF/API sourcing, GDUFA fees, vertical integration, shortage history |
| `Paper/Quality Risk Index/Quality_Risk_Index_Analysis.docx` | Clinical claims data, IQVIA volume approach, prescription trends |
| `Paper/Quality Risk Index/Latent Quality Risk.docx` | Time-varying latent quality state model (probit + random walk) — theoretical basis |
| `Paper/Quality Risk Index/Notes.docx` | Scope expansion to 14 APIs × 129 facilities, tree-based models, HMM |
| `Paper/Quality Risk Index/Extended Abstract (2)_MSOM 2026.pdf` | Price decline signals, Herfindahl competition index, clinical effectiveness, supply chain |

---

## 11. Merge Key Notes

**FEI normalization** (critical — inconsistent formats across files):
```python
df['FEI'] = df['FEI'].astype(str).str.split('.').str[0].str.strip()
```
Some files store FEI as float (e.g., `3.008298e+09`). Always strip decimal and whitespace before merging.

**ANDA normalization** (for FAERS linkage):
```python
ndc_xwalk['appl_no_str'] = (
    ndc_xwalk['application_num']
    .astype(str)
    .str.replace(r'\D', '', regex=True)
    .str.zfill(6)
)
faers['appl_no_str'] = faers['appl_no'].astype(str).str.zfill(6)
```

---

## 12. Open Decisions for Next Session

1. **Which candidate variables to add to the score?** (Section 9 above) — User has not yet decided. Options range from minor tweaks (add `ever_contamination` to D_reg) to substantial additions (new supply chain domain D_sc).

2. **Add a 4th domain?** A supply chain / structural domain could capture country-of-manufacture risk, vertical integration, and price pressure. Would change normalization denominator from 75 to 100.

3. **Expand to other APIs?** The Notes.docx mentions plans to expand from metformin to 14 APIs × 129 facilities. The pipeline is already modular — the Q&As spine just needs updating.

4. **Rethink NDMA prediction gap?** The MQRI does not predict NDMA well (ρ=0.142, p=0.574). If NDMA prediction is important, country-of-manufacture or chemical process variables (not in current FDA data) may be needed.

5. **Manuscript integration?** The latent quality state model (probit + random walk from `Latent Quality Risk.docx`) is a separate methodological approach. Decision pending on whether to integrate or keep MQRI as a standalone index.

---

## 13. Tech Stack

| Tool | Use |
|------|-----|
| Python (pandas, scipy, numpy) | Pipeline, data merging, scoring, validation |
| pptxgenjs (Node.js) | PowerPoint generation |
| Plotly.js | Interactive HTML dashboard |
| Google Drive | Primary data storage |
| Claude (Cowork) | Development environment |

---

*Last updated: Session April 2026. Pipeline version: 2.0 (Valisure-free design).*
