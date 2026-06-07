# Drug Shortage Prediction from Quality Signals

A reproducible pipeline that asks: **can pre-shortage quality signals (FAERS adverse events, FDA recalls, Valisure DoD scores, Redica inspection events) predict whether a drug will enter shortage in the following year?**

The work continues the advisor's MSOM 2026 paper on the Drug Quality Risk Index (QRI) and the latent-quality-risk framework. We use the University of Utah Drug Information Service (UUDIS) shortage list as the outcome and four upstream quality signals as features.

## TL;DR — what we found

1. **Signals lead shortages.** In drug-years immediately preceding a shortage onset, mean FAERS severity is ~1.7× higher and CGMP-related recall counts are ~32× higher than in drug-years that are *not* followed by a shortage. The lift is monotone in lookback for FAERS severity (event-study figures `lead_time_*.png`).
2. **A simple logit + RF gets meaningful but not great discrimination.** On the full 2,808-drug UUtah panel (n=21,868 drug-years, 1,591 next-year shortage events), cross-validated AUC ≈ **0.695** (L2 logit) / **0.665** (RF), with average precision ~0.13–0.14 against a base rate of 7.3%.
3. **Structural risk dominates.** The strongest single feature is **`parenteral_ever`** (drug is ever delivered parenterally), followed by `sole_source_ever`. Quality signals add discrimination on top of those structural flags but do not replace them.
4. **Valisure DoD score correlates negatively with shortage frequency at the drug level** (worse score → more shortage years), but the pilot has only 12 drugs with Valisure ground truth, so this is suggestive, not conclusive.

## Pipeline

```
m01_drug_universe   → pilot (13 Valisure APIs) + broader (2,808 UUtah drugs)
m02_uutah_panel     → drug-year shortage outcomes (1988 drugs × 12 years)
m03_faers_features  → FAERS reports/serious/severity per drug-year (vectorized regex match)
m04_recall_features → FDA recall counts by class + reason bucket per drug-year
m05_valisure_scores → mean/min/max DoD score + #failing manufacturers per pilot drug
m06_redica_features → Redica OAI / 483 / warning-letter counts per drug-year (via FEI map)
m07_panel_assembly  → join all sources, build rolling-3yr windows, lag outcome
m08_eda             → Spearman, lead-time means, temporal trend
m09_model           → statsmodels Logit (HC0) + sklearn L2 Logit + RandomForest, GroupKFold by drug
m10_lead_time       → event-study trajectories at T-4..T-1 vs control baseline
```

Run end-to-end:
```bash
cd code
python3 main.py            # full pipeline
python3 main.py --step 09  # one step
```

All outputs land under `outputs/` (figures, tables, models) and intermediates under `data/`.

## Master panel

- Shape: **23,856 rows × 59 columns** (drug × year, 2013–2024)
- Drugs covered: **1,988** unique normalized drug names
- Pilot rows (`has_valisure==1`): **144** rows across **12** drugs
- Outcome: `y_next_year_shortage` (shortage onset in year t+1)
- Base rate: **7.28%**

Engineered features include rolling 3-year windows (`_w3` suffix), prior-shortage flags, and structural attributes (`sole_source_ever`, `parenteral_ever`).

## Headline results

### Lead-time lift (year t signal, conditional on year t+1 outcome)

| Signal | y_next=0 mean | y_next=1 mean | Lift |
|---|---|---|---|
| FAERS reports | 67.7 | 105.8 | **1.56×** |
| FAERS serious | 56.0 | 86.1 | **1.54×** |
| FAERS severity score | 70.9 | 118.0 | **1.66×** |
| FDA recalls total | 0.0009 | 0.0113 | **12.7×** |
| FDA recalls CGMP | 0.0003 | 0.0094 | **31.9×** |
| Redica OAI | 0.0024 | 0.0069 | **2.92×** |
| Redica warning letters | 0.0012 | 0.0031 | **2.66×** |

CGMP recalls and OAI events are very rare in absolute terms but disproportionately concentrated in drug-years preceding shortage — the high lifts are real but rest on small counts.

### Prediction (cross-validated, GroupKFold by drug)

| Scope | n | Events | Model | AUC | AvgPrec | Brier |
|---|---|---|---|---|---|---|
| Pilot (12 drugs, Valisure features) | 132 | 29 | L2 Logit | 0.634 | 0.284 | 0.224 |
| Pilot | 132 | 29 | RandomForest | 0.593 | 0.278 | 0.206 |
| Broader (1,988 drugs) | 21,868 | 1,591 | L2 Logit | **0.695** | 0.138 | 0.217 |
| Broader | 21,868 | 1,591 | RandomForest | 0.665 | 0.127 | 0.177 |

The pilot suffers from tiny n (29 events across 12 drugs); the broader scope is more reliable as a baseline. The "lift" of CV AUC of 0.695 over a no-signal model (~0.5) is modest but consistent with the pre-onset lifts above.

### Random Forest feature importance (broader)

The top six features by RF importance:
1. `parenteral_ever` (0.246) — structural
2. `sole_source_ever` (0.159) — structural
3. `faers_n_serious_w3` (0.104)
4. `faers_n_reports_w3` (0.103)
5. `faers_n_reports` (0.096)
6. `faers_severity_score_w3` (0.086)

Recall features barely register at scale — see limitations.

### Pilot logit (interpretable, n=132)

Top |z| coefficients: `parenteral_ever` (z=2.46, p=0.014, OR=7.4), `valisure_min_score` (z=−1.6), `prior_shortage_w3` (z=1.5). Pseudo-R² = 0.272, LLR p = 0.018. The constant and most signal coefficients have large standard errors because of the small sample.

## Figures

Generated under `outputs/figures/`:
- `eda_valisure_vs_shortage.png` — pilot drugs scatter, Valisure score vs # shortage starts
- `eda_lead_time_distributions.png` — histograms of year-t signals split by year-t+1 outcome
- `eda_temporal_trend.png` — annual shortage starts and signal trends
- `lead_time_pilot.png`, `lead_time_broader.png` — event-study trajectories T-4..T
- `roc_pilot.png`, `roc_broader.png` — ROC curves
- `feature_importance_pilot.png`, `feature_importance_broader.png` — RF importance

## Limitations and known issues

1. **Sparse recall matching.** Module 4 (`m04_recall_features.py`) matched only ~22 rows across 9 drugs because the recall product-description tokenizer is too strict. Most recalls in the FDA recall file describe products like "metformin hcl extended-release tablets 500 mg" — multi-token strings whose head token isn't always picked up by the current matcher. Fixing this would likely change the recall coefficients materially (the lift numbers above use only the matched recalls).
2. **Broader logit standard errors are NaN.** Statsmodels Logit converged on the broader panel (LLR p ≈ 2.2e-152) but the Hessian was near-singular, so `bse` is NaN. The L2-penalized logit and RF estimates are unaffected, but the unpenalized broader coefficients should be read as point estimates only.
3. **Valisure ground truth is tiny.** Only 13 Valisure-scored APIs, and 12 of them are present in the UUtah list. Any pilot-only inference is suggestive at best.
4. **Redica → drug map is thin.** Redica is keyed on FEI numbers; we join through the Valisure NDC↔FEI map, which only resolves to **126 facilities and 14 drugs**. The broader UUtah panel mostly has missing Redica features (filled with 0).
5. **Outcome definition is binary onset.** `y_next_year_shortage = shortage_started.shift(-1)`. We are not modeling shortage duration, severity, or recurrence.
6. **No causal claim.** Everything here is associational. The lead-time event study shows that signals rise *before* shortage onset, but it does not establish causation — common shocks (e.g., a contaminated factory) can drive both.

## Files

```
99 - Outputs - Shortage Prediction/
├── README.md                    ← you are here
├── code/                        ← pipeline
│   ├── config.py
│   ├── utils.py
│   ├── m01_drug_universe.py
│   ├── m02_uutah_panel.py
│   ├── m03_faers_features.py
│   ├── m04_recall_features.py
│   ├── m05_valisure_scores.py
│   ├── m06_redica_features.py
│   ├── m07_panel_assembly.py
│   ├── m08_eda.py
│   ├── m09_model.py
│   ├── m10_lead_time.py
│   └── main.py
├── data/                        ← intermediate parquets
│   ├── pilot_drugs.parquet
│   ├── uutah_unique_drugs.parquet
│   ├── uutah_drug_year_panel.parquet
│   ├── faers_drug_year.parquet
│   ├── recall_drug_year.parquet
│   ├── valisure_drug.parquet
│   ├── redica_fei_year.parquet
│   ├── redica_drug_year.parquet
│   └── master_panel.parquet     ← final analysis table
├── outputs/
│   ├── figures/                 ← 9 PNGs
│   ├── tables/                  ← 5 CSVs (EDA + lead-time)
│   └── models/                  ← coefs, metrics, RF importance
└── logs/                        ← per-module logs
```

## Suggested next steps

1. **Fix recall matching** — rebuild `m04` with the same vectorized-regex strategy as `m03`, expanding head tokens. This is the highest-value fix.
2. **Broader scope robust SEs** — refit broader logit with L2 from `statsmodels` or with bootstrap-clustered SEs at the drug level, so the unpenalized model is interpretable.
3. **Add manufacturer concentration** — HHI of suppliers per drug from the NDC↔Application↔Labeler map; the advisor's QRI paper treats this as a primary risk factor.
4. **Survival / time-to-event** — replace the next-year binary with a Cox or discrete-time hazard model.
5. **Calibration plot + decision threshold** — current Brier scores (0.18–0.22) suggest miscalibration; a reliability diagram would tell us whether the L2 logit can be used for ranking vs absolute probability.
