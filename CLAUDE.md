# Drug Shortage Project — Claude Context

## Overview
Research project building two interconnected tools:
1. **Manufacturing Quality Risk Index (MQRI)** — a composite quality risk score for generic drug manufacturers
2. **Drug Shortage Prediction Model** — predicts likelihood of drug shortage using regulatory, quality, and market signals

The project started with Metformin as a case study but the current focus is **14 drugs independently tested by Valisure**. Valisure provides ground-truth quality failure data that anchors the analysis. Additional data sources (e.g., MarketScan commercial claims) are planned for future phases.

## Repository & Data Location
- **Code (GitHub):** https://github.com/arsahebi/drug-shortage
- **All files live in Google Drive** (this folder syncs locally via Google Drive for Desktop):
  `/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/`
- Data files (`.csv`, `.xlsx`, `.parquet`, etc.) are excluded from git — they stay only in Drive.
- Code files (`.py`, `.R`, `.ipynb`, `.sas`) are tracked in git.

## Project Root Structure
```
Project - Drug Shortage/
├── Data/                          ← main analysis workspace (numbered data source folders + output pipelines)
├── Code/                          ← standalone R/Python scripts (EDA, early Metformin work)
├── Paper/                         ← manuscript drafts
├── Presentation/                  ← slides
├── Proposal/                      ← grant/project proposals
├── Lit Review/                    ← literature notes
├── Meeting/                       ← meeting notes
├── Conferences/
├── Project Management/
├── Reference Library/
├── Resource/
├── lib/                           ← JavaScript libraries (vis.js, tom-select) for dashboards
└── .venv/                         ← Python virtual environment (not tracked in git)
```

## Data Folder Structure
Each numbered folder = one data source. Raw data files stay in Drive; processed code lives in `processed/code/` subfolders.

| Folder | Data Source | Notes |
|--------|-------------|-------|
| `00 - ERD` | Entity Relationship Diagram | DB schema PDF |
| `01 - Orange Book` | FDA Orange Book | Generic drug approvals |
| `02 - DailyMed - Labels` | DailyMed | Drug label XMLs (large zip) |
| `03 - FDA - NDC` | FDA NDC Directory | product.csv / package.csv |
| `04 - Medicaid - SDUD` | Medicaid State Drug Utilization Data | Monthly Medicaid volumes |
| `04_06 - QA - Volumes` | QA | Cross-check IQVIA vs SDUD |
| `04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)` | Built panel | IQVIA + SDUD + NADAC merged |
| `04_11 - Build - Monthly Panel (SDUD+NADAC)` | Built panel | SDUD + NADAC (no IQVIA) |
| `05 - Firm Level` | FDA DRLS | Firm-level regulatory status |
| `06 - IQVIA` | IQVIA | Commercial sales/volume data |
| `06 - Metformin Data` | Metformin case study | Quality signals + IQVIA dashboard |
| `07 - Redica` | Redica Systems | Third-party inspection/quality ratings |
| `08 - Valisure` | Valisure | Independent drug testing results — **primary quality outcome for current 14-drug focus** |
| `11 - Medicaid - NADAC` | CMS NADAC | Medicaid drug pricing |
| `12 - FDA - 483` | FDA Form 483 | Inspection observation letters (PDFs → structured) |
| `13 - Processed DailyMed` | DailyMed processed | Parsed label data |
| `14 - FDA - Inspection` | FDA OASIS | Facility inspection records |
| `15 - FDA - Adverse Event` | FAERS | FDA adverse event reports |
| `16 - FDA - FEI` | FDA FEI | Facility Establishment Identifier; annual self-ID generics lists |
| `17 - NDC, FEI Mapping` | Built | NDC ↔ FEI crosswalk from DailyMed labels |
| `18 - MCCPDC` | MCCPDC | Multi-source claims data |
| `19 - ProPublica` | ProPublica rx-inspector | Facility inspection public data |
| `20 - Market Scan` | Truven MarketScan | Commercial claims — planned for future phases |
| `21 - FDA - Warning Letter` | FDA | Warning letters by facility |
| `22 - FDA - Recall` | FDA | Drug recall records |
| `23 - FDA - Import Refusal` | FDA | Import refusal records |
| `24 - UUtah - Drug Shortage` | University of Utah | Drug shortage database (outcome variable) |
| `25 - Parent Firm Name` | Built | Parent company name mapping |

## Output Folders (`99 - Outputs - *`)
| Folder | Purpose |
|--------|---------|
| `99 - Outputs - Text Analysis/` | FDA 483 text extraction pipeline (LLM-based) |
| `99 - Outputs - Graphs/` | Metformin JAMA figures + statistical tests |
| `99 - Outputs - MQRI/` | Manufacturing Quality Risk Index pipeline |
| `99 - Outputs - Shortage Prediction/` | Full shortage prediction ML pipeline (m01–m10 modules) |
| `99 - Outputs - Dashboards/` | EDA dashboards (Redica + IQVIA) |

## Key Pipelines

### Text Analysis Pipeline (`99 - Outputs - Text Analysis/`)
Extracts structured signals from FDA 483 observation text.
```
01_build_combined_dataset.py      ← combine 483 + inspection data
02_build_interactive_network.py   ← CFR co-occurrence network
03_build_interactive_dashboard.py ← dashboard
04_extract_observation_signals.py ← LLM extraction of signals
05_aggregate_fei_features.py      ← aggregate to FEI level
07_merge_text_signals.py          ← merge with other features
eval/evaluate_extraction.py       ← evaluate LLM extraction quality
```

### Shortage Prediction Pipeline (`99 - Outputs - Shortage Prediction/code/`)
Modular pipeline; each module outputs a parquet used by the next.
```
config.py          ← paths and settings
main.py            ← runs m01–m10 (annual)
main_monthly.py    ← runs mm01–mm07 (monthly)
m01_drug_universe  ← define drug set
m02_uutah_panel    ← shortage outcome (UUtah)
m03_faers_features ← adverse events features
m04_recall_features← recall features
m05_valisure_scores← Valisure quality scores
m06_redica_features← Redica inspection features
m07_panel_assembly ← merge all features
m08_eda            ← exploratory analysis
m09_model          ← ML model
m10_lead_time      ← lead time analysis
```

### MQRI Pipeline (`99 - Outputs - MQRI/`)
Manufacturing Quality Risk Index. Current version: `20260408_v02_mqri_pipeline.py`.

## Code Conventions
- Python scripts are prefixed with date: `YYYYMMDD_description.py`
- Older/deprecated versions go in `old_not_current_pipeline/` subfolders
- Each data source folder has a `processed/code/` subfolder for scripts that process it
- Data paths use the full Google Drive path (not relative) — update `config.py` or path constants when running on a new machine

## Git Workflow

Solo project — commit directly to `main`. No feature branches or PRs.

### Commits
- One logical change per commit
- Message format: `<verb> <what>` in imperative mood, e.g.:
  - `Add FAERS monthly aggregation pipeline`
  - `Fix FEI deduplication in 483 extraction`
  - `Refactor Redica loader to handle all 14 drugs`
- Keep messages under 72 characters; add a blank line + detail if needed

### Reviewing changes
Use VS Code's built-in diff view — zero token cost:
1. Open this folder in VS Code: `File → Open Folder`
2. Source Control panel (Cmd+Shift+G) lists changed files
3. Click any file → inline diff; click the split-editor icon (top-right) → side-by-side
4. After committing, past diffs are in the **Timeline** panel (bottom of Explorer sidebar)
