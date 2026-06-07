# FDA Form 483 Processing

This folder contains FDA Form 483 PDFs and the processed 483 tables used by the
project's text-analysis pipeline.

## Active Pipeline

Raw PDFs live in:

```text
Data/12 - FDA - 483/raw/
```

Run the main extractor after adding new PDFs:

```bash
python "Data/12 - FDA - 483/processed/code/20260316_483_comprehensive_extraction.py"
```

The script writes these active outputs:

```text
Data/12 - FDA - 483/processed/483_pdf_inventory.csv
Data/12 - FDA - 483/processed/483_observations.csv
Data/12 - FDA - 483/processed/483_fei_features.csv
```

## Active Code

```text
Data/12 - FDA - 483/processed/code/20260316_483_comprehensive_extraction.py
```

Main script. It:

- inventories each PDF by FEI, filename, inspection date, page count, and text length
- extracts raw PDF text with `pdfplumber`
- splits text into Form 483 observations
- removes repeated page/header/footer text from observation bodies
- creates observation-level columns such as CFR citations, example counts, and `has_*` signal flags
- aggregates observation-level signals into one FEI-level feature row

```text
Data/12 - FDA - 483/processed/code/20260318_patch_inventory_dates.py
```

Optional helper. It only patches missing `insp_date` values in
`483_pdf_inventory.csv` using the date parser from the main extractor. In normal
use, rerun the main extractor instead.

## Main Outputs

`483_pdf_inventory.csv`: one row per PDF.

Important columns include:

- `fei`
- `filename`
- `insp_date`
- `n_pages`
- `n_chars`
- `extractable`

`483_observations.csv`: one row per extracted observation.

Important columns include:

- `fei`, `filename`, `insp_date`
- `obs_num`, `obs_header`, `obs_body`
- `obs_text_raw`, `obs_text_clean`, `obs_text`
- `cfr_codes`, `n_cfrs`
- `n_examples`
- `has_repeat`, `has_systemic`, `has_wl_ref`, `has_data_integrity`
- `has_contamination`, `has_oos_oot`, `has_patient_risk`
- `has_quality_unit`, `has_investigation`, `has_documentation`
- `has_laboratory`, `has_equipment_facility`, `has_process_control`

`483_fei_features.csv`: one row per FEI, aggregated from
`483_observations.csv`.

This is the 483 feature table currently read by:

```text
Data/99 - Outputs - Text Analysis/01_build_combined_dataset.py
```

`01_build_combined_dataset.py` also reads `483_pdf_inventory.csv`; it does not
read `483_observations.csv` directly.

## Archived Files

Older OCR, JSON, LLM-cleaning, and exploratory signal files are kept here:

```text
Data/12 - FDA - 483/processed/old_not_current_pipeline/
```

Those files are retained for reference, but they are not the current source for
the combined dataset.

