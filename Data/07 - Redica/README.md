# 07 - Redica

Redica Systems site-level inspection and enforcement data. Written 2026-10-06 from a
read of every file and script in this folder.

## The one file most of the project uses

`processed/redica_all_drugs_combined.csv` is one row per FDA inspection event (FEI x event
date) for the 127 FEIs behind the 14 Valisure APIs. Columns: FEI, Redica site id and name,
event date, NAI/VAI/OAI classification, 483 flag, 483 critical/major/other counts, warning
letter flag, and site-level totals from the Data Availability sheet.

Built by `processed/20260505_redica_all_drugs_combined.py`. Read by:

- `Analysis/Shortage Prediction/code/config.py` (`REDICA_CSV`), used by m14/m17/m19
- `Analysis/Text Analysis/vai_signal_validation/01_build_inspection_panel.py`
- `Data/08 - Valisure/processed/20260717_build_fei_ndc_anda_crosswalk.py`
- `Data/08 - Valisure/processed/20260930_new_ndc_fei_coverage.py`
- `Data/08 - Valisure/processed/code/20261001_link_ndc_fei.py`

## Refreshing it when Redica sends new data

The script needs three files from a Redica delivery. They are the same three sheets Redica
exports for any site list:

| Script constant | Current file in `raw/` | What it is |
|---|---|---|
| `REDICA_DETAILED` | `Valisure14_Sites_Red_Flag_Events.xlsx` | Event-level audit trail, one row per red-flag attribute (2,663 rows). Classification and 483 counts come from here |
| `DATA_AVAILABILITY` | `Valisure14_Sites_Data_Availability.xlsx` | One row per site with totals (inspections, 483s, warning letters, import alerts) |
| `SITE_LIST` | `Valisure14_Site_List.xlsx` | Redica site id to FEI (127 rows) |

Steps:

1. Put the new delivery in its own folder under `raw/`, e.g. `raw/2027-01 delivery/`. Keep
   the old files where they are.
2. Run the script with that folder:

   ```
   cd "Data/07 - Redica/processed"
   python 20260505_redica_all_drugs_combined.py "../raw/2027-01 delivery"
   ```

   It picks the one file in the folder matching each of `*Red_Flag_Events*.xlsx`,
   `*Data_Availability*.xlsx` and `*Site_List*.xlsx`, prints which files it used, and stops
   with a message if a pattern matches zero files or more than one. Run with no folder, it
   uses the three `Valisure14_*` files above.
3. It overwrites `redica_all_drugs_combined.csv` and
   `redica_all_drugs_combined_mismatches_report.csv`.
4. Look at the mismatch report. It lists sites where the count of 483s found in the event
   file differs from Redica's own `483s Issued` total. The notes at the bottom of the script
   explain the usual reasons (non-drug industry tags, missing audit rows, multi-date 483s).

The Text Analysis pipeline reads a fourth Redica file, the 483 observation text:
`Valisure14_FDA_483_Observations_WL_Deficiencies_OSU.xlsx` (sheet
`FDA-483s Obs + WL Deficiencies`, 1,152 observations), via
`Analysis/Text Analysis/00_load_redica_obs.py`. Refresh that one at the same time.

Classification rules the script applies, so they stay consistent across refreshes:

- keep rows where the agency list includes `US - FDA` and the industry list is
  `Human Drugs` or empty
- group by site + event date; take the Drug Quality Assurance program's NAI/VAI/OAI when
  present, otherwise the worst of OAI > VAI > NAI
- with no formal classification, a warning letter or "Non-Compliant" means OAI and
  "Compliant" means NAI

## Other files in `processed/`

| File | What it is | Used by |
|---|---|---|
| `legacy/20260630_build_valisure_fei_inspection_history.py` | One-off from 2026-06-30 for the Metformin paper. Merged the old Sept 2025 metformin export with the June 2026 Valisure14 export into one event history for the ~25 metformin FEIs, flagging each inspection as in both exports, old only or new only (`Insp_coverage`) | Nothing reads its output. **Cannot run as written**, see below |
| `legacy/valisure_fei_inspection_history.csv` | Output of the script above: 235 events, 24 FEIs | Nothing |
| `redica_all_drugs_combined_mismatches_report.csv` | QA output of the main script | You, after a refresh |
| `legacy/ReadMe.rtf` | Copy of the discrepancy notes at the bottom of the main script | Nothing |
| `ndc_fei_73_v4.xlsx` | 2025 metformin NDC to FEI map (73 NDCs) | MQRI v01 and v02 |
| `Metformin Manufacturer Site Score Table.xlsx` | Redica delivery, June 2025: metformin site scores from events 2018-01-01 to 2021-09-01 | Nothing in code |
| `Metformin NDCs labelers DUNS mfgr with Redica data.xlsx` | Redica delivery, June 2025: metformin NDCs linked to Redica site ids | Nothing in code |
| `redica_merged_recalls.xlsx` | Oct 2025 metformin work, Redica sites joined to recalls | Nothing in code |

### What "panel" did in the inspection-history script

`PANEL_CSV` (`metformin_panel_v1.csv`) was the June 2026 Metformin analysis panel. The
script used it for three things only:

1. the FEI universe: events were kept only for FEIs in the panel (`panel_feis`)
2. `Firm` and `CountryCode` labels per FEI
3. `FEI_in_old`, whether the FEI was in the older Redica export

That panel was replaced by the July 2026 step 1-6 pipeline in `Analysis/Metformin Paper/`
and no longer exists. The other two inputs are also stale names:
`METFORMIN_SITE_RED_FLAG_EVENTS.xlsx` is now
`raw/MetformoinValisure_Site_Red_Flag_Events_RedicaSep25.xlsx` and `Site List.xlsx` is now
`raw/Valisure14_Site_List.xlsx`. It is **not part of the refresh** above; the 14-API file
comes only from `20260505_redica_all_drugs_combined.py`.

## What is in `raw/`, by delivery

| Delivery | Files | Scope |
|---|---|---|
| June 2025, metformin (email from Yelena, see `docs/Redica Data (email note).txt`) | 16 files named `1000xxxxx - Firm [City _ Country].xlsx`, one per site, red-flag events and site score | 16 metformin sites. Read by the Dec 2025 dashboard in `Analysis/Dashboards/processed/code/`, which globs every `.xlsx` under `raw/` |
| Sept 2025, metformin + Valisure | `MetformoinValisure_Site_Red_Flag_Events_RedicaSep25.xlsx`, `..._Agg_Score_RedicaSep25.xlsx`, `..._Site_List_RedicaSep25.xlsx` | Events and scores for 19 FEIs; the site list has all 127 |
| June 2026, 14 Valisure APIs | `Valisure14_Sites_Red_Flag_Events.xlsx`, `Valisure14_Sites_Data_Availability.xlsx`, `Valisure14_Site_List.xlsx`, `Valisure14_FDA_483_Observations_WL_Deficiencies_OSU.xlsx` | 127 FEIs. **The current inputs** |
| July 2026, metformin refresh | `MetfrmoinValisure_Red_Flag_Events_RedicaJuly26.xlsx`, `..._Data_Availability_RedicaJuly26.xlsx`, `..._FEI_RedicaID_Mapping_RedicaJuly26.xlsx`, `..._483_Obs_WL_Def_RedicaJuly26.xlsx`, `metformin_2025/MetforminFEI-Redica-July26.xlsx` | 29 metformin FEIs. Read by `Analysis/Metformin Paper/processed/code/step2_build_panel_july26.py` and `build_variant_graphs.py` |

`docs/` holds Redica's data documentation, field definitions and sample site reports.
