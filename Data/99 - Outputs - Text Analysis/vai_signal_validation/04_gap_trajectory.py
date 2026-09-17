"""
04_gap_trajectory.py
----------------------------------------------------------------------------
Rebuild of the INFORMS 2026 "AEs Rise in the Year Before FDA Arrives" backup
slide, using the CURRENT validated pipeline.

Original: old_not_current_pipeline/pdf_source_and_superseded/ae_validation/
00_slide_stats.py::slide12_gap_trajectory() (unmodified code, moved
2026-09-16). Ported here to read from 01_build_inspection_panel.py's panel
(current text extraction, FDA-primary OAI/VAI source) instead of the
INFORMS-era panel; the gap computation itself is unchanged (still derived
from the Redica raw inspection-event files, which this analysis has never
depended on the 483-text pipeline for).

Method: for each inspection in the panel, compute gap_yr = years since that
facility's previous FDA inspection (any classification), from the Redica
raw event log -- not the 483-text-scored subset, the full inspection
history. Bucket into <1yr, 1-2yr, 2-3.5yr, >3.5yr. Within each bucket,
average the AE counts at each quarterly lag (n_ae_tm4 ... n_ae_tp4) and
compute:
  pre_rise = mean(AE at Q0) / mean(AE at Q-4)   -- were AEs already
             elevated relative to a year before the inspection?
  persist  = mean(AE at Q+4) / mean(AE at Q0)   -- did AEs fall, stay flat,
             or keep rising after the inspection?

The 1-2yr bucket is the "clean baseline" per the original interpretation:
Q-4 falls after the facility's prior inspection (AEs suppressed by that
visit), so pre_rise there isn't contaminated by an even-earlier enforcement
action.

Usage
-----
  python 04_gap_trajectory.py            drug-level AE panel (matches the
                                          slide's "manufacturer-specific
                                          AEs" caption)
  python 04_gap_trajectory.py --anda-ae  ANDA-specific AE panel

Outputs
-------
  outputs/tables/gap_trajectory.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DATA = ROOT / "Data"
OUT      = HERE / "outputs"
OUT_TABS = OUT / "tables"
PANEL      = OUT / "fei_ae_panel_inspection_centered.parquet"
PANEL_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

REDICA_XLSX = DATA / "07 - Redica" / "raw" / "Valisure14_Sites_Red_Flag_Events.xlsx"
SITE_LIST   = DATA / "07 - Redica" / "raw" / "Valisure14_Site_List.xlsx"


def _compute_gap_years() -> pd.DataFrame:
    """FEI x inspection-year -> years since that facility's previous FDA
    inspection (any classification), from the full Redica inspection-event
    log, not just the 483-text-scored subset."""
    sl = pd.read_excel(SITE_LIST)
    ev = pd.read_excel(REDICA_XLSX)
    ev["Event Date"] = pd.to_datetime(ev["Event Date"])

    insp = ev[
        (ev["Event Type"] == "Inspection") &
        (ev["Agency List"].str.contains("US - FDA", na=False))
    ].merge(sl[["Site Redica Id", "FEI"]], on="Site Redica Id", how="left")

    insp = insp.sort_values(["FEI", "Event Date"])
    insp["prev_date"] = insp.groupby("FEI")["Event Date"].shift(1)
    insp["gap_yr"] = (insp["Event Date"] - insp["prev_date"]).dt.days / 365.25
    insp["FEI"] = insp["FEI"].astype(str)
    insp["insp_year"] = insp["Event Date"].dt.year

    return insp[["FEI", "insp_year", "gap_yr"]].drop_duplicates(["FEI", "insp_year"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anda-ae", dest="anda_ae", action="store_true",
                        help="Use ANDA-specific AE panel instead of drug-level panel.")
    args = parser.parse_args()
    panel_path = PANEL_ANDA if args.anda_ae else PANEL

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_inspection_panel.py first.")

    print(f"Loading panel ({'ANDA-specific' if args.anda_ae else 'drug-level'})...")
    ic = pd.read_parquet(panel_path)
    print(f"  {len(ic)} inspection events, {ic['fei'].nunique()} FEIs")

    print("Computing gaps-since-previous-inspection from Redica raw event log...")
    gaps = _compute_gap_years()

    ic_key = ic[["fei", "insp_year"]].copy()
    ic_key["fei"] = ic_key["fei"].astype(str)
    merged = ic_key.merge(gaps, left_on=["fei", "insp_year"], right_on=["FEI", "insp_year"], how="left")
    ic["gap_yr"] = merged["gap_yr"].values
    print(f"  Matched gap for {ic['gap_yr'].notna().sum()}/{len(ic)} inspections")

    bins   = [0, 1, 2, 3.5, np.inf]
    labels = ["<1yr", "1-2yr", "2-3.5yr", ">3.5yr"]
    ic["gap_cat"] = pd.cut(ic["gap_yr"], bins=bins, labels=labels)

    lag_cols = [c for c in ic.columns if c.startswith("n_ae_")]
    grp = ic.groupby("gap_cat", observed=True)[lag_cols].mean().round(3)
    grp["n"] = ic.groupby("gap_cat", observed=True).size()
    if "n_ae_tm4" in grp.columns and "n_ae_t0" in grp.columns and "n_ae_tp4" in grp.columns:
        grp["pre_rise"] = (grp["n_ae_t0"] / grp["n_ae_tm4"].replace(0, np.nan)).round(3)
        grp["persist"]  = (grp["n_ae_tp4"] / grp["n_ae_t0"].replace(0, np.nan)).round(3)

    print(f"\nAE trajectory by gap-since-previous-inspection (n={ic['gap_yr'].notna().sum()} with a matched gap):")
    print(grp.to_string())

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_TABS / ("gap_trajectory_anda.csv" if args.anda_ae else "gap_trajectory.csv")
    grp.to_csv(out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
