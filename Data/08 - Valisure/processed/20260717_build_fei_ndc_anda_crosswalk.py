"""
20260717_build_fei_ndc_anda_crosswalk.py
─────────────────────────────────────────────────────────────────────────────
Build a master crosswalk: FEI | NDC | ANDA | Site Redica ID | API | Labeler,
then compute ANDA-specific FAERS serious AE counts (yearly and quarterly).

Why this matters
────────────────
The prior pipeline joined FAERS to FEIs by drug name (api_key), assigning ALL
drug-level AEs to every manufacturer of that drug. For atorvastatin (15 FEIs)
that means each FEI was assigned all 24,801 serious AEs regardless of who made
the product the patient took. This file enables joining FAERS on appl_no
(= ANDA number), which limits AEs to the specific manufacturer's product.

One ANDA can cover multiple manufacturing sites (FEIs). When that happens, the
FAERS AE count is shared across all sites holding that ANDA — the n_feis_sharing
column flags this so callers can decide how to handle it.

Outputs
───────
  valisure_fei_ndc_anda_crosswalk.csv        — master crosswalk (one row per NDC)
  valisure_anda_faers_ae_counts_yearly.csv   — ANDA × FEI × year serious AE counts
  valisure_anda_faers_ae_counts_quarterly.csv— ANDA × FEI × quarter serious AE counts

Coverage
────────
  128 FEIs, 194 unique application numbers, 501 NDC rows (Valisure-tested only).
  FAERS match rate: ~47.5% of serious AEs in the 14-drug pre-filtered file.
  Unmatched AEs = manufacturers not tested by Valisure, not in scope.
  FEIs with no matched FAERS AEs: ~23 of 98 panel FEIs (usually NDA products
  or ANDAs not referenced in FAERS by reporters).
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data"
HERE = Path(__file__).resolve().parent

VALISURE_XLSX = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"
REDICA_CSV    = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
FAERS_CSV     = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.csv"

OUT_XWALK   = HERE / "valisure_fei_ndc_anda_crosswalk.csv"
OUT_AE_YR   = HERE / "valisure_anda_faers_ae_counts_yearly.csv"
OUT_AE_QTR  = HERE / "valisure_anda_faers_ae_counts_quarterly.csv"

SERIOUS = {
    "Death", "Hospitalization", "Life-threatening", "Disability",
    "Congenital anomaly", "Required intervention", "Other serious",
}


def build_crosswalk() -> pd.DataFrame:
    xwalk = pd.read_excel(VALISURE_XLSX, sheet_name="NDC_FEI Mapping")
    xwalk = xwalk.rename(columns={
        "NDC":                "ndc",
        "NDC9":               "ndc9",
        "Application Number": "application_number",
        "API":                "api",
        "Labeler":            "labeler",
        "FEI_NUMBER":         "fei",
    })
    xwalk["fei"] = pd.to_numeric(xwalk["fei"], errors="coerce").astype("Int64")

    # Strip ANDA/NDA prefix to get numeric application number for FAERS join
    xwalk["appl_num"] = pd.to_numeric(
        xwalk["application_number"].str.replace(r"^[A-Z]+", "", regex=True).str.strip(),
        errors="coerce",
    ).astype("Int64")

    # Add Site Redica ID and site display name from Redica combined file
    rc = pd.read_csv(REDICA_CSV)
    redica_map = (
        rc[["FEI", "Site Redica Id", "Site Display Name"]]
        .drop_duplicates(subset=["FEI"])
        .rename(columns={
            "FEI":              "fei",
            "Site Redica Id":   "site_redica_id",
            "Site Display Name":"site_display_name",
        })
    )
    redica_map["fei"] = pd.to_numeric(redica_map["fei"], errors="coerce").astype("Int64")
    xwalk = xwalk.merge(redica_map, on="fei", how="left")

    col_order = [
        "fei", "site_redica_id", "site_display_name",
        "api", "labeler",
        "ndc", "ndc9", "application_number", "appl_num",
    ]
    xwalk = xwalk[[c for c in col_order if c in xwalk.columns]]

    print(f"Crosswalk: {len(xwalk)} rows, {xwalk['fei'].nunique()} FEIs, "
          f"{xwalk['appl_num'].nunique()} application numbers")
    print(f"FEIs without Site Redica ID: {xwalk['site_redica_id'].isna().sum()}")
    return xwalk


def build_ae_counts(xwalk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ── Load FAERS, filter to serious + our application numbers ──────────────
    faers = pd.read_csv(FAERS_CSV, low_memory=False)
    faers_ser = faers[faers["severity"].isin(SERIOUS)].copy()
    faers_ser["appl_no"] = pd.to_numeric(faers_ser["appl_no"], errors="coerce")
    faers_ser = faers_ser.dropna(subset=["appl_no"])
    faers_ser["appl_no"] = faers_ser["appl_no"].astype(int)

    our_andas = set(xwalk["appl_num"].dropna().astype(int))
    matched = faers_ser[faers_ser["appl_no"].isin(our_andas)].copy()

    total = len(faers_ser)
    print(f"\nFAERS serious rows: {total}")
    print(f"Matched to Valisure ANDAs: {len(matched)} ({100*len(matched)/total:.1f}%)")

    # ── ANDA → FEI map (one ANDA can map to multiple FEIs) ───────────────────
    anda_fei = (
        xwalk[["appl_num", "fei", "api", "labeler", "application_number"]]
        .drop_duplicates(subset=["appl_num", "fei"])
        .rename(columns={"appl_num": "appl_no"})
    )
    anda_fei["appl_no"] = anda_fei["appl_no"].astype("Int64")
    anda_fei["fei"]     = anda_fei["fei"].astype("Int64")

    # Flag ANDAs shared across multiple FEIs
    n_sharing = (anda_fei.groupby("appl_no")["fei"].nunique()
                          .rename("n_feis_sharing_anda").reset_index())
    anda_fei = anda_fei.merge(n_sharing, on="appl_no", how="left")

    matched["appl_no"] = matched["appl_no"].astype("Int64")

    # ── Yearly AE counts ──────────────────────────────────────────────────────
    matched["year"] = pd.to_numeric(matched["year"], errors="coerce").astype("Int64")
    ae_yr = (matched.groupby(["appl_no", "year"], as_index=False)
                    .agg(n_ae_serious=("primaryid", "count")))
    ae_yr = ae_yr.merge(anda_fei, on="appl_no", how="left")

    # ── Quarterly AE counts ───────────────────────────────────────────────────
    ae_qtr = (matched.dropna(subset=["period"])
                     .groupby(["appl_no", "period"], as_index=False)
                     .agg(n_ae_serious=("primaryid", "count")))
    ae_qtr = ae_qtr.merge(anda_fei, on="appl_no", how="left")

    print(f"Yearly output:    {len(ae_yr)} rows, {ae_yr['fei'].nunique()} FEIs")
    print(f"Quarterly output: {len(ae_qtr)} rows, {ae_qtr['fei'].nunique()} FEIs")
    return ae_yr, ae_qtr


def main() -> None:
    print("Building FEI-NDC-ANDA crosswalk...")
    xwalk = build_crosswalk()
    xwalk.to_csv(OUT_XWALK, index=False)
    print(f"Saved → {OUT_XWALK.name}")

    print("\nBuilding ANDA-level FAERS AE counts...")
    ae_yr, ae_qtr = build_ae_counts(xwalk)
    ae_yr.to_csv(OUT_AE_YR, index=False)
    ae_qtr.to_csv(OUT_AE_QTR, index=False)
    print(f"Saved → {OUT_AE_YR.name}")
    print(f"Saved → {OUT_AE_QTR.name}")

    # Quick validation: Sun Pharma atorvastatin (FEI 3002807979, ANDA 076477)
    print("\nValidation — FEI 3002807979 (Sun Pharma, ANDA 076477), yearly:")
    print(ae_yr[ae_yr["fei"] == 3002807979]
          [["year", "n_ae_serious", "n_feis_sharing_anda", "application_number"]]
          .sort_values("year").to_string(index=False))

    # Check how many panel FEIs have quarterly AE data
    import numpy as np
    panel_feis = set(ae_qtr["fei"].dropna().astype(int).unique())
    print(f"\nFEIs with quarterly AE data: {len(panel_feis)}")


if __name__ == "__main__":
    main()
