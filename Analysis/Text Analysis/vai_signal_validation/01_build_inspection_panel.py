"""
01_build_inspection_panel.py
────────────────────────────────────────────────────────────────────────────
Rebuild of the INFORMS 2026 "VAI-only subgroup" inspection panel, using the
CURRENT validated 483-text pipeline instead of the pre-validation one the
INFORMS slides were built on.

What changed vs. the archived version
(../old_not_current_pipeline/pdf_source_and_superseded/ae_validation/01_build_fei_ae_panel.py):
  - TEXT_TS_CSV now points at step02_..._redica_claudesonnet5_v2.csv (Claude
    Sonnet 5, human-eval-validated v2 prompt) instead of the old
    step02_..._redica.csv (pre-fix, pre-validation, likely built with an
    earlier/default model). This is the same file m14/m17 in Shortage
    Prediction use.
  - Two TEXT_FEATURES renamed to match the v2 schema's FDA six-system
    scheme: vc_labcontrols_share -> vc_laboratorycontrolssystem_share,
    n_labcontrols_obs -> n_laboratorycontrolssystem_obs. All other 15
    feature names are unchanged between v1 and v2.
  - Everything else (FAERS matching, SDUD volume, OAI/VAI resolution via
    Redica + FDA Drug QA 3-pass fallback) is unchanged from the original.

OAI/VAI/NAI classification source (updated 2026-10-19):
step00_redica_inspections.csv from 00_load_redica_obs.py. The grade is the
"Drug Quality Assurance" label in column F of Redica's
Valisure14_Sites_Red_Flag_Events.xlsx, matched on site and exact date, and
inspections graded only under another FDA program (devices, bioresearch,
pre-approval, blood, animal drugs) or not tagged Human Drugs are dropped.
This replaces the FDA Inspections Details / redica_all_drugs_combined /
+-30 day 3-pass used through 2026-10-15.

Usage
─────
  python 01_build_inspection_panel.py                 ← drug-level FAERS match
  python 01_build_inspection_panel.py --anda-ae        ← ANDA-specific FAERS match

Outputs
───────
  outputs/fei_ae_panel_inspection_centered.parquet
  outputs/fei_ae_panel_inspection_centered_anda.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent.parent.parent
DATA   = ROOT / "Data"
OUT    = HERE / "outputs"

# Current validated redica v2 text timeseries (Claude Sonnet 5).
TEXT_TS_CSV  = DATA.parent / "Analysis" / "Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
TEXT_TS_PRODSYS = DATA.parent / "Analysis" / "Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2_prodsys.csv"
SYSTEMS = "all"   # overridden by --systems
FAERS_PARQ   = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"
REDICA_INSPECTIONS_CSV = DATA.parent / "Analysis" / "Text Analysis" / "step00_redica_inspections.csv"
SDUD_PARQ    = DATA / "04 - Medicaid - SDUD" / "processed" / "2025-12-18-SDUDcanonical.parquet"
NDC_FEI_CSV  = DATA / "17 - NDC-FEI Linkage" / "processed" / "ndc_fei_from_labels.csv"
ANDA_AE_QTR_CSV = DATA / "08 - Valisure" / "processed" / "valisure_anda_faers_ae_counts_quarterly.csv"
# 2026-09-30 alternative: ProPublica-linked ANDA counts on the expanded drug
# list, restricted to NDCs with exactly one FEI and one ANDA so every adverse
# event traces to a single facility. Built by
# 08 - Valisure/processed/20260930_build_propublica_anda_faers.py
# 2026-10-15: counts each FAERS case once, in its first serious quarter, instead
# of every report version (built by the same script with --count-unit case,
# its default). The per-version file, propublica_fei_ae_quarterly_single.csv,
# is kept; the panels built from it are saved in outputs/ with a
# _saved_before_20261015 suffix.
# FAERS_REACTIONS=quality (environment, 2026-10-19) switches to product-quality
# reports only (built with --reactions quality).
_RX = "_" + os.environ["FAERS_REACTIONS"] if os.environ.get("FAERS_REACTIONS") else ""
PROPUBLICA_AE_QTR_CSV = (DATA / "08 - Valisure" / "processed" /
                         f"propublica_fei_ae_quarterly_single_cases{_RX}.csv")
ANDA_SOURCE = "valisure"   # overridden by --anda-source

OUT_PANEL_INSP      = OUT / "fei_ae_panel_inspection_centered.parquet"
OUT_PANEL_INSP_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"
OUT_PANEL_INSP_PP   = OUT / "fei_ae_panel_inspection_centered_anda_propublica.parquet"

# v2 schema: 2 renames from the original (v1) feature list, rest unchanged.
#
# severity_critmajor_share (Critical+Major), NOT severity_majmod_share.
# Corrected 2026-09-30. majmod was chosen because human-eval accuracy is 90-94%
# for a Major/Moderate collapse against 66-68% for the raw 4-tier, the
# extraction's real confusion being at the Major/Moderate boundary. That reasoning
# was about measurement noise and missed a construct problem: (Major+Moderate)/total
# goes DOWN as an inspection gets worse, because Critical observations sit outside
# the numerator. Measured correlation with severity_critical_share is -0.842, and
# 54.5% of snapshots sit at exactly 1.0, so the feature was both inverted and
# saturated. critmajor collapses the same noisy Major boundary while staying
# ordered by severity. Swapping it in moved text-only RF AUC from 0.619 (p=0.077)
# to 0.648 (p=0.032) on the clean attribution panel, and lifted the VAI-only
# subgroup from 0.415 (below chance) to 0.508.
TEXT_FEATURES = [
    "severity_critmajor_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "cultural_root_cause_share",
    "vc_laboratorycontrolssystem_share",   # was vc_labcontrols_share (v1)
    "vc_qualitysystem_share",
    "n_laboratorycontrolssystem_obs",      # was n_labcontrols_obs (v1)
    "n_qualitysystem_obs",
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]


# ── Text timeseries ───────────────────────────────────────────────────────────

def _load_text_timeseries() -> pd.DataFrame:
    src = TEXT_TS_PRODSYS if SYSTEMS == "product" else TEXT_TS_CSV
    print(f"  text features: {src.name}")
    ts = pd.read_csv(src, low_memory=False)
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts = ts.dropna(subset=["fei"])
    keep = ["fei", "snapshot_date"] + TEXT_FEATURES
    missing = [c for c in TEXT_FEATURES if c not in ts.columns]
    if missing:
        raise KeyError(f"TEXT_FEATURES not found in {TEXT_TS_CSV.name}: {missing}")
    return ts[[c for c in keep if c in ts.columns]].copy()


# ── Valisure FEI map ──────────────────────────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    vm = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    vm.columns = [c.strip() for c in vm.columns]
    api_col = next(c for c in vm.columns if c.lower() == "api")
    fei_col = next(c for c in vm.columns if "fei" in c.lower() and "unique" not in c.lower())
    fm = vm[[api_col, fei_col]].rename(columns={api_col: "api", fei_col: "fei"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
    fm["api_key"] = (fm["api"].str.strip().str.lower()
                               .str.replace(r"[^\w\s]", "", regex=True)
                               .str.split().str[0])
    return fm.dropna(subset=["fei"]).drop_duplicates()


# ── FAERS ─────────────────────────────────────────────────────────────────────

def _load_faers_raw(fei_drug_map: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(FAERS_PARQ)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "prod_ai"])
    SERIOUS = {
        "Death", "Hospitalization", "Life-threatening",
        "Disability", "Congenital anomaly", "Required intervention",
        "Other serious",
    }
    before = len(df)
    df = df[df["severity"].isin(SERIOUS)]
    print(f"  Severity filter: {before} -> {len(df)} rows "
          f"(dropped {before - len(df)} 'No outcome reported')")
    df["api_key"] = df["prod_ai"].str.strip().str.lower().str.split().str[0]
    joined = df.merge(
        fei_drug_map[["api_key", "fei", "api"]].drop_duplicates(),
        on="api_key", how="inner"
    )
    return joined


def _faers_quarterly(joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.dropna(subset=["period"])
    agg = (joined.groupby(["fei", "period"], as_index=False)
                 .agg(n_ae=("primaryid", "count"),
                      n_drug_fei_pairs=("api", "nunique")))
    agg["ae_year"] = agg["period"].str[:4].astype(int)
    agg["ae_qtr"]  = agg["period"].str[-1].astype(int)
    agg["ae_idx"]  = agg["ae_year"] * 4 + agg["ae_qtr"]
    return agg


def _load_anda_ae_quarterly() -> pd.DataFrame:
    path = (PROPUBLICA_AE_QTR_CSV if ANDA_SOURCE == "propublica"
            else ANDA_AE_QTR_CSV)
    print(f"  ANDA AE source: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fei", "period", "n_ae_serious"])
    df["ae_year"] = df["period"].str[:4].astype(int)
    df["ae_qtr"]  = df["period"].str[-1].astype(int)
    df["ae_idx"]  = df["ae_year"] * 4 + df["ae_qtr"]
    agg = (df.groupby(["fei", "ae_idx"], as_index=False)
             .agg(n_ae=("n_ae_serious", "sum")))
    print(f"  ANDA-specific AE quarterly: {len(agg)} FEI x quarter rows, "
          f"{agg['fei'].nunique()} FEIs")
    return agg


# ── Inspection outcomes ───────────────────────────────────────────────────────

# ── SDUD volume ───────────────────────────────────────────────────────────────

def _load_sdud_volume(panel_feis: set) -> pd.DataFrame:
    nf = pd.read_csv(NDC_FEI_CSV, low_memory=False)
    nf["fei"] = pd.to_numeric(nf["FEI_NUMBER"], errors="coerce").astype("Int64")
    nf = nf[nf["fei"].isin(panel_feis)].copy()
    nf["ndc_9"] = nf["manufacture_ndc"].str.strip()
    nf = nf[["ndc_9", "fei"]].dropna().drop_duplicates()
    our_ndcs = set(nf["ndc_9"])
    if not our_ndcs:
        print("    WARNING: No NDCs matched -- SDUD volume will be missing.")
        return pd.DataFrame(columns=["fei", "panel_year", "sdud_units", "sdud_rx"])

    sdud = pd.read_parquet(
        SDUD_PARQ,
        columns=["labeler_code", "product_code", "year", "units_reimbursed", "num_prescriptions"]
    )
    sdud["ndc_9"] = (
        sdud["labeler_code"].str.strip().str.zfill(5) + "-" +
        sdud["product_code"].str.strip().str.zfill(4)
    )
    sdud = sdud[sdud["ndc_9"].isin(our_ndcs)].copy()
    if sdud.empty:
        return pd.DataFrame(columns=["fei", "panel_year", "sdud_units", "sdud_rx"])

    sdud = sdud.merge(nf, on="ndc_9", how="inner")
    sdud["units"]      = pd.to_numeric(sdud["units_reimbursed"], errors="coerce")
    sdud["rx"]         = pd.to_numeric(sdud["num_prescriptions"], errors="coerce")
    sdud["panel_year"] = pd.to_numeric(sdud["year"], errors="coerce").astype("Int64")

    vol = (sdud.groupby(["fei", "panel_year"], as_index=False)
               .agg(sdud_units=("units", "sum"), sdud_rx=("rx", "sum")))
    return vol


# ── Inspection-event-centered panel ────────────────────────────────────────────

def build_inspection_centered(ts: pd.DataFrame, fei_drug_map: pd.DataFrame,
                               use_anda_ae: bool = False) -> None:
    print("\n-- Building INSPECTION-CENTERED quarterly panel --")

    ts = ts.copy()
    ts["insp_year"] = ts["snapshot_date"].dt.year
    ts["insp_qtr"]  = ts["snapshot_date"].dt.quarter
    ts["insp_idx"]  = ts["insp_year"] * 4 + ts["insp_qtr"]
    print(f"  {len(ts)} inspection events, {ts['fei'].nunique()} FEIs")

    if use_anda_ae:
        print("Loading FAERS (ANDA-specific quarterly)...")
        ae_q = _load_anda_ae_quarterly()
    else:
        print("Loading FAERS (drug-level quarterly)...")
        joined = _load_faers_raw(fei_drug_map)
        ae_q   = _faers_quarterly(joined)

    MAX_LAG = 4
    rows = []
    for _, insp in ts.iterrows():
        fei      = insp["fei"]
        insp_idx = insp["insp_idx"]
        fei_ae   = ae_q[ae_q["fei"] == fei].set_index("ae_idx")["n_ae"]

        row = {
            "fei":          fei,
            "insp_date":    insp["snapshot_date"],
            "insp_year":    insp["insp_year"],
            "insp_qtr":     insp["insp_qtr"],
            "insp_period":  f"{insp['insp_year']}Q{insp['insp_qtr']}",
        }
        for f in TEXT_FEATURES:
            if f in insp.index:
                row[f] = insp[f]

        for lag in range(-MAX_LAG, MAX_LAG + 1):
            if lag < 0:
                suffix = f"tm{abs(lag)}"
            elif lag == 0:
                suffix = "t0"
            else:
                suffix = f"tp{lag}"
            target_idx = insp_idx + lag
            row[f"n_ae_{suffix}"] = fei_ae.get(target_idx, np.nan)

        rows.append(row)

    panel = pd.DataFrame(rows)
    print(f"  {len(panel)} inspection-event rows built")

    # ── Inspection grade (2026-10-19) ──
    # Redica's Valisure14_Sites_Red_Flag_Events.xlsx, column F, "Drug Quality
    # Assurance" label only, matched on site and exact date in
    # 00_load_redica_obs.py. Replaces the old FDA/Redica/near-match 3-pass.
    # Unknown grade stays all zeros, as before; 07 treats it as UNKNOWN.
    insp = pd.read_csv(REDICA_INSPECTIONS_CSV)
    insp = insp[insp["keep"]]
    insp["insp_date"] = pd.to_datetime(insp["insp_date"])
    insp["dqa_grade"] = insp["dqa_grade"].fillna("")
    panel["insp_date"] = pd.to_datetime(panel["insp_date"])
    n0 = len(panel)
    panel = panel.merge(insp[["fei", "insp_date", "dqa_grade"]],
                        on=["fei", "insp_date"], how="inner")
    print(f"  {len(panel)} of {n0} inspections are kept drug-quality inspections")
    panel["n_oai"] = (panel["dqa_grade"] == "OAI").astype(int)
    panel["n_vai"] = (panel["dqa_grade"] == "VAI").astype(int)
    panel["n_nai"] = (panel["dqa_grade"] == "NAI").astype(int)
    panel["any_oai"] = panel["n_oai"]
    print(f"  grade: {panel['dqa_grade'].replace('', 'unknown').value_counts().to_dict()}")
    panel = panel.drop(columns=["dqa_grade"])

    print("Loading SDUD volume...")
    panel_feis = set(panel["fei"].dropna().unique())
    vol = _load_sdud_volume(panel_feis)
    vol = vol.rename(columns={"panel_year": "insp_year"})
    panel = panel.merge(vol, on=["fei", "insp_year"], how="left")
    panel["ae_rate_t0"] = np.where(
        panel["sdud_units"] > 0,
        panel["n_ae_t0"] / (panel["sdud_units"] / 1e6),
        np.nan
    )

    ae_window = ["n_ae_tm4", "n_ae_tm3", "n_ae_tm2", "n_ae_tm1",
                 "n_ae_t0",
                 "n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"]

    print(f"\nInspection-centered quarterly summary:")
    print(f"  {len(panel)} inspection events, {panel['fei'].nunique()} FEIs")
    print(f"  OAI inspections: {panel['any_oai'].sum()} ({panel['any_oai'].mean():.1%})")

    OUT.mkdir(parents=True, exist_ok=True)
    if use_anda_ae:
        out_path = (OUT_PANEL_INSP_PP if ANDA_SOURCE == "propublica"
                    else OUT_PANEL_INSP_ANDA)
    else:
        out_path = OUT_PANEL_INSP
    if SYSTEMS == "product":
        out_path = out_path.with_name(out_path.stem + "_prodsys" + out_path.suffix)
    panel.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anda-ae", action="store_true",
        help="Use ANDA-specific FAERS AE counts instead of drug-level counts.",
    )
    parser.add_argument(
        "--systems", choices=["all", "product"], default="all",
        help="'product' uses the product-proximate text features only "
             "(02_aggregate --systems product).",
    )
    parser.add_argument(
        "--anda-source", choices=["valisure", "propublica"], default="valisure",
        help="'valisure' = hand-built March 2026 mapping, 14 APIs. "
             "'propublica' = ProPublica linkage on the expanded list, "
             "restricted to one-FEI one-ANDA NDCs so attribution is unambiguous.",
    )
    args = parser.parse_args()
    globals()["ANDA_SOURCE"] = args.anda_source
    globals()["SYSTEMS"] = args.systems

    print("Loading text timeseries...")
    ts = _load_text_timeseries()
    print(f"  {len(ts)} inspection rows, {ts['fei'].nunique()} FEIs")

    print("Loading Valisure FEI map...")
    fei_drug_map = _load_fei_drug_map()
    print(f"  {fei_drug_map['api_key'].nunique()} APIs, {fei_drug_map['fei'].nunique()} FEIs")

    build_inspection_centered(ts, fei_drug_map, use_anda_ae=args.anda_ae)


if __name__ == "__main__":
    main()
