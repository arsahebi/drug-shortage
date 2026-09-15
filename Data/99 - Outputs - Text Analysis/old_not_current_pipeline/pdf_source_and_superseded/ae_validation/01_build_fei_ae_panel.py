"""
01_build_fei_ae_panel.py
────────────────────────────────────────────────────────────────────────────
Build FEI × time panel joining LLM text signals with FAERS adverse event
counts and SDUD Medicaid volume for each facility.

Usage
─────
  python 01_build_fei_ae_panel.py --granularity yearly      ← default
  python 01_build_fei_ae_panel.py --granularity quarterly
  python 01_build_fei_ae_panel.py --granularity both

Design
──────
Text features (as-of join):
  For each FEI × time period t, use the most recent inspection snapshot
  with snapshot_date ≤ end of t. If no snapshot exists yet, FEI-period
  is excluded.

AE outcome (FAERS):
  Pre-filtered parquet: 14 Valisure drugs, serious AEs, 2015–2026.
  Joined via api_key → Valisure FEI map → FEI.
  Yearly:    aggregated to FEI × year (lags: t-2 to t+2 years)
  Quarterly: aggregated to FEI × quarter (lags: t-4 to t+4 quarters = ±1 year)

Volume control (SDUD):
  Medicaid utilization (units + prescriptions) joined via NDC → FEI.
  Used to compute AE rate = n_ae / sdud_units (controls for facility size).

Outputs
───────
  outputs/fei_ae_panel.parquet             — yearly panel
  outputs/fei_ae_panel_quarterly.parquet   — quarterly panel
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent.parent.parent
DATA   = ROOT / "Data"
OUT    = HERE / "outputs"

TEXT_TS_CSV  = DATA / "99 - Outputs - Text Analysis" / "step02_483_fei_text_features_timeseries_redica.csv"
FAERS_PARQ   = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"
REDICA_COMBINED = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
FDA_INSP_XLSX   = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
SDUD_PARQ    = DATA / "04 - Medicaid - SDUD" / "processed" / "2025-12-18-SDUDcanonical.parquet"
NDC_FEI_CSV  = DATA / "17 - NDC, FEI Mapping" / "ndc_fei_from_labels.csv"
ANDA_AE_QTR_CSV = DATA / "08 - Valisure" / "processed" / "valisure_anda_faers_ae_counts_quarterly.csv"

OUT_PANEL_YR        = OUT / "fei_ae_panel.parquet"
OUT_PANEL_QTR       = OUT / "fei_ae_panel_quarterly.parquet"
OUT_PANEL_INSP      = OUT / "fei_ae_panel_inspection_centered.parquet"
OUT_PANEL_INSP_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

TEXT_FEATURES = [
    "severity_critmajor_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "cultural_root_cause_share",
    "vc_labcontrols_share",
    "vc_qualitysystem_share",
    "n_labcontrols_obs",
    "n_qualitysystem_obs",
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]

PANEL_YEARS   = list(range(2018, 2026))
PANEL_PERIODS = [f"{y}Q{q}" for y in range(2015, 2026) for q in range(1, 5)]

_QTR_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


# ── Text timeseries ───────────────────────────────────────────────────────────

def _load_text_timeseries() -> pd.DataFrame:
    ts = pd.read_csv(TEXT_TS_CSV, low_memory=False)
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts = ts.dropna(subset=["fei"])
    keep = ["fei", "snapshot_date"] + TEXT_FEATURES
    return ts[[c for c in keep if c in ts.columns]].copy()


def _as_of_join_yearly(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in PANEL_YEARS:
        cutoff = pd.Timestamp(year, 12, 31)
        eligible = ts[ts["snapshot_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        latest = (eligible.sort_values("snapshot_date")
                          .groupby("fei", as_index=False).last())
        latest["panel_year"] = year
        rows.append(latest)
    return pd.concat(rows, ignore_index=True)


def _quarter_end(period: str) -> pd.Timestamp:
    y, q = int(period[:4]), int(period[-1])
    m, d = _QTR_END[q]
    return pd.Timestamp(y, m, d)


def _as_of_join_quarterly(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PANEL_PERIODS:
        cutoff = _quarter_end(period)
        eligible = ts[ts["snapshot_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        latest = (eligible.sort_values("snapshot_date")
                          .groupby("fei", as_index=False).last())
        latest["panel_period"] = period
        latest["panel_year"]   = int(period[:4])
        latest["panel_qtr"]    = int(period[-1])
        rows.append(latest)
    return pd.concat(rows, ignore_index=True)


# ── Valisure FEI map ──────────────────────────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    vm = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    vm.columns = [c.strip() for c in vm.columns]
    api_col = next(c for c in vm.columns if c.lower() == "api")
    fei_col = next(c for c in vm.columns if "fei" in c.lower() and "unique" not in c.lower())
    fm = vm[[api_col, fei_col]].rename(columns={api_col: "api", fei_col: "fei"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
    # Strip punctuation before taking first word so "Ampicillin; Sulbactam" → "ampicillin"
    # instead of "ampicillin;" (which would never match FAERS prod_ai).
    fm["api_key"] = (fm["api"].str.strip().str.lower()
                               .str.replace(r"[^\w\s]", "", regex=True)
                               .str.split().str[0])
    return fm.dropna(subset=["fei"]).drop_duplicates()


# ── FAERS ─────────────────────────────────────────────────────────────────────

def _load_faers_raw(fei_drug_map: pd.DataFrame) -> pd.DataFrame:
    """Join FAERS to FEI map on first word of prod_ai (lowercased).

    Match rate: ~99.1% of pre-filtered FAERS rows join successfully.
    The remaining ~0.9% are combination products whose prod_ai backslash-
    concatenates two drugs (e.g. "METFORMIN\\SITAGLIPTIN") so the first
    word resolves to the combo string rather than the single API name.
    Switching to an any-word match recovers those rows but mis-attributes
    them — "ATORVASTATIN CALCIUM" contains "calcium", which would match
    Calcium Gluconate FEIs instead of Atorvastatin ones. First-word-only
    is the safer tradeoff; the unmatched volume is negligible.
    """
    df = pd.read_parquet(FAERS_PARQ)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "prod_ai"])
    # Keep serious outcomes only. "No outcome reported" (14.6% of rows) is not
    # patient harm and would dilute the signal we are trying to predict.
    SERIOUS = {
        "Death", "Hospitalization", "Life-threatening",
        "Disability", "Congenital anomaly", "Required intervention",
        "Other serious",
    }
    before = len(df)
    df = df[df["severity"].isin(SERIOUS)]
    print(f"  Severity filter: {before} → {len(df)} rows "
          f"(dropped {before - len(df)} 'No outcome reported')")
    df["api_key"] = df["prod_ai"].str.strip().str.lower().str.split().str[0]
    joined = df.merge(
        fei_drug_map[["api_key", "fei", "api"]].drop_duplicates(),
        on="api_key", how="inner"
    )
    return joined


def _faers_yearly(joined: pd.DataFrame) -> pd.DataFrame:
    return (joined.groupby(["fei", "year"], as_index=False)
                  .agg(n_ae=("primaryid", "count"),
                       n_drug_fei_pairs=("api", "nunique")))


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
    """Load ANDA-specific quarterly FAERS counts built by 20260717_build_fei_ndc_anda_crosswalk.py.

    Returns the same shape as _faers_quarterly(): fei, ae_idx, n_ae.
    When a single FEI holds multiple ANDAs, counts are summed across ANDAs.
    FEIs with no ANDA match in FAERS naturally return no rows here, so
    those inspection events will have NaN AEs in the panel.
    """
    df = pd.read_csv(ANDA_AE_QTR_CSV, low_memory=False)
    df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fei", "period", "n_ae_serious"])
    df["ae_year"] = df["period"].str[:4].astype(int)
    df["ae_qtr"]  = df["period"].str[-1].astype(int)
    df["ae_idx"]  = df["ae_year"] * 4 + df["ae_qtr"]
    agg = (df.groupby(["fei", "ae_idx"], as_index=False)
             .agg(n_ae=("n_ae_serious", "sum")))
    print(f"  ANDA-specific AE quarterly: {len(agg)} FEI×quarter rows, "
          f"{agg['fei'].nunique()} FEIs")
    return agg


# ── Lag builders ──────────────────────────────────────────────────────────────

def _add_lags_yearly(panel: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    """
    5-year window centered on inspection year.
    tm2/tm1 = pre-inspection; t0/t1/t2 = post-inspection.
    """
    ae = ae.rename(columns={"year": "ae_year", "n_ae": "n_ae_raw"})
    for lag, suffix in [(-2, "tm2"), (-1, "tm1"), (0, "t0"), (1, "t1"), (2, "t2")]:
        ae_lag = ae.copy()
        ae_lag["panel_year"] = ae_lag["ae_year"] - lag
        ae_lag = ae_lag.rename(columns={"n_ae_raw": f"n_ae_{suffix}"})[
            ["fei", "panel_year", f"n_ae_{suffix}"]
        ]
        panel = panel.merge(ae_lag, on=["fei", "panel_year"], how="left")
    fei_pairs = ae.groupby("fei")["n_drug_fei_pairs"].max().reset_index()
    panel = panel.merge(fei_pairs, on="fei", how="left")
    return panel


def _add_lags_quarterly(panel: pd.DataFrame, ae_q: pd.DataFrame) -> pd.DataFrame:
    """
    9-quarter window: t-4 → t+4 (±1 year in quarterly resolution).
    Suffixes: tm4 tm3 tm2 tm1 t0 tp1 tp2 tp3 tp4.
    """
    panel = panel.copy()
    panel["panel_idx"] = panel["panel_year"] * 4 + panel["panel_qtr"]

    for lag in range(-4, 5):
        if lag < 0:
            suffix = f"tm{abs(lag)}"
        elif lag == 0:
            suffix = "t0"
        else:
            suffix = f"tp{lag}"
        ae_lag = ae_q[["fei", "ae_idx", "n_ae", "n_drug_fei_pairs"]].copy()
        ae_lag["panel_idx"] = ae_lag["ae_idx"] - lag
        ae_lag = ae_lag.rename(columns={"n_ae": f"n_ae_{suffix}"})[
            ["fei", "panel_idx", f"n_ae_{suffix}"]
        ]
        panel = panel.merge(ae_lag, on=["fei", "panel_idx"], how="left")

    fei_pairs = ae_q.groupby("fei")["n_drug_fei_pairs"].max().reset_index()
    panel = panel.merge(fei_pairs, on="fei", how="left")
    panel = panel.drop(columns=["panel_idx"])
    return panel


# ── Inspection outcomes ───────────────────────────────────────────────────────

def _load_inspection_outcomes() -> pd.DataFrame:
    """Year-level aggregation for yearly/quarterly panel builders."""
    df = pd.read_csv(REDICA_COMBINED)
    df.columns = [c.strip() for c in df.columns]
    df["fei"]  = pd.to_numeric(df["FEI"], errors="coerce").astype("Int64")
    df["year"] = pd.to_datetime(df["Event Date"], errors="coerce").dt.year
    df = df.dropna(subset=["fei", "year", "Classification"])
    df["is_oai"] = (df["Classification"].str.upper() == "OAI").astype(int)
    df["is_vai"] = (df["Classification"].str.upper() == "VAI").astype(int)
    df["is_nai"] = (df["Classification"].str.upper() == "NAI").astype(int)
    agg = (df.groupby(["fei", "year"], as_index=False)
             .agg(n_oai=("is_oai", "sum"), n_vai=("is_vai", "sum"), n_nai=("is_nai", "sum")))
    agg["any_oai"] = (agg["n_oai"] > 0).astype(int)
    return agg  # columns: fei, year, n_oai, n_vai, n_nai, any_oai


def _load_inspection_outcomes_by_date() -> pd.DataFrame:
    """Redica outcomes keyed by (fei, insp_date).
    Includes classifications inferred from enforcement signals (Warning Letter
    / Non-Compliant → OAI; Compliant → NAI) via the build script.
    Returns fei, insp_date (normalized to midnight), n_oai, n_vai, n_nai, any_oai.
    """
    df = pd.read_csv(REDICA_COMBINED)
    df.columns = [c.strip() for c in df.columns]
    df["fei"]       = pd.to_numeric(df["FEI"], errors="coerce").astype("Int64")
    df["insp_date"] = pd.to_datetime(df["Event Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["fei", "insp_date", "Classification"])
    cls = df["Classification"].str.upper()
    df["n_oai"]   = (cls == "OAI").astype(int)
    df["n_vai"]   = (cls == "VAI").astype(int)
    df["n_nai"]   = (cls == "NAI").astype(int)
    df["any_oai"] = df["n_oai"]
    return df[["fei", "insp_date", "n_oai", "n_vai", "n_nai", "any_oai"]].copy()


def _load_fda_drug_outcomes() -> pd.DataFrame:
    """FDA Drug Quality Assurance inspection outcomes.
    Used as fallback for inspections not matched in Redica Red Flag Events.
    Returns fei, insp_date (normalized), n_oai, n_vai, n_nai, any_oai.
    """
    fda = pd.read_excel(FDA_INSP_XLSX,
                        usecols=["FEI Number", "Inspection End Date",
                                 "Classification", "Project Area"])
    fda = fda[fda["Project Area"] == "Drug Quality Assurance"].copy()
    fda["fei"]       = pd.to_numeric(fda["FEI Number"], errors="coerce").astype("Int64")
    fda["insp_date"] = pd.to_datetime(fda["Inspection End Date"], errors="coerce").dt.normalize()
    fda = fda.dropna(subset=["fei", "insp_date", "Classification"])
    cls = fda["Classification"].str.upper()
    fda["n_oai"]   = cls.str.contains("OFFICIAL ACTION").astype(int)
    fda["n_vai"]   = cls.str.contains("VOLUNTARY ACTION").astype(int)
    fda["n_nai"]   = cls.str.contains("NO ACTION").astype(int)
    fda["any_oai"] = fda["n_oai"]
    return (fda[["fei", "insp_date", "n_oai", "n_vai", "n_nai", "any_oai"]]
            .drop_duplicates(subset=["fei", "insp_date"])
            .copy())


# ── SDUD volume ───────────────────────────────────────────────────────────────

def _load_sdud_volume(panel_feis: set) -> pd.DataFrame:
    """
    Build FEI × year Medicaid volume from SDUD via NDC → FEI crosswalk.
    Returns: fei, panel_year, sdud_units, sdud_rx
    """
    print("  Loading NDC-FEI crosswalk…")
    nf = pd.read_csv(NDC_FEI_CSV, low_memory=False)
    nf["fei"] = pd.to_numeric(nf["FEI_NUMBER"], errors="coerce").astype("Int64")
    nf = nf[nf["fei"].isin(panel_feis)].copy()
    # manufacture_ndc is already "LLLLL-PPPP" (5-digit labeler + 4-digit product)
    nf["ndc_9"] = nf["manufacture_ndc"].str.strip()
    nf = nf[["ndc_9", "fei"]].dropna().drop_duplicates()
    our_ndcs = set(nf["ndc_9"])
    print(f"    {len(our_ndcs)} NDC-9 codes for {len(panel_feis)} panel FEIs")

    if not our_ndcs:
        print("    WARNING: No NDCs matched — SDUD volume will be missing.")
        return pd.DataFrame(columns=["fei", "panel_year", "sdud_units", "sdud_rx"])

    print("  Reading SDUD parquet (this may take a moment)…")
    sdud = pd.read_parquet(
        SDUD_PARQ,
        columns=["labeler_code", "product_code", "year", "units_reimbursed", "num_prescriptions"]
    )
    # Build join key matching manufacture_ndc format
    sdud["ndc_9"] = (
        sdud["labeler_code"].str.strip().str.zfill(5) + "-" +
        sdud["product_code"].str.strip().str.zfill(4)
    )
    sdud = sdud[sdud["ndc_9"].isin(our_ndcs)].copy()
    print(f"    SDUD rows matched to panel FEIs: {len(sdud):,}")

    if sdud.empty:
        print("    WARNING: SDUD filter produced no rows — check NDC format.")
        return pd.DataFrame(columns=["fei", "panel_year", "sdud_units", "sdud_rx"])

    sdud = sdud.merge(nf, on="ndc_9", how="inner")
    sdud["units"]      = pd.to_numeric(sdud["units_reimbursed"], errors="coerce")
    sdud["rx"]         = pd.to_numeric(sdud["num_prescriptions"], errors="coerce")
    sdud["panel_year"] = pd.to_numeric(sdud["year"], errors="coerce").astype("Int64")

    vol = (sdud.groupby(["fei", "panel_year"], as_index=False)
               .agg(sdud_units=("units", "sum"), sdud_rx=("rx", "sum")))
    return vol


def _load_sdud_volume_quarterly(panel_feis: set) -> pd.DataFrame:
    """
    FEI × quarter Medicaid volume.
    Returns: fei, panel_year, panel_qtr, sdud_units, sdud_rx
    """
    print("  Loading NDC-FEI crosswalk (quarterly)…")
    nf = pd.read_csv(NDC_FEI_CSV, low_memory=False)
    nf["fei"] = pd.to_numeric(nf["FEI_NUMBER"], errors="coerce").astype("Int64")
    nf = nf[nf["fei"].isin(panel_feis)].copy()
    nf["ndc_9"] = nf["manufacture_ndc"].str.strip()
    nf = nf[["ndc_9", "fei"]].dropna().drop_duplicates()
    our_ndcs = set(nf["ndc_9"])

    print("  Reading SDUD parquet (quarterly)…")
    sdud = pd.read_parquet(
        SDUD_PARQ,
        columns=["labeler_code", "product_code", "year", "quarter",
                 "units_reimbursed", "num_prescriptions"]
    )
    sdud["ndc_9"] = (
        sdud["labeler_code"].str.strip().str.zfill(5) + "-" +
        sdud["product_code"].str.strip().str.zfill(4)
    )
    sdud = sdud[sdud["ndc_9"].isin(our_ndcs)].copy()
    if sdud.empty:
        return pd.DataFrame(columns=["fei", "panel_year", "panel_qtr", "sdud_units", "sdud_rx"])

    sdud = sdud.merge(nf, on="ndc_9", how="inner")
    sdud["units"]      = pd.to_numeric(sdud["units_reimbursed"], errors="coerce")
    sdud["rx"]         = pd.to_numeric(sdud["num_prescriptions"], errors="coerce")
    sdud["panel_year"] = pd.to_numeric(sdud["year"], errors="coerce").astype("Int64")
    sdud["panel_qtr"]  = pd.to_numeric(sdud["quarter"], errors="coerce").astype("Int64")

    vol = (sdud.groupby(["fei", "panel_year", "panel_qtr"], as_index=False)
               .agg(sdud_units=("units", "sum"), sdud_rx=("rx", "sum")))
    return vol


# ── Inspection count helpers ──────────────────────────────────────────────────

def _inspection_counts_yearly(ts: pd.DataFrame) -> pd.DataFrame:
    ts2 = ts.copy()
    ts2["panel_year"] = ts2["snapshot_date"].dt.year
    return (ts2.groupby(["fei", "panel_year"], as_index=False)
               .size().rename(columns={"size": "n_inspections_in_period"}))


def _inspection_counts_quarterly(ts: pd.DataFrame) -> pd.DataFrame:
    ts2 = ts.copy()
    ts2["panel_year"] = ts2["snapshot_date"].dt.year
    ts2["panel_qtr"]  = ts2["snapshot_date"].dt.quarter
    return (ts2.groupby(["fei", "panel_year", "panel_qtr"], as_index=False)
               .size().rename(columns={"size": "n_inspections_in_period"}))


# ── Build yearly panel ────────────────────────────────────────────────────────

def build_yearly(ts: pd.DataFrame, fei_drug_map: pd.DataFrame) -> None:
    print("\n── Building YEARLY panel ──────────────────────────────────────")

    panel = _as_of_join_yearly(ts)
    print(f"  {len(panel)} FEI × year rows, {panel['fei'].nunique()} FEIs")

    print("Loading FAERS (yearly)…")
    joined = _load_faers_raw(fei_drug_map)
    ae_yr  = _faers_yearly(joined)
    print(f"  {len(ae_yr)} FEI × year AE rows")

    panel = _add_lags_yearly(panel, ae_yr)

    insp_cnt = _inspection_counts_yearly(ts)
    panel = panel.merge(insp_cnt, on=["fei", "panel_year"], how="left")
    panel["n_inspections_in_period"] = panel["n_inspections_in_period"].fillna(0).astype(int)

    outcomes = _load_inspection_outcomes()
    outcomes = outcomes.rename(columns={"year": "panel_year"})
    panel = panel.merge(outcomes, on=["fei", "panel_year"], how="left")
    for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
        panel[col] = panel[col].fillna(0).astype(int)

    print("Loading SDUD volume (yearly)…")
    panel_feis = set(panel["fei"].dropna().unique())
    vol = _load_sdud_volume(panel_feis)
    panel = panel.merge(vol, on=["fei", "panel_year"], how="left")
    # AE rate per million Medicaid units (controls for facility volume)
    # guard against zero volume (Medicaid has no suppression=false rows for some FEI-years)
    panel["ae_rate_t1"] = np.where(
        panel["sdud_units"] > 0,
        panel["n_ae_t1"] / (panel["sdud_units"] / 1e6),
        np.nan
    )
    n_vol = panel["sdud_units"].notna().sum()
    print(f"  SDUD volume matched for {n_vol}/{len(panel)} panel rows ({n_vol/len(panel):.0%})")

    id_cols      = ["fei", "panel_year", "snapshot_date", "n_inspections_in_period"]
    outcome_cols = ["n_oai", "n_vai", "n_nai", "any_oai"]
    ae_cols      = ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2", "n_drug_fei_pairs"]
    vol_cols     = ["sdud_units", "sdud_rx", "ae_rate_t1"]
    final = panel[[c for c in id_cols + outcome_cols + TEXT_FEATURES + ae_cols + vol_cols
                   if c in panel.columns]].copy()

    print(f"\nYearly panel summary:")
    print(f"  Rows: {len(final)}, FEIs: {final['fei'].nunique()}")
    print(f"  OAI rows: {final['any_oai'].sum()} ({final['any_oai'].mean():.1%})")
    print(f"  AE window (mean):")
    for c in ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"]:
        if c in final.columns:
            print(f"    {c}: {final[c].mean():.1f}")
    print(f"  Mean SDUD units/year: {final['sdud_units'].mean():,.0f}")
    print(f"  ae_rate_t1 (per M units): {final['ae_rate_t1'].mean():.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PANEL_YR, index=False)
    print(f"\nSaved → {OUT_PANEL_YR}")


# ── Build quarterly panel ─────────────────────────────────────────────────────

def build_quarterly(ts: pd.DataFrame, fei_drug_map: pd.DataFrame) -> None:
    print("\n── Building QUARTERLY panel ───────────────────────────────────")

    panel = _as_of_join_quarterly(ts)
    print(f"  {len(panel)} FEI × quarter rows, {panel['fei'].nunique()} FEIs")

    print("Loading FAERS (quarterly)…")
    joined = _load_faers_raw(fei_drug_map)
    ae_q   = _faers_quarterly(joined)
    print(f"  {len(ae_q)} FEI × quarter AE rows")

    panel = _add_lags_quarterly(panel, ae_q)

    insp_cnt = _inspection_counts_quarterly(ts)
    panel = panel.merge(insp_cnt, on=["fei", "panel_year", "panel_qtr"], how="left")
    panel["n_inspections_in_period"] = panel["n_inspections_in_period"].fillna(0).astype(int)

    # For quarterly panel, collapse inspection outcomes to FEI level (ever OAI / total counts)
    # to avoid cartesian product from fiscal-year vs calendar-quarter mismatch
    outcomes = _load_inspection_outcomes()
    fei_outcomes = (outcomes.groupby("fei", as_index=False)
                            .agg(n_oai=("n_oai", "sum"),
                                 n_vai=("n_vai", "sum"),
                                 n_nai=("n_nai", "sum")))
    fei_outcomes["any_oai"] = (fei_outcomes["n_oai"] > 0).astype(int)
    panel = panel.merge(fei_outcomes, on="fei", how="left")
    for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
        panel[col] = panel[col].fillna(0).astype(int)

    print("Loading SDUD volume (quarterly)…")
    panel_feis = set(panel["fei"].dropna().unique())
    vol_q = _load_sdud_volume_quarterly(panel_feis)
    panel = panel.merge(vol_q, on=["fei", "panel_year", "panel_qtr"], how="left")
    panel["ae_rate_t0"] = np.where(
        panel["sdud_units"] > 0,
        panel["n_ae_t0"] / (panel["sdud_units"] / 1e6),
        np.nan
    )
    n_vol = panel["sdud_units"].notna().sum()
    print(f"  SDUD volume matched for {n_vol}/{len(panel)} rows ({n_vol/len(panel):.0%})")

    id_cols      = ["fei", "panel_period", "panel_year", "panel_qtr",
                    "snapshot_date", "n_inspections_in_period"]
    outcome_cols = ["n_oai", "n_vai", "n_nai", "any_oai"]
    ae_cols      = ["n_ae_tm4", "n_ae_tm3", "n_ae_tm2", "n_ae_tm1",
                    "n_ae_t0",
                    "n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4",
                    "n_drug_fei_pairs"]
    vol_cols     = ["sdud_units", "sdud_rx", "ae_rate_t0"]

    final = panel[[c for c in id_cols + outcome_cols + TEXT_FEATURES + ae_cols + vol_cols
                   if c in panel.columns]].copy()

    print(f"\nQuarterly panel summary:")
    print(f"  Rows: {len(final)}, FEIs: {final['fei'].nunique()}")
    print(f"  Quarters covered: {final['panel_period'].min()} → {final['panel_period'].max()}")
    print(f"  AE window (mean):")
    for c in ["n_ae_tm4", "n_ae_tm2", "n_ae_t0", "n_ae_tp2", "n_ae_tp4"]:
        if c in final.columns:
            print(f"    {c}: {final[c].mean():.1f}")
    print(f"  Mean SDUD units/quarter: {final['sdud_units'].mean():,.0f}")

    OUT.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PANEL_QTR, index=False)
    print(f"\nSaved → {OUT_PANEL_QTR}")


# ── Inspection-event-centered quarterly panel ─────────────────────────────────

def build_inspection_centered(ts: pd.DataFrame, fei_drug_map: pd.DataFrame,
                               use_anda_ae: bool = False) -> None:
    """
    Build one row per inspection event with AE counts in quarters t-4 → t+4
    relative to the actual inspection quarter.

    This directly answers: "at what quarterly lag does the inspection impact
    on AEs appear?" and "how long does any elevation persist?"

    Unlike the calendar-quarter panel, every row here is aligned to a specific
    inspection event — so averaging across rows gives the true inspection-centered
    trajectory rather than a diluted calendar average.
    """
    print("\n── Building INSPECTION-CENTERED quarterly panel ───────────────")

    # One row per inspection snapshot (246 rows)
    ts = ts.copy()
    ts["insp_year"] = ts["snapshot_date"].dt.year
    ts["insp_qtr"]  = ts["snapshot_date"].dt.quarter
    ts["insp_idx"]  = ts["insp_year"] * 4 + ts["insp_qtr"]  # quarter index for arithmetic
    print(f"  {len(ts)} inspection events, {ts['fei'].nunique()} FEIs")

    if use_anda_ae:
        print("Loading FAERS (ANDA-specific quarterly)…")
        ae_q = _load_anda_ae_quarterly()
    else:
        print("Loading FAERS (drug-level quarterly)…")
        joined = _load_faers_raw(fei_drug_map)
        ae_q   = _faers_quarterly(joined)   # has fei, period, ae_idx, n_ae

    # For each inspection, join AE counts at lags -4 to +4 quarters
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
        # text features from this inspection
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

    # ── Inspection outcome resolution (3-pass) ───────────────────────────────
    # Pass 1: Redica exact date (includes inferred OAI from Warning Letter /
    #         Non-Compliant, inferred NAI from Compliant)
    panel["_date_key"] = pd.to_datetime(panel["insp_date"]).dt.normalize()
    outcomes = _load_inspection_outcomes_by_date().rename(columns={"insp_date": "_date_key"})
    panel = panel.merge(outcomes, on=["fei", "_date_key"], how="left")
    n_after_redica = panel["any_oai"].isna().sum()
    print(f"  After Redica exact match: {n_after_redica} unmatched")

    # Pass 2 & 3: FDA Drug QA fallback (exact date, then ±30-day nearest)
    # Covers cases where Redica uses inspection start date but FDA uses end date.
    if n_after_redica > 0:
        print("  Loading FDA Drug QA fallback…")
        fda = _load_fda_drug_outcomes()

        def _apply_fda_fill(panel, fda_rows, label):
            """Fill unmatched panel rows from fda_rows via (fei, _date_key) join.
            Uses reset_index / set_index to preserve original panel row indices."""
            unmatched_mask = panel["any_oai"].isna()
            if unmatched_mask.sum() == 0:
                return panel, 0
            fda_keyed = fda_rows.rename(columns={"insp_date": "_date_key"})
            tmp = (panel.loc[unmatched_mask, ["fei", "_date_key"]]
                   .reset_index()
                   .merge(fda_keyed, on=["fei", "_date_key"], how="left")
                   .set_index("index"))
            n_filled = 0
            for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
                hits = tmp.index[tmp[col].notna()]
                if len(hits):
                    panel.loc[hits, col] = tmp.loc[hits, col]
                    n_filled = max(n_filled, len(hits))
            if n_filled:
                print(f"  {label} filled {n_filled} rows")
            return panel, n_filled

        # Pass 2: exact FDA date
        panel, _ = _apply_fda_fill(panel, fda, "FDA exact match")

        # Pass 3: ±30-day nearest FDA match (start-vs-end date gap)
        still_unmatched = panel["any_oai"].isna()
        if still_unmatched.sum() > 0:
            near_rows = []
            for idx, row in panel[still_unmatched].iterrows():
                fei = row["fei"]; idate = row["_date_key"]
                candidates = fda[fda["fei"] == fei].copy()
                if candidates.empty:
                    continue
                candidates["gap"] = (candidates["insp_date"] - idate).abs().dt.days
                best = candidates[candidates["gap"] <= 30].nsmallest(1, "gap")
                if best.empty:
                    continue
                b = best.iloc[0]
                cls = "OAI" if b["n_oai"] else ("VAI" if b["n_vai"] else "NAI")
                print(f"    FDA near match: FEI {fei} text={idate.date()} "
                      f"FDA={b['insp_date'].date()} gap={int(b['gap'])}d → {cls}")
                for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
                    panel.at[idx, col] = b[col]
                near_rows.append(idx)
            if near_rows:
                print(f"  FDA near match (±30d) filled {len(near_rows)} rows")

    panel = panel.drop(columns=["_date_key"])
    n_remaining = panel["any_oai"].isna().sum()
    if n_remaining:
        print(f"  {n_remaining} inspections still unresolved after all passes "
              f"(no Redica or FDA Drug QA record) — defaulted to 0")
    for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
        panel[col] = panel[col].fillna(0).astype(int)

    # Join SDUD yearly volume (FEI × inspection year)
    print("Loading SDUD volume…")
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
    print(f"  AE trajectory (mean, all inspections):")
    for c in ae_window:
        if c in panel.columns:
            print(f"    {c}: {panel[c].mean():.1f}  (n={panel[c].notna().sum()})")

    # Show trajectory by OAI status
    print(f"\n  By inspection outcome (any_oai):")
    for oai in [0, 1]:
        sub = panel[panel["any_oai"] == oai]
        tag = ["VAI/NAI", "OAI"][oai]
        mid = sub["n_ae_t0"].mean()
        pre = sub["n_ae_tm4"].mean()
        post = sub["n_ae_tp4"].mean()
        print(f"    {tag} ({len(sub)} insp): tm4={pre:.0f} → t0={mid:.0f} → tp4={post:.0f}  "
              f"pre_rise={mid/pre:.3f}  persist={post/mid:.3f}" if pre > 0 else f"    {tag}: pre=0")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PANEL_INSP_ANDA if use_anda_ae else OUT_PANEL_INSP
    panel.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularity",
        choices=["yearly", "quarterly", "inspection", "all"],
        default="yearly",
        help=(
            "yearly = FEI×year prediction panel (default); "
            "quarterly = FEI×calendar-quarter prediction panel; "
            "inspection = inspection-event-centered quarterly panel (trajectory analysis); "
            "all = build all three"
        )
    )
    parser.add_argument(
        "--anda-ae", action="store_true",
        help="Use ANDA-specific FAERS AE counts (from crosswalk) instead of drug-level counts. "
             "Only applies to the inspection-centered panel."
    )
    args = parser.parse_args()

    print("Loading text timeseries…")
    ts = _load_text_timeseries()
    print(f"  {len(ts)} inspection rows, {ts['fei'].nunique()} FEIs")

    print("Loading Valisure FEI map…")
    fei_drug_map = _load_fei_drug_map()
    print(f"  {fei_drug_map['api_key'].nunique()} APIs, {fei_drug_map['fei'].nunique()} FEIs")

    if args.granularity in ("yearly", "all"):
        build_yearly(ts, fei_drug_map)

    if args.granularity in ("quarterly", "all"):
        build_quarterly(ts, fei_drug_map)

    if args.granularity in ("inspection", "all"):
        build_inspection_centered(ts, fei_drug_map, use_anda_ae=args.anda_ae)


if __name__ == "__main__":
    main()
