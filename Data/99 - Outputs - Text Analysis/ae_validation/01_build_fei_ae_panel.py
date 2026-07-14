"""
01_build_fei_ae_panel.py
────────────────────────────────────────────────────────────────────────────
Build a FEI × year panel joining LLM text signals with FAERS adverse event
counts attributed to each facility.

Design
──────
Text features: FEI × inspection-level timeseries
  (483_fei_text_features_timeseries_redica.csv, one row per inspection).
  For each FEI × calendar year t, we use an as-of join: the most recent
  inspection snapshot with snapshot_date ≤ Dec 31 of year t.
  If no snapshot exists for a FEI in year t, that FEI-year is excluded.

AE outcome: FAERS serious AEs attributed to FEI
  Step 1 — Load the pre-filtered FAERS parquet (14 Valisure drugs, serious AEs).
  Step 2 — Match the prod_ai first word to Valisure canonical API name.
  Step 3 — Join Valisure API → FEI via the FEI mapping sheet.
  Step 4 — Aggregate to FEI × year → n_ae (count of serious AE reports).

Lags: for each FEI × year t we record AE counts across a 5-year window:
  t-2, t-1 (pre-inspection) and t0, t+1, t+2 (post-inspection).
  Pre-inspection lags test whether AEs were already elevated before FDA
  caught the problem (supports pre-distribution of bad product hypothesis).
  Post-inspection lags test correction speed and persistence.

Output
──────
  outputs/fei_ae_panel.parquet  — analysis-ready panel

Columns
───────
  fei, year
  n_inspections_in_year  : 483 inspections issued to this FEI in year t
  snapshot_date          : date of the inspection used for text features
  n_oai, n_vai, n_nai    : inspection outcome counts (from Inspections Details.xlsx)
  any_oai                : 1 if FEI received ≥1 OAI in this panel year
  [TEXT_FEATURES]        : 17 LLM signal shares + Layer 5 counts/joints
  n_ae_tm2               : FAERS serious AEs two years BEFORE inspection
  n_ae_tm1               : …one year before
  n_ae_t0                : …in inspection year
  n_ae_t1                : …one year after
  n_ae_t2                : …two years after
  n_drug_fei_pairs       : distinct drug-FEI pairs (market presence proxy)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent.parent.parent   # Project - Drug Shortage/
DATA   = ROOT / "Data"
OUT    = HERE / "outputs"

TEXT_TS_CSV  = DATA / "99 - Outputs - Text Analysis" / "step02_483_fei_text_features_timeseries_redica.csv"
FAERS_PARQ   = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"
INSP_DETAILS = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"

OUT_PANEL = OUT / "fei_ae_panel.parquet"

# ── Text features included in the panel ──────────────────────────────────────
TEXT_FEATURES = [
    # Layer 3: LLM signal shares
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
    # Layer 5: raw counts (intensity alongside proportion)
    "n_labcontrols_obs",
    "n_qualitysystem_obs",
    # Layer 5: joint co-occurrence flags (two-failure-mode hypothesis)
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]

PANEL_YEARS = list(range(2018, 2026))   # years for which we build panel rows


# ── Step 1: Load and reshape text timeseries ─────────────────────────────────

def _load_text_timeseries() -> pd.DataFrame:
    ts = pd.read_csv(TEXT_TS_CSV, low_memory=False)
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["year"] = ts["snapshot_date"].dt.year
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts = ts.dropna(subset=["fei"])
    keep = ["fei", "snapshot_date", "year"] + TEXT_FEATURES
    return ts[[c for c in keep if c in ts.columns]].copy()


def _as_of_join(ts: pd.DataFrame, panel_years: list[int]) -> pd.DataFrame:
    """For each FEI × panel year, pick the most recent snapshot ≤ Dec 31 of that year."""
    rows = []
    for year in panel_years:
        cutoff = pd.Timestamp(year, 12, 31)
        eligible = ts[ts["snapshot_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        # keep latest snapshot per FEI
        latest = (
            eligible.sort_values("snapshot_date")
                    .groupby("fei", as_index=False)
                    .last()
        )
        latest["panel_year"] = year
        rows.append(latest)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"year": "insp_year"})
    return panel


# ── Step 2: Build FEI × year AE counts ───────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    """Valisure API → FEI mapping, normalized to lowercase API names."""
    vm = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    vm.columns = [c.strip() for c in vm.columns]
    api_col = next((c for c in vm.columns if c.lower() == "api"), None)
    fei_col = next((c for c in vm.columns if "fei" in c.lower() and "unique" not in c.lower()), None)
    if not api_col or not fei_col:
        raise ValueError(f"Cannot find API/FEI columns in Valisure map. Found: {vm.columns.tolist()}")
    fm = vm[[api_col, fei_col]].rename(columns={api_col: "api", fei_col: "fei"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
    fm["api_key"] = fm["api"].str.strip().str.lower().str.split().str[0]
    return fm.dropna(subset=["fei"]).drop_duplicates()


def _load_faers_fei_year(fei_drug_map: pd.DataFrame) -> pd.DataFrame:
    """FAERS → drug-year AE counts → FEI-year AE counts."""
    if not FAERS_PARQ.exists():
        print(f"  WARNING: FAERS parquet not found at {FAERS_PARQ}. AE columns will be NaN.")
        return pd.DataFrame(columns=["fei", "year", "n_ae", "n_drug_fei_pairs"])

    df = pd.read_parquet(FAERS_PARQ)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "prod_ai"])

    # extract first word of prod_ai as matching key
    df["api_key"] = df["prod_ai"].str.strip().str.lower().str.split().str[0]

    # join to FEI map on api_key
    joined = df.merge(
        fei_drug_map[["api_key", "fei", "api"]].drop_duplicates(),
        on="api_key", how="inner"
    )

    # aggregate to FEI × year
    agg = (
        joined.groupby(["fei", "year"], as_index=False)
              .agg(n_ae=("primaryid", "count"),
                   n_drug_fei_pairs=("api", "nunique"))
    )
    return agg


# ── Step 3: Build lagged panel ────────────────────────────────────────────────

def _add_lags(panel: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    """
    Join AE counts at pre-inspection lags (-2, -1) and post-inspection lags (0, 1, 2).

    panel_year = inspection snapshot year (t=0).
    n_ae_tm2 = AEs two years BEFORE inspection  (tests pre-existing elevation)
    n_ae_tm1 = AEs one year  BEFORE inspection
    n_ae_t0  = AEs in inspection year
    n_ae_t1  = AEs one year  AFTER  (tests short-run correction effect)
    n_ae_t2  = AEs two years AFTER  (tests long-run persistence)
    """
    ae = ae.rename(columns={"year": "ae_year", "n_ae": "n_ae_raw"})
    # negative lag = pre-inspection (ae_year = panel_year + |lag|)
    # positive lag = post-inspection (ae_year = panel_year - lag)
    for lag, suffix in [(-2, "tm2"), (-1, "tm1"), (0, "t0"), (1, "t1"), (2, "t2")]:
        ae_lag = ae.copy()
        ae_lag["panel_year"] = ae_lag["ae_year"] - lag
        ae_lag = ae_lag.rename(columns={"n_ae_raw": f"n_ae_{suffix}"})[["fei", "panel_year", f"n_ae_{suffix}"]]
        panel = panel.merge(ae_lag, on=["fei", "panel_year"], how="left")
    # market presence proxy: max n_drug_fei_pairs across all years for this FEI
    fei_pairs = ae.groupby("fei")["n_drug_fei_pairs"].max().reset_index()
    panel = panel.merge(fei_pairs, on="fei", how="left")
    return panel


# ── Step 3b: Load OAI/VAI/NAI outcomes from Inspections Details ───────────────

def _load_inspection_outcomes() -> pd.DataFrame:
    """
    Load FDA inspection classifications from Inspections Details.xlsx.
    Aggregate to FEI × fiscal year: n_oai, n_vai, n_nai, any_oai.
    """
    df = pd.read_excel(INSP_DETAILS)
    df.columns = [c.strip() for c in df.columns]
    df["fei"]  = pd.to_numeric(df["FEI Number"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fei", "year", "Classification"])

    # keep drug/pharma inspections only (optional filter — broad inclusion)
    drug_mask = df["Project Area"].str.contains("Drug", na=False, case=False)
    df = df[drug_mask].copy()

    df["is_oai"] = df["Classification"].str.contains("OAI", na=False).astype(int)
    df["is_vai"] = df["Classification"].str.contains("VAI", na=False).astype(int)
    df["is_nai"] = df["Classification"].str.contains("NAI", na=False).astype(int)

    agg = (
        df.groupby(["fei", "year"], as_index=False)
          .agg(n_oai=("is_oai", "sum"),
               n_vai=("is_vai", "sum"),
               n_nai=("is_nai", "sum"))
    )
    agg["any_oai"] = (agg["n_oai"] > 0).astype(int)
    agg = agg.rename(columns={"year": "panel_year"})
    return agg


# ── Step 4: Count inspections per FEI × year ─────────────────────────────────

def _inspection_counts(ts: pd.DataFrame) -> pd.DataFrame:
    ts2 = ts.copy()
    ts2["year"] = ts2["snapshot_date"].dt.year
    counts = ts2.groupby(["fei", "year"], as_index=False).size().rename(
        columns={"year": "panel_year", "size": "n_inspections_in_year"}
    )
    return counts


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading text timeseries…")
    ts = _load_text_timeseries()
    print(f"  {len(ts)} inspection rows, {ts['fei'].nunique()} FEIs")

    print("Building as-of text feature panel…")
    panel = _as_of_join(ts, PANEL_YEARS)
    print(f"  {len(panel)} FEI × year rows across {panel['fei'].nunique()} FEIs")

    print("Loading Valisure FEI map…")
    fei_drug_map = _load_fei_drug_map()
    print(f"  {len(fei_drug_map)} API-FEI pairs, {fei_drug_map['api_key'].nunique()} unique APIs")

    print("Loading FAERS and building FEI × year AE counts…")
    ae = _load_faers_fei_year(fei_drug_map)
    print(f"  {len(ae)} FEI × year AE rows, {ae['fei'].nunique()} FEIs")

    print("Merging lags (t0, t1, t2)…")
    panel = _add_lags(panel, ae)

    print("Adding inspection counts per year…")
    insp_counts = _inspection_counts(ts)
    panel = panel.merge(insp_counts, on=["fei", "panel_year"], how="left")
    panel["n_inspections_in_year"] = panel["n_inspections_in_year"].fillna(0).astype(int)

    print("Loading inspection outcomes (OAI/VAI/NAI)…")
    outcomes = _load_inspection_outcomes()
    print(f"  {len(outcomes)} FEI × year rows with outcome data, {outcomes['fei'].nunique()} FEIs")
    panel = panel.merge(outcomes, on=["fei", "panel_year"], how="left")
    for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
        panel[col] = panel[col].fillna(0).astype(int)

    # Final column order
    id_cols      = ["fei", "panel_year", "snapshot_date", "n_inspections_in_year"]
    outcome_cols = ["n_oai", "n_vai", "n_nai", "any_oai"]
    ae_cols      = ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2", "n_drug_fei_pairs"]
    final        = panel[id_cols + outcome_cols + TEXT_FEATURES + ae_cols].copy()

    n_with_ae  = final["n_ae_t1"].notna().sum()
    n_with_oai = final["any_oai"].sum()
    n_pre      = final["n_ae_tm1"].notna().sum()
    print(f"\nPanel summary:")
    print(f"  Total rows (FEI × year):    {len(final)}")
    print(f"  Rows with pre (t-1) AE:     {n_pre}")
    print(f"  Rows with post (t+1) AE:    {n_with_ae}")
    print(f"  FEIs with any AE data:      {final[final['n_ae_t1'].notna()]['fei'].nunique()}")
    print(f"  FEI-years with OAI:         {n_with_oai} ({n_with_oai/len(final):.1%})")
    print(f"\n  Full window mean AEs (all FEIs):")
    for col in ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"]:
        print(f"    {col}: {final[col].mean():.1f} (n={final[col].notna().sum()})")

    OUT.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PANEL, index=False)
    print(f"\nSaved → {OUT_PANEL}")


if __name__ == "__main__":
    main()
