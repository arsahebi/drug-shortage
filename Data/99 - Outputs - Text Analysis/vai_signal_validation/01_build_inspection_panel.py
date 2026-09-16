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

Known limitation (as of 2026-09-16): the FDA Inspection raw source
(Data/14 - FDA - Inspection/raw/Inspections Details.xlsx and
Inspections Citations Details.xlsx) is dated November 20, 2025, about 10
months old. If a facility's OAI/VAI status changed since then, the VAI-only
subgroup here is defined on a stale classification. Re-run this script after
that raw data is refreshed.

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
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent.parent.parent
DATA   = ROOT / "Data"
OUT    = HERE / "outputs"

# Current validated redica v2 text timeseries (Claude Sonnet 5).
TEXT_TS_CSV  = DATA / "99 - Outputs - Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
FAERS_PARQ   = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"
REDICA_COMBINED = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
FDA_INSP_XLSX   = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
SDUD_PARQ    = DATA / "04 - Medicaid - SDUD" / "processed" / "2025-12-18-SDUDcanonical.parquet"
NDC_FEI_CSV  = DATA / "17 - NDC, FEI Mapping" / "ndc_fei_from_labels.csv"
ANDA_AE_QTR_CSV = DATA / "08 - Valisure" / "processed" / "valisure_anda_faers_ae_counts_quarterly.csv"

OUT_PANEL_INSP      = OUT / "fei_ae_panel_inspection_centered.parquet"
OUT_PANEL_INSP_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

# v2 schema: 2 renames from the original (v1) feature list, rest unchanged.
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
    ts = pd.read_csv(TEXT_TS_CSV, low_memory=False)
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
    df = pd.read_csv(ANDA_AE_QTR_CSV, low_memory=False)
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

def _load_inspection_outcomes_by_date() -> pd.DataFrame:
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

    # ── Inspection outcome resolution (3-pass) ───────────────────────────────
    panel["_date_key"] = pd.to_datetime(panel["insp_date"]).dt.normalize()
    outcomes = _load_inspection_outcomes_by_date().rename(columns={"insp_date": "_date_key"})
    panel = panel.merge(outcomes, on=["fei", "_date_key"], how="left")
    n_after_redica = panel["any_oai"].isna().sum()
    print(f"  After Redica exact match: {n_after_redica} unmatched")

    if n_after_redica > 0:
        print("  Loading FDA Drug QA fallback...")
        fda = _load_fda_drug_outcomes()

        def _apply_fda_fill(panel, fda_rows, label):
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

        panel, _ = _apply_fda_fill(panel, fda, "FDA exact match")

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
                      f"FDA={b['insp_date'].date()} gap={int(b['gap'])}d -> {cls}")
                for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
                    panel.at[idx, col] = b[col]
                near_rows.append(idx)
            if near_rows:
                print(f"  FDA near match (+/-30d) filled {len(near_rows)} rows")

    panel = panel.drop(columns=["_date_key"])
    n_remaining = panel["any_oai"].isna().sum()
    if n_remaining:
        print(f"  {n_remaining} inspections still unresolved after all passes "
              f"(no Redica or FDA Drug QA record) -- defaulted to 0")
    for col in ["n_oai", "n_vai", "n_nai", "any_oai"]:
        panel[col] = panel[col].fillna(0).astype(int)

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
    out_path = OUT_PANEL_INSP_ANDA if use_anda_ae else OUT_PANEL_INSP
    panel.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anda-ae", action="store_true",
        help="Use ANDA-specific FAERS AE counts instead of drug-level counts.",
    )
    args = parser.parse_args()

    print("Loading text timeseries...")
    ts = _load_text_timeseries()
    print(f"  {len(ts)} inspection rows, {ts['fei'].nunique()} FEIs")

    print("Loading Valisure FEI map...")
    fei_drug_map = _load_fei_drug_map()
    print(f"  {fei_drug_map['api_key'].nunique()} APIs, {fei_drug_map['fei'].nunique()} FEIs")

    build_inspection_centered(ts, fei_drug_map, use_anda_ae=args.anda_ae)


if __name__ == "__main__":
    main()
