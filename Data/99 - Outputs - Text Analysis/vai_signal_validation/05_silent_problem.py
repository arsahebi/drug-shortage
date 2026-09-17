"""
05_silent_problem.py
----------------------------------------------------------------------------
Rebuild of the INFORMS 2026 "The Silent Problem: Same VAI Label, Very
Different Patient Outcomes" slide, using the CURRENT validated pipeline.

Original: old_not_current_pipeline/pdf_source_and_superseded/ae_validation/
03_text_only_model.py::_trajectory_analysis() (unmodified code, moved
2026-09-16). Ported here with the v2 feature renames and reading the panel
built by 01_build_inspection_panel.py (current text extraction, FDA-primary
OAI/VAI source).

Method: segment facilities by OAI status x a composite technical-signal
score, then compare AE trajectories.
  tech_score = mean of z-scored values across 5 features (Lab Controls +
               Data Integrity cluster): vc_laboratorycontrolssystem_share,
               data_integrity_llm_share, joint_labcontrols_dataintegrity,
               joint_contamination_labcontrols, n_laboratorycontrolssystem_obs.
               NOTE: the original slide caption says "top-quartile Lab
               Controls share" -- the actual code is this 5-feature
               composite, not the single Lab Controls share column. Kept
               as-is here for a faithful replication.
  Groups (mutually exclusive, FEI-level):
    OAI-ever         -- received >=1 OAI in this dataset
    High-signal VAI   -- top quartile of tech_score, never OAI
    Low-signal VAI    -- bottom three quartiles, never OAI
  For each group: mean AE count at each quarterly lag, pre_rise (Q0/Q-4),
  persist (Q+4/Q0).
  Named-facility table: individual High-signal VAI FEIs ranked by AE
  persistence (Q+4/Q0), mapped to Labeler/API via the Valisure NDC_FEI
  Mapping sheet (this mapping step was done manually for the INFORMS deck;
  automated here).

Usage
-----
  python 05_silent_problem.py --anda-ae   (matches the INFORMS slide, which
                                            used the ANDA-specific panel --
                                            confirmed via the repo's git
                                            history)

Outputs
-------
  outputs/tables/silent_problem_groups.csv
  outputs/tables/silent_problem_flagged_facilities.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DATA = ROOT / "Data"
OUT      = HERE / "outputs"
OUT_TABS = OUT / "tables"
PANEL      = OUT / "fei_ae_panel_inspection_centered.parquet"
PANEL_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"

TECH_FEATURES = [
    "vc_laboratorycontrolssystem_share",
    "data_integrity_llm_share",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "n_laboratorycontrolssystem_obs",
]

AE_COLS = ["n_ae_tm4", "n_ae_tm2", "n_ae_t0", "n_ae_tp2", "n_ae_tp4"]


def _build_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Same as 02_vai_signal_model.py: drop rows with zero AE records in the
    post-inspection window rather than zero-filling."""
    df = df.copy()
    ae_cols = ["n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"]
    has_any = df[ae_cols].notna().any(axis=1)
    df = df[has_any].copy()
    return df


def _trajectory_analysis(df: pd.DataFrame, fei_ever_oai: pd.Series):
    df = df.copy()
    tech_cols = [c for c in TECH_FEATURES if c in df.columns]
    if not tech_cols:
        raise KeyError(f"None of TECH_FEATURES found in panel: {TECH_FEATURES}")

    df["tech_score"] = df[tech_cols].fillna(0).apply(zscore).mean(axis=1)

    fei_tech = df.groupby("fei")["tech_score"].mean()
    q75 = fei_tech.quantile(0.75)

    def _group(fei):
        if fei in fei_ever_oai.index and fei_ever_oai[fei] == 1:
            return "OAI-ever"
        if fei_tech.get(fei, 0) >= q75:
            return "High-signal VAI"
        return "Low-signal VAI"

    df["group"] = df["fei"].map(_group)

    rows = []
    for grp in ["OAI-ever", "High-signal VAI", "Low-signal VAI"]:
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        mm4 = sub["n_ae_tm4"].mean()
        mm2 = sub["n_ae_tm2"].mean()
        m0  = sub["n_ae_t0"].mean()
        mp2 = sub["n_ae_tp2"].mean()
        mp4 = sub["n_ae_tp4"].mean()
        pre_rise = (m0 / mm4) if (mm4 and mm4 > 0) else np.nan
        persist  = (mp4 / m0) if (m0 and m0 > 0) else np.nan
        rows.append({
            "group": grp, "n_feis": sub["fei"].nunique(), "n_rows": len(sub),
            "mean_ae_tm4": round(mm4, 1), "mean_ae_tm2": round(mm2, 1),
            "mean_ae_t0": round(m0, 1), "mean_ae_tp2": round(mp2, 1),
            "mean_ae_tp4": round(mp4, 1),
            "pre_rise_t0_tm4": round(pre_rise, 3) if pd.notna(pre_rise) else np.nan,
            "persist_tp4_t0": round(persist, 3) if pd.notna(persist) else np.nan,
        })
    traj_df = pd.DataFrame(rows)

    hs_vai = df[df["group"] == "High-signal VAI"].copy()
    if hs_vai.empty:
        return traj_df, pd.DataFrame(), q75

    fei_agg = hs_vai.groupby("fei", as_index=False).agg(
        mean_ae_tm4=("n_ae_tm4", "mean"),
        mean_ae_t0=("n_ae_t0", "mean"),
        mean_ae_tp2=("n_ae_tp2", "mean"),
        mean_ae_tp4=("n_ae_tp4", "mean"),
        mean_tech_score=("tech_score", "mean"),
        n_inspections=("fei", "count"),
    )
    fei_agg["persist_tp4_t0"] = (fei_agg["mean_ae_tp4"] / fei_agg["mean_ae_t0"]).round(3)
    fei_agg["ae_rising"] = fei_agg["persist_tp4_t0"] > 1.0
    fei_agg = fei_agg.sort_values("persist_tp4_t0", ascending=False).round(1)

    return traj_df, fei_agg, q75


def _map_fei_to_labeler_api(feis: list[int]) -> pd.DataFrame:
    """FEI -> Labeler/API, for naming flagged facilities. Modal (most common)
    API/Labeler pair per FEI, since one FEI can appear under multiple
    NDC/Labeler combinations."""
    fm = pd.read_excel(VALISURE_FEI, sheet_name="NDC_FEI Mapping")
    fm.columns = [c.strip() for c in fm.columns]
    fm["FEI_NUMBER"] = pd.to_numeric(fm["FEI_NUMBER"], errors="coerce")
    fm = fm.dropna(subset=["FEI_NUMBER"])
    fm["FEI_NUMBER"] = fm["FEI_NUMBER"].astype(int)
    fm = fm[fm["FEI_NUMBER"].isin(feis)]
    modal = (fm.groupby("FEI_NUMBER")
               .agg(labeler=("Labeler", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
                    api=("API", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))
               .reset_index()
               .rename(columns={"FEI_NUMBER": "fei"}))
    return modal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anda-ae", dest="anda_ae", action="store_true",
                        help="Use ANDA-specific AE panel (matches the original INFORMS slide).")
    args = parser.parse_args()
    panel_path = PANEL_ANDA if args.anda_ae else PANEL

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_inspection_panel.py first.")

    print(f"Loading panel ({'ANDA-specific' if args.anda_ae else 'drug-level'})...")
    df = pd.read_parquet(panel_path)
    df = _build_outcome(df)
    print(f"  {len(df)} inspection events (with real AE data in the outcome window), {df['fei'].nunique()} FEIs")

    fei_ever_oai = df.groupby("fei")["any_oai"].max()

    print("\nRunning trajectory analysis (OAI-ever / High-signal VAI / Low-signal VAI)...")
    traj_df, flagged, q75 = _trajectory_analysis(df, fei_ever_oai)
    print(f"  High-signal VAI threshold: tech_score >= {q75:.3f} (top quartile)")
    print(f"\nGroups:\n{traj_df.to_string(index=False)}")

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    traj_df.to_csv(OUT_TABS / "silent_problem_groups.csv", index=False)
    print(f"\nSaved -> {OUT_TABS / 'silent_problem_groups.csv'}")

    if not flagged.empty:
        names = _map_fei_to_labeler_api(flagged["fei"].astype(int).tolist())
        flagged = flagged.merge(names, on="fei", how="left")
        flagged.to_csv(OUT_TABS / "silent_problem_flagged_facilities.csv", index=False)
        print(f"\nHigh-signal VAI facilities, ranked by AE persistence (Q+4/Q0):")
        print(flagged[["fei", "labeler", "api", "mean_ae_t0", "mean_ae_tp4",
                        "persist_tp4_t0", "n_inspections"]].to_string(index=False))
        print(f"\nSaved -> {OUT_TABS / 'silent_problem_flagged_facilities.csv'}")


if __name__ == "__main__":
    main()
