"""
Module M12 — Text-signal validation grid.

For every LLM/regex-extracted 483 text feature (snapshot level, n=79),
test forward predictive power against multiple outcomes and horizons:

  Binary, facility level (same FEI):
    esc_h   = OAI classification or Warning Letter within h months
    rec_h   = drug recall within h months
  Continuous, drug level (mean across the FEI's APIs, monthly panel):
    sh_dur_h        = mean months in shortage during the next h months
    sh_dur_delta_h  = forward minus backward (controls drug base rate)
    faers_ser_delta_h = mean monthly FAERS serious count, fwd minus bwd

Stats: median split (hi/lo rates + lift) for binary outcomes;
Spearman rho for continuous outcomes. Exploratory — n=79 snapshots.

Output: OUT_TABS/text_signal_grid.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import DATA, OUT_DATA, OUT_TABS, REDICA_CSV, VALISURE_FEI  # noqa: E402

TIMESERIES_CSV = DATA / "99 - Outputs - Text Analysis" / "483_fei_text_features_timeseries.csv"
RECALL_XLSX = DATA / "22 - FDA - Recall" / "raw" / "Recall Data.xlsx"

HORIZONS = [12, 24, 36]

FEATURE_GROUPS = {
    "llm_flag": [
        "repeat_llm_share", "patient_risk_llm_share",
        "data_integrity_llm_share", "contamination_llm_share",
        "investigation_llm_share",
    ],
    "llm_only": [
        "repeat_llm_only_share", "patient_risk_llm_only_share",
        "data_integrity_llm_only_share", "contamination_llm_only_share",
        "investigation_llm_only_share",
    ],
    "severity": ["severity_critical_share", "severity_major_share",
                 "severity_moderate_share", "severity_minor_share",
                 "severity_critmajor_share"],
    "scope": ["scope_singlebatch_share", "scope_multipleproducts_share",
              "scope_facilitywide_share", "scope_unclear_share"],
    "cross_repeat": ["repeat_cross_insp_share"],
    "remediation": ["remediation_strong_share", "remediation_partial_share",
                    "remediation_weak_share", "remediation_none_share"],
    "root_cause": ["capital_root_cause_share", "cultural_root_cause_share",
                   "mixed_root_cause_share", "unclear_root_cause_share"],
    "violation_cat": ["vc_labcontrols_share", "vc_productioncontrols_share",
                      "vc_buildingsequipment_share", "vc_orgpersonnel_share",
                      "vc_packaginglabeling_share", "vc_recordsreports_share",
                      "vc_qualitysystem_share", "vc_other_share"],
    "regex": ["wl_ref_regex_share", "oos_oot_regex_share", "quality_unit_regex_share",
              "laboratory_regex_share", "equipment_facility_regex_share",
              "process_control_regex_share"],
    "volume": ["n_obs_total", "mean_confidence"],
}


def load_snapshots() -> tuple[pd.DataFrame, dict[int, list[str]]]:
    b = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    b = b[["API", "FEI_NUMBER"]].dropna(subset=["FEI_NUMBER"])
    b["FEI_NUMBER"] = b["FEI_NUMBER"].astype("int64")
    fei_apis = {int(f): sorted(g["API"].str.strip().unique()) for f, g in b.groupby("FEI_NUMBER")}

    ts = pd.read_csv(TIMESERIES_CSV)
    ts = ts[ts["fei"].isin(fei_apis)].copy()
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    return ts.reset_index(drop=True), fei_apis


def attach_outcomes(ts: pd.DataFrame, fei_apis: dict[int, list[str]]) -> pd.DataFrame:
    ev = pd.read_csv(REDICA_CSV)
    ev["Event Date"] = pd.to_datetime(ev["Event Date"], errors="coerce")
    esc = ev[(ev["Classification"] == "OAI") | (ev["Warning Letter"] == 1)][["FEI", "Event Date"]]

    rec = pd.read_excel(RECALL_XLSX)
    rec["FEI Number"] = pd.to_numeric(rec["FEI Number"], errors="coerce")
    rec = rec[(rec["FEI Number"].isin(fei_apis)) & (rec["Product Type"] == "Drugs")].copy()
    rec["d"] = pd.to_datetime(rec["Center Classification Date"], errors="coerce")

    mp = pd.read_csv(OUT_DATA / "master_panel_monthly.csv")
    mp["midx"] = mp["year"] * 12 + mp["month"]
    panel_min, panel_max = mp["midx"].min(), mp["midx"].max()

    def fwd_any(frame, feicol, datecol, fei, d0, h):
        f = frame[frame[feicol] == fei][datecol].dropna()
        return bool(((f > d0) & (f <= d0 + pd.Timedelta(days=30.44 * h))).any())

    def drug_window_mean(apis, col, m0, m1):
        """Mean of `col` per drug-month across the FEI's APIs in [m0, m1]."""
        m0c, m1c = max(m0, panel_min), min(m1, panel_max)
        if m1c - m0c < 5:  # require >=6 months of panel coverage
            return np.nan
        sub = mp[(mp["drug_norm"].isin(apis)) & (mp["midx"] >= m0c) & (mp["midx"] <= m1c)]
        return float(sub[col].mean()) if len(sub) else np.nan

    out_rows = []
    for r in ts.itertuples():
        d0 = r.snapshot_date
        midx0 = d0.year * 12 + d0.month
        apis = fei_apis.get(int(r.fei), [])
        row = {}
        for h in HORIZONS:
            row[f"esc_{h}"] = fwd_any(esc, "FEI", "Event Date", r.fei, d0, h)
            row[f"rec_{h}"] = fwd_any(rec, "FEI Number", "d", r.fei, d0, h)
            fwd_sh = drug_window_mean(apis, "shortage_ongoing", midx0 + 1, midx0 + h)
            bwd_sh = drug_window_mean(apis, "shortage_ongoing", midx0 - h, midx0 - 1)
            row[f"sh_dur_{h}"] = fwd_sh * h if pd.notna(fwd_sh) else np.nan
            row[f"sh_dur_delta_{h}"] = (fwd_sh - bwd_sh) * h if pd.notna(fwd_sh) and pd.notna(bwd_sh) else np.nan
            fwd_fa = drug_window_mean(apis, "faers_n_serious", midx0 + 1, midx0 + h)
            bwd_fa = drug_window_mean(apis, "faers_n_serious", midx0 - h, midx0 - 1)
            row[f"faers_ser_delta_{h}"] = fwd_fa - bwd_fa if pd.notna(fwd_fa) and pd.notna(bwd_fa) else np.nan
        out_rows.append(row)
    return pd.concat([ts.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    bin_outcomes = [f"{p}_{h}" for p in ["esc", "rec"] for h in HORIZONS]
    cont_outcomes = [f"{p}_{h}" for p in ["sh_dur", "sh_dur_delta", "faers_ser_delta"]
                     for h in HORIZONS]
    for group, feats in FEATURE_GROUPS.items():
        for feat in feats:
            if feat not in df.columns:
                continue
            x = df[feat].astype(float)
            med = x.median()
            hi_mask = x > med
            for out in bin_outcomes:
                y = df[out].astype(float)
                hi, lo = y[hi_mask], y[~hi_mask]
                hi_r = float(hi.mean()) if len(hi) else np.nan
                lo_r = float(lo.mean()) if len(lo) else np.nan
                results.append({
                    "group": group, "feature": feat, "outcome": out, "type": "binary",
                    "n_hi": int(hi_mask.sum()), "n_lo": int((~hi_mask).sum()),
                    "hi_rate": round(hi_r, 3), "lo_rate": round(lo_r, 3),
                    "lift": round(hi_r / lo_r, 2) if lo_r else np.nan,
                    "effect": round(hi_r - lo_r, 3),
                })
            for out in cont_outcomes:
                y = df[out].astype(float)
                ok = x.notna() & y.notna()
                rho = float(x[ok].corr(y[ok], method="spearman")) if ok.sum() >= 20 else np.nan
                results.append({
                    "group": group, "feature": feat, "outcome": out, "type": "continuous",
                    "n_hi": int(ok.sum()), "n_lo": 0,
                    "hi_rate": np.nan, "lo_rate": np.nan, "lift": np.nan,
                    "effect": round(rho, 3) if pd.notna(rho) else np.nan,
                })
    return pd.DataFrame(results)


def main():
    ts, fei_apis = load_snapshots()
    print(f"{len(ts)} snapshots, {ts['fei'].nunique()} FEIs")
    df = attach_outcomes(ts, fei_apis)
    for h in HORIZONS:
        print(f"h={h}: esc rate={df[f'esc_{h}'].mean():.3f}  rec rate={df[f'rec_{h}'].mean():.3f}  "
              f"sh_dur valid n={df[f'sh_dur_delta_{h}'].notna().sum()}  "
              f"faers valid n={df[f'faers_ser_delta_{h}'].notna().sum()}")
    grid = run_grid(df)
    path = OUT_TABS / "text_signal_grid.csv"
    grid.to_csv(path, index=False)
    print(f"\ngrid saved: {path} ({len(grid)} feature-outcome cells)")

    print("\n=== TOP BINARY (lift, n_hi>=10, hi_rate>lo_rate) ===")
    b = grid[(grid["type"] == "binary") & (grid["n_hi"] >= 10) & (grid["effect"] > 0)]
    print(b.sort_values("lift", ascending=False).head(20)[
        ["group", "feature", "outcome", "n_hi", "hi_rate", "lo_rate", "lift"]].to_string(index=False))

    print("\n=== TOP CONTINUOUS (|Spearman rho|) ===")
    c = grid[(grid["type"] == "continuous") & grid["effect"].notna()]
    c = c.reindex(c["effect"].abs().sort_values(ascending=False).index)
    print(c.head(20)[["group", "feature", "outcome", "n_hi", "effect"]].to_string(index=False))


if __name__ == "__main__":
    main()
