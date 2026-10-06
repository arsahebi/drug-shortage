"""
00_slide_stats.py
─────────────────────────────────────────────────────────────────────────────
Computes and prints every statistic shown in the INFORMS 2026 slides
(20260714_informs_slides.tex), organised slide by slide.

Run from the project root or from this folder:
  python 00_slide_stats.py

All heavy outputs (ablation AUCs, trajectory tables, cluster summaries)
are loaded from the CSV files written by scripts 03 and 04. Statistics
that require raw data (gap analysis, SDUD, panel counts, Spearman
correlations) are recomputed here from source files.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[3]          # project root
HERE  = Path(__file__).resolve().parent
OUT   = HERE / "outputs"
TABS  = OUT / "tables"

PANEL_PARQ   = OUT / "fei_ae_panel.parquet"
INSP_PANEL   = OUT / "fei_ae_panel_inspection_centered.parquet"
REDICA_XLSX  = ROOT / "Data/07 - Redica/raw/Valisure14_Sites_Red_Flag_Events.xlsx"
SITE_LIST    = ROOT / "Data/07 - Redica/raw/Valisure14_Site_List.xlsx"
REDICA_COMB  = ROOT / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"

ABLATION_CSV = TABS / "ablation_metrics.csv"
TRAJ_CSV     = TABS / "ae_trajectory_by_group.csv"
CLUSTER_CSV  = TABS / "ae_cluster_summary.csv"
CLUSTER_FEIS = TABS / "ae_cluster_fei_list.csv"
VAL_CSV      = OUT  / "models" / "validation_vs_redica.csv"


def _h(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def _sh(title: str) -> None:
    print(f"\n  --- {title} ---")


# ── Slide 4: Study Design ─────────────────────────────────────────────────────

def slide4_study_design(panel: pd.DataFrame) -> None:
    _h("SLIDE 4 — Study Design")

    n_feis  = panel["fei"].nunique()
    n_rows  = len(panel)
    n_years = panel["panel_year"].nunique() if "panel_year" in panel.columns else None

    # Valisure site list
    sl = pd.read_excel(SITE_LIST)
    n_valisure_feis = sl["FEI"].nunique()

    # Redica combined (all with 483 text)
    rc = pd.read_csv(REDICA_COMB)
    n_redica_feis = rc["fei"].nunique() if "fei" in rc.columns else "?"

    # Inspection events in panel (inspection-centered)
    ic = pd.read_parquet(INSP_PANEL) if INSP_PANEL.exists() else None
    n_insp_events = len(ic) if ic is not None else 246

    print(f"  Valisure FEI mapping total:          {n_valisure_feis}")
    print(f"  FEIs with 483 text + FAERS (panel):  {n_feis}")
    print(f"  Inspection events in panel:           {n_insp_events}")
    print(f"  FEI × year rows:                      {n_rows}")
    if n_years:
        print(f"  Calendar years covered:               {n_years}")

    # Inspections per FEI from Redica
    ev = pd.read_excel(REDICA_XLSX)
    ev["Event Date"] = pd.to_datetime(ev["Event Date"])
    insp = ev[
        (ev["Event Type"] == "Inspection") &
        (ev["Agency List"].str.contains("US - FDA", na=False))
    ].merge(sl[["Site Redica Id","FEI"]], on="Site Redica Id", how="left")
    insp_panel_feis = insp[insp["FEI"].isin(panel["fei"].unique())]
    per_fei = insp_panel_feis.groupby("FEI").size()
    print(f"  Inspections per FEI (all FDA types):  mean={per_fei.mean():.1f}  "
          f"median={per_fei.median():.0f}  range={per_fei.min()}–{per_fei.max()}")


# ── Slide 5: Inspection Frequency + Targeting ────────────────────────────────

def slide5_inspection_gaps(panel: pd.DataFrame) -> None:
    _h("SLIDE 5 — Inspection Gaps and the Targeting Problem")

    sl = pd.read_excel(SITE_LIST)
    ev = pd.read_excel(REDICA_XLSX)
    ev["Event Date"] = pd.to_datetime(ev["Event Date"])

    insp = ev[
        (ev["Event Type"] == "Inspection") &
        (ev["Agency List"].str.contains("US - FDA", na=False))
    ].merge(sl[["Site Redica Id","FEI"]], on="Site Redica Id", how="left")

    panel_feis = set(panel["fei"].unique())

    for label, subset_feis in [("All 127 FEIs", set(sl["FEI"])),
                                ("98 panel FEIs", panel_feis)]:
        _sh(f"Gap stats — {label}")
        sub = insp[insp["FEI"].isin(subset_feis)].sort_values(["FEI","Event Date"]).copy()
        sub["prev"] = sub.groupby("FEI")["Event Date"].shift(1)
        sub["gap_yr"] = (sub["Event Date"] - sub["prev"]).dt.days / 365.25
        gaps = sub.dropna(subset=["gap_yr"]).copy()
        gaps["year"] = gaps["Event Date"].dt.year

        pre   = gaps[gaps["year"] < 2020]["gap_yr"]
        covid = gaps[(gaps["year"] >= 2020) & (gaps["year"] <= 2022)]["gap_yr"]
        post  = gaps[gaps["year"] >= 2023]["gap_yr"]

        print(f"    Total events: {len(sub)}, gap pairs: {len(gaps)}")
        for lbl, g in [("Pre-COVID (<2020)", pre),
                        ("COVID (2020-2022)", covid),
                        ("Post-COVID (2023+)", post),
                        ("All", gaps["gap_yr"])]:
            if len(g) == 0:
                continue
            print(f"    {lbl}  n={len(g)}:")
            print(f"      Median={g.median():.2f}yr  Mean={g.mean():.2f}yr  "
                  f"P75={g.quantile(0.75):.2f}yr  >2yr={(g>2).mean():.1%}  "
                  f">3.5yr={(g>3.5).mean():.1%}")

    # Per-year inspection counts (slide 5 left column table)
    _sh("Per-year inspection counts — 98 panel FEIs (slide 5 table)")
    sub_panel = insp[insp["FEI"].isin(panel_feis)].copy()
    sub_panel["year"] = sub_panel["Event Date"].dt.year
    yearly = sub_panel.groupby("year").size()
    print("    Year-by-year (2015+):")
    for yr, n in yearly[yearly.index >= 2015].items():
        print(f"      {yr}: {n}")

    # Three-period summary for the slide table
    bins   = [0, 2019, 2022, 9999]
    labels = ["Pre-COVID (2015–2019)", "COVID (2020–2022)", "Post-COVID (2023+)"]
    sub_panel["period"] = pd.cut(sub_panel["year"], bins=bins, labels=labels)
    period_df = (
        sub_panel[sub_panel["year"] >= 2015]
        .groupby("period", observed=True)
        .agg(n_inspections=("Event Date", "count"), n_years=("year", "nunique"))
        .assign(avg_per_yr=lambda d: (d["n_inspections"] / d["n_years"]).round(0))
    )
    print("\n    Three-period summary (slide table):")
    print(period_df.to_string())

    _sh("Panel inspection-centered events (246 with 483 text)")
    if INSP_PANEL.exists():
        ic = pd.read_parquet(INSP_PANEL)
        print(f"    Total: {len(ic)}")
        if "gap_to_prev_yr" in ic.columns:
            g2 = ic["gap_to_prev_yr"].dropna()
            pre2   = ic.loc[ic["insp_year"] < 2020, "gap_to_prev_yr"].dropna() \
                     if "insp_year" in ic.columns else pd.Series(dtype=float)
            covid2 = ic.loc[(ic["insp_year"] >= 2020) & (ic["insp_year"] <= 2022),
                            "gap_to_prev_yr"].dropna() \
                     if "insp_year" in ic.columns else pd.Series(dtype=float)
            post2  = ic.loc[ic["insp_year"] >= 2023, "gap_to_prev_yr"].dropna() \
                     if "insp_year" in ic.columns else pd.Series(dtype=float)
            for lbl, g in [("Pre-COVID (<2020)", pre2),
                            ("COVID (2020-2022)", covid2),
                            ("Post-COVID (2023+)", post2),
                            ("All", g2)]:
                if len(g):
                    print(f"    {lbl}  n={len(g)}: Median={g.median():.2f}yr")
    else:
        print("    fei_ae_panel_inspection_centered.parquet not found — run 01_build_fei_ae_panel.py first")


# ── Slide 7: LLM Validation ───────────────────────────────────────────────────

def slide7_validation() -> None:
    _h("SLIDE 7 — LLM Extraction Validation")
    if VAL_CSV.exists():
        v = pd.read_csv(VAL_CSV)
        print(v.to_string(index=False))
    else:
        print("  Hardcoded from evaluate_extraction.py output:")
        print("  Severity agreement (4-tier):         67.5%  (random baseline 25%)")
        print("  Violation category agreement (8-cl): 58.6%  (random baseline 12.5%)")
        print("  DI flag  Precision: 0.307  Recall: 0.859  F1: 0.452")
        print("  Observations compared: 1,067")


# ── Slide 9: Spearman Correlations ───────────────────────────────────────────

def slide9_spearman(panel: pd.DataFrame) -> None:
    _h("SLIDE 9 — Spearman Correlations with Future AEs")

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

    if "n_ae_t1" not in panel.columns:
        print("  n_ae_t1 not in panel — skipping")
        return

    results = []
    for feat in TEXT_FEATURES:
        if feat not in panel.columns:
            continue
        mask = panel[feat].notna() & panel["n_ae_t1"].notna()
        if mask.sum() < 10:
            continue
        rho, pval = stats.spearmanr(panel.loc[mask, feat], panel.loc[mask, "n_ae_t1"])
        results.append({"feature": feat, "rho": round(rho, 3), "pval": round(pval, 4),
                        "n": int(mask.sum())})

    df_r = pd.DataFrame(results).sort_values("rho", ascending=False)
    print(f"  n FEI-years: {panel['n_ae_t1'].notna().sum()}")
    print(df_r.to_string(index=False))


# ── Slide 10: Panel Stats ─────────────────────────────────────────────────────

def slide10_panel_stats(panel: pd.DataFrame) -> None:
    _h("SLIDE 10 — Panel Construction Stats")

    print(f"  Total FEI × year rows:   {len(panel)}")
    print(f"  Unique FEIs:             {panel['fei'].nunique()}")

    if "ae_high_t1" in panel.columns:
        base = panel["ae_high_t1"].mean()
        print(f"  Outcome base rate:       {base:.1%}")

    if "any_oai" in panel.columns:
        n_oai = (panel["any_oai"] == 1).sum()
        print(f"  Rows with OAI:           {n_oai} ({n_oai/len(panel):.1%})")

    if "sdud_units" in panel.columns:
        cov = panel["sdud_units"].notna().mean()
        print(f"  SDUD FEI-year coverage:  {cov:.1%}")
        print(f"  Mean annual SDUD units:  {panel['sdud_units'].mean()/1e6:.0f}M")


# ── Slide 11: Ablation AUC ────────────────────────────────────────────────────

def slide11_ablation() -> None:
    _h("SLIDE 11 — Model Ablation Results")
    if ABLATION_CSV.exists():
        abl = pd.read_csv(ABLATION_CSV)
        print(abl.to_string(index=False))
    else:
        print("  ablation_metrics.csv not found — run 03_text_only_model.py first")
        print("  Hardcoded from last run:")
        rows = [
            ("A", "Text only (17 LLM)",      0.611, 0.569),
            ("B", "Text + insp count",        0.607, 0.577),
            ("C", "Insp count only",          0.545, 0.542),
            ("D", "Text, VAI-only FEIs",      0.555, 0.555),
            ("E", "Text, OAI-ever FEIs",      0.560, 0.484),
        ]
        print(f"  {'Config':6} {'Features':30} {'LR_AUC':>8} {'RF_AUC':>8}")
        for r in rows:
            print(f"  {r[0]:6} {r[1]:30} {r[2]:8.3f} {r[3]:8.3f}")
        print("  Text-only vs insp-count delta: +0.066 AUC")


# ── Slide 12: Gap × Trajectory ───────────────────────────────────────────────

def slide12_gap_trajectory() -> None:
    _h("SLIDE 12 — Inspection Gap × AE Trajectory (inspection-centered panel)")
    if not INSP_PANEL.exists():
        print("  Panel not found — run 01_build_fei_ae_panel.py --granularity inspection")
        return

    ic = pd.read_parquet(INSP_PANEL)
    ic.columns = [c.lower() for c in ic.columns]

    # Gap column: try common names, else compute from Redica on the fly
    gap_col = next((c for c in ic.columns
                    if ("gap" in c and ("yr" in c or "year" in c or "day" in c))
                    or c in ("gap_to_prev_yr", "insp_gap_yr", "days_since_prev")), None)

    ae_tm4 = next((c for c in ic.columns if "tm4" in c), None)
    ae_t0  = next((c for c in ic.columns if c == "n_ae_t0"), None)
    ae_tp4 = next((c for c in ic.columns if "tp4" in c), None)

    print(f"  Total inspections: {len(ic)}")
    print(f"  Columns: {[c for c in ic.columns if 'ae' in c or 'gap' in c or 'oai' in c]}")

    if gap_col is None:
        # Recompute gaps from Redica for the inspection dates in the panel
        print("  Gap column not in panel — computing from Redica inspection dates")
        sl = pd.read_excel(SITE_LIST)
        ev = pd.read_excel(REDICA_XLSX)
        ev["Event Date"] = pd.to_datetime(ev["Event Date"])
        insp_all = ev[
            (ev["Event Type"] == "Inspection") &
            (ev["Agency List"].str.contains("US - FDA", na=False))
        ].merge(sl[["Site Redica Id", "FEI"]], on="Site Redica Id", how="left")
        insp_all = insp_all.sort_values(["FEI", "Event Date"])
        insp_all["prev_date"] = insp_all.groupby("FEI")["Event Date"].shift(1)
        insp_all["gap_yr"] = (insp_all["Event Date"] - insp_all["prev_date"]).dt.days / 365.25

        # Identify inspection year from panel
        date_col = next((c for c in ic.columns if "date" in c or "year" in c.lower()), None)
        year_col = next((c for c in ic.columns if "year" in c), None)
        fei_col  = next((c for c in ic.columns if c == "fei"), None)

        if fei_col and year_col:
            ic_key = ic[[fei_col, year_col]].copy()
            ic_key.columns = ["FEI", "insp_year"]
            ic_key["FEI"] = ic_key["FEI"].astype(str)
            insp_all["FEI"] = insp_all["FEI"].astype(str)
            insp_all["insp_year"] = insp_all["Event Date"].dt.year
            merged = ic_key.merge(
                insp_all[["FEI", "insp_year", "gap_yr"]].drop_duplicates(["FEI","insp_year"]),
                on=["FEI","insp_year"], how="left"
            )
            ic["gap_yr"] = merged["gap_yr"].values
            gap_col = "gap_yr"
            print(f"  Matched gap for {ic['gap_yr'].notna().sum()}/{len(ic)} inspections")

    if gap_col and gap_col in ic.columns:
        ic[gap_col] = pd.to_numeric(ic[gap_col], errors="coerce")
        bins   = [0, 1, 2, 3.5, np.inf]
        labels = ["<1yr", "1-2yr", "2-3.5yr", ">3.5yr"]
        ic["gap_cat"] = pd.cut(ic[gap_col], bins=bins, labels=labels)

        lag_cols = [c for c in ic.columns if c.startswith("n_ae_")]
        if lag_cols:
            grp = ic.groupby("gap_cat", observed=True)[lag_cols].mean().round(1)
            grp["n"] = ic.groupby("gap_cat", observed=True).size()
            if ae_tm4 and ae_t0 and ae_tp4:
                grp["pre_rise"] = (grp[ae_t0] / grp[ae_tm4].replace(0, np.nan)).round(3)
                grp["persist"]  = (grp[ae_tp4] / grp[ae_t0].replace(0, np.nan)).round(3)
            print(grp.to_string())


# ── Slide 13: Yearly Trajectory ───────────────────────────────────────────────

def slide13_yearly_trajectory() -> None:
    _h("SLIDE 13 — Yearly AE Trajectory by Group")
    if TRAJ_CSV.exists():
        traj = pd.read_csv(TRAJ_CSV)
        print(traj.to_string(index=False))
    else:
        print("  ae_trajectory_by_group.csv not found — run 03_text_only_model.py first")
        print("  Hardcoded from last run:")
        print("  Group          tm2   tm1    t0    t1    t2  pre_rise  persist")
        print("  All           2533  2659  2727  2413  2343     —        —")
        print("  OAI-ever      2542  2673  2716  2415  2357    1.069    0.868")
        print("  High-sig VAI  3360  3585  3738  3343  3286    1.112    0.879")
        print("  Low-sig VAI   2237  2328  2374  2084  2001    1.061    0.843")


# ── Slide 14: Quarterly Trajectory ───────────────────────────────────────────

def slide14_quarterly_trajectory() -> None:
    _h("SLIDE 14 — Quarterly Trajectory (inspection-centered)")
    if not INSP_PANEL.exists():
        print("  Panel not found.")
        return

    ic = pd.read_parquet(INSP_PANEL)
    ic.columns = [c.lower() for c in ic.columns]

    # Identify AE quarterly columns (t-4 to t+4)
    qtr_ae = sorted([c for c in ic.columns if c.startswith("n_ae_q") or
                     (c.startswith("n_ae_") and any(x in c for x in
                      ["tm4","tm3","tm2","tm1","t0","tp1","tp2","tp3","tp4"]))],
                    key=lambda x: x)

    # Binary OAI indicator: any inspection at this FEI was OAI
    oai_count_col = next((c for c in ic.columns if c == "n_oai"), None)
    any_oai_col   = next((c for c in ic.columns if c == "any_oai"), None)

    if oai_count_col and any_oai_col is None:
        ic["ever_oai"] = (ic[oai_count_col] > 0).astype(int)
        oai_bin_col = "ever_oai"
    elif any_oai_col:
        oai_bin_col = any_oai_col
    else:
        oai_bin_col = None

    if not qtr_ae:
        print(f"  No quarterly AE columns found. Columns: {[c for c in ic.columns if 'ae' in c]}")
        return

    print(f"  Total inspections: {len(ic)}")
    print(f"  OAI binary column: {oai_bin_col}")
    print(f"  Quarterly AE columns: {qtr_ae}")

    all_means = ic[qtr_ae].mean().round(0)
    print("\n  All inspections (mean AE per quarter):")
    print(all_means.to_string())

    t0_col  = next((c for c in qtr_ae if c == "n_ae_t0"),  None)
    tm4_col = next((c for c in qtr_ae if "tm4" in c),      None)
    tp4_col = next((c for c in qtr_ae if "tp4" in c),      None)

    if oai_bin_col:
        grp = ic.groupby(oai_bin_col)[qtr_ae].mean().round(0)
        grp.index = ["VAI/NAI (OAI=0)", "OAI-ever (OAI=1)"]
        print("\n  By OAI status:")
        print(grp.to_string())

        if t0_col and tm4_col and tp4_col:
            print("\n  pre_rise and persist by group:")
            for lbl, mask in [("VAI/NAI", ic[oai_bin_col]==0), ("OAI-ever", ic[oai_bin_col]==1)]:
                sub = ic[mask]
                t0  = sub[t0_col].mean()
                tm4 = sub[tm4_col].mean()
                tp4 = sub[tp4_col].mean()
                n   = len(sub)
                print(f"    {lbl}  n={n}:  pre_rise={t0/tm4:.3f}  persist={tp4/t0:.3f}")


# ── Slide 15 & 16: Clustering ─────────────────────────────────────────────────

def slide15_16_clustering() -> None:
    _h("SLIDE 15 & 16 — Trajectory Clustering + Strategic Leniency")
    if CLUSTER_CSV.exists():
        cl = pd.read_csv(CLUSTER_CSV)
        print("  Cluster summary:")
        print(cl.to_string(index=False))
    else:
        print("  ae_cluster_summary.csv not found — run 04_ae_trajectory_clustering.py first")

    if CLUSTER_FEIS.exists():
        feis = pd.read_csv(CLUSTER_FEIS)
        flat_rising = feis[feis["cluster"] == "Flat/rising"].sort_values(
            "tech_score", ascending=False)
        print(f"\n  Flat/rising FEIs ({len(flat_rising)} total):")
        cols = [c for c in ["fei","any_oai","n_ae_t0","n_ae_t2","persist","tech_score",
                             "vc_labcontrols_share","data_integrity_llm_share"] if c in feis.columns]
        print(flat_rising[cols].to_string(index=False))


# ── Slide 17: SDUD Volume ─────────────────────────────────────────────────────

def slide17_sdud_volume(panel: pd.DataFrame) -> None:
    _h("SLIDE 17 — SDUD Volume Control")

    if "sdud_units" not in panel.columns:
        print("  sdud_units not in panel")
        return

    cov = panel["sdud_units"].notna().mean()
    mean_units = panel["sdud_units"].mean()
    print(f"  FEI-year coverage:    {cov:.1%}")
    print(f"  Mean annual units:    {mean_units/1e6:.0f}M")
    print(f"  Median annual units:  {panel['sdud_units'].median()/1e6:.0f}M")

    if "ae_rate_t1" in panel.columns:
        rate = panel["ae_rate_t1"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  Median ae_rate_t1:    {rate.median():.0f} AEs/M units")
        print(f"  Mean ae_rate_t1:      {rate.mean():.0f}  (heavily skewed)")

    if "any_oai" in panel.columns:
        for oai_val, lbl in [(0, "Never-OAI"), (1, "Ever-OAI")]:
            sub = panel[panel["any_oai"] == oai_val]
            mean_u = sub["sdud_units"].mean()
            if "ae_rate_t1" in panel.columns:
                rate = sub["ae_rate_t1"].replace([np.inf,-np.inf], np.nan).dropna().mean()
                print(f"  {lbl}: mean SDUD={mean_u/1e6:.0f}M  ae_rate={rate:.0f}/M")
            else:
                print(f"  {lbl}: mean SDUD={mean_u/1e6:.0f}M")

    # SDUD raw data size (approximate)
    sdud_path = ROOT / "Data/04 - Medicaid - SDUD"
    if sdud_path.exists():
        sdud_files = list(sdud_path.glob("**/*.parquet")) + list(sdud_path.glob("**/*.csv"))
        print(f"  SDUD source files found: {len(sdud_files)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading panel…")
    panel = pd.read_parquet(PANEL_PARQ)
    print(f"  {len(panel)} rows, {panel['fei'].nunique()} FEIs")

    slide4_study_design(panel)
    slide5_inspection_gaps(panel)
    slide7_validation()
    slide9_spearman(panel)
    slide10_panel_stats(panel)
    slide11_ablation()
    slide12_gap_trajectory()
    slide13_yearly_trajectory()
    slide14_quarterly_trajectory()
    slide15_16_clustering()
    slide17_sdud_volume(panel)

    print("\n" + "="*70)
    print("  Done. Cross-check each number against the slide before presenting.")
    print("="*70)


if __name__ == "__main__":
    main()
