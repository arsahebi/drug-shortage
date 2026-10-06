"""
08_product_cell_analysis.py
────────────────────────────────────────────────────────────────────────────
Product-level test of whether 483 text lines up with patient behavior in
MarketScan, replacing the plant-pooled pre/post design in 01b/02.

Why a new unit (checked 2026-10-06 before writing this script)
  - Plant-pooled abandonment, standardized against same-product peers, barely
    agrees between CCAE and MDCR (Spearman 0.11, 55 plants), so it is not a
    stable plant trait.
  - The within-product abandonment percentile of a labeler x exact-product
    cell (ingredient, route, form, strength) agrees across cohorts at 0.36
    (386 cells). This is the unit behind the MarketScan team's Qual Score.
  - 39% of 483 observations concern sterile / injectable operations, which
    cannot affect the oral products patients fill at a pharmacy.

Design (A: unit)
  Unit: labeler x exact product, attributed to one finished-dose plant with
  the same bridge as 01b (ProPublica single plant, then Valisure / new
  crosswalks, then DailyMed manufacture sites).
  Patient window: 2016Q3-2022Q4 in both cohorts (MDCR break at 2023; CCAE
  cut at the same point so the cohorts cover the same period).
  Products: oral solids only (TAB, CAP and the modified-release forms below),
  five acute APIs dropped. Tacrolimus ointment, solutions, powders etc. out.
  Outcome: the cell's percentile among ALL labelers of that exact product
  (bridged or not) with >= 300 switches in that cohort; the product needs
  >= 3 such labelers. Higher = worse. Percentile = (rank - 0.5) / n.
    primary    : abandonment (stops - dose change - coverage loss) / switches,
                 mean of CCAE and MDCR percentiles, cell must qualify in both
    secondary  : same, CCAE only (more cells)
    secondary  : net ER visits per 1,000 switches (90d after - 90d before)
    secondary  : net failure-mode dx visits per 1,000 switches
  Text window: the plant's inspections dated 2018-2022 (overlapping the
  patient window). Sensitivity: all inspections 2018-2026.

Design (B: matched observations and pre-specified hypotheses)
  Only observations that can bear on oral products are kept: an observation
  is dropped if it describes sterile / injectable operations (regex
  STERILE_RE) unless it also mentions oral solid manufacturing (ORAL_RE).
  Plant exposures are shares of the kept observations:
    H1 lab_share       : violation_category == LaboratoryControlsSystem
    H2 di_share        : data_integrity_flag_llm
    H3 product_test_share : OOS/OOT regex flag, or text citing dissolution,
                          assay, potency, content uniformity or impurities
    H4 H3 x modified-release : the H3 effect is larger for modified-release
                          products (T12, T24, TER, CER, ECT, TCP), where a
                          release failure is felt by the patient
  Expected sign: positive (more of these problems -> patients abandon the
  product more than peers' patients do).
  Comparators (not hypotheses): critical+major severity share, and the
  FDA classification (plant had any OAI in the text window).

Estimation
  OLS: percentile ~ z(exposure) + product fixed effects, SEs clustered by
  plant (one plant can carry several cells). Coefficient = percentile points
  per 1 SD of the exposure. H4 adds MR and z(H3) x MR. Holm correction over
  H1-H4 on the primary outcome only. Robustness: Spearman between plant mean
  percentile and the exposure (one row per plant).

Results (first run 2026-10-06; see outputs/tables/product_cell_results.md)
  2016Q3-2022Q4, text 2018-2022, 143 cells / 32 plants: H1 lab +9.4 pct pts
  per SD (Holm p 0.022), H2 DI +10.8 (Holm p 0.001); plant-permutation
  two-sided p 0.011 / 0.0025; present in CCAE and MDCR separately; the OAI
  flag is null (-3.8). H3/H4 null. Removing atorvastatin cuts H2 to +3.6.
  The sterile filter makes no difference. NOT replicated in 2023Q1-2024Q3
  CCAE with either 2018-2022 or 2023-2026 text, but the outcome itself is
  only weakly stable over time (cell percentile rho 0.10-0.29 between
  periods), and H1/H2 were picked after a first look at the main window.

Outputs
  outputs/product_cell_panel.parquet
  outputs/tables/product_cell_results.csv
  outputs/tables/product_cell_results.md
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT  = HERE / "outputs"
TABS = OUT / "tables"

STEP01 = ROOT / "Analysis" / "Text Analysis" / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
PANEL  = OUT / "fei_ae_panel_inspection_centered.parquet"   # OAI/VAI/NAI per inspection, all 246

# Reuse the plant bridge from 01b so both analyses attribute rows identically.
_spec = importlib.util.spec_from_file_location("ms01b", HERE / "01b_build_marketscan_outcomes.py")
ms01b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms01b)

COHORTS = ["CCAE", "MDCR"]
FIRST_QI, LAST_QI = 2016 * 4 + 3, 2022 * 4 + 4
MIN_SW, MIN_PEERS = 300, 3
MR_FORMS = {"T12", "T24", "TER", "CER", "ECT", "TCP"}
ORAL_FORMS = {"TAB", "CAP"} | MR_FORMS
PRODUCT_KEY = ["ingredient", "route", "form", "strength"]

STERILE_RE = (r"steril|aseptic|inject|parenteral|\bvials?\b|lyophili|media fill|"
              r"endotoxin|iso 5|cleanroom|grade a\b|bioburden|ophthalmic")
ORAL_RE = r"tablet|capsule|oral solid|compress|granulat|blend|coating|encapsul"
PRODUCT_TEST_RE = r"dissolution|\bassay\b|potency|content uniformity|impurit"

HYPOTHESES = ["lab_share", "di_share", "product_test_share"]
COMPARATORS = ["critmajor_share", "any_oai"]


def _load_cells(cohorts=COHORTS, first_qi=FIRST_QI, last_qi=LAST_QI) -> pd.DataFrame:
    """One row per labeler x exact product x cohort with outcome percentiles."""
    xws = ms01b._load_crosswalks()
    out = []
    for c in cohorts:
        d = pd.read_csv(ms01b.MS_DIR / f"manufacturer_product_quarter_{c}.csv",
                        dtype={"plant_fei": str, "manufacturer_labeler_code": str, "ndc9_list": str})
        d = d[~d["ingredient"].str.startswith(ms01b.ACUTE_PREFIXES) & d["form"].isin(ORAL_FORMS)].copy()
        d["qi"] = d["quarter"].map(ms01b._qidx)
        d = d[(d["qi"] >= first_qi) & (d["qi"] <= last_qi)].copy()
        d["adj_stops"] = d["n_stops"] - d["n_stops_dose_change"] - d["n_coverage_ended"]
        d["er_net"] = d["er_visits_after_90d"] - d["er_visits_before_90d"]
        d["dx_net"] = d["dx_visits_after_90d"] - d["dx_visits_before_90d"]

        # Bridge each labeler-product row to one plant (same order as 01b).
        single = d["plant_fei"].notna() & ~d["plant_fei"].fillna("").str.contains(";")
        d["fei"] = d["plant_fei"].where(single)
        for _, mp in xws:
            need = d["fei"].isna()
            hit = d.loc[need, "ndc9_list"].map(lambda l: ms01b._single_fei(l, mp)).dropna()
            d.loc[hit.index, "fei"] = hit

        g = (d.groupby(["manufacturer_labeler_code"] + PRODUCT_KEY)
               .agg(n_sw=("n_switches", "sum"), adj=("adj_stops", "sum"),
                    er=("er_net", "sum"), dx=("dx_net", "sum"),
                    fei=("fei", lambda s: s.dropna().iloc[0] if s.notna().any() and s.dropna().nunique() == 1 else None))
               .reset_index())
        g = g[g["n_sw"] >= MIN_SW].copy()
        g["ab_rate"] = g["adj"] / g["n_sw"]
        g["er_rate"] = 1000 * g["er"] / g["n_sw"]
        g["dx_rate"] = 1000 * g["dx"] / g["n_sw"]
        g["n_peers"] = g.groupby(PRODUCT_KEY)["n_sw"].transform("size")
        g = g[g["n_peers"] >= MIN_PEERS].copy()
        for m in ["ab", "er", "dx"]:
            r = g.groupby(PRODUCT_KEY)[f"{m}_rate"].rank(method="average")
            g[f"{m}_pct"] = (r - 0.5) / g["n_peers"]
        g["cohort"] = c
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _plant_exposures(year_max: int, year_min: int = 2018) -> pd.DataFrame:
    d = pd.read_csv(STEP01)
    d["insp_date"] = pd.to_datetime(d["insp_date"])
    d = d[(d["insp_date"].dt.year >= year_min) & (d["insp_date"].dt.year <= year_max)].copy()
    t = d["obs_text_clean"].fillna("").str.lower()
    sterile = t.str.contains(STERILE_RE) & ~t.str.contains(ORAL_RE)
    n_all = d.groupby("fei").size()
    d = d[~sterile].copy()
    t = t[~sterile]
    d["lab"] = d["violation_category"].eq("LaboratoryControlsSystem")
    d["di"] = d["data_integrity_flag_llm"].astype(str).str.lower().eq("true")
    d["ptest"] = d["has_oos_oot_regex"].astype(str).str.lower().eq("true") | t.str.contains(PRODUCT_TEST_RE)
    d["critmajor"] = d["severity_tier"].isin(["Critical", "Major"])
    e = d.groupby("fei").agg(n_obs_oral=("lab", "size"), lab_share=("lab", "mean"),
                             di_share=("di", "mean"), product_test_share=("ptest", "mean"),
                             critmajor_share=("critmajor", "mean"))
    e["n_obs_all"] = n_all.reindex(e.index)

    p = pd.read_parquet(PANEL, columns=["fei", "insp_date", "any_oai"])
    yr = pd.to_datetime(p["insp_date"]).dt.year
    p = p[(yr >= year_min) & (yr <= year_max)]
    e["any_oai"] = p.groupby("fei")["any_oai"].max().reindex(e.index).fillna(0).astype(int)
    e.index = e.index.astype("int64").astype(str)
    return e


def _build_panel(cells: pd.DataFrame, expo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Analysis frames keyed by outcome name."""
    linked = cells[cells["fei"].isin(expo.index)].copy()
    key = ["manufacturer_labeler_code"] + PRODUCT_KEY
    wide = linked.pivot_table(index=key + ["fei"], columns="cohort",
                              values=["ab_pct", "er_pct", "dx_pct"], aggfunc="first")
    wide.columns = [f"{v}_{c.lower()}" for v, c in wide.columns]
    wide = wide.reset_index()
    wide["product"] = wide[PRODUCT_KEY].astype(str).agg(" | ".join, axis=1)
    wide["mr"] = wide["form"].isin(MR_FORMS).astype(int)
    wide = wide.merge(expo, left_on="fei", right_index=True, how="left")

    frames = {}
    both = (wide.dropna(subset=["ab_pct_ccae", "ab_pct_mdcr"]).copy()
            if "ab_pct_mdcr" in wide.columns else None)
    for m in ["ab", "er", "dx"] if both is not None else []:
        f = both.copy()
        f["y"] = (f[f"{m}_pct_ccae"] + f[f"{m}_pct_mdcr"]) / 2
        frames[{"ab": "abandonment (CCAE+MDCR)", "er": "net ER (CCAE+MDCR)",
                "dx": "net failure-mode dx (CCAE+MDCR)"}[m]] = f
    f = wide.dropna(subset=["ab_pct_ccae"]).copy()
    f["y"] = f["ab_pct_ccae"]
    frames["abandonment (CCAE only)"] = f
    return frames


def _fit(f: pd.DataFrame, x: str, interaction: bool = False) -> dict:
    f = f.dropna(subset=["y", x]).copy()
    sd = f[x].std()
    if f["fei"].nunique() < 8 or not sd > 0:
        return {"n_cells": len(f), "n_plants": f["fei"].nunique()}
    f["z"] = (f[x] - f[x].mean()) / sd
    formula = "y ~ z * mr + C(product)" if interaction else "y ~ z + C(product)"
    term = "z:mr" if interaction else "z"
    m = smf.ols(formula, data=f).fit(cov_type="cluster", cov_kwds={"groups": f["fei"]})
    plant = f.groupby("fei").agg(y=("y", "mean"), x=(x, "first"))
    rho, prho = spearmanr(plant["x"], plant["y"])
    lo, hi = m.conf_int().loc[term]
    return {"n_cells": len(f), "n_plants": f["fei"].nunique(),
            "beta_pct_pts": 100 * m.params[term], "ci_lo": 100 * lo, "ci_hi": 100 * hi,
            "p": m.pvalues[term], "plant_rho": rho, "plant_rho_p": prho}


def _holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        adj[k] = running
    return adj


def main() -> None:
    print("Building labeler x exact-product cells...")
    cells = _load_cells()
    for c in COHORTS:
        cc = cells[cells["cohort"] == c]
        print(f"  {c}: {len(cc)} cells >= {MIN_SW} switches in products with >= {MIN_PEERS} labelers, "
              f"{cc['fei'].notna().sum()} bridged to a plant")

    rows, panels = [], []
    for window, year_max in [("2018-2022", 2022), ("2018-2026 (sensitivity)", 2026)]:
        expo = _plant_exposures(year_max)
        frames = _build_panel(cells, expo)
        for outcome, f in frames.items():
            tests = [(h, h, False) for h in HYPOTHESES] + [("H4 product_test x MR", "product_test_share", True)]
            tests += [(cmp, cmp, False) for cmp in COMPARATORS]
            for label, x, inter in tests:
                r = _fit(f, x, interaction=inter)
                rows.append({"text_window": window, "outcome": outcome, "exposure": label,
                             "role": "comparator" if label in COMPARATORS else "hypothesis", **r})
        prim = frames["abandonment (CCAE+MDCR)"].assign(text_window=window)
        panels.append(prim)

    # Replication in 2023Q1-2024Q3 (CCAE only; MDCR stops at 2022). Added
    # after the main run: H1/H2 were chosen after a first look at the
    # 2016-2022 cells, so the main window is not a clean confirmatory test.
    rep_cells = _load_cells(["CCAE"], 2023 * 4 + 1, 2024 * 4 + 3)
    for window, (ymin, ymax) in [("replication: text 2018-2022 -> CCAE 2023-24", (2018, 2022)),
                                 ("replication: text 2023-2026 -> CCAE 2023-24", (2023, 2026))]:
        expo = _plant_exposures(ymax, ymin)
        f = _build_panel(rep_cells, expo)["abandonment (CCAE only)"]
        for label in HYPOTHESES + COMPARATORS:
            rows.append({"text_window": window, "outcome": "abandonment (CCAE only)", "exposure": label,
                         "role": "comparator" if label in COMPARATORS else "hypothesis", **_fit(f, label)})

    res = pd.DataFrame(rows)
    hyp = (res["role"] == "hypothesis") & (res["outcome"] == "abandonment (CCAE+MDCR)")
    res["p_holm"] = np.nan
    for w in res["text_window"].unique():
        k = hyp & (res["text_window"] == w) & res["p"].notna()
        res.loc[k, "p_holm"] = _holm(res.loc[k, "p"])

    OUT.mkdir(parents=True, exist_ok=True)
    TABS.mkdir(parents=True, exist_ok=True)
    pd.concat(panels).to_parquet(OUT / "product_cell_panel.parquet", index=False)
    res.to_csv(TABS / "product_cell_results.csv", index=False)

    show = ["outcome", "exposure", "n_cells", "n_plants", "beta_pct_pts", "ci_lo", "ci_hi",
            "p", "p_holm", "plant_rho", "plant_rho_p"]
    md = ["# Product-cell analysis: 483 text vs within-product patient outcomes", "",
          "Unit: labeler x exact product (oral solids), bridged to one plant. Outcome: percentile",
          "among all labelers of the same product (higher = worse), 2016Q3-2022Q4, >= 300 switches.",
          "beta = percentile points per 1 SD of the exposure, product fixed effects, SEs clustered",
          "by plant. Holm over H1-H4 on the primary outcome. plant_rho = Spearman, one row per plant.", ""]
    for w in res["text_window"].unique():
        md += [f"## Text window {w}", "", res.loc[res["text_window"] == w, show]
               .to_string(index=False, float_format=lambda v: f"{v:.3f}"), ""]
    (TABS / "product_cell_results.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {TABS / 'product_cell_results.csv'}")


if __name__ == "__main__":
    main()
