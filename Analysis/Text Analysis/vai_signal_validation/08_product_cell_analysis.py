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

All 17 fixed features (added 2026-10-06)
  The same model run for every step02 feature (the list 02 and
  valisure_validation use), in the main window, its two halves, and three
  7-quarter CCAE windows including 2023Q1-2024Q3. Diagnostics: cross-cohort
  agreement at plant vs cell level, period stability of the cell percentile,
  and CCAE abandonment by quarter.

Outputs
  outputs/product_cell_panel.parquet
  outputs/tables/product_cell_results.csv
  outputs/tables/product_cell_results.md
  outputs/tables/product_cell_robustness.csv
  outputs/tables/product_cell_all17.csv / .md
  outputs/tables/product_cell_diagnostics.csv
  outputs/tables/marketscan_ccae_quarterly_abandonment.csv
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

TEXT_TS = ROOT / "Analysis" / "Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
# The 17 fixed step02 features used by 02 and by valisure_validation (same list, same order).
_spec02 = importlib.util.spec_from_file_location("m02", HERE / "02_vai_signal_model.py")
m02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(m02)
FIXED17 = list(m02.TEXT_FEATURES)

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


def _expo17(year_min: int, year_max: int) -> pd.DataFrame:
    """Plant mean of the 17 fixed step02 features over its inspections in the
    window (all observations, no route filter, matching the Valisure matrix),
    plus the plant's any-OAI flag in the same window."""
    ts = pd.read_csv(TEXT_TS, parse_dates=["snapshot_date"])
    yr = ts["snapshot_date"].dt.year
    ts = ts[(yr >= year_min) & (yr <= year_max)].copy()
    ts["fei"] = ts["fei"].astype("int64").astype(str)
    e = ts.groupby("fei")[FIXED17].mean()
    p = pd.read_parquet(PANEL, columns=["fei", "insp_date", "any_oai"])
    yr = pd.to_datetime(p["insp_date"]).dt.year
    p = p[(yr >= year_min) & (yr <= year_max)]
    e["any_oai"] = (p.groupby(p["fei"].astype("int64").astype(str))["any_oai"].max()
                    .reindex(e.index).fillna(0))
    return e


def _all17_rows(cells: pd.DataFrame, expo: pd.DataFrame, frame: str, window: str) -> list[dict]:
    f = _build_panel(cells, expo)[frame]
    rows = [{"window": window, "outcome": frame, "feature": x, **_fit(f, x)}
            for x in FIXED17 + ["any_oai"]]
    t = pd.DataFrame(rows)
    k = (t["feature"] != "any_oai") & t["p"].notna()
    t.loc[k, "p_holm17"] = _holm(t.loc[k, "p"])
    return t.to_dict("records")


def _diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-quality checks behind the design and the 2023-24 replication.

    1. Cross-cohort agreement (CCAE vs MDCR, 2016Q3-2022Q4) at two units:
       plant-pooled standardized abandonment ratio (observed / peer-expected
       stops, plants with >= 1,000 switches in each cohort) and the
       within-product percentile of labeler x exact-product cells.
    2. Period stability of the CCAE cell percentile across three windows.
    3. CCAE abandonment and coverage-loss share by quarter (oral, non-acute,
       all labelers), which shows the 2023Q3-Q4 drop.
    """
    rows = []
    xws = ms01b._load_crosswalks()
    sar = {}
    for c in COHORTS:
        agg, _ = ms01b._load_cohort(c, xws)
        a = agg.reset_index()
        a = a[(a["qi"] >= FIRST_QI) & (a["qi"] <= LAST_QI)].groupby("fei")[
            ["adj_stops_peer", "expected_stops", "n_switches"]].sum()
        a = a[a["n_switches"] >= 1000]
        sar[c] = a["adj_stops_peer"] / a["expected_stops"]
    j = pd.concat(sar, axis=1, join="inner")
    rho, pv = spearmanr(j["CCAE"], j["MDCR"])
    rows.append({"check": "CCAE vs MDCR agreement, plant-pooled SAR", "n": len(j), "rho": rho, "p": pv})

    cells = _load_cells()
    key = ["manufacturer_labeler_code"] + PRODUCT_KEY
    w = cells.pivot_table(index=key, columns="cohort", values="ab_pct").dropna()
    rho, pv = spearmanr(w["CCAE"], w["MDCR"])
    rows.append({"check": "CCAE vs MDCR agreement, cell within-product percentile", "n": len(w), "rho": rho, "p": pv})

    periods = [("2016Q3-2019Q4", 2016 * 4 + 3, 2019 * 4 + 4), ("2020Q1-2022Q4", 2020 * 4 + 1, 2022 * 4 + 4),
               ("2023Q1-2024Q3", 2023 * 4 + 1, 2024 * 4 + 3)]
    pc = {n: _load_cells(["CCAE"], a, b).set_index(key)["ab_pct"] for n, a, b in periods}
    names = [n for n, _, _ in periods]
    for i in range(3):
        for k in range(i + 1, 3):
            x = pd.concat([pc[names[i]], pc[names[k]]], axis=1, join="inner")
            rho, pv = spearmanr(x.iloc[:, 0], x.iloc[:, 1])
            rows.append({"check": f"CCAE cell percentile stability, {names[i]} vs {names[k]}",
                         "n": len(x), "rho": rho, "p": pv})

    d = pd.read_csv(ms01b.MS_DIR / "manufacturer_product_quarter_CCAE.csv")
    d = d[~d["ingredient"].str.startswith(ms01b.ACUTE_PREFIXES) & d["form"].isin(ORAL_FORMS)].copy()
    d["adj"] = d["n_stops"] - d["n_stops_dose_change"] - d["n_coverage_ended"]
    q = d.groupby("quarter").agg(n_switches=("n_switches", "sum"), adj_stops=("adj", "sum"),
                                 n_stops=("n_stops", "sum"), n_coverage_ended=("n_coverage_ended", "sum"))
    q["abandonment"] = q["adj_stops"] / q["n_switches"]
    q["coverage_share_of_stops"] = q["n_coverage_ended"] / q["n_stops"]
    return pd.DataFrame(rows), q.reset_index()


def _robustness(f: pd.DataFrame, n_perm: int = 2000) -> pd.DataFrame:
    """H1/H2 on the primary frame: plant-level permutation test (exposures
    shuffled across plants, cells keep their plant), each cohort alone on the
    same cells, and leave-one-API-out."""
    def beta(df, x):
        df = df.dropna(subset=["y", x]).copy()
        df["z"] = (df[x] - df[x].mean()) / df[x].std()
        return 100 * smf.ols("y ~ z + C(product)", data=df).fit().params["z"]

    rows, rng = [], np.random.default_rng(1)
    for x in ["lab_share", "di_share"]:
        b0 = beta(f, x)
        plants = f["fei"].unique()
        vals = f.groupby("fei")[x].first().reindex(plants).values
        null = np.array([beta(f.assign(**{x: f["fei"].map(dict(zip(plants, rng.permutation(vals))))}), x)
                         for _ in range(n_perm)])
        rows.append({"exposure": x, "check": f"plant permutation ({n_perm}), two-sided p",
                     "value": float((np.abs(null) >= abs(b0)).mean())})
        for coh in ["ccae", "mdcr"]:
            rows.append({"exposure": x, "check": f"{coh.upper()} only, same cells: beta",
                         "value": beta(f.assign(y=f[f"ab_pct_{coh}"]), x)})
        for ing in sorted(f["ingredient"].unique()):
            rows.append({"exposure": x, "check": f"leave out {ing}: beta",
                         "value": beta(f[f["ingredient"] != ing], x)})
    return pd.DataFrame(rows)


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
        if year_max == 2022:
            rob = _robustness(frames["abandonment (CCAE+MDCR)"])

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
    rob.to_csv(TABS / "product_cell_robustness.csv", index=False)

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
    md += ["## H1/H2 robustness (text 2018-2022, primary outcome)", "",
           rob.to_string(index=False, float_format=lambda v: f"{v:.4f}"), ""]
    (TABS / "product_cell_results.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {TABS / 'product_cell_results.csv'}")

    # ── All 17 fixed features (added 2026-10-06, same family as the Valisure
    # matrix; Holm over 17 within each window). Exploratory: these windows
    # reuse data already examined.
    print("\nAll 17 fixed features...")
    e1822 = _expo17(2018, 2022)
    a17 = []
    a17 += _all17_rows(cells, e1822, "abandonment (CCAE+MDCR)", "main 2016Q3-2022Q4, both cohorts")
    for name, a, b in [("half A 2016Q3-2019Q4, both cohorts", 2016 * 4 + 3, 2019 * 4 + 4),
                       ("half B 2020Q1-2022Q4, both cohorts", 2020 * 4 + 1, 2022 * 4 + 4)]:
        a17 += _all17_rows(_load_cells(COHORTS, a, b), e1822, "abandonment (CCAE+MDCR)", name)
    for name, a, b in [("CCAE 7q 2019Q1-2020Q3", 2019 * 4 + 1, 2020 * 4 + 3),
                       ("CCAE 7q 2021Q1-2022Q3", 2021 * 4 + 1, 2022 * 4 + 3),
                       ("CCAE 7q 2023Q1-2024Q3 (replication)", 2023 * 4 + 1, 2024 * 4 + 3)]:
        a17 += _all17_rows(_load_cells(["CCAE"], a, b), e1822, "abandonment (CCAE only)", name)
    a17 = pd.DataFrame(a17)
    a17.to_csv(TABS / "product_cell_all17.csv", index=False)

    diag, quarters = _diagnostics()
    diag.to_csv(TABS / "product_cell_diagnostics.csv", index=False)
    quarters.to_csv(TABS / "marketscan_ccae_quarterly_abandonment.csv", index=False)

    md = ["# Product-cell analysis: all 17 fixed text features", "",
          "Plant mean of the 17 step02 features over 2018-2022 inspections (all observations). Same",
          "unit, outcome and model as product_cell_results.md. Holm over the 17 within each window.", ""]
    s17 = ["feature", "n_cells", "n_plants", "beta_pct_pts", "ci_lo", "ci_hi", "p", "p_holm17", "plant_rho"]
    for w in a17["window"].unique():
        md += [f"## {w}", "", a17.loc[a17["window"] == w, s17]
               .to_string(index=False, float_format=lambda v: f"{v:.3f}"), ""]
    md += ["## Diagnostics", "", diag.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
           "## CCAE quarterly abandonment (oral, non-acute, all labelers)", "",
           quarters.to_string(index=False, float_format=lambda v: f"{v:.4f}"), ""]
    (TABS / "product_cell_all17.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {TABS / 'product_cell_all17.csv'}")


if __name__ == "__main__":
    main()
