"""
11_quality_score_analysis.py
────────────────────────────────────────────────────────────────────────────
Do the 17 fixed 483 text features line up with the MarketScan team's
quality_score_v2, the externally validated product-quality score?

Why this outcome
  08 used our own rebuilt abandonment percentile, and its data-integrity result
  depends on the coverage subtraction in the numerator (+11.0 pct pts per SD
  with it, +6.3, p 0.12, with the dose-change adjustment only). The vendor's
  score was delivered on 2026-10-14 as one new column. It is reported to agree
  with Valisure / DoD laboratory flags in 87% of within-product pairs
  (docs/Qual_Score_Explained.pdf), and it sits at the same labeler x exact
  product unit as 08.

What the score is (from the vendor document, then checked here)
  score = 0.60 rank(abandonment) + 0.26 rank(ER + failure-mode visits)
          + 0.14 rank(plant does on-site API manufacture), each rank a
  percentile within ingredient | route | form | strength; higher = worse.
  Claims window 2016-07 to 2022-06. So the score is NOT independent of the
  abandonment definition: 60% of it is the vendor's own abandonment rank,
  built at the patient level with a coverage-loss correction. The 14% API
  part is a plant attribute that could track text features through plant
  type. Both caveats travel with any result here.

Design (mirrors 08)
  Unit: labeler x exact product with a score. Oral solids, five acute APIs
    dropped. Each row is bridged to a finished-dose plant with 08's bridge
    (the delivered single plant_fei, then the Valisure / new crosswalks); a
    cell keeps a plant only if all its bridged rows agree on one.
  Text: plant mean of the 17 step02 features over its inspections, 2018-2022
    (overlapping the claims window); sensitivity 2018-2026. any_oai is a
    comparator, not a hypothesis.
  Model: OLS score ~ z(feature) + product fixed effects, SEs clustered by
    plant; coefficient = score points x 100 per SD (percentile points). Holm
    over the 17. Also without fixed effects, and the plant-level Spearman of
    plant mean score against the feature. Sensitivity: cells clearing the
    vendor's 300-switch bar in both cohorts.
  All 17 are reported. Data integrity and lab controls were picked after a
  look at abandonment, so on this outcome they are not privileged.

Diagnostics written alongside
  - whether the score is constant within cell and identical across cohorts
  - how much of the score product fixed effects explain (is it already a
    within-product scale?)
  - Spearman of the score against 08's abandonment percentile under each of
    the three numerators, which shows which definition the score follows
  - cell flow from scored cells to the analysis sample

Results (first run 2026-10-15)
  Score checks: constant within cell, identical in CCAE and MDCR on all 718
  cells; product fixed effects explain 3.6% of it (product means 0.46-0.75),
  so it is already close to a within-product scale. It tracks 08's CCAE
  abandonment percentile at Spearman 0.56 without the coverage subtraction
  and 0.45 with it (539 cells).
  Flow: 718 scored cells -> 660 oral non-acute -> 367 bridged to one plant
  (293 unbridged) -> 222 cells / 38 plants with 2018-2022 text. The bridge,
  not text coverage, is the binding constraint.
  No feature survives Holm in any spec; the smallest raw p is 0.23 in the
  main spec. Data integrity +0.9 pct pts per SD (95% CI -2.2 to 4.1, 222
  cells, 38 plants); lab-controls share +1.8 (-1.7 to 5.4); any_oai -1.5
  (-5.3 to 2.3). Same picture without fixed effects, at the vendor's
  300-switch bar (137 cells, 32 plants), and with 2018-2026 text (257 cells,
  44 plants, smallest p 0.07 for quality-system share, negative sign).
  So 08's data-integrity association does not carry over to the vendor's
  score.

Outputs
  outputs/tables/quality_score_flow.csv
  outputs/tables/quality_score_diagnostics.csv
  outputs/tables/quality_score_results.csv
  outputs/tables/quality_score_results.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TABS = HERE / "outputs" / "tables"

_spec = importlib.util.spec_from_file_location("m08", HERE / "08_product_cell_analysis.py")
m08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08)
ms01b, FIXED17, PRODUCT_KEY, ORAL_FORMS = m08.ms01b, m08.FIXED17, m08.PRODUCT_KEY, m08.ORAL_FORMS

KEY = ["manufacturer_labeler_code"] + PRODUCT_KEY
SCORE_FIRST_QI, SCORE_LAST_QI = 2016 * 4 + 3, 2022 * 4 + 2   # vendor window 2016-07 to 2022-06
VENDOR_MIN_SW = 300


def load_scored_cells() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    flow, diag = [], []
    xws = ms01b._load_crosswalks()
    rows = []
    for c in ["CCAE", "MDCR"]:
        d = pd.read_csv(ms01b.MS_DIR / f"manufacturer_product_quarter_{c}.csv",
                        dtype={"plant_fei": str, "manufacturer_labeler_code": str, "ndc9_list": str})
        d["cohort"] = c
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)

    # Score constant within cell, and the same in both cohorts?
    per = d.dropna(subset=["quality_score_v2"]).groupby(KEY + ["cohort"])["quality_score_v2"]
    diag.append({"check": "max distinct score values within a cell x cohort",
                 "value": int(per.nunique().max())})
    w = per.first().unstack("cohort")
    both = w.dropna()
    diag.append({"check": "cells scored in both cohorts", "value": len(both)})
    diag.append({"check": "cells where CCAE and MDCR scores differ",
                 "value": int((~np.isclose(both["CCAE"], both["MDCR"])).sum())})

    score = d.groupby(KEY)["quality_score_v2"].first().dropna().rename("score")
    flow.append({"step": "cells with a quality_score_v2 (either cohort)", "n_cells": len(score),
                 "n_products": score.reset_index()[PRODUCT_KEY].drop_duplicates().shape[0]})

    d = d[~d["ingredient"].str.startswith(ms01b.ACUTE_PREFIXES) & d["form"].isin(ORAL_FORMS)].copy()
    score = score[score.index.isin(d.set_index(KEY).index.unique())]
    flow.append({"step": "oral solids, five acute APIs dropped", "n_cells": len(score),
                 "n_products": score.reset_index()[PRODUCT_KEY].drop_duplicates().shape[0]})

    # Bridge each row to one plant, exactly as 08 does.
    single = d["plant_fei"].notna() & ~d["plant_fei"].fillna("").str.contains(";")
    d["fei"] = d["plant_fei"].where(single)
    for _, mp in xws:
        need = d["fei"].isna()
        hit = d.loc[need, "ndc9_list"].map(lambda l: ms01b._single_fei(l, mp)).dropna()
        d.loc[hit.index, "fei"] = hit

    d["qi"] = d["quarter"].map(ms01b._qidx)
    inwin = d[(d["qi"] >= SCORE_FIRST_QI) & (d["qi"] <= SCORE_LAST_QI)]
    sw = inwin.groupby(KEY + ["cohort"])["n_switches"].sum().unstack("cohort").fillna(0)
    g = d.groupby(KEY).agg(n_fei=("fei", lambda s: s.dropna().nunique()),
                           fei=("fei", lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else None))
    cells = score.to_frame().join(g, how="left").join(sw.add_prefix("sw_"), how="left").reset_index()
    cells[["sw_CCAE", "sw_MDCR"]] = cells[["sw_CCAE", "sw_MDCR"]].fillna(0)
    cells["vendor_bar"] = (cells["sw_CCAE"] >= VENDOR_MIN_SW) & (cells["sw_MDCR"] >= VENDOR_MIN_SW)
    cells["product"] = cells[PRODUCT_KEY].astype(str).agg(" | ".join, axis=1)
    cells["mr"] = cells["form"].isin(m08.MR_FORMS).astype(int)

    flow.append({"step": "bridged to exactly one plant", "n_cells": int(cells["fei"].notna().sum()),
                 "n_plants": int(cells["fei"].nunique()),
                 "note": f"{int((cells['n_fei'] > 1).sum())} cells conflict across rows, "
                         f"{int((cells['n_fei'] == 0).sum())} have no bridge"})
    return cells, flow, diag


def score_scale_diagnostics(cells: pd.DataFrame) -> list[dict]:
    out = []
    m = smf.ols("score ~ C(product)", data=cells).fit()
    out.append({"check": "R2 of score on product fixed effects (all scored oral cells)", "value": m.rsquared,
                "n": len(cells)})
    pm = cells.groupby("product")["score"].mean()
    out.append({"check": "product mean score: min / median / max",
                "value": f"{pm.min():.3f} / {pm.median():.3f} / {pm.max():.3f}", "n": len(pm)})
    r = cells.groupby("product")["score"].rank()
    n = cells.groupby("product")["score"].transform("size")
    rho, _ = spearmanr(cells["score"], (r - 0.5) / n)
    out.append({"check": "Spearman of score with its own within-product percentile", "value": rho, "n": len(cells)})
    out.append({"check": "share of cells whose score equals (rank-0.5)/n within product (+-0.01)",
                "value": float(np.isclose(cells["score"], (r - 0.5) / n, atol=0.01).mean()), "n": len(cells)})
    return out


def numerator_diagnostics(cells: pd.DataFrame) -> list[dict]:
    """Which abandonment definition does the score follow? Compare with 08's
    CCAE within-product abandonment percentile under each numerator, on the
    vendor's claims window."""
    out = []
    for num in ["coverage_adjusted", "coverage_adjusted_clipped", "no_coverage_adjustment"]:
        ab = m08._load_cells(["CCAE"], SCORE_FIRST_QI, SCORE_LAST_QI, numerator=num)
        j = cells.merge(ab[KEY + ["ab_pct"]], on=KEY, how="inner")
        rho, p = spearmanr(j["score"], j["ab_pct"])
        out.append({"check": f"Spearman(score, CCAE abandonment pct, numerator={num})",
                    "value": rho, "p": p, "n": len(j)})
    return out


def fit(f: pd.DataFrame, x: str, fe: bool = True) -> dict:
    f = f.dropna(subset=["y", x]).copy()
    sd = f[x].std()
    if f["fei"].nunique() < 8 or not sd > 0:
        return {"n_cells": len(f), "n_plants": f["fei"].nunique()}
    f["z"] = (f[x] - f[x].mean()) / sd
    m = smf.ols("y ~ z + C(product)" if fe else "y ~ z", data=f).fit(
        cov_type="cluster", cov_kwds={"groups": f["fei"]})
    lo, hi = m.conf_int().loc["z"]
    plant = f.groupby("fei").agg(y=("y", "mean"), x=(x, "first"))
    rho, prho = spearmanr(plant["x"], plant["y"])
    # cells that identify the within-product slope: products with >= 2 linked plants
    ident = f.groupby("product")["fei"].transform("nunique") >= 2
    return {"n_cells": len(f), "n_plants": f["fei"].nunique(), "n_products": f["product"].nunique(),
            "n_cells_identifying": int(ident.sum()) if fe else len(f),
            "beta_pct_pts": 100 * m.params["z"], "ci_lo": 100 * lo, "ci_hi": 100 * hi,
            "p": m.pvalues["z"], "plant_rho": rho, "plant_rho_p": prho}


def run(cells: pd.DataFrame, expo: pd.DataFrame, spec: str, fe: bool = True) -> pd.DataFrame:
    f = cells[cells["fei"].isin(expo.index)].merge(expo, left_on="fei", right_index=True)
    f = f.assign(y=f["score"])
    t = pd.DataFrame([{"spec": spec, "feature": x, **fit(f, x, fe)} for x in FIXED17 + ["any_oai"]])
    k = (t["feature"] != "any_oai") & t["p"].notna()
    t["p_holm17"] = np.nan
    t.loc[k, "p_holm17"] = m08._holm(t.loc[k, "p"])
    return t


def main() -> None:
    TABS.mkdir(parents=True, exist_ok=True)
    cells, flow, diag = load_scored_cells()
    diag += score_scale_diagnostics(cells)
    diag += numerator_diagnostics(cells)

    results = []
    for window, (ymin, ymax) in [("text 2018-2022", (2018, 2022)), ("text 2018-2026", (2018, 2026))]:
        expo = m08._expo17(ymin, ymax)
        linked = cells[cells["fei"].isin(expo.index)]
        multi = linked.groupby("product")["fei"].transform("nunique") >= 2
        flow.append({"step": f"plant has 483 text ({window})", "n_cells": len(linked),
                     "n_plants": linked["fei"].nunique(), "n_products": linked["product"].nunique(),
                     "note": f"{int(multi.sum())} cells in {linked.loc[multi, 'product'].nunique()} products "
                             "with >= 2 linked plants identify the fixed-effects slope"})
        results.append(run(cells, expo, f"{window}, product FE"))
        results.append(run(cells, expo, f"{window}, no FE", fe=False))
        if ymax == 2022:
            vb = cells[cells["vendor_bar"]]
            flow.append({"step": f"... and >= {VENDOR_MIN_SW} switches in both cohorts (vendor bar)",
                         "n_cells": int(vb["fei"].isin(expo.index).sum()),
                         "n_plants": vb.loc[vb["fei"].isin(expo.index), "fei"].nunique()})
            results.append(run(vb, expo, f"{window}, product FE, vendor 300-switch bar"))

    res = pd.concat(results, ignore_index=True)
    pd.DataFrame(flow).to_csv(TABS / "quality_score_flow.csv", index=False)
    pd.DataFrame(diag).to_csv(TABS / "quality_score_diagnostics.csv", index=False)
    res.to_csv(TABS / "quality_score_results.csv", index=False)

    cols = ["feature", "n_cells", "n_plants", "n_cells_identifying", "beta_pct_pts", "ci_lo", "ci_hi",
            "p", "p_holm17", "plant_rho", "plant_rho_p"]
    lines = ["# 483 text features vs MarketScan quality_score_v2", "",
             "Source: `11_quality_score_analysis.py`. Outcome: the vendor's quality score (higher = worse), "
             "labeler x exact product, oral solids, five acute APIs dropped. beta = score percentile points "
             "per 1 SD of the feature, SEs clustered by plant. Holm over the 17 features within each spec. "
             "any_oai is a comparator and is not in the Holm family.", "",
             "## Cell flow", "", "```", pd.DataFrame(flow).to_string(index=False), "```", "",
             "## Diagnostics", "", "```", pd.DataFrame(diag).to_string(index=False), "```", ""]
    for spec, t in res.groupby("spec", sort=False):
        lines += [f"## {spec}", "", "```", t[[c for c in cols if c in t]].to_string(index=False, float_format="%.3f"),
                  "```", ""]
    (TABS / "quality_score_results.md").write_text("\n".join(lines))

    print(pd.DataFrame(flow).to_string(index=False))
    print(pd.DataFrame(diag).to_string(index=False))
    for spec, t in res.groupby("spec", sort=False):
        print(f"\n== {spec}")
        print(t[[c for c in cols if c in t]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
