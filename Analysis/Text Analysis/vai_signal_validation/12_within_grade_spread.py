"""
12_within_grade_spread.py
─────────────────────────────────────────────────────────────────────────────
How much of the variation in the 483 text does FDA's three-level grade
capture? (2026-10-19)

No outcome is involved. For the drug-quality inspections with a known grade
(NAI / VAI / OAI, from step00_redica_inspections.csv), each of the 17 text
signals is summarized by grade, and we ask:

  1. share of the signal's variation explained by the grade
     (eta squared from a one-way ANOVA: between-grade / total sum of squares)
  2. Kruskal-Wallis p for "the three grades differ at all"
  3. overlap: share of VAI inspections at or above the OAI median
     (if the grade sorted inspections by this signal, this would be near 0)

Outputs: outputs/tables/within_grade_spread.csv
"""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TEXT = HERE.parent
STEP02 = TEXT / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
INSP = TEXT / "step00_redica_inspections.csv"
OUT = HERE / "outputs" / "tables" / "within_grade_spread.csv"

spec = importlib.util.spec_from_file_location("m02", HERE / "02_vai_signal_model.py")
m02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m02)
FEATURES = list(m02.TEXT_FEATURES)


def main() -> None:
    # one row per inspection: text features + grade
    ts = pd.read_csv(STEP02, parse_dates=["snapshot_date"])
    insp = pd.read_csv(INSP, parse_dates=["insp_date"])
    insp = insp[insp["keep"] & insp["dqa_grade"].isin(["NAI", "VAI", "OAI"])]
    d = ts.merge(insp[["fei", "insp_date", "dqa_grade"]],
                 left_on=["fei", "snapshot_date"], right_on=["fei", "insp_date"])
    print(f"inspections with a known grade and text: {len(d)}, "
          f"{d['fei'].nunique()} factories  {d['dqa_grade'].value_counts().to_dict()}")

    rows = []
    for f in FEATURES:
        if f not in d.columns:
            print(f"  [skip] {f} not in step02")
            continue
        x = d[[f, "dqa_grade"]].dropna()
        groups = [x.loc[x["dqa_grade"] == g, f] for g in ("NAI", "VAI", "OAI")]
        total_ss = ((x[f] - x[f].mean()) ** 2).sum()
        between_ss = sum(len(g) * (g.mean() - x[f].mean()) ** 2 for g in groups)
        eta2 = between_ss / total_ss if total_ss > 0 else np.nan
        kw_p = stats.kruskal(*groups).pvalue if x[f].nunique() > 1 else np.nan
        vai, oai = groups[1], groups[2]
        rows.append({
            "feature": f,
            "n": len(x),
            "mean_NAI": groups[0].mean(), "mean_VAI": vai.mean(), "mean_OAI": oai.mean(),
            "sd_VAI": vai.std(),
            "eta2_grade": eta2,
            "kruskal_p": kw_p,
            "share_VAI_at_or_above_OAI_median": (vai >= oai.median()).mean(),
        })
    t = pd.DataFrame(rows).sort_values("eta2_grade", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    print(f"\nmedian share of variation explained by grade: {t['eta2_grade'].median():.3f}; "
          f"max {t['eta2_grade'].max():.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
