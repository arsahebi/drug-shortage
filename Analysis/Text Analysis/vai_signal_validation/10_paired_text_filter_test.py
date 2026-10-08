"""
10_paired_text_filter_test.py
────────────────────────────────────────────────────────────────────────────
Does restricting 483 text to product-proximate observations help, holding the
sample fixed?

Why this script exists
  02 --systems product scores the product-system features on 119 inspections,
  while the all-systems run uses 143: an inspection with no product-system
  observation drops out entirely. Comparing 0.63 against 0.66 across those two
  runs confounds the filter with the sample. This restricts both feature sets
  to the 119 inspections present in BOTH panels and scores them on identical
  partitions, so the only thing that changes is which observations feed the
  features.

Method
  Same repeated-partition CV as 09 (100 random 5-fold partitions of
  facilities, same LR and RF, FAERS relative outcome ae_rise_next4q).

Result (first run 2026-10-08; 119 inspections, 62 FEIs, base rate 55.5%)
  LR: all systems 0.489, product systems 0.568, filter wins on 96% of
      partitions, median gap +0.081
  RF: all systems 0.625, product systems 0.629, filter wins on 57% of
      partitions, median gap +0.005
  So the filter substitutes for model capacity: it buys the linear model most
  of what the forest already finds on its own. The unpaired comparison in 09
  (all systems 0.557 on 143 rows) understates the filter's benefit to LR
  because the extra 24 inspections, not the extra text, carry that difference.

Output
  outputs/tables/paired_text_filter.csv
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
R = 100


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m02 = _load("m02", "02_vai_signal_model.py")
m09 = _load("m09", "09_repeated_cv_robustness.py")
TEXT = list(m02.TEXT_FEATURES)


def _panel(path: Path) -> pd.DataFrame:
    df, _ = m02._build_outcome_relative(pd.read_parquet(path))
    df = df[df[TEXT].notna().all(axis=1)].copy()
    df["k"] = list(zip(df["fei"].astype("int64"),
                       pd.to_datetime(df["insp_date"]).dt.normalize()))
    return df


def main() -> None:
    all_sys = _panel(m02.PANEL_PP)
    prod_sys = _panel(m02.PANEL_PP.with_name(m02.PANEL_PP.stem + "_prodsys" + m02.PANEL_PP.suffix))
    common = sorted(set(all_sys["k"]) & set(prod_sys["k"]))
    print(f"all-systems panel {len(all_sys)}, product-systems panel {len(prod_sys)}, "
          f"common {len(common)} inspections")

    A = all_sys[all_sys["k"].isin(common)].set_index("k").loc[common].reset_index()
    B = prod_sys[prod_sys["k"].isin(common)].set_index("k").loc[common].reset_index()
    assert (A["fei"].values == B["fei"].values).all(), "panels misaligned"

    y = A["ae_rise_next4q"].values.astype(int)
    groups = A["fei"].astype("int64").values
    print(f"paired sample: {len(A)} inspections, {len(np.unique(groups))} FEIs, "
          f"base rate {y.mean():.1%}")

    Xs = {"all systems": A[TEXT].values.astype(float),
          "product systems": B[TEXT].values.astype(float)}
    res = Parallel(n_jobs=-1)(delayed(m09._one_rep)(r, Xs, y, groups) for r in range(R))
    per = pd.DataFrame([row for rep in res for row in rep])
    wide = per.pivot_table(index=["rep", "model"], columns="features", values="auc")

    rows = []
    for mdl in ["LR", "RF"]:
        x = wide.xs(mdl, level="model")
        gap = x["product systems"] - x["all systems"]
        rows.append({"model": mdl, "n": len(A), "n_feis": len(np.unique(groups)),
                     "auc_all_systems": x["all systems"].median(),
                     "auc_product_systems": x["product systems"].median(),
                     "share_filter_wins": (gap > 0).mean(),
                     "median_gap": gap.median(),
                     "gap_p2.5": gap.quantile(0.025), "gap_p97.5": gap.quantile(0.975)})
        print(f"  {mdl}: all {rows[-1]['auc_all_systems']:.3f}  "
              f"product {rows[-1]['auc_product_systems']:.3f}  "
              f"filter wins on {rows[-1]['share_filter_wins']:.0%} of partitions  "
              f"median gap {rows[-1]['median_gap']:+.3f}")

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "paired_text_filter.csv", index=False)
    print(f"\nSaved -> {OUT / 'tables' / 'paired_text_filter.csv'}")


if __name__ == "__main__":
    main()
