"""
09_repeated_cv_robustness.py
────────────────────────────────────────────────────────────────────────────
Fold-split robustness for the inspection-level AUC comparisons in 02 and 07.

Why (found 2026-10-06)
  02 and 07 use one GroupKFold split. The split depends on the scikit-learn
  version: with identical data and code, sklearn 1.9.0 (project .venv)
  reproduces the saved tables (07 text-only LR 0.592, p 0.007; 02 all-systems
  RF 0.648, p 0.032; product-systems RF 0.639, p 0.009), while sklearn 1.6.1
  gives 0.540 (p 0.23), 0.647 (p 0.005) and 0.583 (p 0.17). A single split is
  therefore not a stable summary at this sample size.

What this does
  For each comparison, R random 5-fold partitions of the FACILITIES (FEIs
  shuffled, then dealt round-robin so each fold holds the same number of
  facilities; all inspections of a facility stay together), the same LR and
  RF as 02/07, mean fold AUC per partition. Text and the FDA baseline are
  scored on the SAME partitions, so the paired share "text beats FDA" is a
  like-for-like comparison. The partition generator is numpy-seeded and does
  not depend on the sklearn version.

Comparisons
  F1  FAERS, 123 inspections with a known FDA class (07's sample and outcome)
      text (17 features) vs FDA 3-class dummies vs OAI flag
  F2  FAERS, clean ANDA panel, all systems (02 --anda-ae --anda-source
      propublica, relative outcome)            text vs OAI flag
  F3  same, product-system text (--systems product)  text vs OAI flag
  M1  MarketScan aband_excess, CCAE, >=300 switches each side (02 --outcome
      aband_excess on the all-systems panel)   text vs OAI flag
  M2  MarketScan er_rise, same rules             text vs OAI flag

Outputs
  outputs/tables/repeated_cv_summary.csv / .md
  outputs/models/repeated_cv_per_partition.csv
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
R = 100
K = 5
SEED = 42


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m02 = _load("m02", "02_vai_signal_model.py")
m07 = _load("m07", "07_three_class_and_vai_probe.py")
TEXT = list(m02.TEXT_FEATURES)


def _models():
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                random_state=SEED, n_jobs=1)
    return {"LR": lr, "RF": rf}


def _partition(groups: np.ndarray, rep: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + rep)
    uniq = rng.permutation(np.unique(groups))
    fold_of = {g: i % K for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def _one_rep(rep: int, Xs: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray) -> list[dict]:
    folds = _partition(groups, rep)
    out = []
    for fs, X in Xs.items():
        for mname, model in _models().items():
            aucs = []
            for k in range(K):
                te = folds == k
                if y[te].sum() in (0, te.sum()) or y[~te].sum() in (0, (~te).sum()):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X[~te], y[~te])
                    aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
            out.append({"rep": rep, "features": fs, "model": mname,
                        "auc": float(np.mean(aucs)) if aucs else np.nan, "n_folds": len(aucs)})
    return out


def _comparisons() -> dict[str, tuple[pd.DataFrame, str, dict[str, list[str] | str]]]:
    comps = {}

    p = m07.load()
    k = p[p["fda_class"] != "UNKNOWN"].copy()
    for c in ["NAI", "VAI", "OAI"]:
        k[f"cls_{c}"] = (k["fda_class"] == c).astype(float)
    comps["F1 FAERS, known FDA class (07)"] = (
        k, "rise", {"text": TEXT, "FDA 3-class": ["cls_NAI", "cls_VAI", "cls_OAI"], "OAI flag": ["any_oai"]})

    for label, path in [("F2 FAERS, all systems (02)", m02.PANEL_PP),
                        ("F3 FAERS, product systems (02)",
                         m02.PANEL_PP.with_name(m02.PANEL_PP.stem + "_prodsys" + m02.PANEL_PP.suffix))]:
        df, col = m02._build_outcome_relative(pd.read_parquet(path))
        df = df[df[TEXT].notna().all(axis=1)]
        comps[label] = (df, col, {"text": TEXT, "OAI flag": ["any_oai"]})

    for label, outcome in [("M1 MarketScan aband_excess, CCAE (02)", "aband_excess"),
                           ("M2 MarketScan er_rise, CCAE (02)", "er_rise")]:
        df, col = m02._build_outcome_marketscan(pd.read_parquet(m02.PANEL_PP), outcome, "CCAE", 300)
        df = df[df[TEXT].notna().all(axis=1)]
        comps[label] = (df, col, {"text": TEXT, "OAI flag": ["any_oai"]})
    return comps


def main() -> None:
    print(f"scikit-learn {sklearn.__version__}; {R} random facility partitions x {K} folds")
    per, summ = [], []
    for label, (df, ycol, fsets) in _comparisons().items():
        y = df[ycol].values.astype(int)
        groups = df["fei"].astype("int64").values
        Xs = {fs: df[cols].values.astype(float) for fs, cols in fsets.items()}
        print(f"\n{label}: n={len(df)}, FEIs={len(np.unique(groups))}, base rate {y.mean():.1%}")
        res = Parallel(n_jobs=-1)(delayed(_one_rep)(r, Xs, y, groups) for r in range(R))
        r = pd.DataFrame([row for rep in res for row in rep]).assign(comparison=label)
        per.append(r)
        piv = r.pivot_table(index=["rep", "model"], columns="features", values="auc")
        for (fs, mname), g in r.groupby(["features", "model"]):
            a = g["auc"].dropna()
            row = {"comparison": label, "n": len(df), "n_feis": len(np.unique(groups)),
                   "base_rate": y.mean(), "features": fs, "model": mname,
                   "auc_median": a.median(), "auc_p2.5": a.quantile(0.025),
                   "auc_p97.5": a.quantile(0.975), "share_partitions_above_0.5": (a > 0.5).mean()}
            if fs == "text":
                for base in [b for b in fsets if b != "text"]:
                    d = piv.xs(mname, level="model")
                    row[f"share_text_beats_{base}"] = (d["text"] > d[base]).mean()
            summ.append(row)
            print(f"  {fs:12s} {mname}: median AUC {a.median():.3f} "
                  f"[{a.quantile(0.025):.3f}, {a.quantile(0.975):.3f}], >0.5 in {(a > 0.5).mean():.0%}")

    summ = pd.DataFrame(summ)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "tables" / "repeated_cv_summary.csv", index=False)
    pd.concat(per).to_csv(OUT / "models" / "repeated_cv_per_partition.csv", index=False)
    md = ["# Repeated-partition CV robustness", "",
          f"scikit-learn {sklearn.__version__}. {R} random 5-fold partitions of facilities; "
          "median and 2.5-97.5 percentile of the mean fold AUC across partitions. "
          "share_text_beats_* = share of partitions where text scored higher than that baseline "
          "on the identical split.", "",
          summ.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    (OUT / "tables" / "repeated_cv_summary.md").write_text("\n".join(md))
    print(f"\nSaved -> {OUT / 'tables' / 'repeated_cv_summary.csv'}")


if __name__ == "__main__":
    main()
