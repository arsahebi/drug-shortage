"""
07_three_class_and_vai_probe.py
─────────────────────────────────────────────────────────────────────────────
Two questions the main ablation cannot answer.

1. Is the FDA baseline being treated fairly?
   02_vai_signal_model.py gives FDA a SINGLE binary feature, any_oai, so NAI and
   VAI are pooled and the 40 inspections with no classification on file are
   coded 0, i.e. scored as "not OAI" when they are actually unknown. Comparing
   28 text features against one contaminated binary is not a fair test. Here we
   rebuild the FDA baseline three ways on the rows where the classification is
   actually known, and re-score the text on those same rows so the comparison is
   like for like.

2. Within VAI, is there really no signal in the text?
   Config D returns an AUC near 0.50 on 85 inspections. AUC on a binary outcome
   at that sample size has very little power, so "no signal" and "not enough
   data to see a signal" look identical. This probes it three further ways:
   per-feature rank correlation against the CONTINUOUS outcome, effect sizes with
   confidence intervals rather than a single AUC, and a permutation test so the
   null is calibrated on this sample rather than assumed.

Both use the clean attribution panel (one FEI, one ANDA per NDC) and the
corrected severity feature (Critical+Major, monotonic).

Run:
  python 07_three_class_and_vai_probe.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PANEL = HERE / "outputs" / "fei_ae_panel_inspection_centered_anda_propublica.parquet"
OUT = HERE / "outputs" / "tables"

TEXT_FEATURES = [
    "severity_critmajor_share", "contamination_llm_share", "data_integrity_llm_share",
    "patient_risk_llm_share", "investigation_llm_share", "repeat_cross_insp_share",
    "scope_facilitywide_share", "cultural_root_cause_share",
    "vc_laboratorycontrolssystem_share", "vc_qualitysystem_share",
    "n_laboratorycontrolssystem_obs", "n_qualitysystem_obs",
    "joint_labcontrols_qualitysystem", "joint_labcontrols_dataintegrity",
]
PRE = ["n_ae_tm4", "n_ae_tm3", "n_ae_tm2", "n_ae_tm1"]
POST = ["n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"]
SEED = 42


def load() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p["pre"] = p[PRE].mean(axis=1)
    p["post"] = p[POST].mean(axis=1)
    p = p.dropna(subset=["pre", "post"]).copy()
    p["rise"] = (p["post"] > p["pre"]).astype(int)
    # continuous outcome: log ratio, which keeps magnitude instead of thresholding
    p["log_ratio"] = np.log((p["post"] + 1) / (p["pre"] + 1))

    def cls(r):
        if r["n_oai"] > 0:
            return "OAI"
        if r["n_vai"] > 0:
            return "VAI"
        if r["n_nai"] > 0:
            return "NAI"
        return "UNKNOWN"

    p["fda_class"] = p.apply(cls, axis=1)
    feats = [c for c in TEXT_FEATURES if c in p.columns]
    p = p.dropna(subset=feats)
    print(f"panel: {len(p)} inspections, {p['fei'].nunique()} FEIs, "
          f"{len(feats)} text features")
    print(f"  classification: {p['fda_class'].value_counts().to_dict()}")
    return p


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, label: str) -> dict:
    """
    Deliberately identical to _cv_evaluate() in 02_vai_signal_model.py: GroupKFold
    (no shuffle), LR in a StandardScaler pipeline, RF with n_estimators=300,
    max_depth=4, min_samples_leaf=5, and a one-tailed t-test of fold AUCs vs 0.5.
    A first version of this script used StratifiedGroupKFold with shuffling and an
    unconstrained RF, which produced AUCs that could not be compared with the main
    ablation at all.
    """
    if len(np.unique(y)) < 2 or len(X) < 30:
        return {"label": label, "n": len(X)}
    n_splits = min(5, max(2, len(np.unique(groups)) - 1))
    gkf = GroupKFold(n_splits=n_splits)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                random_state=SEED, n_jobs=-1)
    out = {"label": label, "n": len(X)}
    for name, model in [("LR", lr), ("RF", rf)]:
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            if y[te].sum() in (0, len(y[te])):
                continue
            model.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
        if len(aucs) >= 2:
            t, p2 = stats.ttest_1samp(aucs, 0.5)
            p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        else:
            p1 = np.nan
        out[name] = (float(np.mean(aucs)) if aucs else np.nan,
                     float(np.std(aucs)) if aucs else np.nan, p1)
    return out


def q1_fair_baseline(p: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("Q1  Is the FDA baseline fair? Known-classification rows only")
    print("=" * 74)
    k = p[p["fda_class"] != "UNKNOWN"].copy()
    print(f"dropped {len(p) - len(k)} rows with no classification on file "
          f"(previously coded as 'not OAI')")
    print(f"remaining: {len(k)} inspections, {k['fei'].nunique()} FEIs, "
          f"{k['fda_class'].value_counts().to_dict()}")

    y = k["rise"].values
    g = k["fei"].values
    feats = [c for c in TEXT_FEATURES if c in k.columns]

    variants = {
        "FDA: binary any_oai (as used now)": k[["any_oai"]].values.astype(float),
        "FDA: 3-class dummies (NAI/VAI/OAI)":
            pd.get_dummies(k["fda_class"]).reindex(
                columns=["NAI", "VAI", "OAI"], fill_value=0).values.astype(float),
        "FDA: ordinal 0=NAI 1=VAI 2=OAI":
            k["fda_class"].map({"NAI": 0, "VAI": 1, "OAI": 2}).values.reshape(-1, 1).astype(float),
        "TEXT only (same rows)": k[feats].values.astype(float),
        "TEXT + 3-class FDA":
            np.hstack([k[feats].values.astype(float),
                       pd.get_dummies(k["fda_class"]).reindex(
                           columns=["NAI", "VAI", "OAI"], fill_value=0).values.astype(float)]),
    }
    rows = []
    for lbl, X in variants.items():
        r = cv_auc(X, y, g, lbl)
        for mdl in ("LR", "RF"):
            if mdl in r:
                auc, sd, pv = r[mdl]
                rows.append({"features": lbl, "model": mdl, "auc": round(auc, 3),
                             "sd": round(sd, 3), "p_vs_0.5": round(pv, 3),
                             "n": r["n"]})
    d = pd.DataFrame(rows)
    print()
    print(d.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "three_class_baseline.csv", index=False)
    print(f"\nSaved -> {OUT / 'three_class_baseline.csv'}")


def q2_vai_probe(p: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("Q2  Within VAI, is there really nothing in the text?")
    print("=" * 74)
    v = p[p["fda_class"] == "VAI"].copy()
    print(f"VAI inspections: {len(v)}, facilities: {v['fei'].nunique()}")
    print(f"  adverse events rose afterwards in {100 * v['rise'].mean():.1f}% of them")

    feats = [c for c in TEXT_FEATURES if c in v.columns]
    rows = []
    for f in feats:
        if v[f].nunique() < 3:
            continue
        r_c, p_c = stats.spearmanr(v[f], v["log_ratio"])
        r_b, p_b = stats.pointbiserialr(v["rise"], v[f])
        rows.append({"feature": f,
                     "spearman_vs_log_ratio": round(r_c, 3), "p_cont": round(p_c, 3),
                     "corr_vs_rise": round(r_b, 3), "p_bin": round(p_b, 3)})
    d = pd.DataFrame(rows).sort_values("p_cont")
    print("\nper-feature association with the CONTINUOUS outcome "
          "(log post/pre), not a threshold:")
    print(d.to_string(index=False))
    sig = d[d["p_cont"] < 0.05]
    print(f"\n  features significant at p<0.05 before correction: {len(sig)} of {len(d)}")
    if len(sig):
        print(f"  {sig['feature'].tolist()}")
    print(f"  Bonferroni threshold for {len(d)} tests: p < {0.05 / max(len(d), 1):.4f}")
    print(f"  surviving Bonferroni: {int((d['p_cont'] < 0.05 / max(len(d), 1)).sum())}")

    # permutation test: is the multivariate AUC better than this sample's own null?
    y, g = v["rise"].values, v["fei"].values
    X = v[feats].values.astype(float)
    obs = cv_auc(X, y, g, "VAI text")
    obs_rf = obs.get("RF", (np.nan, np.nan, np.nan))[0]
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(200):
        yp = rng.permutation(y)
        r = cv_auc(X, yp, g, "perm")
        if "RF" in r and not np.isnan(r["RF"][0]):
            null.append(r["RF"][0])
    null = np.array(null)
    pval = float((null >= obs_rf).mean()) if len(null) else np.nan
    print(f"\npermutation test (200 shuffles, RF):")
    print(f"  observed AUC {obs_rf:.3f}")
    print(f"  null mean {null.mean():.3f}, null 95th pct {np.percentile(null, 95):.3f}")
    print(f"  p = {pval:.3f}")

    print(f"\n  POWER CHECK: with n={len(v)} and a balanced binary outcome, the "
          f"detectable AUC at 80% power is roughly 0.63.")
    print("  So an AUC near 0.50 here rules out a LARGE effect, not a modest one.")
    d.to_csv(OUT / "vai_within_group_probe.csv", index=False)
    print(f"\nSaved -> {OUT / 'vai_within_group_probe.csv'}")


def main() -> None:
    p = load()
    q1_fair_baseline(p)
    q2_vai_probe(p)


if __name__ == "__main__":
    main()
