# %%
"""
Module 9 — Exploratory prediction benchmark.

The panel is now Valisure-only: 14 APIs over 2015–2024. Because the usable
modeling sample is small (~126 drug-years, ~19 next-year shortage events), this
module is intentionally modest. It fits cross-validated benchmark models only;
results should be read as exploratory ranking/discrimination checks, not as a
stable predictive model.

Target:
    y_next_year_shortage = shortage_started at year t+1

Outputs:
    outputs/models/metrics_valisure.csv
    outputs/models/coefs_l2_valisure.csv
    outputs/models/rf_importance_valisure.csv
    outputs/figures/roc_valisure.png
    outputs/figures/feature_importance_valisure.png
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    LogisticRegression = None
    RandomForestClassifier = None
    roc_auc_score = None
    average_precision_score = None
    roc_curve = None
    brier_score_loss = None
    GroupKFold = None
    StandardScaler = None

from config import OUT_DATA, OUT_FIGS, OUT_MODELS, OUT_LOGS, SEED
from utils import get_logger, read_table

log = get_logger("m09_model", OUT_LOGS / "m09_model.log")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Features: one or two signals per source.
# Recalls excluded — concurrent/lagging w.r.t. shortage onset, not leading indicators.
FEATURES = [
    # FAERS adverse-event signals
    "faers_severity_score",
    "faers_severity_score_w3",
    # Redica inspection / regulatory signals
    "redica_n_oai",
    "redica_n_oai_w3",
    # Valisure independent quality test scores (time-invariant cross-section)
    "valisure_mean_score",
    "valisure_min_score",
    "valisure_n_failing",
    # Drug structural attributes
    "parenteral_ever",
    # Shortage history
    "prior_shortage_t",
    "prior_shortage_w3",
    # 483 text features — time-aware LLM/regex shares (most-recent snapshot per FEI,
    # drug-level mean; m12 grid identifies which features carry forward signal)
    "severity_critmajor_share",       # Critical+Major severity (4-tier recalibrated)
    "scope_facilitywide_share",       # facility-wide vs isolated failure
    "scope_multipleproducts_share",
    "cultural_root_cause_share",      # management/training root cause
    "capital_root_cause_share",       # equipment/facility root cause
    "remediation_none_share",         # no remediation → longer shortage duration
    "remediation_weak_share",
    "repeat_llm_share",               # LLM: any repeat violation finding
    "contamination_llm_share",
    "data_integrity_llm_share",
    "investigation_llm_share",
    "repeat_llm_only_share",          # semantic lift: LLM flagged, regex missed
    "contamination_llm_only_share",
    "oos_oot_regex_share",
    "wl_ref_regex_share",
    "repeat_cross_insp_share",        # algorithmic: same deficiency cited across inspections
    "vc_labcontrols_share",
    "vc_buildingsequipment_share",
]

TEXT_FEATURE_COLS = [
    "severity_critmajor_share", "scope_facilitywide_share", "scope_multipleproducts_share",
    "cultural_root_cause_share", "capital_root_cause_share",
    "remediation_none_share", "remediation_weak_share",
    "repeat_llm_share", "contamination_llm_share", "data_integrity_llm_share", "investigation_llm_share",
    "repeat_llm_only_share", "contamination_llm_only_share",
    "oos_oot_regex_share", "wl_ref_regex_share",
    "repeat_cross_insp_share", "vc_labcontrols_share", "vc_buildingsequipment_share",
]
# Ablation baseline: same feature set minus the LLM-derived text features
FEATURES_NO_TEXT = [f for f in FEATURES if f not in TEXT_FEATURE_COLS]


def _prep(panel: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = panel.dropna(subset=["y_next_year_shortage"]).copy()
    feats_in_panel = [f for f in features if f in df.columns]
    missing = set(features) - set(feats_in_panel)
    if missing:
        log.warning("Features missing from panel (dropped): %s", missing)
    X = df[feats_in_panel].fillna(0).astype(float)
    y = df["y_next_year_shortage"].astype(int)
    groups = df["drug_norm"]
    return X, y, groups


def cv_metrics(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_factory, n_splits: int = 5):
    """Grouped K-fold CV by drug, so the same drug never appears in train+test."""
    n_groups = groups.nunique()
    n_splits = min(n_splits, max(2, n_groups - 1))
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            preds[te] = y.iloc[tr].mean()
            continue
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, preds) if y.sum() > 0 else float("nan")
    ap  = average_precision_score(y, preds) if y.sum() > 0 else float("nan")
    bs  = brier_score_loss(y, preds)
    return preds, dict(auc=auc, average_precision=ap, brier=bs, n=len(y), events=int(y.sum()))


def run_valisure(panel: pd.DataFrame):
    scope = "valisure"
    log.info("=== Valisure-only exploratory benchmark ===")
    if LogisticRegression is None:
        msg = "scikit-learn is not installed; skipping m09 exploratory models"
        log.warning(msg)
        pd.DataFrame([{"scope": scope, "status": "skipped", "reason": msg}]).to_csv(
            OUT_MODELS / "metrics_valisure.csv", index=False
        )
        return

    X, y, groups = _prep(panel, FEATURES)
    if y.sum() < 5 or len(X) < 20:
        log.warning("Insufficient events (n=%d, events=%d); skipping", len(X), int(y.sum()))
        return

    log.info("Modeling rows=%d | events=%d | drugs=%d | features=%d",
             len(X), int(y.sum()), groups.nunique(), X.shape[1])

    # CV: L2 logit (full features)
    Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    preds_l2, met_l2 = cv_metrics(Xz, y, groups,
        lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced", random_state=SEED))
    log.info("L2 Logit CV: AUC=%.3f AP=%.3f Brier=%.3f", met_l2["auc"], met_l2["average_precision"], met_l2["brier"])

    l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced", random_state=SEED)
    l2.fit(Xz, y)
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef_standardized": l2.coef_[0],
        "odds_ratio_per_sd": np.exp(l2.coef_[0]),
    }).sort_values("coef_standardized", key=lambda s: s.abs(), ascending=False)
    coefs.to_csv(OUT_MODELS / "coefs_l2_valisure.csv", index=False)

    # CV: RandomForest (full features)
    preds_rf, met_rf = cv_metrics(X, y, groups,
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1))
    log.info("RandomForest CV: AUC=%.3f AP=%.3f Brier=%.3f", met_rf["auc"], met_rf["average_precision"], met_rf["brier"])

    # ---- Ablation: without 483 text indices (TRI / SCRI / IRWI / QCI) ----
    text_feats_present = [f for f in TEXT_FEATURE_COLS if f in X.columns]
    if text_feats_present:
        X_no, y_no, grp_no = _prep(panel, FEATURES_NO_TEXT)
        Xz_no = pd.DataFrame(StandardScaler().fit_transform(X_no), columns=X_no.columns)
        _, met_l2_no = cv_metrics(Xz_no, y_no, grp_no,
            lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced", random_state=SEED))
        _, met_rf_no = cv_metrics(X_no, y_no, grp_no,
            lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1))
        delta_l2 = met_l2["auc"] - met_l2_no["auc"]
        delta_rf = met_rf["auc"] - met_rf_no["auc"]
        log.info("AUC delta (with vs without 483 text indices) — L2 Logit: %+.3f  RF: %+.3f",
                 delta_l2, delta_rf)
        ablation = pd.DataFrame([
            {"model": "L2_logit",
             "auc_with_text": met_l2["auc"], "auc_without_text": met_l2_no["auc"], "auc_delta": delta_l2},
            {"model": "RandomForest",
             "auc_with_text": met_rf["auc"], "auc_without_text": met_rf_no["auc"], "auc_delta": delta_rf},
        ])
        ablation.to_csv(OUT_MODELS / "text_features_ablation.csv", index=False)
        log.info("Ablation saved to text_features_ablation.csv")

    pd.DataFrame([
        {"scope":scope, "model":"L2_logit",     **met_l2},
        {"scope":scope, "model":"RandomForest", **met_rf},
    ]).to_csv(OUT_MODELS / "metrics_valisure.csv", index=False)

    # ROC plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for preds, name in [(preds_l2, "L2 Logit"), (preds_rf, "RandomForest")]:
        if y.sum() > 0:
            fpr, tpr, _ = roc_curve(y, preds)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, preds):.3f})")
    ax.plot([0,1],[0,1],"--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — predict shortage at year t+1 ({scope})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUT_FIGS / f"roc_{scope}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # RF feature importance
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    fi = fi.sort_values("importance", ascending=True)
    fi.to_csv(OUT_MODELS / f"rf_importance_{scope}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(fi))))
    ax.barh(fi["feature"], fi["importance"])
    ax.set_xlabel("RandomForest importance")
    ax.set_title(f"Feature importance ({scope})")
    fig.savefig(OUT_FIGS / f"feature_importance_{scope}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    panel = read_table(OUT_DATA / "master_panel.parquet")
    run_valisure(panel)


if __name__ == "__main__":
    main()

# %%
