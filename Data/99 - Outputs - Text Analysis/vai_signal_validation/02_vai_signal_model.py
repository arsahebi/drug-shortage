"""
02_vai_signal_model.py
────────────────────────────────────────────────────────────────────────────
Rebuild of the INFORMS 2026 "Do 483 Text Signals Carry Information? Evidence
From the VAI-Only Subgroup" analysis (Paper/Text Analysis/20260714_informs_slides.tex),
using the CURRENT validated 483-text pipeline (see 01_build_inspection_panel.py
for exactly what changed vs. the archived version).

Unit of observation: one inspection event.
Features: 17 LLM text signals from the 483 observations at that inspection.

Outcome (updated 2026-09-16, --outcome relative is now the default): binary
-- did this facility's own AE count rise after the inspection, compared to
its own AE count before it? A variance decomposition found the original
outcome (--outcome global: above-median AEs pooled across ALL facilities
and drugs) is confounded: 88.6% of the variance in raw AE count across
inspection-events is BETWEEN facilities (which facility/drug this is), only
11.4% is WITHIN a facility over time. A global median split mostly labels
"is this a high-volume facility," which FEI-grouped CV correctly refuses to
let a model exploit, leaving little for any feature to explain. The
relative outcome (ae_rise_next4q = post-inspection AE count > pre-
inspection AE count, same facility, same window length) cancels the
facility/drug-size confound out entirely. Both are still implemented,
see _build_outcome_global() and _build_outcome_relative(), for comparison.

Five configurations, matching the INFORMS slide table exactly:
  A. Text only, full sample
  B. Text + inspection outcome (OAI) flag
  C. OAI flag only
  D. Text only, VAI-only facilities (never received OAI in this panel) --
     the flagship INFORMS claim (0.656, p=0.046 in the original slide)
  E. Text only, OAI-ever facilities (for contrast)

Models: Logistic Regression (L2) and Random Forest.
Evaluation: FEI-grouped 5-fold CV.

Outputs
───────
  outputs/models/ablation_metrics.csv
  outputs/tables/model_summary.md
  outputs/figures/ablation_auc_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats

HERE      = Path(__file__).resolve().parent
OUT       = HERE / "outputs"
OUT_TABS  = OUT / "tables"
OUT_FIGS  = OUT / "figures"
OUT_MOD   = OUT / "models"
PANEL      = OUT / "fei_ae_panel_inspection_centered.parquet"
PANEL_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

# v2 schema: same 2 renames as 01_build_inspection_panel.py.
# severity_majmod_share (Major+Moderate collapsed), not severity_critmajor_share
# (Critical+Major collapsed): human-eval accuracy (eval/results_and_notes/
# 20260916_LLM_Extraction_Validation_Report.docx, Section 2) is 90-94% for the
# Major/Moderate collapse vs. 66-68% for the raw 4-tier -- the extraction's real
# confusion is at the Major/Moderate boundary, not Critical/Major.
TEXT_FEATURES = [
    "severity_majmod_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "cultural_root_cause_share",
    "vc_laboratorycontrolssystem_share",
    "vc_qualitysystem_share",
    "n_laboratorycontrolssystem_obs",
    "n_qualitysystem_obs",
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]

SEED = 42


def _build_outcome_global(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Original outcome (kept for comparison): above-median AEs in the 4
    quarters after inspection, split against a single median pooled across
    ALL facilities and ALL drugs. Restrict to inspections with at least one
    real (non-imputed) AE count in the post-inspection window before
    summing. Reproduces the INFORMS slide's "176 inspections, 78 FEIs with
    ANDA-matched AEs" cohort exactly (verified against --anda-ae): an
    inspection whose FEI has zero ANDA-matched AE records anywhere in
    tp1..tp4 is excluded, not zero-filled. Within a kept row, a still-
    missing individual quarter is treated as 0 (the FEI does have AE
    tracking; that specific quarter is a true zero).

    KNOWN CONFOUND (found 2026-09-16): a global median split on raw AE
    count is dominated by which facility/drug this is -- 88.6% of the
    variance in n_ae_next4q across inspection-events is BETWEEN facilities,
    only 11.4% is WITHIN a facility over time. GroupKFold CV (which holds
    out entire facilities) correctly refuses to let a model exploit that,
    which leaves little for any feature, text or otherwise, to explain.
    See _build_outcome_relative() for the fix.
    """
    df = df.copy()
    ae_cols = ["n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"]
    has_any_ae_data = df[ae_cols].notna().any(axis=1)
    dropped = (~has_any_ae_data).sum()
    if dropped:
        print(f"  Dropping {dropped} inspections with zero AE records in the "
              f"post-inspection window (not zero-filling)")
    df = df[has_any_ae_data].copy()
    df["n_ae_next4q"] = df[ae_cols].fillna(0).sum(axis=1)
    df["ae_high_next4q"] = (df["n_ae_next4q"] > df["n_ae_next4q"].median()).astype(int)
    return df, "ae_high_next4q"


def _build_outcome_relative(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-facility relative-change outcome (added 2026-09-16 to fix the
    confound documented in _build_outcome_global): compare each inspection's
    facility to ITSELF, pre- vs. post-inspection, instead of splitting
    against a median pooled across every facility and drug. This cancels
    out facility/drug size entirely, since both windows come from the same
    FEI.

    ae_rise_next4q = 1 if this facility's AE count in the 4 quarters after
    the inspection is higher than its own AE count in the 4 quarters
    before it. Requires real (non-imputed) AE data on both sides; an
    inspection with no real AE record on either side is dropped, not
    zero-filled, same no-zero-fill standard as the rest of this session's
    work.
    """
    df = df.copy()
    pre_cols  = ["n_ae_tm4", "n_ae_tm3", "n_ae_tm2", "n_ae_tm1"]
    post_cols = ["n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"]
    has_pre  = df[pre_cols].notna().any(axis=1)
    has_post = df[post_cols].notna().any(axis=1)
    keep = has_pre & has_post
    dropped = (~keep).sum()
    if dropped:
        print(f"  Dropping {dropped} inspections missing real AE data on the "
              f"pre- or post-inspection side (not zero-filling)")
    df = df[keep].copy()
    df["n_ae_pre4q"]  = df[pre_cols].fillna(0).sum(axis=1)
    df["n_ae_post4q"] = df[post_cols].fillna(0).sum(axis=1)
    df["ae_rise_next4q"] = (df["n_ae_post4q"] > df["n_ae_pre4q"]).astype(int)
    return df, "ae_rise_next4q"


def _cv_evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, label: str,
                  n_splits: int = 5) -> list[dict]:
    n_splits = min(n_splits, max(2, len(np.unique(groups)) - 1))
    gkf = GroupKFold(n_splits=n_splits)
    lr  = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
    rf  = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                  random_state=SEED, n_jobs=-1)
    results = []
    for model_name, model in [("LR", lr), ("RF", rf)]:
        aucs, aps = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
                prob = model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, prob))
            aps.append(average_precision_score(y_te, prob))
        # One-tailed t-test of fold AUCs vs. 0.5, matching the INFORMS slide's
        # own stated test ("p from one-tailed t-test vs. 0.5 across 5 CV folds").
        if len(aucs) >= 2:
            t_stat, p_two = stats.ttest_1samp(aucs, 0.5)
            p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        else:
            p_one = np.nan
        results.append({
            "config": label,
            "model":  model_name,
            "auc":    np.mean(aucs) if aucs else np.nan,
            "auc_std": np.std(aucs) if aucs else np.nan,
            "p_vs_0.5": p_one,
            "ap":     np.mean(aps)  if aps  else np.nan,
            "n_folds": len(aucs),
            "n":      len(y),
        })
    return results


def plot_ablation_bar(metrics: pd.DataFrame, out_path: Path, outcome_label: str = "relative") -> None:
    lr = metrics[metrics["model"] == "LR"].copy()
    rf = metrics[metrics["model"] == "RF"].copy()
    configs = lr["config"].tolist()
    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_lr = ax.bar(x - w/2, lr["auc"], w, label="Logistic Reg.", color="#2563eb", alpha=0.85)
    bars_rf = ax.bar(x + w/2, rf["auc"], w, label="Random Forest", color="#d97706", alpha=0.85)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Random (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Mean AUC (GroupKFold CV)", fontsize=10)
    outcome_desc = ("AE count rose vs. this facility's own pre-inspection window"
                     if outcome_label == "relative" else
                     "above-median AEs in 4 quarters after inspection (pooled across facilities)")
    ax.set_title(f"VAI-signal validation rerun (current text pipeline)\noutcome: {outcome_desc}", fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    for bar in [*bars_lr, *bars_rf]:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation bar -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAI-only text-signal AE prediction, rerun with current data")
    parser.add_argument("--anda-ae", dest="anda_ae", action="store_true",
                        help="Use ANDA-specific AE panel instead of drug-level panel")
    parser.add_argument("--outcome", choices=["global", "relative"], default="relative",
                        help="global = original above-pooled-median split (confounded by "
                             "facility/drug size, see _build_outcome_global). relative = "
                             "per-facility pre-vs-post comparison, the 2026-09-16 fix "
                             "(default).")
    args = parser.parse_args()
    panel_path = PANEL_ANDA if args.anda_ae else PANEL

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_inspection_panel.py first.")

    print(f"Loading panel ({'ANDA-specific' if args.anda_ae else 'drug-level'})...")
    df = pd.read_parquet(panel_path)
    if args.outcome == "relative":
        df, outcome_col = _build_outcome_relative(df)
    else:
        df, outcome_col = _build_outcome_global(df)
    print(f"  {len(df)} inspection events, {df['fei'].nunique()} FEIs")
    print(f"  Outcome definition: {args.outcome} ({outcome_col})")

    complete_mask = df[TEXT_FEATURES].notna().all(axis=1) & df[outcome_col].notna()
    df = df[complete_mask].copy()
    print(f"  Complete rows for modeling: {len(df)} (FEIs: {df['fei'].nunique()})")

    if len(df) < 20:
        print("  Too few rows for cross-validation. Check panel construction.")
        return

    X_text  = df[TEXT_FEATURES].values
    y       = df[outcome_col].values.astype(int)
    groups  = df["fei"].astype(int).values
    print(f"  Outcome base rate: {y.mean():.1%}")

    print("\nConfig A: Text only, full sample...")
    results_A = _cv_evaluate(X_text, y, groups, "A: Text only (full sample)")

    X_oai_flag = df[["any_oai"]].values.astype(float)
    print("\nConfig C: OAI flag only...")
    results_C = _cv_evaluate(X_oai_flag, y, groups, "C: OAI flag only")

    print("\nConfig B: Text + OAI flag...")
    X_both = np.hstack([X_text, X_oai_flag])
    results_B = _cv_evaluate(X_both, y, groups, "B: Text + OAI")

    # ── Config D: VAI-only facilities (the INFORMS flagship claim) ──
    results_D = []
    fei_ever_oai = df.groupby("fei")["any_oai"].max()
    vai_feis = fei_ever_oai[fei_ever_oai == 0].index
    df_vai = df[df["fei"].isin(vai_feis)].copy()
    print(f"\nConfig D: VAI-only facilities ({df_vai['fei'].nunique()} FEIs, {len(df_vai)} rows)...")
    if len(df_vai) >= 20 and df_vai[outcome_col].nunique() > 1:
        X_vai   = df_vai[TEXT_FEATURES].values
        y_vai   = df_vai[outcome_col].values.astype(int)
        grp_vai = df_vai["fei"].astype(int).values
        results_D = _cv_evaluate(X_vai, y_vai, grp_vai, "D: VAI-only (text)")
    else:
        print("  Too few rows or single-class outcome -- skipping Config D.")

    # ── Config E: OAI-ever facilities only (for contrast with D) ──
    results_E = []
    fei_has_oai = fei_ever_oai[fei_ever_oai == 1].index
    df_oai_sub = df[df["fei"].isin(fei_has_oai)].copy()
    print(f"\nConfig E: OAI-ever facilities ({df_oai_sub['fei'].nunique()} FEIs, {len(df_oai_sub)} rows)...")
    if len(df_oai_sub) >= 20 and df_oai_sub[outcome_col].nunique() > 1:
        X_oai_e   = df_oai_sub[TEXT_FEATURES].values
        y_oai_e   = df_oai_sub[outcome_col].values.astype(int)
        grp_oai_e = df_oai_sub["fei"].astype(int).values
        results_E = _cv_evaluate(X_oai_e, y_oai_e, grp_oai_e, "E: OAI-ever (text)")
    else:
        print("  Too few rows or single-class outcome -- skipping Config E.")

    all_results = results_A + results_B + results_C + results_D + results_E
    metrics = pd.DataFrame(all_results)

    OUT_MOD.mkdir(parents=True, exist_ok=True)
    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.outcome == "relative" else "_global"
    metrics.to_csv(OUT_MOD / f"ablation_metrics{suffix}.csv", index=False)
    print(f"\nResults:\n{metrics[['config','model','auc','p_vs_0.5','n_folds','n']].to_string(index=False)}")

    md_lines = [
        "# VAI-signal validation rerun -- model summary",
        "",
        f"Outcome definition: {args.outcome} ({outcome_col})",
        f"Panel: {len(df)} inspection events, {df['fei'].nunique()} unique FEIs",
        f"Outcome base rate: {y.mean():.1%}",
        "",
        metrics[["config", "model", "auc", "p_vs_0.5", "ap"]].to_string(index=False),
        "",
        "AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.",
        "p_vs_0.5: one-tailed t-test of the 5 fold-level AUCs against 0.5, matching",
        "the INFORMS slide's own stated test.",
    ]
    (OUT_TABS / f"model_summary{suffix}.md").write_text("\n".join(md_lines))

    plot_ablation_bar(metrics, OUT_FIGS / f"ablation_auc_bar{suffix}.png", outcome_label=args.outcome)
    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
