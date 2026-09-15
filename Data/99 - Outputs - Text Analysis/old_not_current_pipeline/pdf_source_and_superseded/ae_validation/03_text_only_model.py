"""
03_text_only_model.py
────────────────────────────────────────────────────────────────────────────
Predictive model: do LLM text signals forecast above-median adverse event
volume at the facility level, 4 quarters after inspection?

Unit of observation: one inspection event (246 total, 98 FEIs, 2018–2026).
Features: 17 LLM text signals from the 483 observations at that inspection.
Outcome: binary — were AEs in the 4 quarters after inspection above median?
         ae_high_next4q = 1 if sum(n_ae_tp1..tp4) > median.

Three model configurations (ablation):
  A. Text only       — 17 LLM signals from 483 text
  B. Text + OAI flag — Text + whether this inspection was OAI (structured baseline)
  C. OAI flag only   — just the OAI/VAI/NAI outcome (what structured data alone tells you)

Models: Logistic Regression (L2) and Random Forest.
Evaluation: FEI-grouped 5-fold CV (all inspections of the same facility stay
            in one fold — prevents facility-level data leakage).

Outputs
───────
  outputs/models/ablation_metrics.csv        — AUC/AP by model config
  outputs/figures/ablation_auc_bar.png       — bar chart of AUC by config
  outputs/figures/rf_feature_importance.png  — RF feature importances (text-only)
  outputs/tables/model_summary.md            — plain-text summary table
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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    _SKLEARN = True
except ModuleNotFoundError:
    _SKLEARN = False
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
OUT       = HERE / "outputs"
OUT_TABS  = OUT / "tables"
OUT_FIGS  = OUT / "figures"
OUT_MOD   = OUT / "models"
PANEL      = OUT / "fei_ae_panel_inspection_centered.parquet"
PANEL_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

# Inspection features from shortage prediction pipeline (optional enrichment)
SP_CODE   = HERE.parent.parent.parent / "Data" / "99 - Outputs - Shortage Prediction" / "code"

TEXT_FEATURES = [
    # Layer 3: LLM signal shares
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
    # Layer 5: raw counts (intensity alongside proportion)
    "n_labcontrols_obs",
    "n_qualitysystem_obs",
    # Layer 5: joint co-occurrence flags (two-failure-mode hypothesis)
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]

FEATURE_LABELS = {
    "severity_critmajor_share":        "Severity Maj+Crit share",
    "contamination_llm_share":         "Contamination flag rate",
    "data_integrity_llm_share":        "Data integrity flag rate",
    "patient_risk_llm_share":          "Patient risk flag rate",
    "investigation_llm_share":         "Invest. failure flag rate",
    "repeat_cross_insp_share":         "Repeat obs. rate",
    "scope_facilitywide_share":        "Scope: facility-wide share",
    "cultural_root_cause_share":       "Root cause: Cultural share",
    "vc_labcontrols_share":            "Domain: Lab controls share",
    "vc_qualitysystem_share":          "Domain: Quality system share",
    "n_labcontrols_obs":               "# Lab control obs.",
    "n_qualitysystem_obs":             "# Quality system obs.",
    "joint_labcontrols_qualitysystem": "Joint: LabCtrl + QualSys",
    "joint_labcontrols_dataintegrity": "Joint: LabCtrl + DI",
    "joint_contamination_labcontrols": "Joint: Contam + LabCtrl",
    "joint_qualitysystem_production":  "Joint: QualSys + Prod",
    "multi_domain_insp":               "Multi-domain inspection",
}

SEED = 42


# ── Outcome construction ──────────────────────────────────────────────────────

def _build_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: ae_high_next4q = 1 if total AEs in Q+1..Q+4 > median across all inspections."""
    df = df.copy()
    df["n_ae_next4q"] = (
        df["n_ae_tp1"].fillna(0) + df["n_ae_tp2"].fillna(0)
        + df["n_ae_tp3"].fillna(0) + df["n_ae_tp4"].fillna(0)
    )
    df["ae_high_next4q"] = (df["n_ae_next4q"] > df["n_ae_next4q"].median()).astype(int)
    return df


# ── Inspection feature loading (optional) ─────────────────────────────────────

def _try_load_inspection_features(feis: pd.Series, years: pd.Series) -> pd.DataFrame | None:
    """
    Attempt to load OAI/VAI/WL counts from the Redica CSV used by the
    shortage prediction pipeline. Returns None if not available.
    """
    try:
        import sys
        sys.path.insert(0, str(SP_CODE))
        from config import REDICA_CSV
        if not REDICA_CSV.exists():
            return None
        red = pd.read_csv(REDICA_CSV, low_memory=False)
        red.columns = [c.strip() for c in red.columns]
        # look for OAI / warning letter columns
        oai_col = next((c for c in red.columns if "oai" in c.lower()), None)
        wl_col  = next((c for c in red.columns if "warning" in c.lower()), None)
        fei_col = next((c for c in red.columns if "fei" in c.lower()), None)
        yr_col  = next((c for c in red.columns if "year" in c.lower()), None)
        if not all([oai_col, fei_col, yr_col]):
            return None
        sub = red[[fei_col, yr_col, oai_col] + ([wl_col] if wl_col else [])].copy()
        sub = sub.rename(columns={fei_col: "fei", yr_col: "year",
                                   oai_col: "n_oai", **({"wl_col": "n_wl"} if wl_col else {})})
        sub["fei"]  = pd.to_numeric(sub["fei"], errors="coerce").astype("Int64")
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
        return sub.dropna(subset=["fei", "year"])
    except Exception:
        return None


# ── Cross-validation ──────────────────────────────────────────────────────────

def _cv_evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 label: str, n_splits: int = 5) -> dict:
    """GroupKFold CV; returns mean AUC and AP."""
    if not _SKLEARN:
        return {"model": label, "auc": np.nan, "ap": np.nan, "n": len(y)}

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
        results.append({
            "config": label,
            "model":  model_name,
            "auc":    np.mean(aucs) if aucs else np.nan,
            "ap":     np.mean(aps)  if aps  else np.nan,
            "n_folds": len(aucs),
            "n":      len(y),
        })
    return results


def _rf_importances(X: np.ndarray, y: np.ndarray,
                    feature_names: list[str]) -> pd.DataFrame:
    if not _SKLEARN:
        return pd.DataFrame()
    rf = RandomForestClassifier(n_estimators=500, max_depth=4,
                                 min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y)
    return (pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
              .sort_values("importance", ascending=False))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_ablation_bar(metrics: pd.DataFrame, out_path: Path) -> None:
    lr = metrics[metrics["model"] == "LR"].copy()
    rf = metrics[metrics["model"] == "RF"].copy()

    configs = lr["config"].tolist()
    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_lr = ax.bar(x - w/2, lr["auc"], w, label="Logistic Reg.", color="#2563eb", alpha=0.85)
    bars_rf = ax.bar(x + w/2, rf["auc"], w, label="Random Forest", color="#d97706", alpha=0.85)

    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Random (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Mean AUC (GroupKFold CV)", fontsize=10)
    ax.set_title("Ablation: text vs. inspection features for AE prediction\n(outcome: above-median AEs in 4 quarters after inspection)",
                 fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9)

    for bar in [*bars_lr, *bars_rf]:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation bar → {out_path}")


def plot_ae_trajectory(traj: pd.DataFrame, out_path: Path) -> None:
    """
    Line plot: mean AE count at tm2/tm1/t0/t1/t2 for each facility group.
    Full 5-point window centered on inspection year shows pre-existing
    elevation and post-inspection correction speed.
    """
    colors  = {"OAI-ever": "#dc2626", "High-signal VAI": "#d97706", "Low-signal VAI": "#2563eb"}
    markers = {"OAI-ever": "o", "High-signal VAI": "s", "Low-signal VAI": "^"}
    lags    = ["tm4", "tm2", "t0", "tp2", "tp4"]
    xlabels = ["Q−4\n(1yr before)", "Q−2\n(6mo before)", "Q0\n(inspection)", "Q+2\n(6mo after)", "Q+4\n(1yr after)"]
    x       = np.arange(len(lags))

    fig, ax = plt.subplots(figsize=(7, 4))
    for grp in ["OAI-ever", "High-signal VAI", "Low-signal VAI"]:
        sub = traj[traj["group"] == grp]
        if sub.empty:
            continue
        means = [sub[f"mean_ae_{lag}"].values[0] for lag in lags]
        ns    = sub["n_feis"].values[0]
        ax.plot(x, means, marker=markers[grp], color=colors[grp],
                linewidth=2, label=f"{grp} (n={ns} FEIs)")

    ax.axvline(x=2, color="gray", linewidth=1.0, linestyle="--", alpha=0.6, label="Inspection")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean serious AEs per FEI-year")
    ax.set_title("AE trajectory around inspection year by facility group\n"
                 "(High-signal = top-quartile LabControls+DI text features)", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved AE trajectory → {out_path}")


def plot_rf_importance(imp: pd.DataFrame, out_path: Path) -> None:
    imp = imp.copy()
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
    imp = imp.sort_values("importance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp["label"], imp["importance"], color="#2563eb", alpha=0.8)
    ax.set_xlabel("RF Feature Importance (MDI)", fontsize=10)
    ax.set_title("Text-only model: feature importances\n(predicting above-median AEs, t+1)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved RF importance → {out_path}")


# ── Trajectory analysis ───────────────────────────────────────────────────────

# Technical signal features that theory says should drive AEs (Chain 1)
_TECH_FEATURES = [
    "vc_labcontrols_share",
    "data_integrity_llm_share",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "n_labcontrols_obs",
]


def _trajectory_analysis(
    df: pd.DataFrame,
    fei_ever_oai: "pd.Series",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment facilities by OAI status × technical signal level, then compare
    AE trajectories at t0, t1, t2.

    Groups:
      OAI-ever          — received at least one OAI (forced correction)
      High-signal VAI   — top-quartile composite technical score, never OAI
      Low-signal VAI    — bottom three quartiles, never OAI

    Returns:
      traj_df   — one row per group: mean AE at t0/t1/t2 and persistence ratio
      flagged   — individual High-signal VAI FEIs sorted by AE persistence
    """
    df = df.copy()

    # Composite technical score: z-score mean of available tech features
    tech_cols = [c for c in _TECH_FEATURES if c in df.columns]
    if not tech_cols:
        return pd.DataFrame(), pd.DataFrame()

    from scipy.stats import zscore as _zscore
    df["tech_score"] = df[tech_cols].fillna(0).apply(_zscore).mean(axis=1)

    # Assign group per FEI (use max tech_score across years for the FEI)
    fei_tech = df.groupby("fei")["tech_score"].mean()
    q75 = fei_tech.quantile(0.75)

    def _group(fei):
        if fei in fei_ever_oai.index and fei_ever_oai[fei] == 1:
            return "OAI-ever"
        if fei_tech.get(fei, 0) >= q75:
            return "High-signal VAI"
        return "Low-signal VAI"

    df["group"] = df["fei"].map(_group)

    rows = []
    for grp in ["OAI-ever", "High-signal VAI", "Low-signal VAI"]:
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        mm4 = sub["n_ae_tm4"].mean() if "n_ae_tm4" in sub.columns else np.nan
        mm2 = sub["n_ae_tm2"].mean() if "n_ae_tm2" in sub.columns else np.nan
        m0  = sub["n_ae_t0"].mean()  if "n_ae_t0"  in sub.columns else np.nan
        mp2 = sub["n_ae_tp2"].mean() if "n_ae_tp2" in sub.columns else np.nan
        mp4 = sub["n_ae_tp4"].mean() if "n_ae_tp4" in sub.columns else np.nan
        # pre-elevation: Q0 vs Q-4 (1yr before)
        pre_rise = (m0 / mm4) if (mm4 and mm4 > 0) else np.nan
        # persistence: Q+4 vs Q0
        persist  = (mp4 / m0) if (m0 and m0 > 0) else np.nan
        rows.append({
            "group":           grp,
            "n_feis":          sub["fei"].nunique(),
            "n_rows":          len(sub),
            "mean_ae_tm4":     round(mm4, 1),
            "mean_ae_tm2":     round(mm2, 1),
            "mean_ae_t0":      round(m0, 1),
            "mean_ae_tp2":     round(mp2, 1),
            "mean_ae_tp4":     round(mp4, 1),
            "pre_rise_t0_tm4": round(pre_rise, 3),
            "persist_tp4_t0":  round(persist, 3),
        })
    traj_df = pd.DataFrame(rows)

    # Flag individual high-signal VAI facilities: ranked by persistence
    hs_vai = df[df["group"] == "High-signal VAI"].copy()
    if hs_vai.empty:
        return traj_df, pd.DataFrame()

    agg_dict = dict(
        mean_ae_t0=("n_ae_t0",   "mean"),
        mean_ae_tp2=("n_ae_tp2", "mean"),
        mean_ae_tp4=("n_ae_tp4", "mean"),
        mean_tech_score=("tech_score", "mean"),
        n_inspections=("fei",    "count"),
    )
    if "n_ae_tm4" in hs_vai.columns:
        agg_dict["mean_ae_tm4"] = ("n_ae_tm4", "mean")
    fei_agg = hs_vai.groupby("fei", as_index=False).agg(**agg_dict)
    fei_agg["persist_tp4_t0"] = (fei_agg["mean_ae_tp4"] / fei_agg["mean_ae_t0"]).round(3)
    fei_agg["ae_rising"] = fei_agg["persist_tp4_t0"] > 1.0
    fei_agg = fei_agg.sort_values("persist_tp4_t0", ascending=False).round(1)

    print(f"  Groups: {traj_df[['group','n_feis']].to_string(index=False)}")
    print(f"  High-signal VAI threshold: tech_score ≥ {q75:.3f} (top quartile)")
    return traj_df, fei_agg


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Text-signal AE prediction model")
    parser.add_argument("--anda-ae", dest="anda_ae", action="store_true",
                        help="Use ANDA-specific AE panel instead of drug-level panel")
    args = parser.parse_args()
    panel_path = PANEL_ANDA if args.anda_ae else PANEL

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_fei_ae_panel.py first.")

    print(f"Loading panel ({'ANDA-specific' if args.anda_ae else 'drug-level'})…")
    df = pd.read_parquet(panel_path)
    df = _build_outcome(df)
    print(f"  {len(df)} inspection events, {df['fei'].nunique()} FEIs")

    # restrict to rows with complete text features and outcome
    complete_mask = df[TEXT_FEATURES].notna().all(axis=1) & df["ae_high_next4q"].notna()
    df = df[complete_mask].copy()
    print(f"  Complete rows for modeling: {len(df)} (FEIs: {df['fei'].nunique()})")

    if len(df) < 20:
        print("  Too few rows for cross-validation. Check panel construction.")
        return

    X_text  = df[TEXT_FEATURES].fillna(0).values
    y       = df["ae_high_next4q"].values.astype(int)
    groups  = df["fei"].astype(int).values

    print(f"  Outcome: {y.mean():.1%} above-median AEs in Q+1..Q+4")

    # ── Config A: Text only ──
    print("\nConfig A: Text only…")
    results_A = _cv_evaluate(X_text, y, groups, "A: Text only")

    # ── RF importances (text only, full data) ──
    print("Fitting full RF for feature importances (text only)…")
    imp = _rf_importances(X_text, y, TEXT_FEATURES)

    # ── Config C: OAI flag only (structured inspection outcome as baseline) ──
    X_oai_flag = df[["any_oai"]].fillna(0).values.astype(float)
    print("\nConfig C: OAI flag only (structured inspection outcome baseline)…")
    results_C = _cv_evaluate(X_oai_flag, y, groups, "C: OAI flag only")

    # ── Config B: Text + OAI flag ──
    print("\nConfig B: Text + OAI flag…")
    X_both = np.hstack([X_text, X_oai_flag])
    results_B = _cv_evaluate(X_both, y, groups, "B: Text + OAI")

    # ── Config D: VAI-only facilities (OAI moderation hypothesis) ──
    # Facilities that received only VAI (never OAI in any year in this panel)
    # had bad 483 signals but no forced corrective action — do their signals
    # predict sustained AEs better than the full population?
    results_D = []
    if "any_oai" in df.columns:
        fei_ever_oai = df.groupby("fei")["any_oai"].max()
        vai_feis = fei_ever_oai[fei_ever_oai == 0].index
        df_vai = df[df["fei"].isin(vai_feis)].copy()
        print(f"\nConfig D: VAI-only facilities ({df_vai['fei'].nunique()} FEIs, {len(df_vai)} rows)…")
        if len(df_vai) >= 20 and df_vai["ae_high_next4q"].nunique() > 1:
            X_vai   = df_vai[TEXT_FEATURES].fillna(0).values
            y_vai   = df_vai["ae_high_next4q"].values.astype(int)
            grp_vai = df_vai["fei"].astype(int).values
            results_D = _cv_evaluate(X_vai, y_vai, grp_vai, "D: VAI-only (text)")
        else:
            print("  Too few rows or single-class outcome — skipping Config D.")
    else:
        print("\nConfig D skipped — any_oai column not in panel (run 01 first).")

    # ── Config E: OAI-ever facilities only (for contrast with D) ──
    results_E = []
    if "any_oai" in df.columns:
        fei_has_oai = fei_ever_oai[fei_ever_oai == 1].index
        df_oai_sub = df[df["fei"].isin(fei_has_oai)].copy()
        print(f"\nConfig E: OAI-ever facilities ({df_oai_sub['fei'].nunique()} FEIs, {len(df_oai_sub)} rows)…")
        if len(df_oai_sub) >= 20 and df_oai_sub["ae_high_next4q"].nunique() > 1:
            X_oai_e   = df_oai_sub[TEXT_FEATURES].fillna(0).values
            y_oai_e   = df_oai_sub["ae_high_next4q"].values.astype(int)
            grp_oai_e = df_oai_sub["fei"].astype(int).values
            results_E = _cv_evaluate(X_oai_e, y_oai_e, grp_oai_e, "E: OAI-ever (text)")
        else:
            print("  Too few rows or single-class outcome — skipping Config E.")

    # ── Trajectory analysis (AE persistence by facility group) ───────────────
    # Technical signal composite: mean z-score of top LabControls/DI features.
    # Segments VAI facilities into high-signal vs low-signal to test whether
    # bad-signal VAI facilities show sustained (non-declining) AE trajectory,
    # unlike OAI facilities where corrective action should reduce AEs at t+1/t+2.
    print("\nTrajectory analysis: AE persistence by facility group…")
    traj_df, flagged = _trajectory_analysis(df, fei_ever_oai if "any_oai" in df.columns else pd.Series(dtype=int))

    # ── Compile and save ──
    all_results = results_A + results_B + results_C + results_D + results_E
    metrics = pd.DataFrame(all_results)

    OUT_MOD.mkdir(parents=True, exist_ok=True)
    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(OUT_MOD / "ablation_metrics.csv", index=False)
    print(f"\nAblation results:\n{metrics[['config','model','auc','ap','n_folds']].to_string(index=False)}")

    if not traj_df.empty:
        traj_df.to_csv(OUT_TABS / "ae_trajectory_by_group.csv", index=False)
        print(f"\nAE trajectory by group:\n{traj_df.to_string(index=False)}")
    if not flagged.empty:
        flagged.to_csv(OUT_TABS / "high_signal_vai_flagged.csv", index=False)
        print(f"\nHigh-signal VAI facilities (candidates for strategic leniency):")
        print(flagged.to_string(index=False))

    # Markdown summary
    md_lines = [
        "# Text-signal AE prediction — model summary",
        "",
        f"Panel: {len(df)} inspection events, {df['fei'].nunique()} unique FEIs",
        f"Outcome: above-median AEs in Q+1..Q+4 after inspection (base rate {y.mean():.1%})",
        "",
        metrics[["config", "model", "auc", "ap"]].to_string(index=False),
        "",
        "AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.",
        "",
        "## AE trajectory by facility group",
        traj_df.to_string(index=False) if not traj_df.empty else "(not available)",
    ]
    (OUT_TABS / "model_summary.md").write_text("\n".join(md_lines))

    plot_ablation_bar(metrics, OUT_FIGS / "ablation_auc_bar.png")
    if not imp.empty:
        plot_rf_importance(imp, OUT_FIGS / "rf_feature_importance.png")
    if not traj_df.empty:
        plot_ae_trajectory(traj_df, OUT_FIGS / "ae_trajectory_by_group.png")

    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
