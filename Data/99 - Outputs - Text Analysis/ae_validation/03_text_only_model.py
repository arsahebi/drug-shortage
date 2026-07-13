"""
03_text_only_model.py
────────────────────────────────────────────────────────────────────────────
Predictive model: do LLM text signals forecast above-median adverse event
volume at the facility level (FEI × year, predicting year t+1)?

This script answers the core research question for the paper:
  "Text features extracted by LLM from 483 observations predict
   facility-level patient harm, measured by FAERS serious AE counts,
   one year in advance."

Three model configurations are compared (ablation):
  A. Text only       — LLM signals from 483 text
  B. Text + Insp     — Text + inspection outcome counts (OAI/VAI/WL)
  C. Insp only       — Inspection counts only (baseline)

Binary outcome: ae_high_t1 = 1 if FEI's n_ae_t1 > per-drug-FEI-pair median.

Models: Logistic Regression (L2) and Random Forest.
Evaluation: GroupKFold cross-validation grouped by FEI (no data leakage).

Outputs
───────
  outputs/models/ablation_metrics.csv        — AUC/AP by model config
  outputs/figures/ablation_auc_bar.png       — bar chart of AUC by config
  outputs/figures/rf_feature_importance.png  — RF feature importances (text-only)
  outputs/tables/model_summary.md            — plain-text summary table
"""

from __future__ import annotations

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
PANEL     = OUT / "fei_ae_panel.parquet"

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
    """Binary: ae_high_t1 = 1 if n_ae_t1 > median across all FEI-years."""
    df = df.copy()
    df["ae_high_t1"] = (df["n_ae_t1"] > df["n_ae_t1"].median()).astype(int)
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
    ax.set_title("Ablation: text vs. inspection features for AE prediction\n(outcome: above-median AEs at facility, t+1)",
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(f"Panel not found: {PANEL}\nRun 01_build_fei_ae_panel.py first.")

    print("Loading panel…")
    df = pd.read_parquet(PANEL)
    df = _build_outcome(df)

    # restrict to rows with complete text features and t+1 AE outcome
    complete_mask = df[TEXT_FEATURES].notna().all(axis=1) & df["ae_high_t1"].notna()
    df = df[complete_mask].copy()
    print(f"  Complete rows for modeling: {len(df)} (FEIs: {df['fei'].nunique()})")

    if len(df) < 20:
        print("  Too few rows for cross-validation. Check panel construction.")
        return

    X_text  = df[TEXT_FEATURES].fillna(0).values
    y       = df["ae_high_t1"].values.astype(int)
    groups  = df["fei"].astype(int).values

    print(f"  Outcome: {y.mean():.1%} above-median AE rate")

    # ── Config A: Text only ──
    print("\nConfig A: Text only…")
    results_A = _cv_evaluate(X_text, y, groups, "A: Text only")

    # ── RF importances (text only, full data) ──
    print("Fitting full RF for feature importances (text only)…")
    imp = _rf_importances(X_text, y, TEXT_FEATURES)

    # ── Config C: Inspection only (inspection count as minimal baseline) ──
    X_insp_fallback = df[["n_inspections_in_year"]].fillna(0).values
    print("\nConfig C: Inspection count only (fallback baseline)…")
    results_C = _cv_evaluate(X_insp_fallback, y, groups, "C: Insp only")

    # ── Config B: Text + Inspection ──
    print("\nConfig B: Text + Inspection…")
    X_both = np.hstack([X_text, X_insp_fallback])
    results_B = _cv_evaluate(X_both, y, groups, "B: Text + Insp")

    # ── Compile and save ──
    all_results = results_A + results_B + results_C
    metrics = pd.DataFrame(all_results)

    OUT_MOD.mkdir(parents=True, exist_ok=True)
    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(OUT_MOD / "ablation_metrics.csv", index=False)
    print(f"\nAblation results:\n{metrics[['config','model','auc','ap','n_folds']].to_string(index=False)}")

    # Markdown summary
    md_lines = [
        "# Text-signal AE prediction — model summary",
        "",
        f"Panel: {len(df)} FEI × year observations, {df['fei'].nunique()} unique FEIs",
        f"Outcome: above-median serious AE count at t+1 (base rate {y.mean():.1%})",
        "",
        metrics[["config", "model", "auc", "ap"]].to_string(index=False),
        "",
        "AUC > 0.5 = better than random. Group-based CV prevents FEI data leakage.",
    ]
    (OUT_TABS / "model_summary.md").write_text("\n".join(md_lines))

    plot_ablation_bar(metrics, OUT_FIGS / "ablation_auc_bar.png")
    if not imp.empty:
        plot_rf_importance(imp, OUT_FIGS / "rf_feature_importance.png")

    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
