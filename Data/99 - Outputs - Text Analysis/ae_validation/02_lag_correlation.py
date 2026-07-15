"""
02_lag_correlation.py
────────────────────────────────────────────────────────────────────────────
Cross-correlation analysis: do LLM text signals from 483 observations
predict serious adverse event counts at facilities 0, 1, 2 years later?

Method
──────
For each text feature × AE lag pair, compute Spearman rank correlation
across all 645 FEI×year rows. No model, no train/test split — just a
bivariate rank statistic: rank facilities by the feature value, rank them
by AE count at that lag, then correlate the two rank vectors.

  ρ > 0  →  more of this signal in 483 text associates with more AEs later
  ρ < 0  →  more of this signal associates with fewer AEs later
  ρ ≈ 0  →  no monotonic relationship

If the correlation is stronger at lag+1 than at lag 0 (same year), that
supports the predictive interpretation over a reactive one.

Outputs
───────
  outputs/tables/lag_correlation_table.csv
      Spearman ρ and p-value for each feature × lag combination.

  outputs/figures/lag_correlation_heatmap.png
      Heatmap of Spearman ρ values (features × lags).

  outputs/figures/lag_profile_all_features.png
      Line chart: ρ vs lag for each feature.

  outputs/figures/scatter_<feature>_lag1.png
      Individual scatter plots for each text feature vs n_ae_t1.

Debug flags
───────────
  --feature FEAT    run and print diagnostics for one feature only
  --verbose         print per-feature stats (n, null%, mean, std) and each ρ
  --no-plots        skip figure generation (faster for spot-checking)

Examples
────────
  python 02_lag_correlation.py                          # full run
  python 02_lag_correlation.py --verbose                # full run + per-feature stats
  python 02_lag_correlation.py --feature vc_labcontrols_share   # single feature
  python 02_lag_correlation.py --feature vc_labcontrols_share --verbose
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
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
OUT      = HERE / "outputs"
OUT_TABS = OUT / "tables"
OUT_FIGS = OUT / "figures"
PANEL    = OUT / "fei_ae_panel.parquet"

TEXT_FEATURES = [
    # individual shares / rates
    "n_labcontrols_obs",
    "vc_labcontrols_share",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "data_integrity_llm_share",
    "severity_critmajor_share",
    "vc_qualitysystem_share",
    "cultural_root_cause_share",
    "joint_qualitysystem_production",
    "contamination_llm_share",
    # additional
    "n_qualitysystem_obs",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "joint_labcontrols_qualitysystem",
    "multi_domain_insp",
]

FEATURE_LABELS = {
    "severity_critmajor_share":        "Severity: Maj+Crit share",
    "contamination_llm_share":         "Contamination flag rate",
    "data_integrity_llm_share":        "Data integrity flag rate",
    "patient_risk_llm_share":          "Patient risk flag rate",
    "investigation_llm_share":         "Invest. failure flag rate",
    "repeat_cross_insp_share":         "Repeat obs. rate",
    "scope_facilitywide_share":        "Scope: facility-wide share",
    "cultural_root_cause_share":       "Root cause: Cultural share",
    "vc_labcontrols_share":            "Domain: Lab controls share",
    "vc_qualitysystem_share":          "Domain: Quality system share",
    "n_labcontrols_obs":               "Lab controls obs count",
    "n_qualitysystem_obs":             "Quality system obs count",
    "joint_labcontrols_dataintegrity": "Joint: Lab ctrl + DI",
    "joint_contamination_labcontrols": "Joint: Contamination + Lab ctrl",
    "joint_qualitysystem_production":  "Joint: Quality sys + Production",
    "joint_labcontrols_qualitysystem": "Joint: Lab ctrl + Quality sys",
    "multi_domain_insp":               "Multi-domain inspection (≥3)",
}

AE_LAGS = {
    "n_ae_t0": "Lag 0 (same year)",
    "n_ae_t1": "Lag +1 yr",
    "n_ae_t2": "Lag +2 yr",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def _sig_label(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _feature_diagnostics(df: pd.DataFrame, feat: str) -> None:
    """Print distribution stats for a single feature — useful for debugging."""
    col = df[feat]
    n_total = len(col)
    n_null  = col.isna().sum()
    n_valid = n_total - n_null
    print(f"\n  [DEBUG] {feat}")
    print(f"    rows: {n_valid} valid / {n_null} null ({100*n_null/n_total:.1f}% missing)")
    if n_valid > 0:
        print(f"    range: [{col.min():.4f}, {col.max():.4f}]  "
              f"mean={col.mean():.4f}  std={col.std():.4f}")
        q = col.quantile([0.25, 0.50, 0.75])
        print(f"    Q25={q[0.25]:.4f}  median={q[0.50]:.4f}  Q75={q[0.75]:.4f}")
        if col.nunique() <= 5:
            print(f"    value counts: {col.value_counts().to_dict()}")


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_correlation_table(
    df: pd.DataFrame,
    features: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in df.columns:
            print(f"  [WARN] feature not in panel: {feat} — skipping")
            continue
        if verbose:
            _feature_diagnostics(df, feat)
        for ae_col, lag_label in AE_LAGS.items():
            r, p = _spearman(df[feat], df[ae_col])
            n = (df[feat].notna() & df[ae_col].notna()).sum()
            if verbose:
                sig = _sig_label(p) if not np.isnan(p) else ""
                print(f"      vs {lag_label:20s}  ρ={r:+.4f}  p={p:.4f}  n={n}  {sig}")
            rows.append({
                "feature":       feat,
                "feature_label": FEATURE_LABELS.get(feat, feat),
                "lag":           ae_col,
                "lag_label":     lag_label,
                "spearman_r":    round(r, 4) if not np.isnan(r) else np.nan,
                "p_value":       round(p, 4) if not np.isnan(p) else np.nan,
                "sig":           _sig_label(p),
                "n":             n,
            })
    return pd.DataFrame(rows)


def plot_heatmap(corr_tbl: pd.DataFrame, out_path: Path) -> None:
    pivot = corr_tbl.pivot(index="feature_label", columns="lag_label", values="spearman_r")
    pivot_p = corr_tbl.pivot(index="feature_label", columns="lag_label", values="sig")

    col_order = ["Lag 0 (same year)", "Lag +1 yr", "Lag +2 yr"]
    pivot   = pivot.reindex(columns=col_order)
    pivot_p = pivot_p.reindex(columns=col_order)

    lag1_vals = corr_tbl[corr_tbl["lag_label"] == "Lag +1 yr"].set_index("feature_label")["spearman_r"]
    pivot   = pivot.loc[lag1_vals.abs().sort_values(ascending=False).index]
    pivot_p = pivot_p.loc[pivot.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.05)
    im = ax.imshow(pivot.values.astype(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i, row_feat in enumerate(pivot.index):
        for j, col_name in enumerate(col_order):
            val = pivot.loc[row_feat, col_name]
            sig = pivot_p.loc[row_feat, col_name]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > vmax * 0.5 else "black")

    ax.set_title("Spearman ρ: text signal vs FAERS AE count by lag\n(* p<.05  ** p<.01  *** p<.001)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap → {out_path}")


def plot_scatter(df: pd.DataFrame, feature: str, ae_col: str, out_path: Path) -> None:
    sub = df[[feature, ae_col]].dropna()
    if len(sub) < 5:
        return
    r, p = _spearman(sub[feature], sub[ae_col])

    y = np.log1p(sub[ae_col])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(sub[feature], y, alpha=0.4, s=20, color="#2563eb", edgecolors="none")

    z  = np.polyfit(sub[feature], y, 1)
    xr = np.linspace(sub[feature].min(), sub[feature].max(), 100)
    ax.plot(xr, np.polyval(z, xr), color="#dc2626", linewidth=1.5,
            label=f"ρ={r:.2f}{_sig_label(p)}")

    label = FEATURE_LABELS.get(feature, feature)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("log(1 + AEs)", fontsize=9)
    ax.set_title(f"{label}\nvs {AE_LAGS.get(ae_col, ae_col)} (n={len(sub)})", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_lag_profile(corr_tbl: pd.DataFrame, features: list[str], out_path: Path) -> None:
    col_order = ["n_ae_t0", "n_ae_t1", "n_ae_t2"]
    lag_nums  = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(6, 4))
    for feat in features:
        sub  = corr_tbl[corr_tbl["feature"] == feat].set_index("lag")
        vals = [sub.loc[c, "spearman_r"] if c in sub.index else np.nan for c in col_order]
        ax.plot(lag_nums, vals, marker="o", linewidth=1.4, markersize=5,
                label=FEATURE_LABELS.get(feat, feat), alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Lag (years after inspection)", fontsize=10)
    ax.set_ylabel("Spearman ρ with FAERS AE count", fontsize=10)
    ax.set_title("Text signal lead: ρ by lag\n(upward slope = predictive, not reactive)", fontsize=9)
    ax.set_xticks(lag_nums)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved lag profile → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spearman lag correlation: 483 text features vs FAERS AEs")
    p.add_argument("--feature", metavar="FEAT",
                   help="Run diagnostics for a single feature only (name must match column in panel)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-feature distribution stats and each ρ as it is computed")
    p.add_argument("--no-plots", dest="no_plots", action="store_true",
                   help="Skip figure generation (faster for spot-checking correlations)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not PANEL.exists():
        raise FileNotFoundError(f"Panel not found: {PANEL}\nRun 01_build_fei_ae_panel.py first.")

    print("Loading panel…")
    df = pd.read_parquet(PANEL)
    print(f"  {len(df)} rows, {df['fei'].nunique()} FEIs")

    # which features to run
    if args.feature:
        if args.feature not in df.columns:
            avail = [c for c in df.columns if args.feature.lower() in c.lower()]
            print(f"  [ERROR] '{args.feature}' not in panel.")
            if avail:
                print(f"  Did you mean one of: {avail}")
            return
        features = [args.feature]
        print(f"  Running single-feature debug mode: {args.feature}")
    else:
        features = TEXT_FEATURES

    print(f"\nComputing Spearman correlations for {len(features)} feature(s)…")
    corr_tbl = build_correlation_table(df, features, verbose=args.verbose)

    if corr_tbl.empty:
        print("No results — check feature names.")
        return

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    if not args.feature:
        corr_tbl.to_csv(OUT_TABS / "lag_correlation_table.csv", index=False)
        print(f"  Saved → {OUT_TABS / 'lag_correlation_table.csv'}")

    print(f"\nCorrelations at lag+1 (sorted by |ρ|):")
    top = (corr_tbl[corr_tbl["lag"] == "n_ae_t1"]
           .sort_values("spearman_r", key=abs, ascending=False)
           [["feature_label", "spearman_r", "p_value", "sig", "n"]])
    print(top.to_string(index=False))

    if args.no_plots:
        print("\nSkipping plots (--no-plots).")
        return

    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    if not args.feature:
        print("\nPlotting heatmap…")
        plot_heatmap(corr_tbl, OUT_FIGS / "lag_correlation_heatmap.png")
        print("Plotting lag profile…")
        plot_lag_profile(corr_tbl, features, OUT_FIGS / "lag_profile_all_features.png")

    print("Plotting scatter(s) at lag+1…")
    for feat in features:
        fname = feat.replace("_share", "").replace("_llm", "")
        plot_scatter(df, feat, "n_ae_t1", OUT_FIGS / f"scatter_{fname}_lag1.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
