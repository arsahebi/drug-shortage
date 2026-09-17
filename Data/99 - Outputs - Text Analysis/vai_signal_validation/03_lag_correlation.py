"""
03_lag_correlation.py
----------------------------------------------------------------------------
Rebuild of the INFORMS 2026 "Correlation Heatmap" backup slide (all 17 text
features x AE lags), using the CURRENT validated pipeline instead of the
one the INFORMS slides were built on.

Original: old_not_current_pipeline/pdf_source_and_superseded/ae_validation/
02_lag_correlation.py (unmodified code, moved 2026-09-16). Ported here with
the v2 feature renames (severity_majmod_share instead of
severity_critmajor_share -- see LLM Extraction Validation Report, Section 2
-- and the FDA six-system Lab Controls/Quality System column names), and
reading the panel built by 01_build_inspection_panel.py, which already uses
the human-eval-validated text extraction and the FDA-primary OAI/VAI source.

Method: for each text feature x AE lag pair, Spearman rank correlation
across all inspection-event rows (min 10 non-null pairs). No model, no
train/test split -- a bivariate rank statistic. Same interpretation as the
original: rho > 0 means more of this signal associates with more AEs at
that lag; a stronger correlation at Q+4 than Q0 supports a predictive
reading over a purely reactive one.

Usage
-----
  python 03_lag_correlation.py                  drug-level AE panel
  python 03_lag_correlation.py --anda-ae         ANDA-specific AE panel (matches the slide's n=148 caption)

Outputs
-------
  outputs/tables/lag_correlation_table.csv
  outputs/figures/lag_correlation_heatmap.png
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

HERE = Path(__file__).resolve().parent
OUT      = HERE / "outputs"
OUT_TABS = OUT / "tables"
OUT_FIGS = OUT / "figures"
PANEL      = OUT / "fei_ae_panel_inspection_centered.parquet"
PANEL_ANDA = OUT / "fei_ae_panel_inspection_centered_anda.parquet"

# v2 schema: severity_majmod_share replaces severity_critmajor_share (the
# INFORMS-era feature); vc_laboratorycontrolssystem_share and
# n_laboratorycontrolssystem_obs replace the old vc_labcontrols_share /
# n_labcontrols_obs names. All other 14 features are unchanged.
TEXT_FEATURES = [
    "n_laboratorycontrolssystem_obs",
    "vc_laboratorycontrolssystem_share",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "data_integrity_llm_share",
    "severity_majmod_share",
    "vc_qualitysystem_share",
    "cultural_root_cause_share",
    "joint_qualitysystem_production",
    "contamination_llm_share",
    "n_qualitysystem_obs",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "joint_labcontrols_qualitysystem",
    "multi_domain_insp",
]

FEATURE_LABELS = {
    "severity_majmod_share":           "Major/Moderate severity",
    "contamination_llm_share":         "Contamination flag",
    "data_integrity_llm_share":        "Data integrity flag",
    "patient_risk_llm_share":          "Patient risk flag",
    "investigation_llm_share":         "Investigation failure flag",
    "repeat_cross_insp_share":         "Repeat observations",
    "scope_facilitywide_share":        "Facility-wide scope",
    "cultural_root_cause_share":       "Cultural root cause",
    "vc_laboratorycontrolssystem_share": "Lab controls (share)",
    "vc_qualitysystem_share":          "Quality system (share)",
    "n_laboratorycontrolssystem_obs":  "Lab controls (count)",
    "n_qualitysystem_obs":             "Quality system (count)",
    "joint_labcontrols_dataintegrity": "Lab ctrl + DI (joint)",
    "joint_contamination_labcontrols": "Contamination + Lab ctrl (joint)",
    "joint_qualitysystem_production":  "Quality sys + Production (joint)",
    "joint_labcontrols_qualitysystem": "Lab ctrl + Quality sys (joint)",
    "multi_domain_insp":               "Multi-domain (>=3 domains)",
}

AE_LAGS = {
    "n_ae_t0":  "Q0 (inspection qtr)",
    "n_ae_tp1": "Q+1 (0-3 mo after)",
    "n_ae_tp2": "Q+2 (3-6 mo after)",
    "n_ae_tp4": "Q+4 (9-12 mo after)",
}


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


def build_correlation_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in df.columns:
            print(f"  [WARN] feature not in panel: {feat} -- skipping")
            continue
        for ae_col, lag_label in AE_LAGS.items():
            r, p = _spearman(df[feat], df[ae_col])
            n = (df[feat].notna() & df[ae_col].notna()).sum()
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


def plot_heatmap(corr_tbl: pd.DataFrame, out_path: Path, top_n: int = 12, n_label: str = "") -> None:
    pivot   = corr_tbl.pivot(index="feature_label", columns="lag_label", values="spearman_r")
    pivot_p = corr_tbl.pivot(index="feature_label", columns="lag_label", values="sig")

    col_order = list(AE_LAGS.values())
    pivot   = pivot.reindex(columns=col_order)
    pivot_p = pivot_p.reindex(columns=col_order)

    sort_lag  = col_order[-1]
    lag_vals  = corr_tbl[corr_tbl["lag_label"] == sort_lag].set_index("feature_label")["spearman_r"]
    top_index = lag_vals.abs().sort_values(ascending=False).head(top_n).index
    pivot     = pivot.loc[top_index]
    pivot_p   = pivot_p.loc[top_index]

    short_labels = [v.split(" ")[0] for v in col_order]

    plt.rcParams.update({"font.family": "sans-serif"})
    n_rows = len(pivot)
    fig, ax = plt.subplots(figsize=(8, max(4, n_rows * 0.52)), constrained_layout=True)

    vals = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmax = max(abs(finite).max() if len(finite) else 0.05, 0.05)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)

    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i, row_feat in enumerate(pivot.index):
        for j, col_name in enumerate(col_order):
            val = pivot.loc[row_feat, col_name]
            sig = pivot_p.loc[row_feat, col_name]
            if np.isfinite(val):
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:+.2f}{sig}", ha="center", va="center",
                        fontsize=8.5, color=color)

    n_str = f", n={n_label}" if n_label else ""
    ax.set_title(
        "Spearman rho: 483 text signals vs. FAERS AE counts by lag after inspection\n"
        f"* p<.05   ** p<.01   *** p<.001   (top 12 by |rho| at Q+4{n_str})",
        fontsize=10, pad=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  Saved heatmap -> {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anda-ae", dest="anda_ae", action="store_true",
                        help="Use ANDA-specific AE panel (matches the INFORMS slide's n=148 caption).")
    args = parser.parse_args()
    panel_path = PANEL_ANDA if args.anda_ae else PANEL

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}\nRun 01_build_inspection_panel.py first.")

    print(f"Loading panel ({'ANDA-specific' if args.anda_ae else 'drug-level'})...")
    df = pd.read_parquet(panel_path)
    print(f"  {len(df)} rows, {df['fei'].nunique()} FEIs")

    print(f"\nComputing Spearman correlations for {len(TEXT_FEATURES)} features...")
    corr_tbl = build_correlation_table(df, TEXT_FEATURES)

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    corr_tbl.to_csv(OUT_TABS / "lag_correlation_table.csv", index=False)
    print(f"  Saved -> {OUT_TABS / 'lag_correlation_table.csv'}")

    print("\nCorrelations at Q+4 / 1-year-after (sorted by |rho|):")
    top = (corr_tbl[corr_tbl["lag"] == "n_ae_tp4"]
           .sort_values("spearman_r", key=abs, ascending=False)
           [["feature_label", "spearman_r", "p_value", "sig", "n"]])
    print(top.to_string(index=False))

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    n_obs = int(corr_tbl["n"].median())
    plot_heatmap(corr_tbl, OUT_FIGS / "lag_correlation_heatmap.png", n_label=str(n_obs))

    print("\nDone.")


if __name__ == "__main__":
    main()
