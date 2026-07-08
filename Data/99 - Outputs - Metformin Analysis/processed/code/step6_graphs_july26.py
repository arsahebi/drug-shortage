# %%
"""
Step 6 (July 2026 refresh) — Analysis Graphs
=============================================
Reads step5_analysis_panel_july26.csv (336 rows: NDC11 × TestYear).

Figures produced
----------------
  Figure 1  Quality by Country
              Panels: DMF | NDMA | Difference Factor
              Bar height = mean; restricted to IND / CHN / USA

  Figure 2  Volume by Prior Inspection Outcome
              Box-plot (IQVIA extended units, log scale)
              Points jittered and colored by country

  Figure 3  Quality vs Volume scatter
              Rows: DMF | NDMA | Difference Factor
              Columns: one per year (2020 | 2022 | 2024)
              Color = country; Spearman ρ annotation

Figures saved to ~/Desktop/MetforminFigures_July26/
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, kruskal
from itertools import combinations

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.size"]    = 11

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
STEP5   = BASE / "Data/99 - Outputs - Metformin Analysis/processed/step5_analysis_panel_july26.csv"
OUT_DIR = Path.home() / "Desktop" / "MetforminFigures_July26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
COUNTRY_ORDER  = ["IND", "CHN", "USA"]
COUNTRY_LABELS = {"IND": "India", "CHN": "China", "USA": "United States"}
COUNTRY_COLORS = {"IND": "#ef4444", "CHN": "#f59e0b", "USA": "#3b82f6"}

OUTCOME_ORDER  = ["NAI", "VAI", "OAI"]
OUTCOME_COLORS = {"NAI": "#22c55e", "VAI": "#f59e0b", "OAI": "#ef4444"}
OUTCOME_LABELS = {"NAI": "NAI (0)", "VAI": "VAI (1.5)", "OAI": "OAI (3.5)"}

DMF_COL  = "DMF (ng/DAY) Valisure"
NDMA_COL = "NDMA (ng/DAY) Valisure"
DIFF_COL = "Difference Factor"
VOL_COL  = "iqvia_extended_units"

TEST_YEARS = [2020, 2022, 2024]


# ── helpers ────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, format=ext, bbox_inches="tight", dpi=150)
    print(f"  Saved → {OUT_DIR / name}.pdf / .png")


def _n_label(ax, x_pos, n_values, y_frac=0.03, fontsize=9) -> None:
    """Place 'n=X' below each category on the x-axis in axes coordinates."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, n in zip(x_pos, n_values):
        ax.text(x, y_frac, f"n={int(n)}", transform=trans,
                ha="center", va="bottom", fontsize=fontsize, color="#374151")


def _block_bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                               clusters: np.ndarray,
                               n_boot: int = 2000, seed: int = 42) -> dict:
    """
    NDC-cluster block bootstrap for Spearman ρ.
    Resamples whole NDC clusters (same NDC across test years) with replacement.
    Adapted from old statistical_tests.py.
    Returns dict with rho, p_naive, p_boot, ci_lo, ci_hi, n_obs, n_clusters.
    """
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym, cm = x[mask], y[mask], np.asarray(clusters, dtype=str)[mask]

    rho_obs, p_obs = spearmanr(xm, ym)
    unique_cl = np.unique(cm)
    n_cl = len(unique_cl)

    boot_rhos = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_cl, size=n_cl, replace=True)
        idx = np.concatenate([np.where(cm == c)[0] for c in sampled])
        if len(idx) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = spearmanr(xm[idx], ym[idx])
        if np.isfinite(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 10:
        return {"rho": rho_obs, "p_naive": p_obs, "p_boot": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "n_obs": int(mask.sum()), "n_clusters": n_cl}

    boot_rhos = np.array(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    shifted = boot_rhos - np.mean(boot_rhos)
    p_boot = float(np.mean(np.abs(shifted) >= abs(rho_obs)))
    p_boot = max(p_boot, 1.0 / n_boot)

    return {"rho": rho_obs, "p_naive": p_obs, "p_boot": p_boot,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_obs": int(mask.sum()), "n_clusters": n_cl}


def _spearman_annotation(ax, x, y, ndc_clusters=None, fontsize=9) -> None:
    """
    Annotate scatter plot with Spearman ρ.
    If ndc_clusters provided: shows NDC-cluster bootstrap p and 95% CI.
    Otherwise falls back to plain Spearman.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 5:
        ax.text(0.04, 0.96, f"n={n} (too few)", transform=ax.transAxes,
                va="top", fontsize=fontsize, color="#6b7280")
        return

    if ndc_clusters is not None:
        res = _block_bootstrap_spearman(x, y, ndc_clusters)
        rho = res["rho"]
        p_b = res["p_boot"]
        ci_lo, ci_hi = res["ci_lo"], res["ci_hi"]
        sig = "**" if (np.isfinite(p_b) and p_b < 0.01) else (
              "*"  if (np.isfinite(p_b) and p_b < 0.05) else (
              "."  if (np.isfinite(p_b) and p_b < 0.10) else ""))
        if np.isfinite(p_b):
            p_str = f"p={p_b:.3f}" if p_b >= 0.001 else "p<0.001"
        else:
            p_str = "p=n/a"
        ci_str = (f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"
                  if np.isfinite(ci_lo) else "")
        txt = f"ρ={rho:+.2f} {ci_str}\n{p_str} {sig}\nn={n}"
    else:
        rho, p = spearmanr(x[mask], y[mask])
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        txt = f"ρ={rho:+.2f}\n{p_str}\nn={n}"

    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7))


def _dunn_posthoc(groups: dict, adjust: str = "bonferroni") -> pd.DataFrame:
    """Dunn (1964) pairwise post-hoc after Kruskal-Wallis (Bonferroni by default)."""
    labels = list(groups.keys())
    all_vals, all_grp = [], []
    for lbl, vals in groups.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        all_vals.extend(v.tolist())
        all_grp.extend([lbl] * len(v))

    combined = np.array(all_vals, dtype=float)
    grp_arr  = np.array(all_grp)
    rnks     = stats.rankdata(combined)
    n_total  = len(combined)
    _, tie_counts = np.unique(combined, return_counts=True)
    T_ties = np.sum(tie_counts ** 3 - tie_counts)

    rows = []
    for g1, g2 in combinations(labels, 2):
        n1 = int((grp_arr == g1).sum())
        n2 = int((grp_arr == g2).sum())
        mr1 = float(rnks[grp_arr == g1].mean())
        mr2 = float(rnks[grp_arr == g2].mean())
        se = np.sqrt((n_total * (n_total + 1) / 12.0
                      - T_ties / (12.0 * (n_total - 1)))
                     * (1.0 / n1 + 1.0 / n2))
        se = max(se, 1e-12)
        z = (mr1 - mr2) / se
        p_raw = 2 * stats.norm.sf(abs(z))
        rows.append({"group1": g1, "group2": g2,
                     "z": round(z, 3), "p_raw": round(p_raw, 5)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    k = len(result)
    if adjust == "bonferroni":
        result["p_adj"] = np.minimum(result["p_raw"].values * k, 1.0).round(5)
    else:
        result["p_adj"] = result["p_raw"]
    result["sig"] = result["p_adj"].apply(
        lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "")))
    return result


def _kruskal_p(groups: dict) -> float | None:
    """Kruskal-Wallis p-value across groups (filtered to ≥2 groups with n≥2)."""
    valid = {k: np.array(v, dtype=float) for k, v in groups.items()
             if len([x for x in v if np.isfinite(x)]) >= 2}
    if len(valid) < 2:
        return None
    arrays = [v[np.isfinite(v)] for v in valid.values()]
    try:
        _, p = kruskal(*arrays)
        return p
    except Exception:
        return None


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading step5 panel...")
df = pd.read_csv(STEP5, dtype=str)
for col in [DMF_COL, NDMA_COL, DIFF_COL, VOL_COL,
            "iqvia_trx", "sdud_num_prescriptions", "sdud_units_reimbursed",
            "prior_score", "n_lots"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df["TestYear"] = pd.to_numeric(df["TestYear"], errors="coerce").astype("Int64")
print(f"  {len(df):,} rows | {df['NDC11'].nunique()} NDC11s")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Quality by Country (bar chart, IND / CHN / USA)
# Panels: DMF | NDMA | Difference Factor
# Each bar = mean across tested NDC11s in that country
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig1_quality_by_country() -> None:
    print("\nPlotting Figure 1 — Quality by Country...")

    metrics = [
        (DMF_COL,  "DMF (ng/day)",          "{:,.0f}", [2020, 2022, 2024]),
        (NDMA_COL, "NDMA (ng/day)",         "{:,.1f}", [2020, 2022]),
        (DIFF_COL, "Difference Factor",      "{:.3f}",  [2024]),
    ]

    bar_color  = "#93c5fd"
    edge_color = "#2563eb"
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4.8))

    d_core = df[df["CountryCode"].isin(COUNTRY_ORDER)].copy()

    for ax, (col, ylabel, fmt, years) in zip(axes, metrics):
        sub = d_core[d_core["TestYear"].isin(years) & d_core[col].notna()].copy()

        g = (
            sub.groupby("CountryCode", as_index=False)
            .agg(mean=(col, "mean"), n=(col, "count"))
        )
        g["CountryCode"] = pd.Categorical(g["CountryCode"], categories=COUNTRY_ORDER, ordered=True)
        g = g.sort_values("CountryCode").reset_index(drop=True)

        x = np.arange(len(g))
        bars = ax.bar(x, g["mean"], color=bar_color, edgecolor=edge_color,
                      linewidth=1.0, zorder=2)

        # value labels on top of bars
        for bar, val in zip(bars, g["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in g["CountryCode"]])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        # n= labels below x-axis ticks
        _n_label(ax, x, g["n"], y_frac=0.02)

        # Kruskal-Wallis p
        groups = {cc: sub.loc[sub["CountryCode"] == cc, col].dropna().values
                  for cc in COUNTRY_ORDER}
        p = _kruskal_p(groups)
        if p is not None:
            p_str = f"KW p={p:.3f}" if p >= 0.001 else "KW p<0.001"
            ax.set_title(f"{p_str}", fontsize=9, color="#374151")

        years_str = "+".join(str(y) for y in years)
        ax.set_xlabel(f"Manufacturing Country  (years tested: {years_str})")

    plt.suptitle("Figure 1 — Mean Contamination by Manufacturing Country",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "Figure1_Quality_by_Country")
    plt.close(fig)
    print_fig1_stats(d_core)


def print_fig1_stats(d_core: pd.DataFrame) -> None:
    print("\n── Figure 1 statistics ──")
    for col, label, years in [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]:
        sub = d_core[d_core["TestYear"].isin(years) & d_core[col].notna()]
        groups = {cc: sub.loc[sub["CountryCode"] == cc, col].dropna().values
                  for cc in COUNTRY_ORDER if (sub["CountryCode"] == cc).any()}
        p = _kruskal_p(groups)
        print(f"  {label}: KW p={p:.4f}" if p is not None else f"  {label}: KW n/a")
        for cc in COUNTRY_ORDER:
            vals = sub.loc[sub["CountryCode"] == cc, col].dropna()
            if len(vals):
                print(f"    {cc}: n={len(vals)}  mean={vals.mean():.2f}  median={vals.median():.2f}")
        if p is not None and len(groups) >= 2:
            dunn = _dunn_posthoc(groups)
            if not dunn.empty:
                print(f"    Dunn post-hoc (Bonferroni):")
                print(dunn[["group1","group2","z","p_raw","p_adj","sig"]].to_string(index=False, col_space=10))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Volume by Prior Inspection Outcome
# Box plot (log scale) with jittered points colored by country
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig2_volume_by_outcome() -> None:
    print("\nPlotting Figure 2 — Volume by Prior Inspection Outcome...")

    sub = df[
        df["prior_outcome"].notna() &
        df[VOL_COL].notna() &
        (df[VOL_COL] > 0) &
        df["CountryCode"].isin(COUNTRY_ORDER)
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos = {out: i for i, out in enumerate(OUTCOME_ORDER)}
    n_vals = []

    for out in OUTCOME_ORDER:
        d_out = sub[sub["prior_outcome"] == out]
        n_vals.append(len(d_out))
        xi = x_pos[out]

        # box plot data
        vals = d_out[VOL_COL].values
        if len(vals) > 0:
            bp = ax.boxplot(vals, positions=[xi], widths=0.45,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor="#e0e7ff", color="#4f46e5"),
                            medianprops=dict(color="#1e1b4b", linewidth=2),
                            whiskerprops=dict(color="#4f46e5"),
                            capprops=dict(color="#4f46e5"))

        # jittered points by country
        rng = np.random.default_rng(42)
        for cc in COUNTRY_ORDER:
            d_cc = d_out[d_out["CountryCode"] == cc]
            if d_cc.empty:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(d_cc))
            ax.scatter(xi + jitter, d_cc[VOL_COL].values,
                       c=COUNTRY_COLORS[cc], s=40, alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER])
    ax.set_xlabel("Prior Inspection Outcome (prior_score)")
    ax.set_ylabel("IQVIA Extended Units (log scale)")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    _n_label(ax, list(x_pos.values()), n_vals, y_frac=0.01)

    # Kruskal-Wallis p
    groups = {out: sub.loc[sub["prior_outcome"] == out, VOL_COL].dropna().values
              for out in OUTCOME_ORDER}
    p = _kruskal_p(groups)
    if p is not None:
        p_str = f"KW p={p:.3f}" if p >= 0.001 else "KW p<0.001"
        ax.set_title(f"Market Volume by FDA Inspection Outcome  ({p_str})",
                     fontsize=11, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=COUNTRY_COLORS[cc], label=COUNTRY_LABELS[cc],
               markeredgecolor="white", markeredgewidth=0.5, markersize=8)
        for cc in COUNTRY_ORDER
    ]
    ax.legend(handles=legend_handles, title="Country", loc="upper right")
    plt.tight_layout()
    _save(fig, "Figure2_Volume_by_Outcome")
    plt.close(fig)
    print_fig2_stats(sub)


def print_fig2_stats(sub: pd.DataFrame) -> None:
    print("\n── Figure 2 statistics ──")
    groups = {out: sub.loc[sub["prior_outcome"] == out, VOL_COL].dropna().values
              for out in OUTCOME_ORDER}
    p = _kruskal_p(groups)
    print(f"  Kruskal-Wallis p = {p:.4f}" if p is not None else "  KW n/a")
    for out in OUTCOME_ORDER:
        vals = groups[out]
        if len(vals):
            print(f"  {out}: n={len(vals)}  median={np.median(vals):,.0f}  mean={np.mean(vals):,.0f}")
    print("  Pairwise Mann-Whitney:")
    for a, b in combinations(OUTCOME_ORDER, 2):
        va, vb = groups.get(a, []), groups.get(b, [])
        if len(va) >= 2 and len(vb) >= 2:
            from scipy.stats import mannwhitneyu
            _, p_mw = mannwhitneyu(va, vb, alternative="two-sided")
            print(f"    {a} vs {b}: p={p_mw:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Quality vs Volume scatter
# Rows: DMF | NDMA | Difference Factor
# Columns: by test year (only years where that metric is non-null)
# Color: manufacturing country
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig3_quality_vs_volume() -> None:
    print("\nPlotting Figure 3 — Quality vs Volume scatter...")

    metric_rows = [
        (DMF_COL,  "DMF (ng/day)",          [2020, 2022, 2024], "symlog", 1.0),
        (NDMA_COL, "NDMA (ng/day)",         [2020, 2022],       "symlog", 1.0),
        (DIFF_COL, "Difference Factor",      [2024],             "linear", None),
    ]

    n_rows = len(metric_rows)
    # one column per year across all metrics
    all_years = sorted({yr for _, _, years, _, _ in metric_rows for yr in years})
    n_cols = len(all_years)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows),
                             squeeze=False)

    d_core = df[df["CountryCode"].isin(COUNTRY_ORDER) & df[VOL_COL].notna() & (df[VOL_COL] > 0)].copy()

    for row_i, (qcol, ylabel, years, xscale, linthresh) in enumerate(metric_rows):
        for col_j, yr in enumerate(all_years):
            ax = axes[row_i][col_j]
            if yr not in years:
                ax.axis("off")
                continue

            sub = d_core[
                (d_core["TestYear"] == yr) &
                d_core[qcol].notna()
            ].copy()

            for cc in COUNTRY_ORDER:
                d_cc = sub[sub["CountryCode"] == cc]
                if d_cc.empty:
                    continue
                ax.scatter(d_cc[qcol].values, d_cc[VOL_COL].values,
                           s=55, alpha=0.70,
                           c=COUNTRY_COLORS[cc],
                           edgecolor="white", linewidth=0.4, zorder=3)

            # axes scale
            if xscale == "symlog" and linthresh is not None:
                ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_yscale("log")

            ax.set_xlabel(ylabel)
            ax.set_ylabel("IQVIA Extended Units" if col_j == 0 else "")
            ax.set_title(f"Year {yr}", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            _spearman_annotation(ax, sub[qcol].values.astype(float),
                                 sub[VOL_COL].values.astype(float),
                                 ndc_clusters=sub["NDC11"].values)

    # shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=COUNTRY_COLORS[cc], label=COUNTRY_LABELS[cc],
               markeredgecolor="white", markeredgewidth=0.5, markersize=9)
        for cc in COUNTRY_ORDER
    ]
    fig.legend(handles=legend_handles, title="Country",
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, framealpha=0.9)
    fig.suptitle("Figure 3 — Quality Metrics vs Market Volume",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "Figure3_Quality_vs_Volume")
    plt.close(fig)
    print_fig3_stats(d_core)


def print_fig3_stats(d_core: pd.DataFrame) -> None:
    print("\n── Figure 3 Spearman ρ (NDC-cluster bootstrap, 2000 resamples) ──")
    metric_rows = [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]
    for qcol, label, years in metric_rows:
        sub = d_core[d_core["TestYear"].isin(years) & d_core[qcol].notna() & d_core[VOL_COL].notna()].copy()
        x = sub[qcol].values.astype(float)
        y = sub[VOL_COL].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            res = _block_bootstrap_spearman(x, y, sub["NDC11"].values)
            sig = "**" if (np.isfinite(res["p_boot"]) and res["p_boot"] < 0.01) else (
                  "*"  if (np.isfinite(res["p_boot"]) and res["p_boot"] < 0.05) else "")
            print(f"  {label} (all years pooled): "
                  f"ρ={res['rho']:+.3f} [{res['ci_lo']:+.3f},{res['ci_hi']:+.3f}]  "
                  f"p_boot={res['p_boot']:.4f}{sig}  p_naive={res['p_naive']:.4f}  "
                  f"n={res['n_obs']} (k={res['n_clusters']} NDCs)")
        for yr in years:
            sy = sub[sub["TestYear"] == yr]
            xyr = sy[qcol].values.astype(float)
            yyr = sy[VOL_COL].values.astype(float)
            m = np.isfinite(xyr) & np.isfinite(yyr)
            if m.sum() >= 5:
                res = _block_bootstrap_spearman(xyr, yyr, sy["NDC11"].values)
                sig = "**" if (np.isfinite(res["p_boot"]) and res["p_boot"] < 0.01) else (
                      "*"  if (np.isfinite(res["p_boot"]) and res["p_boot"] < 0.05) else "")
                print(f"    {yr}: ρ={res['rho']:+.3f} [{res['ci_lo']:+.3f},{res['ci_hi']:+.3f}]  "
                      f"p_boot={res['p_boot']:.4f}{sig}  n={res['n_obs']} (k={res['n_clusters']} NDCs)")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Prior Outcome vs Quality (box + jitter by country)
# Rows: DMF | NDMA | Difference Factor
# Shows distribution of quality metric within NAI / VAI / OAI groups
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig4_outcome_vs_quality() -> None:
    print("\nPlotting Figure 4 — Prior Inspection Outcome vs Quality...")

    metrics = [
        (DMF_COL,  "DMF (ng/day)",           [2020, 2022, 2024], "symlog", 1.0),
        (NDMA_COL, "NDMA (ng/day)",          [2020, 2022],       "symlog", 1.0),
        (DIFF_COL, "Difference Factor",       [2024],             "linear", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    d_core = df[df["CountryCode"].isin(COUNTRY_ORDER) & df["prior_outcome"].notna()].copy()
    rng = np.random.default_rng(42)

    for ax, (qcol, ylabel, years, yscale, linthresh) in zip(axes, metrics):
        sub = d_core[d_core["TestYear"].isin(years) & d_core[qcol].notna()].copy()

        n_vals = []
        for xi, out in enumerate(OUTCOME_ORDER):
            d_out = sub[sub["prior_outcome"] == out]
            n_vals.append(len(d_out))
            vals = d_out[qcol].dropna().values
            if len(vals) == 0:
                continue

            ax.boxplot(vals, positions=[xi], widths=0.45,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#e0e7ff", color="#4f46e5"),
                       medianprops=dict(color="#1e1b4b", linewidth=2),
                       whiskerprops=dict(color="#4f46e5"),
                       capprops=dict(color="#4f46e5"))

            for cc in COUNTRY_ORDER:
                d_cc = d_out[d_out["CountryCode"] == cc]
                if d_cc.empty:
                    continue
                jitter = rng.uniform(-0.15, 0.15, size=len(d_cc))
                ax.scatter(xi + jitter, d_cc[qcol].values,
                           c=COUNTRY_COLORS[cc], s=40, alpha=0.75,
                           edgecolor="white", linewidth=0.4, zorder=3)

        if yscale == "symlog" and linthresh is not None:
            ax.set_yscale("symlog", linthresh=linthresh)

        ax.set_xticks(range(len(OUTCOME_ORDER)))
        ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Prior Inspection Outcome")
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        _n_label(ax, range(len(OUTCOME_ORDER)), n_vals, y_frac=0.01)

        # Kruskal-Wallis p
        groups = {out: sub.loc[sub["prior_outcome"] == out, qcol].dropna().values
                  for out in OUTCOME_ORDER}
        p = _kruskal_p(groups)
        if p is not None:
            p_str = f"KW p={p:.3f}" if p >= 0.001 else "KW p<0.001"
            ax.set_title(f"{ylabel}\n{p_str}", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=COUNTRY_COLORS[cc], label=COUNTRY_LABELS[cc],
               markeredgecolor="white", markeredgewidth=0.5, markersize=8)
        for cc in COUNTRY_ORDER
    ]
    fig.legend(handles=legend_handles, title="Country",
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, framealpha=0.9)
    fig.suptitle("Figure 4 — Quality Metrics by Prior Inspection Outcome",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "Figure4_Outcome_vs_Quality")
    plt.close(fig)
    print_fig4_stats(d_core)


def print_fig4_stats(d_core: pd.DataFrame) -> None:
    print("\n── Figure 4 Kruskal-Wallis ──")
    for qcol, label, years in [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]:
        sub = d_core[d_core["TestYear"].isin(years) & d_core[qcol].notna()]
        groups = {out: sub.loc[sub["prior_outcome"] == out, qcol].dropna().values
                  for out in OUTCOME_ORDER}
        p = _kruskal_p(groups)
        print(f"  {label}: KW p={p:.4f}" if p is not None else f"  {label}: KW n/a")
        for out in OUTCOME_ORDER:
            vals = groups[out]
            if len(vals):
                print(f"    {out}: n={len(vals)}  mean={np.mean(vals):.2f}  median={np.median(vals):.2f}")
        if p is not None:
            dunn = _dunn_posthoc({k: v for k, v in groups.items() if len(v) >= 2})
            if not dunn.empty:
                print(f"    Dunn post-hoc (Bonferroni):")
                print(dunn[["group1","group2","z","p_raw","p_adj","sig"]].to_string(index=False, col_space=10))


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
plot_fig1_quality_by_country()
plot_fig2_volume_by_outcome()
plot_fig3_quality_vs_volume()
plot_fig4_outcome_vs_quality()

print(f"\nAll figures saved to: {OUT_DIR}")
# %%
