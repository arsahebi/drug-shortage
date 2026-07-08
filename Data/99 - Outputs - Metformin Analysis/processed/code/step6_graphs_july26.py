# %%
from __future__ import annotations
"""
Step 6 (July 2026 refresh) — Analysis Graphs + Statistical Models
==================================================================
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
              Color = country; NDC-cluster bootstrap Spearman ρ

  Figure 4  Quality by Prior Inspection Outcome
              Box + jitter; DMF | NDMA | Difference Factor

Statistical models (after figures)
-----------------------------------
  Country (Fig 1): log(metric) ~ IND + CHN + (1|NDC11)
                   Random NDC intercept (MixedLM) + CGM two-way clustered SE (NDC × FEI)
  Inspection outcome (Fig 2/4): log(DMF) ~ VAI + OAI + (1|NDC11)
                   Same approach; reference = NAI; additional test OAI vs VAI

Figures and model outputs saved to:
  Data/99 - Outputs - Metformin Analysis/processed/outputs/
"""

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
OUT_DIR = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs"
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
# Figure 1 — Market Outcomes by Prior Inspection Outcome (2 panels)
# Left:  NADAC price per unit (blank — not yet in current pipeline)
# Right: IQVIA annual volume (box + jitter by country, log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig1_market_by_outcome() -> None:
    print("\nPlotting Figure 1 — Market Outcomes by Prior Inspection Outcome...")

    sub = df[
        df["prior_outcome"].notna() &
        df[VOL_COL].notna() &
        (df[VOL_COL] > 0) &
        df["CountryCode"].isin(COUNTRY_ORDER)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left panel: NADAC Price (not in pipeline) ─────────────────────────────
    ax_price = axes[0]
    ax_price.set_xlim(-0.5, 2.5)
    ax_price.set_ylim(0.01, 100)
    ax_price.set_yscale("log")
    ax_price.set_xticks([0, 1, 2])
    ax_price.set_xticklabels(["NAI (0)", "VAI (1.5)", "OAI (3.5)"])
    ax_price.text(0.5, 0.5,
                  "NADAC price data\nnot available\nin current pipeline",
                  transform=ax_price.transAxes, ha="center", va="center",
                  fontsize=12, color="#9ca3af",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb",
                            edgecolor="#e5e7eb", alpha=0.9))
    ax_price.set_xlabel("Prior Inspection Outcome (prior_score)")
    ax_price.set_ylabel("Price per Unit ($/unit)")
    ax_price.set_title("Market Price by FDA Inspection Outcome", fontsize=11, fontweight="bold")
    ax_price.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # ── Right panel: IQVIA Volume ─────────────────────────────────────────────
    ax_vol = axes[1]
    x_pos  = {out: i for i, out in enumerate(OUTCOME_ORDER)}
    n_vals = []
    rng    = np.random.default_rng(42)

    for out in OUTCOME_ORDER:
        d_out = sub[sub["prior_outcome"] == out]
        n_vals.append(len(d_out))
        xi   = x_pos[out]
        vals = d_out[VOL_COL].values
        if len(vals) > 0:
            ax_vol.boxplot(vals, positions=[xi], widths=0.45,
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
            ax_vol.scatter(xi + jitter, d_cc[VOL_COL].values,
                           c=COUNTRY_COLORS[cc], s=40, alpha=0.75,
                           edgecolor="white", linewidth=0.4, zorder=3)

    ax_vol.set_yscale("log")
    ax_vol.set_xticks(list(x_pos.values()))
    ax_vol.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER])
    ax_vol.set_xlabel("Prior Inspection Outcome (prior_score)")
    ax_vol.set_ylabel("IQVIA Extended Units (log scale)")
    ax_vol.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax_vol.set_axisbelow(True)
    _n_label(ax_vol, list(x_pos.values()), n_vals, y_frac=0.01)

    groups = {out: sub.loc[sub["prior_outcome"] == out, VOL_COL].dropna().values
              for out in OUTCOME_ORDER}
    p = _kruskal_p(groups)
    if p is not None:
        p_str = f"KW p={p:.3f}" if p >= 0.001 else "KW p<0.001"
        ax_vol.set_title(f"Market Volume by FDA Inspection Outcome  ({p_str})",
                         fontsize=11, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=COUNTRY_COLORS[cc], label=COUNTRY_LABELS[cc],
               markeredgecolor="white", markeredgewidth=0.5, markersize=8)
        for cc in COUNTRY_ORDER
    ]
    ax_vol.legend(handles=legend_handles, title="Country", loc="upper right")

    fig.suptitle(
        "Figure 1 — Relationship between Market Outcomes and Prior FDA Inspection Outcome",
        fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "Figure1_Market_by_Outcome")
    plt.close(fig)
    print_fig1_stats(sub)


def _bootstrap_pairwise(sub: pd.DataFrame, val_col: str, group_col: str,
                         cluster_col: str, groups: list, n_boot: int = 2000) -> None:
    """Print cluster-bootstrap pairwise comparisons between group pairs."""
    sub = sub[[val_col, group_col, cluster_col]].dropna().copy()
    for g1, g2 in combinations(groups, 2):
        d1 = sub[sub[group_col] == g1]
        d2 = sub[sub[group_col] == g2]
        if len(d1) < 3 or len(d2) < 3:
            print(f"    {g1} vs {g2}: insufficient data (n1={len(d1)}, n2={len(d2)})")
            continue
        combined = pd.concat([d1[[val_col, cluster_col]], d2[[val_col, cluster_col]]])
        dummy = np.array([0] * len(d1) + [1] * len(d2), dtype=float)
        res = _block_bootstrap_spearman(
            combined[val_col].values.astype(float), dummy,
            combined[cluster_col].values, n_boot=n_boot)
        sig = " **" if res["p_boot"] < 0.01 else (" *" if res["p_boot"] < 0.05 else "")
        print(f"    {g1} vs {g2}:  n_obs={res['n_obs']}  n_clusters={res['n_clusters']}  "
              f"p_naive={res['p_naive']:.4f}  p_boot_clustered={res['p_boot']:.4f}{sig}")


def print_fig1_stats(sub: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FIGURE 1 – Volume by Prior Inspection Outcome")
    print("=" * 80)
    groups = {out: sub.loc[sub["prior_outcome"] == out, VOL_COL].dropna().values
              for out in OUTCOME_ORDER}
    p = _kruskal_p(groups)

    print("\nMARKET VOLUME — IQVIA Extended Units:")
    print(f"  {'Outcome':>8s}  {'n':>5s}  {'mean':>15s}  {'median':>15s}  {'p25':>15s}  {'p75':>15s}")
    print(f"  {'-'*75}")
    for out in OUTCOME_ORDER:
        vals = groups[out]
        if len(vals):
            print(f"  {out:>8s}  {len(vals):>5d}  {np.mean(vals):>15,.0f}  "
                  f"{np.median(vals):>15,.0f}  "
                  f"{np.percentile(vals,25):>15,.0f}  {np.percentile(vals,75):>15,.0f}")

    print(f"\n  [Approach 1: Independent – Kruskal-Wallis + Dunn (Bonferroni)]")
    print(f"    Kruskal-Wallis: p={p:.5f}" if p is not None else "    KW n/a")
    if p is not None and len([k for k, v in groups.items() if len(v) >= 2]) >= 2:
        dunn = _dunn_posthoc({k: v for k, v in groups.items() if len(v) >= 2})
        if not dunn.empty:
            print(dunn[["group1","group2","z","p_raw","p_adj","sig"]].to_string(index=False, col_space=10))

    print(f"\n  [Approach 2: Cluster-robust Bootstrap by NDC11]")
    _bootstrap_pairwise(sub.copy(), VOL_COL, "prior_outcome", "NDC11", OUTCOME_ORDER)

    print(f"\n  [Approach 3: Cluster-robust Bootstrap by FEI (prior_fei)]")
    if "prior_fei" in sub.columns:
        sub_fei = sub[sub["prior_fei"].notna()].copy()
        _bootstrap_pairwise(sub_fei, VOL_COL, "prior_outcome", "prior_fei", OUTCOME_ORDER)
    else:
        print("    prior_fei column not available")

    print("  [PRICE: NADAC data not yet in pipeline — price statistics pending]")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Market Volume vs Tested Drug Quality (3 panels, all years pooled)
# Panels: DMF | NDMA | Difference Factor
# Color = country; NDC-cluster bootstrap Spearman ρ; red dashed trend line
# ═══════════════════════════════════════════════════════════════════════════════
def _add_trend_line(ax, x: np.ndarray, y: np.ndarray,
                    xscale: str, linthresh: float = 1.0) -> None:
    """Fit log10(y) ~ f(x) and draw a red dashed trend line (matches old JAMA code)."""
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    xf, yf = x[mask], y[mask]
    if len(xf) < 3:
        return
    if xscale == "symlog":
        def T(u):
            u = np.asarray(u, dtype=float); out = u.copy()
            big = u >= linthresh; out[big] = linthresh * (1.0 + np.log10(u[big] / linthresh))
            return out
        def Tinv(t):
            t = np.asarray(t, dtype=float); out = t.copy()
            big = t >= linthresh; out[big] = linthresh * 10 ** (t[big] / linthresh - 1.0)
            return out
        Xfit = T(xf)
    else:
        def T(u):    return np.asarray(u, dtype=float)
        def Tinv(t): return np.asarray(t, dtype=float)
        Xfit = xf
    try:
        Yfit = np.log10(yf)
        b, a = np.polyfit(Xfit, Yfit, 1)
        t_line = np.linspace(np.nanmin(Xfit), np.nanmax(Xfit), 200)
        ax.plot(Tinv(t_line), 10 ** (a + b * t_line), "r--", alpha=0.55, linewidth=2, zorder=2)
    except Exception:
        pass


def plot_fig2_volume_vs_quality() -> None:
    print("\nPlotting Figure 2 — Market Volume vs Tested Drug Quality (all years pooled)...")

    d_core = df[df["CountryCode"].isin(COUNTRY_ORDER) & df[VOL_COL].notna() & (df[VOL_COL] > 0)].copy()

    metrics = [
        (DMF_COL,  "DMF (ng/day)",       [2020, 2022, 2024], "symlog", 1.0),
        (NDMA_COL, "NDMA (ng/day)",      [2020, 2022],       "symlog", 1.0),
        (DIFF_COL, "Difference Factor",   [2024],             "linear", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (qcol, xlabel, years, xscale, linthresh) in zip(axes, metrics):
        sub = d_core[d_core["TestYear"].isin(years) & d_core[qcol].notna()].copy()
        for cc in COUNTRY_ORDER:
            d_cc = sub[sub["CountryCode"] == cc]
            if d_cc.empty:
                continue
            ax.scatter(d_cc[qcol].values, d_cc[VOL_COL].values,
                       s=55, alpha=0.65, c=COUNTRY_COLORS[cc],
                       edgecolor="white", linewidth=0.4, zorder=3)
        _add_trend_line(ax, sub[qcol].values.astype(float), sub[VOL_COL].values.astype(float),
                        xscale=xscale, linthresh=linthresh if linthresh is not None else 1.0)
        if xscale == "symlog" and linthresh is not None:
            ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IQVIA Extended Units")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        _spearman_annotation(ax, sub[qcol].values.astype(float),
                             sub[VOL_COL].values.astype(float),
                             ndc_clusters=sub["NDC11"].values)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=COUNTRY_COLORS[cc], label=COUNTRY_LABELS[cc],
               markeredgecolor="white", markeredgewidth=0.5, markersize=9)
        for cc in COUNTRY_ORDER
    ]
    fig.legend(handles=legend_handles, title="Country",
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, framealpha=0.9)
    fig.suptitle("Figure 2 — Market Volume vs Tested Drug Quality",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "Figure2_Volume_vs_Quality")
    plt.close(fig)
    print_fig2_stats(d_core)


def print_fig2_stats(d_core: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FIGURE 2 – Volume vs Quality  (Spearman ρ, NDC-cluster bootstrap, 2000 resamples)")
    print("=" * 80)
    metrics = [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]
    sub_all = d_core[d_core[VOL_COL].notna() & (d_core[VOL_COL] > 0)].copy()
    print(f"\n  Y = volume (IQVIA extended units)")
    for qcol, label, years in metrics:
        sub = sub_all[sub_all["TestYear"].isin(years) & sub_all[qcol].notna()].copy()
        print(f"    {label}:")
        x = sub[qcol].values.astype(float)
        y = sub[VOL_COL].values.astype(float)
        if np.isfinite(x).sum() >= 5:
            res = _block_bootstrap_spearman(x, y, sub["NDC11"].values)
            print(f"      Naive Spearman: rho={res['rho']:+.4f}  p={res['p_naive']:.5f}  n={res['n_obs']}")
            if np.isfinite(res["p_boot"]):
                print(f"      Clustered (NDC bootstrap): rho={res['rho']:+.4f}  "
                      f"p_boot={res['p_boot']:.5f}  "
                      f"95%CI=[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
                      f"n_ndcs={res['n_clusters']}")
        else:
            print(f"      n < 5 — skipped")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Price vs Tested Drug Quality (3 panels — NADAC pending)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig3_price_vs_quality() -> None:
    print("\nPlotting Figure 3 — Price vs Quality (NADAC data not in pipeline)...")

    labels = ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, xlabel in zip(axes, labels):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5,
                "NADAC price data\nnot available\nin current pipeline",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#9ca3af",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb",
                          edgecolor="#e5e7eb", alpha=0.9))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Price per Unit ($/unit)")
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    fig.suptitle("Figure 3 — Price vs Tested Drug Quality  [NADAC data pending]",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "Figure3_Price_vs_Quality")
    plt.close(fig)
    print_fig3_stats()


def print_fig3_stats() -> None:
    print("\n" + "=" * 80)
    print("FIGURE 3 – Price vs Quality")
    print("=" * 80)
    print("  NADAC price data not yet integrated into step5 pipeline.")
    print("  Statistics pending. When available, results should include:")
    print("    - Spearman ρ (naive + NDC-cluster bootstrap) for price vs DMF, NDMA, DiffFactor")
    print("    - Old result: NDMA vs price ρ=+0.282, NDC-clustered p=0.013, 95% CI [+0.056, +0.490]")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Drug Quality by Country of Manufacture (3 bar chart panels)
# Panels: DMF | NDMA | Difference Factor
# Each bar = mean across tested NDC11s in that country
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig4_quality_by_country() -> None:
    print("\nPlotting Figure 4 — Drug Quality by Country of Manufacture...")

    metrics = [
        (DMF_COL,  "DMF (ng/day)",       "{:,.0f}", [2020, 2022, 2024]),
        (NDMA_COL, "NDMA (ng/day)",      "{:,.1f}", [2020, 2022]),
        (DIFF_COL, "Difference Factor",   "{:.3f}",  [2024]),
    ]

    bar_color  = "#93c5fd"
    edge_color = "#2563eb"
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4.8))
    d_core     = df[df["CountryCode"].isin(COUNTRY_ORDER)].copy()

    for ax, (col, ylabel, fmt, years) in zip(axes, metrics):
        sub = d_core[d_core["TestYear"].isin(years) & d_core[col].notna()].copy()
        g = (
            sub.groupby("CountryCode", as_index=False)
            .agg(mean=(col, "mean"), n=(col, "count"))
        )
        g["CountryCode"] = pd.Categorical(g["CountryCode"], categories=COUNTRY_ORDER, ordered=True)
        g = g.sort_values("CountryCode").reset_index(drop=True)

        x    = np.arange(len(g))
        bars = ax.bar(x, g["mean"], color=bar_color, edgecolor=edge_color,
                      linewidth=1.0, zorder=2)
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
        _n_label(ax, x, g["n"], y_frac=0.02)

        groups = {cc: sub.loc[sub["CountryCode"] == cc, col].dropna().values
                  for cc in COUNTRY_ORDER}
        p = _kruskal_p(groups)
        if p is not None:
            p_str = f"KW p={p:.3f}" if p >= 0.001 else "KW p<0.001"
            ax.set_title(p_str, fontsize=9, color="#374151")

        years_str = "+".join(str(y) for y in years)
        ax.set_xlabel(f"Manufacturing Country  (years tested: {years_str})")

    plt.suptitle("Figure 4 — Drug Quality by Country of Manufacture",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "Figure4_Quality_by_Country")
    plt.close(fig)
    print_fig4_stats(d_core)


def print_fig4_stats(d_core: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FIGURE 4 – Quality by Country")
    print("=" * 80)
    for col, label, years in [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]:
        sub = d_core[d_core["TestYear"].isin(years) & d_core[col].notna()]
        print(f"\n  [{label}  years={years}]")
        groups = {cc: sub.loc[sub["CountryCode"] == cc, col].dropna().values
                  for cc in COUNTRY_ORDER if (sub["CountryCode"] == cc).any()}
        for cc in COUNTRY_ORDER:
            vals = sub.loc[sub["CountryCode"] == cc, col].dropna()
            if len(vals):
                print(f"    {cc}:  n={len(vals)}  mean={vals.mean():.3g}  median={vals.median():.3g}")

        print(f"\n  [Approach 1: Independent – Kruskal-Wallis + Dunn (Bonferroni)]")
        p = _kruskal_p(groups)
        print(f"    Kruskal-Wallis: p={p:.5f}" if p is not None else "    KW n/a")
        if p is not None and len(groups) >= 2:
            dunn = _dunn_posthoc(groups)
            if not dunn.empty:
                print(dunn[["group1","group2","z","p_raw","p_adj","sig"]].to_string(index=False, col_space=10))

        print(f"\n  [Approach 2: Cluster-robust Bootstrap by NDC11]")
        _bootstrap_pairwise(sub.copy(), col, "CountryCode", "NDC11", COUNTRY_ORDER)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Models — Model B (Primary): RE + Two-Way Clustered SE
#
# Adapted from old statistical_tests_advanced_models.py.
# Column mapping vs old code:
#   Year → TestYear  |  FEI → prior_fei  |  DMF/NDMA/Dissolution → full col names
#   volume → iqvia_extended_units  |  PriorScore_cat → prior_outcome (NAI/VAI/OAI)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not installed — statistical models skipped")


def _cgm_vcov(y: np.ndarray, X: np.ndarray,
              c1: np.ndarray, c2: np.ndarray,
              beta: np.ndarray | None = None) -> np.ndarray:
    """Cameron-Gelbach-Miller (2011) two-way clustered variance-covariance."""
    n, k = X.shape
    b = np.asarray(beta) if beta is not None else np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)

    def _v(clusters):
        g = len(np.unique(clusters))
        dfc = (g / (g - 1)) * (n / (n - k))
        meat = np.zeros((k, k))
        for c in np.unique(clusters):
            idx = clusters == c
            sc = X[idx].T @ e[idx]
            meat += np.outer(sc, sc)
        return bread @ (dfc * meat) @ bread

    inter = np.array([f"{a}__{b}" for a, b in zip(c1, c2)])
    return _v(c1) + _v(c2) - _v(inter)


def _print_coef_table(names: list, params: np.ndarray, se: np.ndarray,
                      dof: int, header: str = "") -> None:
    from scipy.stats import t as t_dist
    t_vals = params / np.where(se > 0, se, np.nan)
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=max(dof, 1))
    lo = params - 1.96 * se
    hi = params + 1.96 * se
    if header:
        print(f"\n  {header}")
    print(f"  {'Var':>14s}  {'Coef':>10s}  {'SE':>8s}  {'t':>7s}  {'p':>8s}  {'95% CI':>24s}  sig")
    print(f"  {'-'*85}")
    for i, name in enumerate(names):
        if np.isnan(params[i]):
            print(f"  {name:>14s}  (dropped)")
            continue
        sig = "**" if p_vals[i] < 0.01 else ("*" if p_vals[i] < 0.05 else ("." if p_vals[i] < 0.10 else ""))
        ci = f"[{lo[i]:+.4f}, {hi[i]:+.4f}]"
        print(f"  {name:>14s}  {params[i]:>10.4f}  {se[i]:>8.4f}  "
              f"{t_vals[i]:>7.3f}  {p_vals[i]:>8.4f}  {ci:>24s}  {sig}")
    # compact inline format (skip intercept)
    print(f"  {'─'*55}")
    for i, name in enumerate(names):
        if name == "const" or np.isnan(params[i]):
            continue
        sig = "**" if p_vals[i] < 0.01 else ("*" if p_vals[i] < 0.05 else ("." if p_vals[i] < 0.10 else ""))
        p_str = f"p={p_vals[i]:.3f}" if p_vals[i] >= 0.001 else "p<0.001"
        sig_str = f" {sig}" if sig else ""
        print(f"    {name}: β={params[i]:+.3f}, SE={se[i]:.3f}, "
              f"95% CI [{lo[i]:+.3f}, {hi[i]:+.3f}], {p_str}{sig_str}")


def _modelB_re_twoway(sub: pd.DataFrame, y_col: str, dummy_names: list,
                      ndc_col: str, fei_col: str, tag: str) -> None:
    """
    Run Model B: MixedLM with random NDC intercept + CGM two-way SE (NDC × FEI).
    Prints results. Reference category is the omitted dummy (see dummy_names).
    """
    if not HAS_STATSMODELS:
        return

    sub = sub.copy()
    y = sub[y_col].values.astype(float)
    X = sm.add_constant(sub[dummy_names].values.astype(float))
    n_obs = len(sub)
    n_ndc = sub[ndc_col].nunique()
    n_fei = sub[fei_col].nunique() if fei_col in sub.columns else 0
    dof   = max(n_obs - len(dummy_names) - 1, 1)

    print(f"\n  n_obs={n_obs}  n_NDC={n_ndc}  n_FEI={n_fei}")

    # Step 1: MixedLM (random NDC intercept)
    beta_re = None
    try:
        formula = f"{y_col} ~ " + " + ".join(dummy_names)
        mlm = smf.mixedlm(formula, data=sub, groups=sub[ndc_col]).fit(reml=True)
        var_re  = float(mlm.cov_re.iloc[0, 0]) if hasattr(mlm, "cov_re") else 0
        var_res = float(mlm.scale)
        icc = var_re / (var_re + var_res) if (var_re + var_res) > 0 else 0
        print(f"  MixedLM: ICC={icc:.4f}  Var(NDC)={var_re:.4f}  Var(resid)={var_res:.4f}")
        beta_re = np.array([mlm.params.get("Intercept", np.nan)] +
                            [mlm.params.get(d, np.nan) for d in dummy_names])
    except Exception as exc:
        print(f"  MixedLM error: {exc} — falling back to OLS beta")

    # Step 2: CGM two-way SE on RE beta (PRIMARY)
    has_fei = fei_col in sub.columns and n_fei >= 2
    has_ndc = n_ndc >= 2
    if beta_re is not None and not np.any(np.isnan(beta_re)) and has_ndc and has_fei:
        try:
            V2  = _cgm_vcov(y, X, sub[ndc_col].values, sub[fei_col].values, beta=beta_re)
            se2 = np.sqrt(np.diag(V2))
            _print_coef_table(["const"] + dummy_names, beta_re, se2, dof,
                              header=f"★ RE + TWO-WAY clustered SE (NDC×FEI)  [{tag}]  — PRIMARY:")
        except Exception as exc:
            print(f"  CGM error: {exc}")
            beta_re = None

    # Fallback: OLS + two-way clustered SE
    if beta_re is None and has_ndc and has_fei:
        ols = sm.OLS(y, X).fit()
        try:
            V2  = _cgm_vcov(y, X, sub[ndc_col].values, sub[fei_col].values)
            se2 = np.sqrt(np.diag(V2))
            _print_coef_table(["const"] + dummy_names, ols.params, se2, dof,
                              header=f"OLS + TWO-WAY clustered SE  [{tag}]:")
        except Exception as exc:
            print(f"  OLS+CGM error: {exc}")


def run_statistical_models() -> None:
    """
    Model B (Primary) for Figure 4 (quality ~ country) and Figure 1 (volume ~ inspection outcome).
    Reference groups: USA (country), NAI (inspection outcome).
    """
    if not HAS_STATSMODELS:
        return

    print("\n" + "═" * 80)
    print("  STATISTICAL MODELS — Model B (RE + Two-Way Clustered SE)")
    print("  Random NDC intercept (MixedLM) + CGM (2011) SE clustered on NDC × prior_fei")
    print("═" * 80)

    d_core = df[df["CountryCode"].isin(COUNTRY_ORDER)].copy()
    d_core["IND"] = (d_core["CountryCode"] == "IND").astype(float)
    d_core["CHN"] = (d_core["CountryCode"] == "CHN").astype(float)

    # ── Figure 1: quality ~ country (reference = USA) ───────────────────────────
    print("\n" + "─" * 80)
    print("  FIGURE 4 — Quality by Country  (reference = USA)")
    print("  model: log1p(metric) ~ IND + CHN + (1|NDC11)")
    print("─" * 80)

    fig1_metrics = [
        (DMF_COL,  "DMF",              [2020, 2022, 2024]),
        (NDMA_COL, "NDMA",             [2020, 2022]),
        (DIFF_COL, "Difference Factor", [2024]),
    ]
    for qcol, label, years in fig1_metrics:
        print(f"\n  [{label}  years={years}]")
        sub = d_core[d_core["TestYear"].isin(years) & d_core[qcol].notna()].copy()
        sub = sub[sub["prior_fei"].notna()].copy()  # need FEI for clustering
        sub["_y"] = np.log1p(sub[qcol].astype(float))
        is_xs = (label == "Difference Factor")  # cross-section: 2024 only
        if is_xs:
            print("  (Cross-section: 2024 only — FEI-only SE; NDC clustering N/A)")
            if not sub.empty and HAS_STATSMODELS:
                y = sub["_y"].values
                X = sm.add_constant(sub[["IND", "CHN"]].values.astype(float))
                try:
                    r = sm.OLS(y, X).fit(cov_type="cluster",
                                          cov_kwds={"groups": sub["prior_fei"].values})
                    _print_coef_table(["const", "IND", "CHN"],
                                      r.params, r.bse, max(len(y) - 3, 1),
                                      header=f"★ FEI-clustered SE  [{label}]  — PRIMARY (cross-section):")
                except Exception as exc:
                    print(f"  FEI-clustered error: {exc}")
        else:
            _modelB_re_twoway(sub, "_y", ["IND", "CHN"],
                              ndc_col="NDC11", fei_col="prior_fei", tag=label)

        # CHN vs IND: re-parameterize with IND as reference
        sub2 = sub.copy()
        sub2["USA"] = (sub2["CountryCode"] == "USA").astype(float)
        sub2["CHN"] = (sub2["CountryCode"] == "CHN").astype(float)
        if not sub2.empty and sub2["prior_fei"].notna().sum() >= 4:
            if is_xs:
                y2 = sub2["_y"].values
                X2 = sm.add_constant(sub2[["USA", "CHN"]].values.astype(float))
                try:
                    r2 = sm.OLS(y2, X2).fit(cov_type="cluster",
                                              cov_kwds={"groups": sub2["prior_fei"].values})
                    _print_coef_table(["const", "USA_vs_IND", "CHN_vs_IND"],
                                      r2.params, r2.bse, max(len(y2) - 3, 1),
                                      header=f"★ CHN vs IND  [{label}]:")
                except Exception as exc:
                    print(f"  CHN vs IND error: {exc}")
            else:
                _modelB_re_twoway(sub2, "_y", ["USA", "CHN"],
                                  ndc_col="NDC11", fei_col="prior_fei",
                                  tag=f"{label} CHN_vs_IND")

    # ── Figure 4: quality ~ inspection outcome (reference = NAI) ────────────────
    print("\n" + "─" * 80)
    print("  QUALITY ~ INSPECTION OUTCOME  (reference = NAI; related to Fig 1 right panel)")
    print("  model: log1p(metric) ~ VAI + OAI + (1|NDC11)")
    print("─" * 80)

    d_insp = df[
        df["CountryCode"].isin(COUNTRY_ORDER) &
        df["prior_outcome"].notna() &
        df["prior_fei"].notna()
    ].copy()
    d_insp["VAI"] = (d_insp["prior_outcome"] == "VAI").astype(float)
    d_insp["OAI"] = (d_insp["prior_outcome"] == "OAI").astype(float)

    for qcol, label, years in fig1_metrics:
        print(f"\n  [{label}  years={years}]")
        sub = d_insp[d_insp["TestYear"].isin(years) & d_insp[qcol].notna()].copy()
        sub["_y"] = np.log1p(sub[qcol].astype(float))
        is_xs = (label == "Difference Factor")
        if is_xs:
            if not sub.empty and HAS_STATSMODELS:
                y = sub["_y"].values
                X = sm.add_constant(sub[["VAI", "OAI"]].values.astype(float))
                try:
                    r = sm.OLS(y, X).fit(cov_type="cluster",
                                          cov_kwds={"groups": sub["prior_fei"].values})
                    _print_coef_table(["const", "VAI", "OAI"],
                                      r.params, r.bse, max(len(y) - 3, 1),
                                      header=f"★ FEI-clustered SE  [{label}]  (cross-section):")
                except Exception as exc:
                    print(f"  error: {exc}")
        else:
            _modelB_re_twoway(sub, "_y", ["VAI", "OAI"],
                              ndc_col="NDC11", fei_col="prior_fei", tag=label)
            # OAI vs VAI: re-parameterize with VAI as reference
            sub3 = sub.copy()
            sub3["NAI_d"] = (sub3["prior_outcome"] == "NAI").astype(float)
            sub3["OAI_d"] = (sub3["prior_outcome"] == "OAI").astype(float)
            _modelB_re_twoway(sub3, "_y", ["NAI_d", "OAI_d"],
                              ndc_col="NDC11", fei_col="prior_fei",
                              tag=f"{label} OAI_vs_VAI")

    # ── Figure 2: volume ~ inspection outcome (reference = NAI) ─────────────────
    print("\n" + "─" * 80)
    print("  FIGURE 1 (right panel) — Volume by Inspection Outcome  (reference = NAI)")
    print("  model: log(iqvia_extended_units) ~ VAI + OAI + (1|NDC11)")
    print("─" * 80)

    d_vol = df[
        df["CountryCode"].isin(COUNTRY_ORDER) &
        df["prior_outcome"].notna() &
        df["prior_fei"].notna() &
        df[VOL_COL].notna() &
        (df[VOL_COL] > 0)
    ].copy()
    d_vol["VAI"] = (d_vol["prior_outcome"] == "VAI").astype(float)
    d_vol["OAI"] = (d_vol["prior_outcome"] == "OAI").astype(float)
    d_vol["_y"]  = np.log(d_vol[VOL_COL].astype(float))
    print(f"\n  [log(IQVIA Extended Units)]")
    _modelB_re_twoway(d_vol, "_y", ["VAI", "OAI"],
                      ndc_col="NDC11", fei_col="prior_fei", tag="log(Volume)")

    print("\n" + "═" * 80)
    print("  DONE")
    print("═" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
plot_fig1_market_by_outcome()   # Fig 1: Price (blank) + Volume by inspection outcome
plot_fig2_volume_vs_quality()   # Fig 2: Volume vs DMF | NDMA | DiffFactor (pooled)
plot_fig3_price_vs_quality()    # Fig 3: Price vs Quality (blank — NADAC pending)
plot_fig4_quality_by_country()  # Fig 4: Quality by Country (bar charts)

run_statistical_models()

print(f"\nAll figures saved to: {OUT_DIR}")
# %%
