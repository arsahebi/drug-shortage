# %%
"""
Box-plot drawing style, shared by step6_graphs_july26.py and
build_variant_graphs.py so both produce the figure the manuscript uses
(Metformin J Pharma 2026 10 09 ASF.docx, Figure 1): blue-filled boxes, navy
median, points jittered and colored by country, KW p in the title, country
legend centered below the plot, n beneath each box.

One outcome per figure: volume and price are drawn as separate figures.
outcome_boxplot draws Figure 1 (by inspection outcome, colored by country);
country_boxplot draws Figures 5/5b (by country, colored by inspection
outcome) in the same style.
Drawing only -- no statistics are computed here except the Kruskal-Wallis p
shown in the title, which is the same test the stats logs report.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import kruskal

OUTCOME_ORDER  = ["NAI", "VAI", "OAI"]
COUNTRY_ORDER  = ["IND", "CHN", "USA"]
COUNTRY_LABELS = {"IND": "India", "CHN": "China", "USA": "United States"}
COUNTRY_COLORS = {"IND": "#ef4444", "CHN": "#f59e0b", "USA": "#3b82f6"}
OUTCOME_COLORS = {"NAI": "#22c55e", "VAI": "#f59e0b", "OAI": "#ef4444"}
UNKNOWN_COLOR  = "#9ca3af"


def _kw_title_p(groups):
    valid = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return ""
    p = kruskal(*valid)[1]
    return f"  (KW p={p:.3f})" if p >= 0.001 else "  (KW p<0.001)"


def outcome_boxplot(sub, value_col, title, ylabel, outfiles,
                    outcome_col="prior_outcome", country_col="CountryCode", seed=42):
    """Figure 1: value by prior inspection outcome, points colored by country.
    sub: rows already filtered to the analysis sample. outfiles: list of
    paths to save (e.g. .png and .pdf)."""
    group_boxplot(sub, value_col, outcome_col, OUTCOME_ORDER, OUTCOME_ORDER,
                  "Prior Inspection Outcome", country_col, COUNTRY_ORDER,
                  COUNTRY_COLORS, COUNTRY_LABELS, "Unknown", "Country",
                  title, ylabel, outfiles, seed)


def country_boxplot(sub, value_col, title, ylabel, outfiles,
                    country_col="CountryCode", outcome_col="prior_outcome", seed=42):
    """Figures 5/5b: value by country of manufacture, points colored by prior
    inspection outcome (gray = no prior inspection on record). Same look as
    Figure 1."""
    group_boxplot(sub, value_col, country_col, COUNTRY_ORDER,
                  [COUNTRY_LABELS[c] for c in COUNTRY_ORDER],
                  "Country of Manufacture", outcome_col, OUTCOME_ORDER,
                  OUTCOME_COLORS, {o: o for o in OUTCOME_ORDER},
                  "No prior inspection", "Prior Inspection Outcome",
                  title, ylabel, outfiles, seed)


def group_boxplot(sub, value_col, group_col, group_order, group_labels, xlabel,
                  color_col, color_order, color_map, color_labels, unknown_label,
                  legend_title, title, ylabel, outfiles, seed=42):
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        _draw(sub, value_col, group_col, group_order, group_labels, xlabel,
              color_col, color_order, color_map, color_labels, unknown_label,
              legend_title, title, ylabel, outfiles, seed)


def _draw(sub, value_col, group_col, group_order, group_labels, xlabel,
          color_col, color_order, color_map, color_labels, unknown_label,
          legend_title, title, ylabel, outfiles, seed):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    rng = np.random.default_rng(seed)
    groups, n_vals = [], []
    has_unknown = False
    for xi, g in enumerate(group_order):
        d_g = sub[sub[group_col] == g]
        vals = d_g[value_col].astype(float).values
        groups.append(vals)
        n_vals.append(len(d_g))
        if len(vals):
            ax.boxplot(vals, positions=[xi], widths=0.45,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#e0e7ff", color="#4f46e5"),
                       medianprops=dict(color="#1e1b4b", linewidth=2),
                       whiskerprops=dict(color="#4f46e5"),
                       capprops=dict(color="#4f46e5"))
        known = d_g[color_col].isin(color_order)
        for c in color_order + [None]:
            d_c = d_g[d_g[color_col] == c] if c else d_g[~known]
            if d_c.empty:
                continue
            has_unknown |= c is None
            jitter = rng.uniform(-0.15, 0.15, size=len(d_c))
            ax.scatter(xi + jitter, d_c[value_col].astype(float).values,
                       c=color_map.get(c, UNKNOWN_COLOR), s=40, alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for xi, n in enumerate(n_vals):
        ax.text(xi, 0.01, f"n={int(n)}", transform=trans,
                ha="center", va="bottom", fontsize=9, color="#374151")
    ax.set_title(f"{title}{_kw_title_p(groups)}", fontsize=11, fontweight="bold")

    handles = [Line2D([0], [0], marker="o", linestyle="", color=color_map[c],
                      label=color_labels[c], markeredgecolor="white",
                      markeredgewidth=0.5, markersize=8) for c in color_order]
    if has_unknown:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=UNKNOWN_COLOR,
                              label=unknown_label, markeredgecolor="white",
                              markeredgewidth=0.5, markersize=8))
    # Below the plot area, centered, so it never covers data points
    ax.legend(handles=handles, title=legend_title, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=len(handles), fontsize=9,
              title_fontsize=9, frameon=True)
    fig.tight_layout()
    for p in outfiles:
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
# %%
