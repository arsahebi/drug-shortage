# %%
"""
Figure 1 drawing style, shared by step6_graphs_july26.py and
build_variant_graphs.py so both produce the figure the manuscript uses
(Metformin J Pharma 2026 10 09 ASF.docx, Figure 1): blue-filled boxes, navy
median, points jittered and colored by country, KW p in the title, country
legend below the plot (bottom right), n beneath each box.

One outcome per figure: volume and price are drawn as separate figures.
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
UNKNOWN_COLOR  = "#9ca3af"


def _kw_title_p(groups):
    valid = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return ""
    p = kruskal(*valid)[1]
    return f"  (KW p={p:.3f})" if p >= 0.001 else "  (KW p<0.001)"


def outcome_boxplot(sub, value_col, title, ylabel, outfiles,
                    outcome_col="prior_outcome", country_col="CountryCode", seed=42):
    """sub: rows already filtered to the analysis sample. outfiles: list of
    paths to save (e.g. .png and .pdf)."""
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        _draw(sub, value_col, title, ylabel, outfiles, outcome_col, country_col, seed)


def _draw(sub, value_col, title, ylabel, outfiles, outcome_col, country_col, seed):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    rng = np.random.default_rng(seed)
    groups, n_vals = [], []
    has_unknown = False
    for xi, out in enumerate(OUTCOME_ORDER):
        d_out = sub[sub[outcome_col] == out]
        vals = d_out[value_col].astype(float).values
        groups.append(vals)
        n_vals.append(len(d_out))
        if len(vals):
            ax.boxplot(vals, positions=[xi], widths=0.45,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#e0e7ff", color="#4f46e5"),
                       medianprops=dict(color="#1e1b4b", linewidth=2),
                       whiskerprops=dict(color="#4f46e5"),
                       capprops=dict(color="#4f46e5"))
        known = d_out[country_col].isin(COUNTRY_ORDER)
        for cc in COUNTRY_ORDER + [None]:
            d_cc = d_out[d_out[country_col] == cc] if cc else d_out[~known]
            if d_cc.empty:
                continue
            has_unknown |= cc is None
            jitter = rng.uniform(-0.15, 0.15, size=len(d_cc))
            ax.scatter(xi + jitter, d_cc[value_col].astype(float).values,
                       c=COUNTRY_COLORS.get(cc, UNKNOWN_COLOR), s=40, alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(range(len(OUTCOME_ORDER)))
    ax.set_xticklabels(OUTCOME_ORDER)
    ax.set_xlim(-0.5, len(OUTCOME_ORDER) - 0.5)
    ax.set_xlabel("Prior Inspection Outcome")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for xi, n in enumerate(n_vals):
        ax.text(xi, 0.01, f"n={int(n)}", transform=trans,
                ha="center", va="bottom", fontsize=9, color="#374151")
    ax.set_title(f"{title}{_kw_title_p(groups)}", fontsize=11, fontweight="bold")

    handles = [Line2D([0], [0], marker="o", linestyle="", color=COUNTRY_COLORS[cc],
                      label=COUNTRY_LABELS[cc], markeredgecolor="white",
                      markeredgewidth=0.5, markersize=8) for cc in COUNTRY_ORDER]
    if has_unknown:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=UNKNOWN_COLOR,
                              label="Unknown", markeredgecolor="white",
                              markeredgewidth=0.5, markersize=8))
    # Below the plot area, right-aligned, so it never covers data points
    ax.legend(handles=handles, title="Country", loc="upper right",
              bbox_to_anchor=(1.0, -0.13), ncol=len(handles), fontsize=9,
              title_fontsize=9, frameon=True)
    fig.tight_layout()
    for p in outfiles:
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
# %%
