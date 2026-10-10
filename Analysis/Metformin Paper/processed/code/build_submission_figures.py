# %%
"""
Journal-ready figures for the J Pharm Innov submission (manual NDC-FEI map,
all dosage forms pooled -- the paper's primary analysis).

Journal artwork rules applied (Springer guidelines for J Pharm Innov):
  - no titles inside the illustrations (captions live in the manuscript text)
  - Helvetica/Arial lettering, 8-12 pt at final printed size
  - full-width figures: 174 mm wide, no taller than 234 mm
  - TIFF at 600 dpi (graphs with text count as "combination art", >= 600 dpi)

Same data, filters, and drawing code as build_variant_graphs.py; only size,
font, resolution, format, and titles differ. The Kruskal-Wallis p values that
the internal versions print in the title go in the captions instead.

Output: outputs/submission_figures/
  Figure1.tif  volume by prior inspection outcome; (a) any prior inspection,
               (b) inspections within 3 years before testing only
  Figure2.tif  market volume vs tested quality (DMF, NDMA, dissolution)
  Figure3.tif  mean tested quality by country of manufacture
  Figure4.tif  market volume by country of manufacture
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import build_variant_graphs as b
import fig1_style as fs

MM = 1 / 25.4
WIDTH = 174 * MM
OUT = b.PROC / "outputs" / "submission_figures"

# ── journal settings ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({"font.family": "Arial", "font.size": 9,
                            "pdf.fonttype": 42, "ps.fonttype": 42})
fs.DPI = 600
fs.FIGSIZE = (WIDTH, 115 * MM)
fs.N_IN_TICKS = True
b.STYLE.update({"fig23_size": (WIDTH, 78 * MM), "fig4_size": (WIDTH, 72 * MM),
                "ext": "tif", "titles": False, "fig23_data_frac": 0.72,
                "fig23_ylabel": "IQVIA Extended Units (log scale)",
                "fig4_n_in_ticks": True})


def _noop(_):
    pass


def figure1(df_all, df_recent, path):
    """Two panels side by side sharing one country legend."""
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 100 * MM), sharey=True)
        has_unknown = False
        for ax, d, tag in [(axes[0], df_all, "a"), (axes[1], df_recent, "b")]:
            sub = d[d.prior_outcome.notna() & d[b.VOL_COL].notna() & (d[b.VOL_COL] > 0)]
            _, unk = fs.draw_on(ax, sub, b.VOL_COL, "prior_outcome", fs.OUTCOME_ORDER,
                                fs.OUTCOME_ORDER, "Prior Inspection Outcome", "CountryCode",
                                fs.COUNTRY_ORDER, fs.COUNTRY_COLORS,
                                "IQVIA Extended Units (log scale)")
            has_unknown |= unk
            ax.text(-0.02, 1.02, f"({tag})", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("")
        handles = fs.legend_handles(fs.COUNTRY_ORDER, fs.COUNTRY_COLORS, fs.COUNTRY_LABELS,
                                    "Unknown", has_unknown)
        fig.legend(handles=handles, title="Country", loc="lower center",
                   ncol=len(handles), fontsize=9, title_fontsize=9, frameon=True,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fs.save(fig, [path])
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "_work"
    work.mkdir(exist_ok=True)

    df_all = b.prepare("manual", "all")
    df_recent, _ = b.mask_recent(df_all)

    figure1(df_all, df_recent, OUT / "Figure1.tif")

    b.fig2_3(df_all, work, _noop, b.VOL_COL, "Market Volume (Extended Units)", "Figure2")
    (work / "Figure2.tif").replace(OUT / "Figure2.tif")

    b.fig4(df_all, work, _noop)
    (work / "Figure4_Quality_by_Country.tif").replace(OUT / "Figure3.tif")

    b._by_country(df_all, work, _noop, b.VOL_COL, "Market Volume (Extended Units)",
                  "Figure4", None, "IQVIA Extended Units (log scale)")
    (work / "Figure4.tif").replace(OUT / "Figure4.tif")
    work.rmdir()

    from PIL import Image
    for f in sorted(OUT.glob("Figure*.tif")):
        with Image.open(f) as im:
            w, h = im.size
            dpi = float(im.info.get("dpi", (0, 0))[0])
        print(f"{f.name}: {w}x{h}px, {dpi:.0f} dpi, "
              f"{w / dpi * 25.4:.0f} x {h / dpi * 25.4:.0f} mm")


if __name__ == "__main__":
    main()
# %%
