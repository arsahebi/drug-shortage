"""
Case study figure: FEI 3004554612
Strides Pharma Science Limited (Bangalore, India) -- Vancomycin + Tacrolimus
Three VAI inspections: Aug 2018, May 2019, Dec 2022 (for-cause).
483 signals escalate at each visit; AEs surge after the 2022 for-cause VAI.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

OUT = Path(__file__).resolve().parent

# ── Combined quarterly AEs (Vancomycin + Tacrolimus + KCl) ──────────────────
ae_data = {
    "2017Q1": 78,  "2017Q2": 239, "2017Q3": 115, "2017Q4": 88,
    "2018Q1": 87,  "2018Q2": 160, "2018Q3": 221, "2018Q4": 87,
    "2019Q1": 42,  "2019Q2": 139, "2019Q3": 124, "2019Q4": 62,
    "2020Q1": 61,  "2020Q2": 107, "2020Q3": 176, "2020Q4": 83,
    "2021Q1": 94,  "2021Q2": 92,  "2021Q3": 163, "2021Q4": 69,
    "2022Q1": 44,  "2022Q2": 136, "2022Q3": 184, "2022Q4": 103,
    "2023Q1": 116, "2023Q2": 198, "2023Q3": 242, "2023Q4": 74,
    "2024Q1": 69,  "2024Q2": 212, "2024Q3": 138, "2024Q4": 85,
}

def _qstart(period):
    y, q = int(period[:4]), int(period[-1])
    m = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return pd.Timestamp(y, m, 1)

ts_dates = [mdates.date2num(_qstart(p)) for p in ae_data]
ts_vals  = list(ae_data.values())

# ── Inspections ──────────────────────────────────────────────────────────────
# Aug 2018: pre-approval inspection, 483 issued (minor criticality)
# May 2019: VAI, routine, posted citations
# Dec 2022: VAI, for-cause + routine, major criticality, all observations critical/major
inspections = [
    {"date": pd.Timestamp("2018-08-24"), "label": "VAI",
     "lc": 0.333, "di": 0.333, "severity": "33% Crit/Maj"},
    {"date": pd.Timestamp("2019-05-24"), "label": "VAI",
     "lc": 0.500, "di": 0.500, "severity": "75% Crit/Maj"},
    {"date": pd.Timestamp("2022-12-09"), "label": "VAI*",
     "lc": 0.667, "di": 0.667, "severity": "100% Crit/Maj"},
]

VAI_COLOR = "#d97706"

xlim = (
    mdates.date2num(pd.Timestamp("2017-01-01")),
    mdates.date2num(pd.Timestamp("2025-01-01")),
)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.0, 5.0),
    gridspec_kw={"height_ratios": [2.2, 1.8], "hspace": 0.12},
)

# ── Panel 1: AE trajectory ───────────────────────────────────────────────────
ax1.plot(ts_dates, ts_vals, color="#1e3a5f", linewidth=2.2, zorder=3)
ax1.fill_between(ts_dates, ts_vals, alpha=0.10, color="#1e3a5f", zorder=2)

# Annotate the 2023 peak
t_peak = mdates.date2num(pd.Timestamp("2023-07-01"))
ax1.annotate(
    "Peak: 242/qtr\n(+32% vs. 2018 max)",
    xy=(t_peak, 242), xytext=(t_peak + 300, 255),
    fontsize=7.5, color="#cc0000", style="italic",
    arrowprops=dict(arrowstyle="->", color="#cc0000", lw=0.9),
)

for insp in inspections:
    dn = mdates.date2num(insp["date"])
    ax1.axvline(dn, color=VAI_COLOR, linewidth=1.8, linestyle="--", zorder=4)
    ax1.text(
        dn, 270, insp["label"],
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=VAI_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=VAI_COLOR, linewidth=0.9),
    )

ax1.set_xlim(xlim)
ax1.set_ylim(0, 300)
ax1.set_ylabel("Serious AEs (quarterly)", fontsize=9)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter(""))
ax1.tick_params(axis="x", length=4)
ax1.tick_params(axis="y", labelsize=8.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title(
    "FEI 3004554612 (Strides Pharma, Bangalore) -- Escalating 483 signals, always VAI\n"
    "Vancomycin + Tacrolimus (ANDA-specific serious AEs)",
    fontsize=8.5, loc="left", pad=7,
)
ax1.text(0.01, 0.03, "* For-cause VAI: 100% of observations Critical/Major",
         transform=ax1.transAxes, fontsize=7, color="#6b7280", style="italic")

# ── Panel 2: Text signals ─────────────────────────────────────────────────────
bar_w = 55

for insp in inspections:
    dn = mdates.date2num(insp["date"])
    ax2.axvline(dn, color=VAI_COLOR, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)
    ax2.bar(dn - bar_w, insp["lc"], width=bar_w * 0.92,
            color="#1d4ed8", alpha=0.85, align="edge", zorder=2)
    ax2.bar(dn,          insp["di"], width=bar_w * 0.92,
            color="#059669", alpha=0.80, align="edge", zorder=2)

# Sample VAI/NAI median reference lines
lc_med, di_med = 0.111, 0.300
ax2.axhline(lc_med, color="#1d4ed8", linewidth=1.0, linestyle=":", alpha=0.6, zorder=0)
ax2.axhline(di_med, color="#059669", linewidth=1.0, linestyle=":", alpha=0.6, zorder=0)
ax2.text(xlim[1] - 20, lc_med + 0.03,
         "VAI/NAI median LC: 11%", ha="right", va="bottom",
         fontsize=7, color="#1d4ed8", style="italic")
ax2.text(xlim[1] - 20, di_med + 0.03,
         "VAI/NAI median DI: 30%", ha="right", va="bottom",
         fontsize=7, color="#059669", style="italic")

ax2.set_xlim(xlim)
ax2.set_ylim(0, 1.10)
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_yticklabels(["0%", "50%", "100%"], fontsize=8.5)
ax2.set_ylabel("483 signal\nshare", fontsize=8.5)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.tick_params(axis="x", labelsize=9, length=4)
ax2.tick_params(axis="y", labelsize=8.5)
ax2.spines[["top", "right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#1d4ed8", alpha=0.85, label="Lab Controls"),
    mpatches.Patch(color="#059669", alpha=0.80, label="Data Integrity"),
]
ax2.legend(handles=legend_patches, fontsize=8.5, loc="upper left",
           framealpha=0.85, edgecolor="#cccccc")

out_path = OUT / "fei_case_study_3004554612.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
