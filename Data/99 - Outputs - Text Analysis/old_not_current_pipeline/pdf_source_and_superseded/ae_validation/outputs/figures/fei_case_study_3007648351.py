"""
Case study figure: FEI 3007648351
Mylan Laboratories Limited (Bengaluru, India) — Vancomycin
Four inspections, 2018-2023. All VAI or NAI. Never OAI.
Shows ANDA-specific AE trajectory, contamination/LC text signals, and inspection outcomes.
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

# ── AE quarterly data (ANDA-specific, summed across ANDAs for this FEI) ─────
ae_data = {
    "2017Q1": 26,  "2017Q2": 48,  "2017Q3": 38,  "2017Q4": 27,
    "2018Q1": 60,  "2018Q2": 82,  "2018Q3": 50,  "2018Q4": 74,
    "2019Q1": 63,  "2019Q2": 95,  "2019Q3": 120, "2019Q4": 106,
    "2020Q1": 130, "2020Q2": 83,  "2020Q3": 61,  "2020Q4": 77,
    "2021Q1": 64,  "2021Q2": 38,  "2021Q3": 69,  "2021Q4": 65,
    "2022Q1": 51,  "2022Q2": 70,  "2022Q3": 46,  "2022Q4": 65,
    "2023Q1": 72,  "2023Q2": 67,  "2023Q3": 62,  "2023Q4": 61,
    "2024Q1": 48,  "2024Q2": 72,  "2024Q3": 82,  "2024Q4": 108,
}

def _qstart(period):
    y, q = int(period[:4]), int(period[-1])
    m = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return pd.Timestamp(y, m, 1)

ts_dates = [mdates.date2num(_qstart(p)) for p in ae_data]
ts_vals  = list(ae_data.values())

# ── Inspections ───────────────────────────────────────────────────────────────
# text signals from 483 text panel; 2019 VAI had zero critical/major obs (no text)
inspections = [
    {"date": pd.Timestamp("2018-03-02"), "outcome": "VAI", "lc": 0.25, "di": 0.50, "contam": 0.25, "has_text": True},
    {"date": pd.Timestamp("2019-05-13"), "outcome": "VAI", "lc": None, "di": None, "contam": None, "has_text": False},
    {"date": pd.Timestamp("2020-02-28"), "outcome": "VAI", "lc": 0.40, "di": 0.20, "contam": 0.80, "has_text": True},
    {"date": pd.Timestamp("2023-01-16"), "outcome": "NAI", "lc": 1.00, "di": 1.00, "contam": 0.00, "has_text": True},
]

outcome_colors = {"NAI": "#6b7280", "VAI": "#d97706", "OAI": "#cc0000"}

xlim = (
    mdates.date2num(pd.Timestamp("2017-01-01")),
    mdates.date2num(pd.Timestamp("2025-01-01")),
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.0, 5.0),
    gridspec_kw={"height_ratios": [2.2, 1.8], "hspace": 0.12},
)

# ── Panel 1: AE trajectory ────────────────────────────────────────────────────
ax1.plot(ts_dates, ts_vals, color="#1e3a5f", linewidth=2.2, zorder=3)
ax1.fill_between(ts_dates, ts_vals, alpha=0.10, color="#1e3a5f", zorder=2)

# Annotate the 2020 peak
t_peak = mdates.date2num(pd.Timestamp("2020-01-01"))
ax1.annotate(
    "Peak: 130/qtr\n(2x vs. 2017)",
    xy=(t_peak, 130), xytext=(t_peak + 420, 148),
    fontsize=7.5, color="#cc0000", style="italic",
    arrowprops=dict(arrowstyle="->", color="#cc0000", lw=0.9),
)

for insp in inspections:
    color = outcome_colors[insp["outcome"]]
    dn = mdates.date2num(insp["date"])
    ls = "--" if insp["has_text"] else ":"
    ax1.axvline(dn, color=color, linewidth=1.8, linestyle=ls, zorder=4)
    label = insp["outcome"] if insp["has_text"] else f"{insp['outcome']}*"
    ax1.text(
        dn, 158,
        label,
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=color, linewidth=0.9),
    )

ax1.set_xlim(xlim)
ax1.set_ylim(0, 175)
ax1.set_ylabel("Serious AEs (quarterly)", fontsize=9)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter(""))
ax1.tick_params(axis="x", length=4)
ax1.tick_params(axis="y", labelsize=8.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title(
    "FEI 3007648351 (Mylan Vancomycin) -- Inspection outcomes, 483 text signals, and adverse events",
    fontsize=8.5, loc="left", pad=7,
)
ax1.text(0.01, 0.03, "* VAI with no critical/major 483 observations", transform=ax1.transAxes,
         fontsize=7, color="#6b7280", style="italic")

# ── Panel 2: Text signals ─────────────────────────────────────────────────────
bar_w = 55

for insp in inspections:
    if not insp["has_text"]:
        continue
    color = outcome_colors[insp["outcome"]]
    dn = mdates.date2num(insp["date"])
    ax2.axvline(dn, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)

    ax2.bar(dn - bar_w,     insp["lc"],     width=bar_w * 0.92,
            color="#1d4ed8", alpha=0.85, align="edge", zorder=2)
    ax2.bar(dn,             insp["contam"], width=bar_w * 0.92,
            color="#dc2626", alpha=0.75, align="edge", zorder=2)

# NAI median reference lines across sample (both are 14% so same height; show as one line)
nai_median = 0.143
ax2.axhline(nai_median, color="#888888", linewidth=1.1, linestyle=":", alpha=0.8, zorder=0)
ax2.text(xlim[1] - 20, nai_median + 0.04,
         "NAI median (LC & Contam): 14%", ha="right", va="bottom",
         fontsize=7.5, color="#555555", style="italic")

# Mark the 2019 VAI (no text) with a dotted line only
dn_2019 = mdates.date2num(pd.Timestamp("2019-05-13"))
ax2.axvline(dn_2019, color=outcome_colors["VAI"], linewidth=1.5, linestyle=":", alpha=0.5, zorder=1)

ax2.set_xlim(xlim)
ax2.set_ylim(0, 1.38)
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
    mpatches.Patch(color="#dc2626", alpha=0.75, label="Contamination"),
]
ax2.legend(handles=legend_patches, fontsize=8.5, loc="upper left",
           framealpha=0.85, edgecolor="#cccccc")

out_path = OUT / "fei_case_study_3007648351.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
