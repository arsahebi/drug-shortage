"""
Case study figure: FEI 3002807979
Three inspections, 2018–2022.
Shows AE trajectory, 483 text signals, and inspection outcomes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent

# ── Data ──────────────────────────────────────────────────────────────────────
inspections = [
    {
        "date": pd.Timestamp("2018-09-14"),
        "outcome": "NAI",
        "lc": 0.50,
        "di": 1.00,
        "ae": [247, 289, 296, 457, 465, 466, 633, 632, 698],
    },
    {
        "date": pd.Timestamp("2020-02-07"),
        "outcome": "VAI",
        "lc": 1.00,
        "di": 0.00,
        "ae": [633, 632, 698, 569, 736, 809, 667, 558, 561],
    },
    {
        "date": pd.Timestamp("2022-08-12"),
        "outcome": "OAI",
        "lc": 0.67,
        "di": 0.67,
        "ae": [747, 584, 579, 526, 533, 547, 604, 620, 800],
    },
]

offsets_months = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

outcome_colors = {"NAI": "#6b7280", "VAI": "#d97706", "OAI": "#cc0000"}

# ── Build time series ─────────────────────────────────────────────────────────
records = {}
for insp in inspections:
    for months, ae in zip(offsets_months, insp["ae"]):
        dt = insp["date"] + pd.DateOffset(months=months)
        key = dt.to_period("Q").start_time
        if key not in records:
            records[key] = []
        records[key].append(ae)

ts = pd.Series({k: np.mean(v) for k, v in sorted(records.items())})
ts.index = pd.to_datetime(ts.index)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.5, 4.6),
    gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
    sharex=False,
)

# ── Panel 1: AE trajectory ────────────────────────────────────────────────────
ax1.plot(ts.index, ts.values, color="#1e3a5f", linewidth=2, zorder=3)
ax1.fill_between(ts.index, ts.values, alpha=0.10, color="#1e3a5f", zorder=2)

# shade gap between NAI and VAI to highlight the unmonitored window
t_nai = inspections[0]["date"]
t_vai = inspections[1]["date"]
ax1.axvspan(t_nai, t_vai, alpha=0.07, color="#cc0000", zorder=1)
ax1.text(
    t_nai + (t_vai - t_nai) / 2, 820,
    "AEs rise 50%\nafter NAI",
    ha="center", va="bottom", fontsize=7, color="#cc0000", style="italic",
)

for insp in inspections:
    color = outcome_colors[insp["outcome"]]
    ax1.axvline(insp["date"], color=color, linewidth=1.6, linestyle="--", zorder=4)
    ax1.text(
        insp["date"], 900,
        insp["outcome"],
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.8),
    )

ax1.set_ylabel("Serious AEs (quarterly)", fontsize=8)
ax1.set_ylim(100, 1000)
ax1.set_xlim(pd.Timestamp("2017-06-01"), pd.Timestamp("2024-03-01"))
ax1.tick_params(axis="both", labelsize=7.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("FEI 3002807979 — Inspection outcomes, 483 text signals, and adverse events",
              fontsize=8.5, loc="left", pad=6)

# ── Panel 2: Text signals ─────────────────────────────────────────────────────
bar_w = 60  # days
dates = [insp["date"] for insp in inspections]
lc_vals = [insp["lc"] for insp in inspections]
di_vals = [insp["di"] for insp in inspections]

import matplotlib.dates as mdates
date_nums = mdates.date2num(dates)
ax2.bar([d - bar_w for d in date_nums], lc_vals, width=bar_w * 0.9,
        color="#1d4ed8", alpha=0.85, label="Lab Controls", align="edge")
ax2.bar([d for d in date_nums], di_vals, width=bar_w * 0.9,
        color="#dc2626", alpha=0.75, label="Data Integrity", align="edge")

for insp in inspections:
    color = outcome_colors[insp["outcome"]]
    ax2.axvline(insp["date"], color=color, linewidth=1.4, linestyle="--", alpha=0.7)

ax2.set_ylim(0, 1.35)
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_yticklabels(["0%", "50%", "100%"], fontsize=7)
ax2.set_ylabel("483 signal\nshare", fontsize=7.5)
ax2.set_xlim(mdates.date2num(pd.Timestamp("2017-06-01")),
             mdates.date2num(pd.Timestamp("2024-03-01")))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.tick_params(axis="both", labelsize=7.5)
ax2.spines[["top", "right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#1d4ed8", alpha=0.85, label="Lab Controls share"),
    mpatches.Patch(color="#dc2626", alpha=0.75, label="Data Integrity share"),
]
ax2.legend(handles=legend_patches, fontsize=7, loc="upper right", framealpha=0.8)

out_path = OUT / "fei_case_study_3002807979.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out_path}")
