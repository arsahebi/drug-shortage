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
import matplotlib.dates as mdates

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

# ── Build deduplicated time series (mdates numbers) ───────────────────────────
records = {}
for insp in inspections:
    for months, ae in zip(offsets_months, insp["ae"]):
        dt = insp["date"] + pd.DateOffset(months=months)
        key = dt.to_period("Q").start_time
        dn = mdates.date2num(key)
        records.setdefault(dn, []).append(ae)

ts_dates = sorted(records)
ts_vals  = [np.mean(records[d]) for d in ts_dates]

xlim = (
    mdates.date2num(pd.Timestamp("2017-06-01")),
    mdates.date2num(pd.Timestamp("2024-03-01")),
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.0, 5.0),
    gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.12},
)

# ── Panel 1: AE trajectory ────────────────────────────────────────────────────
ax1.plot(ts_dates, ts_vals, color="#1e3a5f", linewidth=2.2, zorder=3)
ax1.fill_between(ts_dates, ts_vals, alpha=0.10, color="#1e3a5f", zorder=2)

# shade gap between NAI and VAI
t_nai = mdates.date2num(inspections[0]["date"])
t_vai = mdates.date2num(inspections[1]["date"])
ax1.axvspan(t_nai, t_vai, alpha=0.07, color="#cc0000", zorder=1)
ax1.text(
    (t_nai + t_vai) / 2, 845,
    "AEs +50% after NAI",
    ha="center", va="bottom", fontsize=8, color="#cc0000", style="italic",
)

for insp in inspections:
    color = outcome_colors[insp["outcome"]]
    dn = mdates.date2num(insp["date"])
    ax1.axvline(dn, color=color, linewidth=1.8, linestyle="--", zorder=4)
    ax1.text(
        dn, 912,
        insp["outcome"],
        ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=color, linewidth=0.9),
    )

ax1.set_xlim(xlim)
ax1.set_ylim(100, 1010)
ax1.set_ylabel("Serious AEs (quarterly)", fontsize=9)

# year ticks on upper panel — no labels (lower panel carries them)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter(""))   # hide labels
ax1.tick_params(axis="x", length=4)
ax1.tick_params(axis="y", labelsize=8.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title(
    "FEI 3002807979 — Inspection outcomes, 483 text signals, and adverse events",
    fontsize=9, loc="left", pad=7,
)

# ── Panel 2: Text signals ─────────────────────────────────────────────────────
bar_w = 55  # days in date-number units

for insp in inspections:
    color = outcome_colors[insp["outcome"]]
    dn = mdates.date2num(insp["date"])
    ax2.axvline(dn, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)

    ax2.bar(dn - bar_w, insp["lc"], width=bar_w * 0.92,
            color="#1d4ed8", alpha=0.85, align="edge", zorder=2)
    ax2.bar(dn,          insp["di"], width=bar_w * 0.92,
            color="#dc2626", alpha=0.75, align="edge", zorder=2)

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
    mpatches.Patch(color="#dc2626", alpha=0.75, label="Data Integrity"),
]
ax2.legend(handles=legend_patches, fontsize=8.5, loc="upper right",
           framealpha=0.85, edgecolor="#cccccc")

out_path = OUT / "fei_case_study_3002807979.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved → {out_path}")
