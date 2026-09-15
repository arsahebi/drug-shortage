"""
Case study figure: FEI 3002807979
Sun Pharmaceutical Industries Limited (Mohali, India) -- Atorvastatin
Three inspections in our panel window: NAI (Sep 2018), VAI (Feb 2020), OAI (Aug 2022).
A fourth NAI (May 2019) exists in inspection records but has no 483 text in our dataset.

Story arc:
  Clean baseline (2018 NAI, 2019 NAI) --> Feb 2020 VAI with LC=100%, Contam=100%
  --> AEs rise from 104 to 112 by Q+4 --> FDA returns Aug 2022 --> OAI
  --> AEs spike to 140 by Q+4 (2023-Q3)
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

# ── Quarterly AE time series (ANDA-specific, reconstructed from inspection panel) ──
# Sources: inspection-centered panel rows for 2018 NAI, 2020 VAI, 2022 OAI
# 2021-Q2 interpolated (gap between 2020-VAI tp4 and 2022-OAI tm4)
ae_data = {
    "2017Q3": 55,  "2017Q4": 78,
    "2018Q1": 85,  "2018Q2": 89,  "2018Q3": 85,  "2018Q4": 93,
    "2019Q1": 84,  "2019Q2": 77,  "2019Q3": 76,  "2019Q4": 80,
    "2020Q1": 104, "2020Q2": 91,  "2020Q3": 64,  "2020Q4": 87,
    "2021Q1": 112, "2021Q2": 102, "2021Q3": 93,  "2021Q4": 81,
    "2022Q1": 108, "2022Q2": 73,  "2022Q3": 88,  "2022Q4": 97,
    "2023Q1": 80,  "2023Q2": 74,  "2023Q3": 140,
}

def _qstart(period):
    y, q = int(period[:4]), int(period[-1])
    m = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return pd.Timestamp(y, m, 1)

ts_dates = [mdates.date2num(_qstart(p)) for p in ae_data]
ts_vals  = list(ae_data.values())

# ── Inspections ───────────────────────────────────────────────────────────────
# 2019-05-10 NAI: in inspection records but no 483 text in our dataset
inspections = [
    {"date": pd.Timestamp("2018-09-14"), "outcome": "NAI",
     "lc": 0.50, "contam": 0.00, "has_text": True},
    {"date": pd.Timestamp("2019-05-10"), "outcome": "NAI",
     "lc": None, "contam": None,  "has_text": False},
    {"date": pd.Timestamp("2020-02-07"), "outcome": "VAI",
     "lc": 1.00, "contam": 1.00, "has_text": True},
    {"date": pd.Timestamp("2022-08-12"), "outcome": "OAI",
     "lc": 0.67, "contam": 0.50, "has_text": True},
]

outcome_colors = {"NAI": "#6b7280", "VAI": "#d97706", "OAI": "#cc0000"}

xlim = (
    mdates.date2num(pd.Timestamp("2017-06-01")),
    mdates.date2num(pd.Timestamp("2023-12-01")),
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.0, 5.0),
    gridspec_kw={"height_ratios": [2.2, 1.8], "hspace": 0.12},
)

# ── Panel 1: AE trajectory ────────────────────────────────────────────────────
ax1.plot(ts_dates, ts_vals, color="#1e3a5f", linewidth=2.2, zorder=3)
ax1.fill_between(ts_dates, ts_vals, alpha=0.10, color="#1e3a5f", zorder=2)

# Shade Q+1..Q+4 prediction window after 2020 VAI
t_vai_date  = pd.Timestamp("2020-02-07")
t_vai_q4    = pd.Timestamp("2021-02-07")
t_vai_dn    = mdates.date2num(t_vai_date)
t_vai_q4_dn = mdates.date2num(t_vai_q4)
ax1.axvspan(t_vai_dn, t_vai_q4_dn, alpha=0.07, color="#d97706", zorder=1)

# Q+4 window label -- inside the shade, low in the panel
ax1.text(
    (t_vai_dn + t_vai_q4_dn) / 2, 43,
    "Q+1..Q+4", ha="center", va="bottom",
    fontsize=7, color="#d97706", style="italic",
)

# Mark Q+4 value (2021-Q1 = 112) -- annotate to the right to avoid shade
t_q4_dn = mdates.date2num(pd.Timestamp("2021-01-01"))
ax1.annotate(
    "Q+4 = 112",
    xy=(t_q4_dn, 112), xytext=(t_q4_dn + 220, 138),
    fontsize=7.5, color="#d97706",
    arrowprops=dict(arrowstyle="->", color="#d97706", lw=0.9),
)

# Mark 2023-Q3 peak -- place text to the right of the OAI line
t_peak = mdates.date2num(pd.Timestamp("2023-07-01"))
ax1.annotate(
    "140/qtr",
    xy=(t_peak, 140), xytext=(t_peak + 60, 155),
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
        dn, 163, label,
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=color, linewidth=0.9),
    )

ax1.set_xlim(xlim)
ax1.set_ylim(40, 185)
ax1.set_ylabel("Serious AEs (quarterly)", fontsize=9)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter(""))
ax1.tick_params(axis="x", length=4)
ax1.tick_params(axis="y", labelsize=8.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title(
    "FEI 3002807979 (Sun Pharma, Mohali) -- Atorvastatin\n"
    "NAI baseline, VAI with maximum signals, OAI follow-up",
    fontsize=8.5, loc="left", pad=7,
)
ax1.text(0.01, 0.02, "* NAI with no 483 text in dataset",
         transform=ax1.transAxes, fontsize=7, color="#6b7280", style="italic")

# ── Panel 2: Text signals ─────────────────────────────────────────────────────
bar_w = 50

for insp in inspections:
    if not insp["has_text"]:
        color = outcome_colors[insp["outcome"]]
        dn = mdates.date2num(insp["date"])
        ax2.axvline(dn, color=color, linewidth=1.5, linestyle=":", alpha=0.4, zorder=1)
        continue
    color = outcome_colors[insp["outcome"]]
    dn = mdates.date2num(insp["date"])
    ax2.axvline(dn, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)
    ax2.bar(dn - bar_w, insp["lc"],    width=bar_w * 0.92,
            color="#1d4ed8", alpha=0.85, align="edge", zorder=2)
    ax2.bar(dn,          insp["contam"], width=bar_w * 0.92,
            color="#dc2626", alpha=0.75, align="edge", zorder=2)

# Single combined median reference line (LC~11%, Contam~14% are nearly equal)
med = 0.125
ax2.axhline(med, color="#888888", linewidth=1.0, linestyle=":", alpha=0.6, zorder=0)
ax2.text(xlim[1] - 20, med + 0.04,
         "Sample median (LC~11%, Contam~14%)", ha="right", va="bottom",
         fontsize=7, color="#888888", style="italic")

ax2.set_xlim(xlim)
ax2.set_ylim(0, 1.20)
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

out_path = OUT / "fei_case_study_3002807979.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
