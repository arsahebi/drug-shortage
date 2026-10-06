"""
Module 10 — Valisure lead-time analysis.

For each shortage onset, look backward 1–4 years and compare the average
quality-signal trajectory before and during the shortage-start year.

Output:
  outputs/figures/lead_time_valisure.png
  outputs/tables/lead_time_valisure.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DATA, OUT_FIGS, OUT_TABS, OUT_LOGS
from utils import get_logger, read_table

log = get_logger("m10_leadtime", OUT_LOGS / "m10_leadtime.log")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def build_event_study(panel: pd.DataFrame, signal_cols: list[str], lookback: int = 4) -> pd.DataFrame:
    """For each shortage onset (year T), record signal values at T-lookback..T-1."""
    rows = []
    panel = panel.sort_values(["drug_norm", "year"])
    for drug, g in panel.groupby("drug_norm"):
        g = g.set_index("year")
        onsets = g.index[g["shortage_started"] == 1].tolist()
        for T in onsets:
            for k in range(-lookback, 1):
                yr = T + k
                if yr not in g.index:
                    continue
                row = {"drug_norm": drug, "T": T, "rel_year": k}
                for c in signal_cols:
                    row[c] = g.at[yr, c] if c in g.columns else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def build_control_baseline(panel: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Year-mean of each signal among drug-years that had no shortage in t-3..t+1."""
    p = panel.copy().sort_values(["drug_norm", "year"])
    p["short_lead"] = p.groupby("drug_norm")["shortage_started"].transform(
        lambda s: s.rolling(5, min_periods=1, center=True).max())
    ctrl = p[p["short_lead"] == 0]
    means = ctrl[signal_cols].mean()
    return means


def plot_event_study(es: pd.DataFrame, baseline: pd.Series, signal_cols: list[str], scope: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, col in zip(axes.flat, signal_cols):
        m = es.groupby("rel_year")[col].mean()
        s = es.groupby("rel_year")[col].sem()
        ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label="Pre-shortage drug-years")
        ax.axhline(baseline[col], color="C2", linestyle="--", label="Control baseline (no shortage ±2y)")
        ax.set_xlabel("Years relative to shortage onset (0 = onset year)")
        ax.set_ylabel(f"Mean {col}")
        ax.set_title(col)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Quality-signal trajectory around Valisure-drug shortage starts", y=1.03)
    fig.savefig(OUT_FIGS / f"lead_time_{scope}.png")
    plt.close(fig)
    log.info("Wrote lead_time_%s.png", scope)


def main():
    panel = read_table(OUT_DATA / "master_panel.parquet")

    cols = ["faers_severity_score", "faers_n_serious", "recall_total", "recall_cgmp"]
    cols = [c for c in cols if c in panel.columns]

    es = build_event_study(panel, cols)
    baseline = build_control_baseline(panel, cols)
    if es.empty:
        log.warning("No shortage starts found; skipping lead-time analysis")
        return es

    plot_event_study(es, baseline, cols, "valisure")
    agg = es.groupby("rel_year")[cols].mean().reset_index()
    agg["scope"] = "valisure"
    agg.to_csv(OUT_TABS / "lead_time_valisure.csv", index=False)
    log.info("Wrote lead_time_valisure.csv (%d event-study rows)", len(es))
    return es


if __name__ == "__main__":
    main()
