# %%
"""
Module 8 — Exploratory data analysis on the quality↔shortage link.

Produces:
  outputs/figures/eda_*.png
  outputs/tables/eda_*.csv

Three core EDA questions:
  1. Do drugs with worse Valisure scores have more UUtah shortages?
  2. Do FAERS severity / recall counts in year t lead shortage in t+1?
  3. Within the Valisure pilot, what's the rank correlation?
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import OUT_DATA, OUT_FIGS, OUT_TABS, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR
from utils import get_logger, read_table

log = get_logger("m08_eda", OUT_LOGS / "m08_eda.log")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def eda_valisure_vs_shortage(panel: pd.DataFrame):
    """Pilot drugs only: do worse Valisure scores predict more shortage years?"""
    p = panel.dropna(subset=["valisure_mean_score"]).copy()
    if p.empty:
        log.warning("No drugs with Valisure score; skipping eda_valisure_vs_shortage")
        return
    drug_summary = (p.groupby("drug_norm")
                    .agg(valisure_mean_score=("valisure_mean_score","first"),
                         valisure_min_score=("valisure_min_score","first"),
                         valisure_n_failing=("valisure_n_failing","first"),
                         n_shortage_years=("shortage_started","sum"),
                         n_years_ongoing=("shortage_ongoing","sum"))
                    .reset_index())
    drug_summary.to_csv(OUT_TABS / "eda_valisure_drug_summary.csv", index=False)

    # Spearman correlation (rank-based, robust)
    if len(drug_summary) >= 4:
        for x in ["valisure_mean_score","valisure_min_score","valisure_n_failing"]:
            rho, p_val = stats.spearmanr(drug_summary[x], drug_summary["n_shortage_years"], nan_policy="omit")
            log.info("Spearman %s vs n_shortage_years: rho=%.3f p=%.3f", x, rho, p_val)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(drug_summary["valisure_mean_score"], drug_summary["n_shortage_years"],
                  s=60, alpha=0.8)
    for _, r in drug_summary.iterrows():
        ax[0].annotate(r["drug_norm"], (r["valisure_mean_score"], r["n_shortage_years"]),
                       fontsize=7, alpha=0.7)
    ax[0].set_xlabel("Valisure Mean DoD Score (higher = better quality)")
    ax[0].set_ylabel(f"# Shortage-Start Years ({PANEL_START_YEAR}–{PANEL_END_YEAR})")
    ax[0].set_title("Pilot drugs: Valisure quality vs UUtah shortages")
    ax[0].grid(alpha=0.3)

    ax[1].scatter(drug_summary["valisure_n_failing"], drug_summary["n_shortage_years"],
                  s=60, alpha=0.8, color="C3")
    for _, r in drug_summary.iterrows():
        ax[1].annotate(r["drug_norm"], (r["valisure_n_failing"], r["n_shortage_years"]),
                       fontsize=7, alpha=0.7)
    ax[1].set_xlabel("# Failing Valisure rows (score < 70)")
    ax[1].set_ylabel("# Shortage-Start Years")
    ax[1].set_title("Failing tested rows vs shortage frequency")
    ax[1].grid(alpha=0.3)
    fig.savefig(OUT_FIGS / "eda_valisure_vs_shortage.png")
    plt.close(fig)
    log.info("Wrote eda_valisure_vs_shortage.png")


def eda_signal_lead_time(panel: pd.DataFrame):
    """Does quality signal in year t precede shortage onset in year t+1?"""
    df = panel.dropna(subset=["y_next_year_shortage"]).copy()
    grouped_means = (df.groupby("y_next_year_shortage")
                     .agg(faers_n_reports=("faers_n_reports","mean"),
                          faers_n_serious=("faers_n_serious","mean"),
                          faers_severity_score=("faers_severity_score","mean"),
                          recall_total=("recall_total","mean"),
                          recall_class_I=("recall_class_I","mean"),
                          recall_cgmp=("recall_cgmp","mean"),
                          redica_n_oai=("redica_n_oai","mean"),
                          redica_n_warning_letters=("redica_n_warning_letters","mean"))
                     .T)
    grouped_means.columns = [f"y_next={int(c)}" for c in grouped_means.columns]
    grouped_means["lift"] = (grouped_means.get("y_next=1", 0) /
                             grouped_means.get("y_next=0", np.nan).replace(0, np.nan))
    grouped_means.to_csv(OUT_TABS / "eda_lead_time_means.csv")
    log.info("Lead-time mean comparison:\n%s", grouped_means.to_string())

    # Small Valisure panel: box/strip plots are more readable than overlaid histograms.
    plot_specs = [
        ("faers_severity_score", "FAERS severity score"),
        ("recall_total", "FDA recall count"),
        ("redica_n_oai", "Redica OAI count"),
        ("recall_cgmp", "CGMP recall count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, plot_specs):
        groups = [
            df.loc[df["y_next_year_shortage"] == 0, col].fillna(0),
            df.loc[df["y_next_year_shortage"] == 1, col].fillna(0),
        ]
        values = [np.log1p(g) for g in groups]
        ax.boxplot(values, positions=[0, 1], widths=0.45, patch_artist=True,
                   boxprops={"facecolor": "#d9e8f5", "edgecolor": "#5b7fa3"},
                   medianprops={"color": "#222222"},
                   whiskerprops={"color": "#5b7fa3"},
                   capprops={"color": "#5b7fa3"})
        for x, vals, color in [(0, values[0], "C0"), (1, values[1], "C3")]:
            if len(vals) == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0])
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=22, alpha=0.55, color=color)
        means = [float(v.mean()) if len(v) else np.nan for v in values]
        for x, m in enumerate(means):
            if np.isfinite(m):
                ax.scatter([x], [m], s=55, marker="D", color="#111111", zorder=3)
        ax.set_xticks([0, 1], ["No shortage\nt+1", "Shortage\nt+1"])
        ax.set_ylabel(f"log1p({col})")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Year-t quality signals by next-year shortage outcome", y=1.03)
    fig.savefig(OUT_FIGS / "eda_lead_time_distributions.png")
    plt.close(fig)
    log.info("Wrote eda_lead_time_distributions.png")


def eda_temporal_trend(panel: pd.DataFrame):
    yr = (panel.groupby("year")
          .agg(n_shortage_starts=("shortage_started","sum"),
               n_drugs_in_shortage=("shortage_ongoing","sum"),
               mean_faers_reports=("faers_n_reports","mean"),
               mean_recalls=("recall_total","mean"))
          .reset_index())
    yr.to_csv(OUT_TABS / "eda_temporal_trend.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(yr["year"], yr["n_shortage_starts"], alpha=0.6, color="C3", label="Shortage starts")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Shortage starts", color="C3")
    ax2 = ax1.twinx()
    ax2.plot(yr["year"], yr["mean_faers_reports"], "o-", color="C0", label="Avg FAERS reports per drug")
    ax2.plot(yr["year"], yr["mean_recalls"] * 100, "s--", color="C2", label="Avg recalls per drug × 100")
    ax2.set_ylabel("Mean per-drug signal", color="C0")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    ax1.set_title("Annual shortages and quality signal trends")
    ax1.grid(alpha=0.3)
    fig.savefig(OUT_FIGS / "eda_temporal_trend.png")
    plt.close(fig)
    log.info("Wrote eda_temporal_trend.png")


def main():
    panel = read_table(OUT_DATA / "master_panel.parquet")
    eda_valisure_vs_shortage(panel)
    eda_signal_lead_time(panel)
    eda_temporal_trend(panel)


if __name__ == "__main__":
    main()

# %%
