# %%
"""
Module MM05 — Monthly master panel assembly.

Joins monthly feature sources (mm01–mm04) onto the drug × month base grid
and attaches Valisure quality scores as a STATIC, CROSS-SECTIONAL attribute.

════════════════════════════════════════════════════════════════════════════
IMPORTANT — VALISURE SCORES ARE NOT TIME-VARYING:
    Valisure tested drugs in 2024 only. The scores represent a single
    cross-sectional quality snapshot and have no temporal variation.
    They are joined here as drug-level constants (same value in every month
    for a given drug).  Do NOT use them as lagged monthly predictors or
    include them in any lead-lag analysis.  They are retained solely as a
    static drug attribute for cross-sectional comparisons.
════════════════════════════════════════════════════════════════════════════

IMPORTANT — FAERS RESOLUTION:
    FAERS features are non-zero only in months 1, 4, 7, 10 (quarterly
    approximation).  3-month trailing window sums (*_w3m) are computed
    here; use those instead of raw monthly FAERS values in lead-lag tables.

Output:
    data/master_panel_monthly.csv  (+ .parquet)
    1,680 rows: 14 drugs × 120 months (Jan 2015 – Dec 2024)
"""

from __future__ import annotations
import pandas as pd

from config import OUT_DATA, OUT_LOGS
from utils import get_logger, read_table, write_table

log = get_logger("mm05_panel_monthly", OUT_LOGS / "mm05_panel_monthly.log")


def build_monthly_panel() -> pd.DataFrame:
    # ── Base grid with shortage state ──────────────────────────────────────────
    panel = read_table(OUT_DATA / "uutah_monthly_panel.parquet")
    log.info("Base grid: %d rows, %d drugs", len(panel), panel["drug_norm"].nunique())

    # ── FAERS (quarterly → monthly approximation) ──────────────────────────────
    faers = read_table(OUT_DATA / "faers_monthly.parquet")
    panel = panel.merge(faers, on=["drug_norm", "year", "month"], how="left")
    faers_cols = [c for c in panel.columns if c.startswith("faers_")]
    for c in faers_cols:
        panel[c] = panel[c].fillna(0)

    # ── Recalls ────────────────────────────────────────────────────────────────
    recall = read_table(OUT_DATA / "recall_monthly.parquet")
    panel = panel.merge(recall, on=["drug_norm", "year", "month"], how="left")
    raw_recall_cols = [
        c for c in panel.columns
        if c.startswith("n_recalls") or c.startswith("n_class")
        or c.startswith("n_cgmp")    or c.startswith("n_contam")
        or c.startswith("n_potency") or c.startswith("n_mislabel")
        or c.startswith("n_stability") or c.startswith("n_foreign")
        or c.startswith("n_dissolution")
    ]
    for c in raw_recall_cols:
        panel[c] = panel[c].fillna(0)
    # Rename to recall_ prefix (mirrors annual panel)
    panel = panel.rename(columns={
        "n_recalls_total": "recall_total",
        "n_class_I":       "recall_class_I",
        "n_class_II":      "recall_class_II",
        "n_class_III":     "recall_class_III",
        "n_cgmp":          "recall_cgmp",
        "n_contam":        "recall_contam",
        "n_potency":       "recall_potency",
        "n_mislabel":      "recall_mislabel",
        "n_stability":     "recall_stability",
        "n_foreign":       "recall_foreign",
        "n_dissolution":   "recall_dissolution",
    })

    # ── Recall circularity flag ─────────────────────────────────────────────────
    # Marks months where a recall and an ongoing shortage co-occur for the same
    # drug — these events are likely mechanically linked (circular).
    panel["recall_during_shortage"] = (
        (panel["recall_total"] > 0) & (panel["shortage_ongoing"] == 1)
    ).astype(int)

    # ── Redica ─────────────────────────────────────────────────────────────────
    redica = read_table(OUT_DATA / "redica_monthly.parquet")
    panel = panel.merge(redica, on=["drug_norm", "year", "month"], how="left")
    for c in [c for c in panel.columns if c.startswith("redica_")]:
        panel[c] = panel[c].fillna(0)

    # ── Valisure — STATIC cross-sectional attribute ─────────────────────────────
    # Valisure scores are a 2024 snapshot attached as drug-level constants.
    # They do not vary over time; see module docstring for usage warning.
    val_path = OUT_DATA / "valisure_drug.parquet"
    if val_path.exists() or val_path.with_suffix(".csv").exists():
        val = read_table(val_path)
        val_keep = ["drug_norm", "valisure_mean_score", "valisure_min_score",
                    "valisure_max_score", "valisure_n_companies", "valisure_n_failing"]
        val = val[[c for c in val_keep if c in val.columns]].copy()
        panel = panel.merge(val, on="drug_norm", how="left")
        panel["has_valisure"] = panel["valisure_mean_score"].notna().astype(int)
    else:
        log.warning("valisure_drug.parquet not found — skipping Valisure join. "
                    "Run m05_valisure_scores.py first.")
        panel["has_valisure"] = 0

    # ── 3-month trailing window sums (for sparse signals) ──────────────────────
    # FAERS: use w3m sums in lead-lag analysis (quarters → months approximation).
    # Recalls and Redica: also smooth as a robustness check.
    panel = panel.sort_values(["drug_norm", "year", "month"]).reset_index(drop=True)
    sparse_cols = [
        "faers_n_reports", "faers_n_serious", "faers_severity_score",
        "recall_total", "recall_class_I", "recall_cgmp", "recall_contam", "recall_potency",
        "redica_n_oai", "redica_n_warning_letters", "redica_n_483_critical",
    ]
    sparse_cols = [c for c in sparse_cols if c in panel.columns]
    for c in sparse_cols:
        panel[f"{c}_w3m"] = (panel.groupby("drug_norm")[c]
                              .transform(lambda s: s.rolling(3, min_periods=1).sum()))

    log.info("Final monthly panel shape: %s", panel.shape)
    log.info("  drugs: %d | shortage_start months: %d | shortage_ongoing months: %d",
             panel["drug_norm"].nunique(),
             int(panel["shortage_start"].sum()),
             int(panel["shortage_ongoing"].sum()))
    log.info("  recall events in panel: %d | recall during ongoing shortage: %d",
             int(panel["recall_total"].sum()),
             int(panel["recall_during_shortage"].sum()))
    return panel.sort_values(["drug_norm", "year", "month"]).reset_index(drop=True)


# %%
def main():
    panel = build_monthly_panel()
    write_table(panel, OUT_DATA / "master_panel_monthly.parquet", log)
    return panel


if __name__ == "__main__":
    main()

# %%
