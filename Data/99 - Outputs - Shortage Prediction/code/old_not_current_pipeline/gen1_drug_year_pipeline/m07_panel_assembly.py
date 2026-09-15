# %%
"""
Module 7 — Master Valisure drug-year panel assembly.

We join all shortage and quality-signal sources onto the Valisure API × year
grid produced in m02. The key `drug_norm` is the exact API name from Valisure.
Features at year t (or rolling t-2..t) are used to predict shortage onset at
year t+1.

Target variable:
    y = 1 if shortage_started at year t+1, else 0

Final feature set (all are values at year t, used to predict year t+1):
    faers_n_reports_t, faers_n_serious_t, faers_severity_score_t
    faers_n_reports_w3, faers_n_serious_w3
    recall_total_t, recall_class_I_t, recall_cgmp_t, recall_potency_t
    recall_total_w3, recall_class_I_w3
    redica_n_oai_t, redica_n_wl_t, redica_total_obs_t
    redica_n_oai_w3
    valisure_mean_score, valisure_min_score, valisure_n_failing
    sole_source_ever, parenteral_ever
    prior_shortage_t, prior_shortage_w3
    483 text features (time-aware, most-recent snapshot per FEI ≤ year-end,
    averaged across drug's FEIs): severity_critmajor_share, scope_facilitywide_share,
    scope_multipleproducts_share, cultural_root_cause_share, capital_root_cause_share,
    remediation_none_share, remediation_weak_share, repeat_llm_share,
    contamination_llm_share, data_integrity_llm_share, investigation_llm_share,
    repeat_llm_only_share, contamination_llm_only_share, oos_oot_regex_share,
    wl_ref_regex_share, repeat_cross_insp_share, vc_labcontrols_share,
    vc_buildingsequipment_share, vc_qualitysystem_share, vc_productioncontrols_share
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from config import (OUT_DATA, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR,
                    ROLLING_WINDOW_YEARS, VALISURE_FEI, TEXT_TIMESERIES_CSV)
from utils import get_logger, read_table, write_table

log = get_logger("m07_panel", OUT_LOGS / "m07_panel.log")


def _rolling_sum(df: pd.DataFrame, key: str, cols: list[str], window: int) -> pd.DataFrame:
    """Per-key trailing window sum (inclusive of current year)."""
    df = df.sort_values([key, "year"]).copy()
    for c in cols:
        df[f"{c}_w{window}"] = (df.groupby(key)[c]
                                  .transform(lambda s: s.rolling(window, min_periods=1).sum()))
    return df


def build_panel() -> pd.DataFrame:
    # Base Valisure API × year shortage panel
    panel = read_table(OUT_DATA / "uutah_drug_year_panel.parquet")
    log.info("Base UUtah panel: %d rows, %d drugs", len(panel), panel["drug_norm"].nunique())

    # ---- FAERS ----
    faers = read_table(OUT_DATA / "faers_drug_year.parquet")
    faers["faers_severity_score"] = (
        faers.get("faers_sev_death", 0) * 4 +
        faers.get("faers_sev_life",  0) * 3 +
        faers.get("faers_sev_hosp",  0) * 2 +
        faers.get("faers_sev_disabled", 0) * 1
    )
    panel = panel.merge(faers, on=["drug_norm", "year"], how="left")
    for c in ["faers_n_reports","faers_n_serious","faers_severity_score",
              "faers_sev_death","faers_sev_life","faers_sev_hosp","faers_sev_disabled",
              "faers_sev_congenital","faers_sev_intervention","faers_sev_other_serious",
              "faers_sev_no_outcome"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)

    # ---- Recall ----
    recall = read_table(OUT_DATA / "recall_drug_year.parquet")
    panel = panel.merge(recall, on=["drug_norm", "year"], how="left")
    for c in [c for c in panel.columns if c.startswith("n_recalls") or c.startswith("n_class")
              or c.startswith("n_cgmp") or c.startswith("n_contam") or c.startswith("n_potency")
              or c.startswith("n_mislabel") or c.startswith("n_stability")
              or c.startswith("n_foreign") or c.startswith("n_dissolution")]:
        panel[c] = panel[c].fillna(0)
    panel = panel.rename(columns={
        "n_recalls_total":"recall_total","n_class_I":"recall_class_I",
        "n_class_II":"recall_class_II","n_class_III":"recall_class_III",
        "n_cgmp":"recall_cgmp","n_contam":"recall_contam","n_potency":"recall_potency",
        "n_mislabel":"recall_mislabel","n_stability":"recall_stability",
        "n_foreign":"recall_foreign","n_dissolution":"recall_dissolution",
    })

    # ---- Redica (drug-year level) ----
    redica = read_table(OUT_DATA / "redica_drug_year.parquet")
    panel = panel.merge(redica, on=["drug_norm", "year"], how="left")
    for c in [c for c in panel.columns if c.startswith("redica_")]:
        panel[c] = panel[c].fillna(0)

    # ---- Valisure (time-invariant per API) ----
    val = read_table(OUT_DATA / "valisure_drug.parquet")
    val = val[["drug_norm","valisure_mean_score","valisure_min_score","valisure_max_score",
               "valisure_n_companies","valisure_n_failing"]]
    panel = panel.merge(val, on="drug_norm", how="left")
    panel["has_valisure"] = panel["valisure_mean_score"].notna().astype(int)

    # ---- Stage 1 escalation risk scores (from m12) ----
    # For each (drug_norm, year t): take the most recent p_esc_24 per FEI as of
    # Dec 31 of year t, then aggregate across the drug's FEIs.
    # max_p_esc = worst-facility risk (captures tail risk)
    # mean_p_esc = average across all the drug's FEIs
    # FEIs with no text data contribute nothing to either aggregate.
    bridge = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping",
                           usecols=["API", "FEI_NUMBER"])
    bridge = (bridge.dropna(subset=["FEI_NUMBER"])
              .rename(columns={"API": "drug_norm", "FEI_NUMBER": "fei"}))
    bridge["fei"] = bridge["fei"].astype(int)

    p_esc_path = OUT_DATA / "fei_p_esc.parquet"
    if p_esc_path.exists():
        p_esc = pd.read_parquet(p_esc_path)
        p_esc["fei"] = p_esc["fei"].astype(int)
        p_esc["snapshot_date"] = pd.to_datetime(p_esc["snapshot_date"])

        fei_year_rows = []
        for yr in sorted(panel["year"].unique()):
            cutoff = pd.Timestamp(f"{int(yr)}-12-31")
            avail = p_esc[p_esc["snapshot_date"] <= cutoff]
            if avail.empty:
                continue
            latest = (avail.loc[avail.groupby("fei")["snapshot_date"].idxmax(),
                                 ["fei", "p_esc_24"]]
                          .assign(year=yr))
            fei_year_rows.append(latest)

        if fei_year_rows:
            fei_year_p = pd.concat(fei_year_rows, ignore_index=True)
            drug_year_p = (fei_year_p.merge(bridge, on="fei", how="inner")
                           .groupby(["drug_norm", "year"])["p_esc_24"]
                           .agg(max_p_esc="max", mean_p_esc="mean")
                           .reset_index())
            panel = panel.merge(drug_year_p, on=["drug_norm", "year"], how="left")
            n_cov = panel["max_p_esc"].notna().groupby(panel["drug_norm"]).any().sum()
            log.info("Stage 1 p_esc joined: %d / %d drugs have ≥1 covered year",
                     n_cov, panel["drug_norm"].nunique())
        else:
            log.warning("fei_p_esc.parquet has no rows in panel year range")
    else:
        log.warning("fei_p_esc.parquet not found — run m12 first to generate Stage 1 scores")
            for c in TEXT_COLS:
                panel[c] = np.nan
    else:
        log.warning("TEXT_TIMESERIES_CSV not found; text features will be NaN")
        for c in TEXT_COLS:
            panel[c] = np.nan
    # Leave NaN for drug-years with no snapshot coverage (model handles with fillna(0))

    # ---- Rolling window features ----
    rolling_cols = [
        "faers_n_reports","faers_n_serious","faers_severity_score",
        "recall_total","recall_class_I","recall_cgmp","recall_potency","recall_contam",
        "redica_n_oai","redica_n_warning_letters","redica_n_483_critical",
    ]
    rolling_cols = [c for c in rolling_cols if c in panel.columns]
    panel = _rolling_sum(panel, "drug_norm", rolling_cols, ROLLING_WINDOW_YEARS)

    # ---- Prior shortage history (predictor; uses year t and prior only) ----
    panel = panel.sort_values(["drug_norm","year"]).copy()
    panel["prior_shortage_t"] = (panel.groupby("drug_norm")["shortage_started"].shift(1).fillna(0).astype(int))
    panel["prior_shortage_w3"] = (panel.groupby("drug_norm")["shortage_started"]
                                  .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum().fillna(0)))

    # ---- Drug-level stable attrs (sole_source_ever, parenteral_ever) ----
    stable = (panel.groupby("drug_norm")
              .agg(sole_source_ever=("sole_source","max"),
                   parenteral_ever=("parenteral","max"))
              .reset_index())
    panel = panel.drop(columns=["sole_source","parenteral"]).merge(stable, on="drug_norm", how="left")

    # ---- Outcome: y = shortage_started at year t+1 ----
    panel = panel.sort_values(["drug_norm","year"])
    panel["y_next_year_shortage"] = (panel.groupby("drug_norm")["shortage_started"].shift(-1).astype("Int64"))

    log.info("Final panel shape: %s", panel.shape)
    log.info("Outcome prevalence (y=1): %.3f", panel["y_next_year_shortage"].dropna().mean())
    log.info("Panel year range: %d–%d", panel["year"].min(), panel["year"].max())

    return panel


def main():
    panel = build_panel()
    write_table(panel, OUT_DATA / "master_panel.parquet", log)
    return panel


if __name__ == "__main__":
    main()

# %%
