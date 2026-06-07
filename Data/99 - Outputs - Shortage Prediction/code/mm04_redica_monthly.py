# %%
"""
Module MM04 — Monthly Redica facility-inspection features.

Aggregates Redica inspection events by drug_norm × (year, month), keyed on
the Event Date column.  Drug linkage reuses the same FEI → Valisure API
mapping as the annual pipeline (m06_redica_features.load_fei_to_api).

Output columns per (drug_norm, year, month):
    redica_n_inspections, redica_n_oai, redica_n_vai,
    redica_n_warning_letters, redica_n_483_critical,
    redica_max_total_obs, redica_n_facilities
"""

from __future__ import annotations
import pandas as pd

from config import (
    REDICA_CSV, VALISURE_CSV, VALISURE_FEI,
    OUT_DATA, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("mm04_redica_monthly", OUT_LOGS / "mm04_redica_monthly.log")


def _load_fei_to_api() -> pd.DataFrame:
    """Read Valisure FEI→drug mapping.  Logic mirrors m06_redica_features.load_fei_to_api."""
    fei = pd.read_excel(VALISURE_FEI, sheet_name=0)
    fei.columns = [str(c).strip() for c in fei.columns]
    fei_col = next((c for c in fei.columns if "fei" in c.lower()), None)
    api_col = next((c for c in fei.columns
                    if "api" in c.lower() or "drug" in c.lower() or "molecule" in c.lower()), None)
    if fei_col is None:
        raise ValueError("Could not find FEI column in Valisure FEIs file")
    if api_col is None:
        api_col = [c for c in fei.columns if c != fei_col][0]
        log.warning("No API column identified; using '%s' as API column", api_col)
    out = fei[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "api"})
    out["fei"] = pd.to_numeric(out["fei"], errors="coerce").astype("Int64")
    matcher = ValisureDrugMatcher(load_valisure_api_names(VALISURE_CSV))
    out["drug_norm"] = out["api"].astype(str).map(matcher.match)
    out = out.dropna(subset=["fei", "drug_norm"]).drop_duplicates()
    log.info("FEI→API mapping: %d rows, %d unique FEIs, %d unique drugs",
             len(out), out["fei"].nunique(), out["drug_norm"].nunique())
    return out


def build_redica_monthly() -> pd.DataFrame:
    df = pd.read_csv(REDICA_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df["event_dt"] = pd.to_datetime(df["Event Date"], errors="coerce")
    df["year"]     = df["event_dt"].dt.year.astype("Int64")
    df["month"]    = df["event_dt"].dt.month.astype("Int64")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]
    log.info("Redica events in panel window: %d", len(df))

    df["FEI"] = pd.to_numeric(df["FEI"], errors="coerce").astype("Int64")
    cls = df["Classification"].astype(str).str.strip()
    df["is_oai"]         = (cls == "OAI").astype(int)
    df["is_vai"]         = (cls == "VAI").astype(int)
    df["has_wl"]         = (pd.to_numeric(df.get("Warning Letter", 0), errors="coerce")
                            .fillna(0).astype(int))
    df["n_483_critical"] = (pd.to_numeric(df.get("483 critical", 0), errors="coerce")
                            .fillna(0).astype(int))
    df["total_obs"]      = (pd.to_numeric(df.get("Total Observations", 0), errors="coerce")
                            .fillna(0).astype(int))

    # Aggregate to FEI × month
    fei_month = (df.groupby(["FEI", "year", "month"], as_index=False)
                 .agg(
                     n_inspections=("event_dt",      "count"),
                     n_oai=        ("is_oai",         "sum"),
                     n_vai=        ("is_vai",          "sum"),
                     n_wl=         ("has_wl",          "sum"),
                     n_483_crit=   ("n_483_critical",  "sum"),
                     max_obs=      ("total_obs",       "max"),
                 )
                 .rename(columns={"FEI": "fei"}))

    # Map FEI → drug_norm via Valisure FEI table
    fei_api = _load_fei_to_api()
    merged = fei_month.merge(fei_api, on="fei", how="inner")

    out = (merged.groupby(["drug_norm", "year", "month"], as_index=False)
           .agg(
               redica_n_inspections=    ("n_inspections", "sum"),
               redica_n_oai=            ("n_oai",          "sum"),
               redica_n_vai=            ("n_vai",          "sum"),
               redica_n_warning_letters=("n_wl",           "sum"),
               redica_n_483_critical=   ("n_483_crit",     "sum"),
               redica_max_total_obs=    ("max_obs",        "max"),
               redica_n_facilities=     ("fei",            "nunique"),
           ))
    log.info("Redica monthly rows: %d | drugs: %d | months with data: %d",
             len(out), out["drug_norm"].nunique(), out[["year", "month"]].drop_duplicates().shape[0])
    return out


# %%
def main():
    out = build_redica_monthly()
    write_table(out, OUT_DATA / "redica_monthly.parquet", log)
    return out


if __name__ == "__main__":
    main()

# %%
