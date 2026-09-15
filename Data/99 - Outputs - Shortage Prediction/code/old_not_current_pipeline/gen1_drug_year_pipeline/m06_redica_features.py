# %%
"""
Module 6 — Redica facility-level features per drug-year.

The Redica panel is event-level (FEI × inspection date × classification).
We aggregate to facility-year severity, then to *drug-year* by joining via
Valisure's FEI list (FEIs_March 2026.xlsx) which links FEIs to APIs.

The output drug-year columns:
    redica_n_inspections, redica_n_oai, redica_n_vai,
    redica_n_warning_letters, redica_n_483_critical, redica_max_total_obs
"""

from __future__ import annotations
import pandas as pd

from config import REDICA_CSV, VALISURE_CSV, VALISURE_FEI, OUT_DATA, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("m06_redica", OUT_LOGS / "m06_redica.log")


def load_fei_to_api() -> pd.DataFrame:
    """Read the Valisure FEI map. Schema may vary; we try to be defensive."""
    fei = pd.read_excel(VALISURE_FEI, sheet_name=0)
    fei.columns = [str(c).strip() for c in fei.columns]
    log.info("FEI map columns: %s", list(fei.columns))
    # Heuristic: find an FEI column and an API/drug column
    fei_col = next((c for c in fei.columns if "fei" in c.lower()), None)
    api_col = next((c for c in fei.columns if "api" in c.lower() or "drug" in c.lower() or "molecule" in c.lower()), None)
    if not fei_col:
        raise ValueError("Could not find FEI column in Valisure FEIs file")
    if not api_col:
        # Fallback: assume the first column besides FEI is the API
        api_col = [c for c in fei.columns if c != fei_col][0]
        log.warning("No API column found; using '%s' as API column", api_col)
    out = fei[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "api"})
    out["fei"] = pd.to_numeric(out["fei"], errors="coerce").astype("Int64")
    matcher = ValisureDrugMatcher(load_valisure_api_names(VALISURE_CSV))
    out["drug_norm"] = out["api"].astype(str).map(matcher.match)
    out = out.dropna(subset=["fei", "drug_norm"]).drop_duplicates()
    log.info("FEI→API mapping rows: %d", len(out))
    return out


def aggregate_redica_by_year() -> pd.DataFrame:
    df = pd.read_csv(REDICA_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_datetime(df["Event Date"], errors="coerce").dt.year.astype("Int64")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]

    df["FEI"] = pd.to_numeric(df["FEI"], errors="coerce").astype("Int64")
    cls = df["Classification"].astype(str).str.strip()
    df["is_oai"] = (cls == "OAI").astype(int)
    df["is_vai"] = (cls == "VAI").astype(int)
    df["is_nai"] = (cls == "NAI").astype(int)
    df["has_wl"] = pd.to_numeric(df.get("Warning Letter", 0), errors="coerce").fillna(0).astype(int)
    df["n_483_critical"] = pd.to_numeric(df.get("483 critical", 0), errors="coerce").fillna(0).astype(int)
    df["total_obs"] = pd.to_numeric(df.get("Total Observations", 0), errors="coerce").fillna(0).astype(int)

    agg = df.groupby(["FEI", "year"], as_index=False).agg(
        n_inspections=("Event Date", "count"),
        n_oai=("is_oai", "sum"),
        n_vai=("is_vai", "sum"),
        n_nai=("is_nai", "sum"),
        n_wl=("has_wl", "sum"),
        n_483_critical=("n_483_critical", "sum"),
        max_total_obs=("total_obs", "max"),
    )
    log.info("Redica FEI-year rows: %d (unique FEIs %d)",
             len(agg), agg["FEI"].nunique())
    return agg.rename(columns={"FEI": "fei"})


def build_redica_drug_year(fei_year: pd.DataFrame, fei_api: pd.DataFrame) -> pd.DataFrame:
    merged = fei_year.merge(fei_api, on="fei", how="inner")
    out = merged.groupby(["drug_norm", "year"], as_index=False).agg(
        redica_n_inspections=("n_inspections", "sum"),
        redica_n_oai=("n_oai", "sum"),
        redica_n_vai=("n_vai", "sum"),
        redica_n_warning_letters=("n_wl", "sum"),
        redica_n_483_critical=("n_483_critical", "sum"),
        redica_max_total_obs=("max_total_obs", "max"),
        redica_n_facilities=("fei", "nunique"),
    )
    log.info("Redica drug-year rows: %d (drugs %d)",
             len(out), out["drug_norm"].nunique())
    return out


def main():
    fei_year = aggregate_redica_by_year()
    fei_api  = load_fei_to_api()
    out = build_redica_drug_year(fei_year, fei_api)
    write_table(out, OUT_DATA / "redica_drug_year.parquet", log)
    write_table(fei_year, OUT_DATA / "redica_fei_year.parquet", log)
    return out


if __name__ == "__main__":
    main()

# %%
