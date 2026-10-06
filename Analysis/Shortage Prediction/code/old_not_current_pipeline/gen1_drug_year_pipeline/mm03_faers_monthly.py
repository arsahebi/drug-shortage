# %%
"""
Module MM03 — Monthly FAERS adverse-event features.

RESOLUTION NOTE:
    The processed FAERS file used in this project (faers_valisure_14_drugs_*.csv)
    carries a 'period' column in YYYY-QN format (e.g. "2015Q1") rather than an
    exact date.  Monthly features are therefore approximated by assigning each
    quarter's event count to the FIRST month of that quarter:
        Q1 → January (month 1)
        Q2 → April   (month 4)
        Q3 → July    (month 7)
        Q4 → October (month 10)

    Consequently FAERS columns will be non-zero only in months 1, 4, 7, 10.
    Use the 3-month trailing window sums (*_w3m, added in mm05) for lead-lag
    analysis rather than the raw monthly values.

Output columns per (drug_norm, year, month):
    faers_n_reports, faers_n_serious,
    faers_sev_death, faers_sev_life, faers_sev_hosp, faers_sev_disabled,
    faers_sev_congenital, faers_sev_intervention, faers_sev_other_serious,
    faers_sev_no_outcome, faers_severity_score

Logic mirrors m03_faers_features.py; adds month derivation via 'period'.
"""

from __future__ import annotations
import pandas as pd

from config import (
    FAERS_ALL, VALISURE_CSV,
    OUT_DATA, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("mm03_faers_monthly", OUT_LOGS / "mm03_faers_monthly.log")

USE_COLS = ["primaryid", "drugname", "prod_ai", "year", "severity", "period"]
CHUNK = 200_000


def _classify_severity(s: pd.Series) -> pd.DataFrame:
    sl = s.astype(str).str.lower().fillna("")
    return pd.DataFrame({
        "sev_death":         sl.str.contains(r"death|died|fatal",               regex=True).astype("int8"),
        "sev_life":          sl.str.contains(r"life.?threat",                   regex=True).astype("int8"),
        "sev_hosp":          sl.str.contains(r"hospital",                       regex=True).astype("int8"),
        "sev_disabled":      sl.str.contains(r"disab",                          regex=True).astype("int8"),
        "sev_congenital":    sl.str.contains(r"congenital|birth defect",         regex=True).astype("int8"),
        "sev_intervention":  sl.str.contains(r"required intervention|intervention", regex=True).astype("int8"),
        "sev_other_serious": sl.str.contains(r"other serious",                  regex=True).astype("int8"),
        "sev_no_outcome":    sl.str.contains(r"no outcome reported",            regex=True).astype("int8"),
    })


def _derive_serious_flag(severity: pd.Series) -> pd.Series:
    sl = severity.astype(str).str.lower().str.strip()
    pat = (r"death|died|fatal|life.?threat|hospital|disab|congenital|"
           r"birth defect|required intervention|other serious")
    return sl.str.contains(pat, regex=True, na=False).astype("int8")


def _quarter_to_month(period: pd.Series) -> pd.Series:
    """Map 'YYYY-QN' → first month of quarter (1, 4, 7, 10). Returns Int64."""
    q = pd.to_numeric(period.str.extract(r"Q(\d)", expand=False), errors="coerce")
    return ((q - 1) * 3 + 1).astype("Int64")


def aggregate_faers_monthly(api_names: list[str]) -> pd.DataFrame:
    matcher = ValisureDrugMatcher(api_names)

    agg_rows: list[pd.DataFrame] = []
    rows_seen = 0
    rows_matched = 0

    available_cols: list[str] | None = None
    reader = pd.read_csv(
        FAERS_ALL,
        chunksize=CHUNK,
        low_memory=True,
        dtype={"drugname": "string", "prod_ai": "string",
               "severity": "string", "period": "string"},
    )

    for i, chunk in enumerate(reader):
        # On first chunk, figure out which USE_COLS are actually present
        if available_cols is None:
            available_cols = [c for c in USE_COLS if c in chunk.columns]
            missing = set(USE_COLS) - set(available_cols)
            if missing:
                log.warning("FAERS columns not found (will be skipped): %s", missing)

        chunk = chunk[available_cols].copy()
        rows_seen += len(chunk)

        match_text = chunk.get("prod_ai", pd.Series("", index=chunk.index)).fillna("") + \
                     " " + chunk.get("drugname", pd.Series("", index=chunk.index)).fillna("")
        match = match_text.map(matcher.match)
        hit = chunk.assign(drug_norm=match).dropna(subset=["drug_norm"])
        if hit.empty:
            if (i + 1) % 5 == 0:
                log.info("  chunk %d | seen=%d | matched=%d", i + 1, rows_seen, rows_matched)
            continue
        rows_matched += len(hit)

        hit["year"] = pd.to_numeric(hit["year"], errors="coerce").astype("Int64")

        # Derive month from period (quarterly → first month of quarter)
        if "period" in hit.columns:
            hit["month"] = _quarter_to_month(hit["period"].fillna("").astype(str))
        else:
            # Fallback: cannot determine sub-annual timing; skip
            log.warning("'period' column absent in chunk %d; skipping FAERS monthly split", i)
            continue

        hit = hit[
            (hit["year"]  >= PANEL_START_YEAR) & (hit["year"]  <= PANEL_END_YEAR) &
            hit["month"].notna()
        ]
        if hit.empty:
            continue

        sev = _classify_severity(hit["severity"])
        hit = pd.concat([hit.reset_index(drop=True), sev.reset_index(drop=True)], axis=1)
        hit["serious_flag"] = _derive_serious_flag(hit["severity"])

        g = hit.groupby(["drug_norm", "year", "month"],
                        dropna=False, observed=True).agg(
            faers_n_reports=      ("primaryid",        "count"),
            faers_n_serious=      ("serious_flag",     "sum"),
            faers_sev_death=      ("sev_death",        "sum"),
            faers_sev_life=       ("sev_life",         "sum"),
            faers_sev_hosp=       ("sev_hosp",         "sum"),
            faers_sev_disabled=   ("sev_disabled",     "sum"),
            faers_sev_congenital= ("sev_congenital",   "sum"),
            faers_sev_intervention=("sev_intervention","sum"),
            faers_sev_other_serious=("sev_other_serious","sum"),
            faers_sev_no_outcome= ("sev_no_outcome",   "sum"),
        ).reset_index()
        agg_rows.append(g)

        if (i + 1) % 5 == 0:
            log.info("  chunk %d | seen=%d | matched=%d", i + 1, rows_seen, rows_matched)

    log.info("Total seen=%d | matched=%d", rows_seen, rows_matched)

    if not agg_rows:
        log.warning("No FAERS rows matched any target drug")
        return pd.DataFrame(columns=["drug_norm", "year", "month", "faers_n_reports"])

    out = (pd.concat(agg_rows, ignore_index=True)
           .groupby(["drug_norm", "year", "month"], as_index=False)
           .sum(numeric_only=True))

    # Severity score (same weights as annual pipeline)
    out["faers_severity_score"] = (
        out.get("faers_sev_death",    pd.Series(0, index=out.index)) * 4 +
        out.get("faers_sev_life",     pd.Series(0, index=out.index)) * 3 +
        out.get("faers_sev_hosp",     pd.Series(0, index=out.index)) * 2 +
        out.get("faers_sev_disabled", pd.Series(0, index=out.index)) * 1
    )

    log.info("FAERS monthly rows: %d | drugs: %d | months with data: %d",
             len(out), out["drug_norm"].nunique(), out["month"].nunique())
    return out


# %%
def main():
    api_names = load_valisure_api_names(VALISURE_CSV)
    out = aggregate_faers_monthly(api_names)
    write_table(out, OUT_DATA / "faers_monthly.parquet", log)
    return out


if __name__ == "__main__":
    main()

# %%
