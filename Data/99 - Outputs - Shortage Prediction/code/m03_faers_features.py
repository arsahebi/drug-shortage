# %%
"""
Module 3 — FAERS quality-signal features per drug-year.

The ANDA-linked FAERS panel is ~593 MB. We stream it in modest chunks and
match drugs vectorized via regex on the active-ingredient field. We restrict
the column set we read to keep memory low.

Output columns per (drug_norm, year):
    faers_n_reports
    faers_n_serious
    faers_sev_death, faers_sev_life, faers_sev_hosp, faers_sev_disabled,
    faers_sev_congenital, faers_sev_intervention, faers_sev_other_serious,
    faers_sev_no_outcome
"""

from __future__ import annotations
import pandas as pd

from config import FAERS_ALL, OUT_DATA, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR, VALISURE_CSV
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("m03_faers", OUT_LOGS / "m03_faers.log")

USE_COLS = ["primaryid", "drugname", "prod_ai", "year", "severity"]
CHUNK = 200_000


def _classify_severity(s: pd.Series) -> pd.DataFrame:
    sl = s.astype(str).str.lower().fillna("")
    return pd.DataFrame({
        "sev_death":    sl.str.contains(r"death|died|fatal", regex=True, na=False).astype("int8"),
        "sev_life":     sl.str.contains(r"life.?threat", regex=True, na=False).astype("int8"),
        "sev_hosp":     sl.str.contains(r"hospital", regex=True, na=False).astype("int8"),
        "sev_disabled": sl.str.contains(r"disab", regex=True, na=False).astype("int8"),
        "sev_congenital": sl.str.contains(r"congenital|birth defect", regex=True, na=False).astype("int8"),
        "sev_intervention": sl.str.contains(r"required intervention|intervention", regex=True, na=False).astype("int8"),
        "sev_other_serious": sl.str.contains(r"other serious", regex=True, na=False).astype("int8"),
        "sev_no_outcome": sl.str.contains(r"no outcome reported", regex=True, na=False).astype("int8"),
    })


def _derive_serious_flag(severity: pd.Series) -> pd.Series:
    """FAERS serious outcome flag inferred from the processed severity label."""
    sl = severity.astype(str).str.lower().str.strip()
    serious_pat = (
        r"death|died|fatal|life.?threat|hospital|disab|congenital|"
        r"birth defect|required intervention|other serious"
    )
    return sl.str.contains(serious_pat, regex=True, na=False).astype("int8")


def aggregate_faers(api_names: list[str]) -> pd.DataFrame:
    matcher = ValisureDrugMatcher(api_names)

    log.info("Streaming FAERS file %s (chunk=%d)…", FAERS_ALL.name, CHUNK)
    agg_rows = []
    rows_seen = 0
    rows_matched = 0

    reader = pd.read_csv(FAERS_ALL, usecols=[c for c in USE_COLS],
                         chunksize=CHUNK, low_memory=True,
                         dtype={"drugname":"string","prod_ai":"string","severity":"string"})
    for i, chunk in enumerate(reader):
        rows_seen += len(chunk)

        match_text = chunk["prod_ai"].fillna("") + " " + chunk["drugname"].fillna("")
        match = match_text.map(matcher.match)
        hit = chunk.assign(drug_norm=match).dropna(subset=["drug_norm"])
        if hit.empty:
            if (i + 1) % 5 == 0:
                log.info("  chunk %d | rows_seen=%d | matched_so_far=%d", i+1, rows_seen, rows_matched)
            continue
        rows_matched += len(hit)

        hit["year"] = pd.to_numeric(hit["year"], errors="coerce").astype("Int64")
        hit = hit[(hit["year"] >= PANEL_START_YEAR) & (hit["year"] <= PANEL_END_YEAR)]
        if hit.empty:
            continue

        sev = _classify_severity(hit["severity"])
        hit = pd.concat([hit.reset_index(drop=True), sev.reset_index(drop=True)], axis=1)
        hit["serious_flag"] = _derive_serious_flag(hit["severity"])

        g = hit.groupby(["drug_norm", "year"], dropna=False, observed=True).agg(
            faers_n_reports=("primaryid", "count"),
            faers_n_serious=("serious_flag", "sum"),
            faers_sev_death=("sev_death", "sum"),
            faers_sev_life=("sev_life", "sum"),
            faers_sev_hosp=("sev_hosp", "sum"),
            faers_sev_disabled=("sev_disabled", "sum"),
            faers_sev_congenital=("sev_congenital", "sum"),
            faers_sev_intervention=("sev_intervention", "sum"),
            faers_sev_other_serious=("sev_other_serious", "sum"),
            faers_sev_no_outcome=("sev_no_outcome", "sum"),
        ).reset_index()
        agg_rows.append(g)

        if (i + 1) % 5 == 0:
            log.info("  chunk %d | rows_seen=%d | matched_so_far=%d", i+1, rows_seen, rows_matched)

    if not agg_rows:
        log.warning("No FAERS rows matched targets")
        return pd.DataFrame(columns=["drug_norm","year","faers_n_reports","faers_n_serious",
                                     "faers_sev_death","faers_sev_life","faers_sev_hosp",
                                     "faers_sev_disabled","faers_sev_congenital",
                                     "faers_sev_intervention","faers_sev_other_serious",
                                     "faers_sev_no_outcome"])
    out = pd.concat(agg_rows, ignore_index=True).groupby(
        ["drug_norm", "year"], as_index=False).sum(numeric_only=True)
    log.info("FAERS aggregation rows: %d | drugs covered: %d | years: %d",
             len(out), out["drug_norm"].nunique(), out["year"].nunique())
    return out

# %%
def main():
    api_names = load_valisure_api_names(VALISURE_CSV)
    faers = aggregate_faers(api_names)
    write_table(faers, OUT_DATA / "faers_drug_year.parquet", log)
    return faers


if __name__ == "__main__":
    main()

# %%
