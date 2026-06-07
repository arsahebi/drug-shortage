# %%
"""
Module 2 — UUtah shortage panel for the Valisure API universe.

Parses the EFox UUtah Drug Information Service file into a drug-year panel:

    columns: drug_norm, year, shortage_started, shortage_ongoing, n_starts,
             reason_manufacturing, reason_supply_demand, sole_source, parenteral

`drug_norm` is the exact API name from Valisure, not a generic normalized token.
UUtah formulation strings are matched onto those canonical Valisure names.
"""

from __future__ import annotations
import pandas as pd

from config import VALISURE_CSV, UUTAH_FILE, OUT_DATA, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, to_year, write_table

log = get_logger("m02_uutah", OUT_LOGS / "m02_uutah.log")


def _categorize_reason(reason: str) -> dict[str, int]:
    """Map UUtah free-text reason into a small set of buckets."""
    r = (reason or "").lower()
    return {
        "reason_manufacturing": int(any(k in r for k in
            ["manufactur", "production", "plant", "facility", "quality", "gmp", "fda"])),
        "reason_supply_demand": int(any(k in r for k in
            ["supply", "demand", "increased demand", "shortage of supply"])),
        "reason_raw_material" : int("raw material" in r or "api" in r),
        "reason_business"     : int("business" in r or "discontin" in r),
        "reason_regulatory"   : int("regulatory" in r or "recall" in r),
        "reason_unknown"      : int(r.strip() in ("", "unknown")),
    }


def build_drug_year_panel() -> pd.DataFrame:
    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher = ValisureDrugMatcher(api_names)

    raw = pd.read_excel(UUTAH_FILE, header=1)
    raw = raw.rename(columns={raw.columns[0]: "drug_name"})
    raw["drug_norm"] = raw["drug_name"].map(matcher.match)
    raw = raw.dropna(subset=["drug_norm"]).copy()
    drug_names = (raw.groupby("drug_norm")["drug_name"]
                  .apply(lambda s: "; ".join(sorted(s.dropna().astype(str).unique())))
                  .reset_index())

    # Normalize date columns
    raw["yr_int"] = to_year(raw["yr"])
    raw["date_notified"] = pd.to_datetime(raw["Date Notified"], errors="coerce")
    raw["date_resolved"] = pd.to_datetime(raw["Date Resolved"], errors="coerce")
    raw["start_year"] = raw["date_notified"].dt.year.fillna(raw["yr_int"]).astype("Int64")
    raw["end_year"]   = raw["date_resolved"].dt.year.astype("Int64")

    # Sole source / parenteral flags
    raw["sole_source"] = raw["Sole source yes or no"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "y": 1, "no": 0, "n": 0}).fillna(0).astype(int)
    raw["parenteral"]  = raw["parenteral"].astype(str).str.strip().str.lower().map(
        {"y": 1, "yes": 1, "n": 0, "no": 0}).fillna(0).astype(int)

    reasons = raw["Reason"].astype(str).apply(_categorize_reason).apply(pd.Series)
    raw = pd.concat([raw, reasons], axis=1)

    # Build the drug × year grid for the analysis window
    years = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    grid = pd.MultiIndex.from_product([api_names, years], names=["drug_norm", "year"]).to_frame(index=False)
    grid = grid.merge(drug_names, on="drug_norm", how="left")

    # Per-year shortage onsets
    raw_in_window = raw[(raw["start_year"] >= PANEL_START_YEAR) & (raw["start_year"] <= PANEL_END_YEAR)]
    starts = (raw_in_window.groupby(["drug_norm", "start_year"])
              .agg(n_starts=("drug_name", "size"),
                   reason_manufacturing=("reason_manufacturing", "max"),
                   reason_supply_demand=("reason_supply_demand", "max"),
                   reason_raw_material=("reason_raw_material", "max"),
                   reason_business=("reason_business", "max"),
                   reason_regulatory=("reason_regulatory", "max"),
                   reason_unknown=("reason_unknown", "max"),
                   sole_source=("sole_source", "max"),
                   parenteral=("parenteral", "max"))
              .reset_index()
              .rename(columns={"start_year": "year"}))
    starts["shortage_started"] = 1

    panel = grid.merge(starts, on=["drug_norm", "year"], how="left")
    panel["shortage_started"] = panel["shortage_started"].fillna(0).astype(int)
    panel["n_starts"] = panel["n_starts"].fillna(0).astype(int)
    for c in ["reason_manufacturing", "reason_supply_demand", "reason_raw_material",
              "reason_business", "reason_regulatory", "reason_unknown",
              "sole_source", "parenteral"]:
        panel[c] = panel[c].fillna(0).astype(int)

    # "Ongoing" flag: any shortage active any time during the year
    ongoing_rows = []
    for _, r in raw[["drug_norm", "start_year", "end_year"]].dropna(subset=["drug_norm", "start_year"]).iterrows():
        sy = int(r["start_year"])
        ey = int(r["end_year"]) if pd.notna(r["end_year"]) else PANEL_END_YEAR
        for y in range(max(sy, PANEL_START_YEAR), min(ey, PANEL_END_YEAR) + 1):
            ongoing_rows.append((r["drug_norm"], y))
    ongoing = pd.DataFrame(ongoing_rows, columns=["drug_norm", "year"]).drop_duplicates()
    ongoing["shortage_ongoing"] = 1
    panel = panel.merge(ongoing, on=["drug_norm", "year"], how="left")
    panel["shortage_ongoing"] = panel["shortage_ongoing"].fillna(0).astype(int)

    log.info("Panel shape: %s | unique drugs: %d | years: %d",
             panel.shape, panel["drug_norm"].nunique(),
             panel["year"].nunique())
    log.info("Shortage-start prevalence (rows): %.3f",
             panel["shortage_started"].mean())
    return panel

# %%
def main():
    panel = build_drug_year_panel()
    write_table(panel, OUT_DATA / "uutah_drug_year_panel.parquet", log)
    return panel


if __name__ == "__main__":
    main()

# %%
