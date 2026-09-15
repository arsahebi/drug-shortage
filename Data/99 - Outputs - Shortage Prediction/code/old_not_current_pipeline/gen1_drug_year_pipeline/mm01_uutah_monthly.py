# %%
"""
Module MM01 — Monthly shortage panel from UUtah data.

Grain: drug_norm × (year, month) — Jan 2015 to Dec 2024 (14 × 120 = 1,680 rows).

Columns produced:
    drug_norm, year, month, period (YYYY-MM)
    shortage_start    — 1 if a new shortage episode began this calendar month
    shortage_ongoing  — 1 if any shortage was active at any point this month
    reason_manufacturing, reason_supply_demand, reason_raw_material,
    reason_business, reason_regulatory, reason_unknown
        (from any shortage starting this month; max across concurrent starts)
    sole_source_ever, parenteral_ever  (drug-level stable attrs)

Analogue to m02_uutah_panel.py at annual granularity; extend/replace
nothing in that module — this output lives alongside it.
"""

from __future__ import annotations
import pandas as pd

from config import (
    UUTAH_FILE, VALISURE_CSV,
    OUT_DATA, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("mm01_uutah_monthly", OUT_LOGS / "mm01_uutah_monthly.log")

PANEL_START = pd.Timestamp(PANEL_START_YEAR, 1, 1)
PANEL_END   = pd.Timestamp(PANEL_END_YEAR,  12, 31)


def _categorize_reason(reason: str) -> dict[str, int]:
    """Map UUtah free-text reason into a small set of buckets (mirrors m02)."""
    r = (reason or "").lower()
    return {
        "reason_manufacturing": int(any(k in r for k in
            ["manufactur", "production", "plant", "facility", "quality", "gmp", "fda"])),
        "reason_supply_demand": int(any(k in r for k in
            ["supply", "demand", "increased demand", "shortage of supply"])),
        "reason_raw_material":  int("raw material" in r or " api " in r),
        "reason_business":      int("business" in r or "discontin" in r),
        "reason_regulatory":    int("regulatory" in r or "recall" in r),
        "reason_unknown":       int(r.strip() in ("", "unknown")),
    }


def build_monthly_panel() -> pd.DataFrame:
    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher   = ValisureDrugMatcher(api_names)

    raw = pd.read_excel(UUTAH_FILE, header=1)
    raw = raw.rename(columns={raw.columns[0]: "drug_name"})
    raw["drug_norm"] = raw["drug_name"].map(matcher.match)
    raw = raw.dropna(subset=["drug_norm"]).copy()

    # Parse dates; treat unresolved shortages as ongoing through end of panel window
    raw["date_notified"] = pd.to_datetime(raw["Date Notified"], errors="coerce")
    raw["date_resolved"] = pd.to_datetime(raw["Date Resolved"],  errors="coerce")
    raw["date_resolved"] = raw["date_resolved"].fillna(PANEL_END)

    # Structural flags
    raw["sole_source"] = (raw["Sole source yes or no"].astype(str).str.strip().str.lower()
                          .map({"yes": 1, "y": 1, "no": 0, "n": 0}).fillna(0).astype(int))
    raw["parenteral"]  = (raw["parenteral"].astype(str).str.strip().str.lower()
                          .map({"y": 1, "yes": 1, "n": 0, "no": 0}).fillna(0).astype(int))

    reasons = raw["Reason"].astype(str).apply(_categorize_reason).apply(pd.Series)
    raw = pd.concat([raw, reasons], axis=1)

    # Drug-level stable attrs (ever sole-source or parenteral across all shortage records)
    stable = (raw.groupby("drug_norm")
              .agg(sole_source_ever=("sole_source", "max"),
                   parenteral_ever=("parenteral",   "max"))
              .reset_index())

    # ── Full drug × month grid ──────────────────────────────────────────────────
    months = pd.date_range(start=PANEL_START, end=PANEL_END, freq="MS")
    grid = pd.MultiIndex.from_product(
        [api_names, months], names=["drug_norm", "_month"]
    ).to_frame(index=False)
    grid["year"]   = grid["_month"].dt.year
    grid["month"]  = grid["_month"].dt.month
    grid["period"] = grid["_month"].dt.strftime("%Y-%m")
    grid = grid.drop(columns=["_month"])

    # ── shortage_start: month in which Date Notified falls ─────────────────────
    raw_valid = raw.dropna(subset=["date_notified"]).copy()
    raw_valid["start_year"]  = raw_valid["date_notified"].dt.year.astype("Int64")
    raw_valid["start_month"] = raw_valid["date_notified"].dt.month.astype("Int64")
    raw_in_window = raw_valid[
        (raw_valid["start_year"] >= PANEL_START_YEAR) &
        (raw_valid["start_year"] <= PANEL_END_YEAR)
    ]
    starts = (raw_in_window
              .groupby(["drug_norm", "start_year", "start_month"], as_index=False)
              .agg(
                  n_episodes=          ("drug_name",           "count"),
                  reason_manufacturing=("reason_manufacturing", "max"),
                  reason_supply_demand=("reason_supply_demand", "max"),
                  reason_raw_material= ("reason_raw_material",  "max"),
                  reason_business=     ("reason_business",      "max"),
                  reason_regulatory=   ("reason_regulatory",    "max"),
                  reason_unknown=      ("reason_unknown",       "max"),
              )
              .rename(columns={"start_year": "year", "start_month": "month"}))
    starts["shortage_start"] = 1

    grid = grid.merge(starts, on=["drug_norm", "year", "month"], how="left")
    grid["shortage_start"] = grid["shortage_start"].fillna(0).astype(int)
    grid["n_episodes"]     = grid["n_episodes"].fillna(0).astype(int)
    for c in ["reason_manufacturing", "reason_supply_demand", "reason_raw_material",
              "reason_business", "reason_regulatory", "reason_unknown"]:
        grid[c] = grid[c].fillna(0).astype(int)

    # ── shortage_ongoing: any shortage active at any point during the month ─────
    # Expand each shortage interval to all months it covers, then flag.
    ongoing_rows: list[tuple[str, int, int]] = []
    for _, r in raw_valid[["drug_norm", "date_notified", "date_resolved"]].iterrows():
        start_p = max(r["date_notified"].to_period("M"),
                      PANEL_START.to_period("M"))
        end_p   = min(r["date_resolved"].to_period("M"),
                      PANEL_END.to_period("M"))
        if start_p > end_p:
            continue
        for p in pd.period_range(start_p, end_p, freq="M"):
            ongoing_rows.append((r["drug_norm"], p.year, p.month))

    ongoing = (pd.DataFrame(ongoing_rows, columns=["drug_norm", "year", "month"])
               .drop_duplicates()
               .assign(shortage_ongoing=1))
    grid = grid.merge(ongoing, on=["drug_norm", "year", "month"], how="left")
    grid["shortage_ongoing"] = grid["shortage_ongoing"].fillna(0).astype(int)

    # Stable attrs
    grid = grid.merge(stable, on="drug_norm", how="left")
    grid["sole_source_ever"] = grid["sole_source_ever"].fillna(0).astype(int)
    grid["parenteral_ever"]  = grid["parenteral_ever"].fillna(0).astype(int)

    log.info("Monthly panel shape: %s", grid.shape)
    log.info("Drugs: %d | months: %d | shortage_start months: %d | shortage_ongoing months: %d",
             grid["drug_norm"].nunique(), grid["period"].nunique(),
             int(grid["shortage_start"].sum()), int(grid["shortage_ongoing"].sum()))
    return grid.sort_values(["drug_norm", "year", "month"]).reset_index(drop=True)


# %%
def main():
    panel = build_monthly_panel()
    write_table(panel, OUT_DATA / "uutah_monthly_panel.parquet", log)
    return panel


if __name__ == "__main__":
    main()

# %%
