# %%
"""
Module MM02 — Monthly FDA recall features.

Aggregates recall events by drug_norm × (year, month), keyed on the recall
initiation date (Recall_Date in recall_filtered.csv).

Output columns per drug-month:
    n_recalls_total,
    n_class_I, n_class_II, n_class_III,
    n_cgmp, n_contam, n_potency, n_mislabel, n_stability, n_foreign, n_dissolution

Also writes recall_matched_events_monthly.csv — the individual matched events
with drug_norm, date, reason flags.  Used by mm06 for circularity analysis.

Reason-bucket patterns mirror m04_recall_features.py.
"""

from __future__ import annotations
import pandas as pd

from config import (
    RECALL_FILT, VALISURE_CSV,
    OUT_DATA, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("mm02_recall_monthly", OUT_LOGS / "mm02_recall_monthly.log")

# Reason patterns identical to m04_recall_features.py
REASON_PATTERNS: dict[str, str] = {
    "n_cgmp":        r"\b(?:cgmp|gmp|manufacturing standards|good manufacturing)\b",
    "n_contam":      r"\b(?:contamin\w*|microb\w*|particul\w*|sterility|endotoxin|bacteri\w*|fung\w*)\b",
    "n_potency":     r"\b(?:potency|super[- ]potent|sub[- ]potent|wrong potency|out of specification|oos|assay)\b",
    "n_mislabel":    r"\b(?:mislabel\w*|label\w*|wrong drug|wrong product|incorrect label)\b",
    "n_stability":   r"\b(?:stability|degrad\w*|impurit\w*|nitros\w*|dmf|ndma)\b",
    "n_foreign":     r"\b(?:foreign material|particulate matter|glass|metal particles)\b",
    "n_dissolution": r"\b(?:dissolution|content uniformity)\b",
}


def build_recall_monthly(api_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (aggregated drug-month features, matched individual events)."""
    df = pd.read_csv(RECALL_FILT, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    matcher = ValisureDrugMatcher(api_names)

    df["recall_dt"] = pd.to_datetime(df["Recall_Date"], errors="coerce")
    df["year"]  = df["recall_dt"].dt.year.astype("Int64")
    df["month"] = df["recall_dt"].dt.month.astype("Int64")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]
    log.info("Recall events in panel window: %d", len(df))

    # Match product description to canonical drug names
    df["drug_norm"] = df["Product Description"].astype(str).apply(matcher.match)
    matched = df.dropna(subset=["drug_norm"]).copy()
    log.info("Matched to a target drug: %d events (%.1f%%)",
             len(matched), 100 * len(matched) / max(len(df), 1))

    # Class flags — exact string match to avoid 'Class II' matching inside 'Class III'
    cls_norm = matched["Event Classification"].astype(str).str.strip()
    matched = matched.assign(
        is_I  =(cls_norm == "Class I").astype(int),
        is_II =(cls_norm == "Class II").astype(int),
        is_III=(cls_norm == "Class III").astype(int),
    )

    # Reason flags
    reason = matched["Reason for Recall"].astype(str).str.lower().fillna("")
    for col, pat in REASON_PATTERNS.items():
        matched[col] = reason.str.contains(pat, regex=True, na=False).astype(int)

    # Aggregate to drug-month
    agg = matched.groupby(["drug_norm", "year", "month"], as_index=False).agg(
        n_recalls_total=("Event ID",  "count"),
        n_class_I=      ("is_I",      "sum"),
        n_class_II=     ("is_II",     "sum"),
        n_class_III=    ("is_III",    "sum"),
        n_cgmp=         ("n_cgmp",       "sum"),
        n_contam=       ("n_contam",     "sum"),
        n_potency=      ("n_potency",    "sum"),
        n_mislabel=     ("n_mislabel",   "sum"),
        n_stability=    ("n_stability",  "sum"),
        n_foreign=      ("n_foreign",    "sum"),
        n_dissolution=  ("n_dissolution","sum"),
    )
    log.info("Recall monthly rows: %d | drugs: %d", len(agg), agg["drug_norm"].nunique())

    # Write individual matched events for circularity analysis in mm06
    detail_cols = [
        "drug_norm", "year", "month", "recall_dt",
        "Event ID", "Event Classification", "Reason for Recall",
        "Product Description", "Recalling Firm Name", "FEI Number",
        "is_I", "is_II", "is_III",
        "n_cgmp", "n_contam", "n_potency", "n_mislabel", "n_stability",
        "n_foreign", "n_dissolution",
    ]
    detail = matched[[c for c in detail_cols if c in matched.columns]].copy()
    detail_path = OUT_DATA / "recall_matched_events_monthly.csv"
    detail.to_csv(detail_path, index=False)
    log.info("Wrote %s (%d rows)", detail_path.name, len(detail))

    return agg, detail


# %%
def main():
    api_names = load_valisure_api_names(VALISURE_CSV)
    agg, _ = build_recall_monthly(api_names)
    write_table(agg, OUT_DATA / "recall_monthly.parquet", log)
    return agg


if __name__ == "__main__":
    main()

# %%
