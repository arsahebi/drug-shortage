# %%
"""
Module 4 — FDA Recall features per drug-year.

We use the already-filtered `recall_filtered.csv` (drugs & biologics only),
which gives us per-event firm name, product description, classification
(Class I/II/III), date, and reason for recall. We aggregate at drug-year
granularity using the *Product Description* text to detect molecule mentions.

Output columns:
    n_recalls_total, n_class_I, n_class_II, n_class_III,
    n_cgmp, n_contam, n_potency, n_mislabel, n_stability, n_foreign
"""

from __future__ import annotations
import pandas as pd

from config import RECALL_FILT, OUT_DATA, OUT_LOGS, PANEL_START_YEAR, PANEL_END_YEAR, VALISURE_CSV
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("m04_recalls", OUT_LOGS / "m04_recalls.log")

# Reason keyword maps (mirror the engineered features in recall_fei_features.csv)
REASON_PATTERNS = {
    "n_cgmp":       r"\b(?:cgmp|gmp|manufacturing standards|good manufacturing)\b",
    "n_contam":     r"\b(?:contamin\w*|microb\w*|particul\w*|sterility|endotoxin|bacteri\w*|fung\w*)\b",
    "n_potency":    r"\b(?:potency|super[- ]potent|sub[- ]potent|wrong potency|out of specification|oos|assay)\b",
    "n_mislabel":   r"\b(?:mislabel\w*|label\w*|wrong drug|wrong product|incorrect label)\b",
    "n_stability":  r"\b(?:stability|degrad\w*|impurit\w*|nitros\w*|dmf|ndma)\b",
    "n_foreign":    r"\b(?:foreign material|particulate matter|glass|metal particles)\b",
    "n_dissolution":r"\b(?:dissolution|content uniformity)\b",
}


def build_recall_features(api_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RECALL_FILT, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    matcher = ValisureDrugMatcher(api_names)

    df["year"] = pd.to_datetime(df["Recall_Date"], errors="coerce").dt.year.astype("Int64")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]
    log.info("Recall events in window: %d", len(df))

    df["drug_norm"] = df["Product Description"].astype(str).apply(
        matcher.match)
    matched = df.dropna(subset=["drug_norm"]).copy()
    log.info("Matched to a target drug: %d events (%.1f%%)",
             len(matched), 100 * len(matched) / max(len(df), 1))

    # Class flags
    cls = matched["Event Classification"].astype(str)
    matched = matched.assign(
        is_I  = cls.str.contains("Class I",  na=False).astype(int),
        is_II = cls.str.contains("Class II", na=False).astype(int),
        is_III= cls.str.contains("Class III",na=False).astype(int),
    )
    # Note: 'Class II' also matches inside 'Class III' literally if we used simple contains;
    # tighten by exact tokenization:
    cls_norm = cls.str.strip()
    matched["is_I"]   = (cls_norm == "Class I").astype(int)
    matched["is_II"]  = (cls_norm == "Class II").astype(int)
    matched["is_III"] = (cls_norm == "Class III").astype(int)

    # Reason buckets
    reason = matched["Reason for Recall"].astype(str).str.lower().fillna("")
    for col, pat in REASON_PATTERNS.items():
        matched[col] = reason.str.contains(pat, regex=True, na=False).astype(int)

    agg = matched.groupby(["drug_norm", "year"], as_index=False).agg(
        n_recalls_total=("Event ID", "count"),
        n_class_I=("is_I", "sum"),
        n_class_II=("is_II", "sum"),
        n_class_III=("is_III", "sum"),
        n_cgmp=("n_cgmp", "sum"),
        n_contam=("n_contam", "sum"),
        n_potency=("n_potency", "sum"),
        n_mislabel=("n_mislabel", "sum"),
        n_stability=("n_stability", "sum"),
        n_foreign=("n_foreign", "sum"),
        n_dissolution=("n_dissolution", "sum"),
    )
    log.info("Recall feature rows: %d (drugs %d)", len(agg), agg["drug_norm"].nunique())
    detail_cols = [
        "drug_norm", "year", "Event ID", "Event Classification",
        "Reason for Recall", "Product Description", "Recalling Firm Name",
        "FEI Number", "Recall_Date",
    ]
    matched[[c for c in detail_cols if c in matched.columns]].to_csv(
        OUT_DATA / "recall_matched_events.csv", index=False
    )
    log.info("Wrote recall_matched_events.csv (%d rows)", len(matched))
    return agg


def main():
    out = build_recall_features(load_valisure_api_names(VALISURE_CSV))
    write_table(out, OUT_DATA / "recall_drug_year.parquet", log)
    return out


if __name__ == "__main__":
    main()

# %%
