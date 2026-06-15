"""
03_validate_vs_redica.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates LLM-extracted features against Redica's pre-structured
categorizations on the set of 483 documents present in both sources.

Inputs
  483_observation_context_signals.csv     (PDF+LLM pipeline, Step 1)
  redica_483_observations.csv             (Redica structured, Step 0)

Matching
  Documents are matched on (fei, insp_date) — there is no shared
  observation-level ID, so all comparisons are at the DOCUMENT level
  (aggregating over all observations in the same 483).

Three validation tasks
  1. Severity agreement
       Redica uses 3 tiers: Critical / Major / Other.
       Our LLM uses 4 tiers: Critical / Major / Moderate / Minor.
       Mapping: Critical→Critical, Major→Major, {Moderate,Minor}→Other.
       Metric: per-document agreement on the dominant severity tier;
               per-document Spearman ρ of Critical+Major share;
               overall confusion matrix (our collapsed 3-tier vs Redica).

  2. Violation category agreement
       Redica QSL Area is mapped to our 8-class violation_category schema
       via the table in 00_load_redica_obs.py (stored in redica_vc column).
       Metric: per-document agreement on dominant violation category;
               overall confusion matrix.

  3. Data integrity flag agreement
       Redica: DI Labels list non-empty → document-level DI present.
       Ours:   any observation in the document has data_integrity_flag_llm=True.
       Metric: document-level precision, recall, F1 (Redica = ground truth).

Outputs (written to the same folder as this script)
  redica_validation_doc_level.csv   — one row per overlapping document
  redica_validation_summary.csv     — aggregate metrics table
  (console print of all results)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, classification_report

HERE = Path(__file__).resolve().parent

OUR_CSV    = HERE / "483_observation_context_signals.csv"
REDICA_CSV = HERE / "redica_483_observations.csv"
OUT_DOC    = HERE / "redica_validation_doc_level.csv"
OUT_SUMM   = HERE / "redica_validation_summary.csv"

# Severity collapse: our 4-tier → Redica 3-tier
SEVERITY_COLLAPSE = {
    "Critical": "Critical",
    "Major":    "Major",
    "Moderate": "Other",
    "Minor":    "Other",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def dominant(series: pd.Series) -> str:
    """Most common non-null value in a series, or '' if all null."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) else ""


def crit_major_share(severity_series: pd.Series, valid_values: set) -> float:
    """Share of observations that are Critical or Major (NaN excluded)."""
    s = severity_series.dropna()
    s = s[s.isin(valid_values)]
    if len(s) == 0:
        return np.nan
    return (s.isin({"Critical", "Major"})).mean()


def doc_di_flag(flag_series: pd.Series) -> bool:
    """True if any observation in the document has a DI flag."""
    return bool(flag_series.any())


# ── load & prepare ───────────────────────────────────────────────────────────

def load_our(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only LLM-scored rows
    df = df[df["extraction_status"].isin(["ok", "partial"])].copy()
    df["insp_date"] = pd.to_datetime(df["insp_date"]).dt.date
    df["severity_collapsed"] = df["severity_tier"].map(SEVERITY_COLLAPSE)
    return df


def load_redica(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["insp_date"] = pd.to_datetime(df["insp_date"]).dt.date
    return df


# ── document-level aggregation ───────────────────────────────────────────────

def agg_our(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate our observations to document level (fei, insp_date)."""
    grp = df.groupby(["fei", "insp_date"])
    return pd.DataFrame({
        "our_n_obs":            grp.size(),
        "our_dom_severity":     grp["severity_collapsed"].apply(dominant),
        "our_cm_share":         grp["severity_collapsed"].apply(
                                    lambda s: crit_major_share(s, {"Critical","Major","Other"})),
        "our_dom_vc":           grp["violation_category"].apply(dominant),
        "our_di_flag":          grp["data_integrity_flag_llm"].apply(doc_di_flag),
        "our_contam_flag":      grp["contamination_flag_llm"].apply(doc_di_flag),
        "our_patient_flag":     grp["patient_risk_flag_llm"].apply(doc_di_flag),
        "our_invest_flag":      grp["investigation_flag_llm"].apply(doc_di_flag),
    }).reset_index()


def agg_redica(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Redica observations to document level (fei, insp_date)."""
    grp = df.groupby(["fei", "insp_date"])
    return pd.DataFrame({
        "red_n_obs":    grp.size(),
        "red_dom_sev":  grp["redica_severity"].apply(dominant),
        "red_cm_share": grp["redica_severity"].apply(
                            lambda s: crit_major_share(s, {"Critical","Major","Other"})),
        "red_dom_vc":   grp["redica_vc"].apply(dominant),
        "red_di_flag":  grp["redica_di_flag"].apply(doc_di_flag),
    }).reset_index()


# ── validation tasks ─────────────────────────────────────────────────────────

def task1_severity(docs: pd.DataFrame) -> dict:
    """Severity: dominant-tier agreement + Critical+Major share correlation."""
    sub = docs.dropna(subset=["our_dom_severity", "red_dom_sev"])
    agree = (sub["our_dom_severity"] == sub["red_dom_sev"]).mean()

    cm_sub = docs.dropna(subset=["our_cm_share", "red_cm_share"])
    rho, pval = spearmanr(cm_sub["our_cm_share"], cm_sub["red_cm_share"])
    mae = (cm_sub["our_cm_share"] - cm_sub["red_cm_share"]).abs().mean()

    print("\n── Severity Agreement ──────────────────────────────────────")
    print(f"  Documents compared (dominant tier): {len(sub)}")
    print(f"  Dominant tier agreement: {agree:.1%}")
    print(f"  Documents compared (Crit+Major share): {len(cm_sub)}")
    print(f"  Spearman ρ (Crit+Major share): {rho:.3f}  p={pval:.3f}")
    print(f"  Mean absolute difference in Crit+Major share: {mae:.3f}")

    if len(sub) > 0:
        labels = sorted(set(sub["our_dom_severity"]) | set(sub["red_dom_sev"]))
        cm = confusion_matrix(sub["red_dom_sev"], sub["our_dom_severity"], labels=labels)
        print(f"\n  Confusion matrix  (rows=Redica, cols=ours):")
        print(f"  {'':12s}" + "  ".join(f"{l:>10s}" for l in labels))
        for i, row_label in enumerate(labels):
            print(f"  {row_label:12s}" + "  ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))

    return {
        "metric": ["dominant_tier_agreement", "cm_share_spearman_rho",
                   "cm_share_spearman_p", "cm_share_mae"],
        "value":  [round(agree, 4), round(rho, 4), round(pval, 4), round(mae, 4)],
        "n_docs": [len(sub), len(cm_sub), len(cm_sub), len(cm_sub)],
    }


def task2_violation_cat(docs: pd.DataFrame) -> dict:
    """Violation category: dominant-category agreement."""
    sub = docs.dropna(subset=["our_dom_vc", "red_dom_vc"])
    sub = sub[(sub["our_dom_vc"] != "") & (sub["red_dom_vc"] != "")]
    agree = (sub["our_dom_vc"] == sub["red_dom_vc"]).mean()

    print("\n── Violation Category Agreement ────────────────────────────")
    print(f"  Documents compared: {len(sub)}")
    print(f"  Dominant category agreement: {agree:.1%}")

    if len(sub) > 0:
        labels = sorted(set(sub["our_dom_vc"]) | set(sub["red_dom_vc"]))
        cm = confusion_matrix(sub["red_dom_vc"], sub["our_dom_vc"], labels=labels)
        print(f"\n  Confusion matrix  (rows=Redica, cols=ours):")
        header = "  " + " ".join(f"{l[:12]:>13s}" for l in labels)
        print(header)
        for i, row_label in enumerate(labels):
            print(f"  {row_label[:12]:12s} " +
                  " ".join(f"{cm[i,j]:>13d}" for j in range(len(labels))))

    disagree = sub[sub["our_dom_vc"] != sub["red_dom_vc"]][
        ["fei", "insp_date", "red_dom_vc", "our_dom_vc"]
    ]
    if len(disagree):
        print(f"\n  Disagreement cases ({len(disagree)}):")
        print(disagree.to_string(index=False))

    return {
        "metric": ["dominant_vc_agreement"],
        "value":  [round(agree, 4)],
        "n_docs": [len(sub)],
    }


def task3_di_flag(docs: pd.DataFrame) -> dict:
    """Data integrity flag: document-level precision / recall / F1."""
    sub = docs.dropna(subset=["our_di_flag", "red_di_flag"])
    y_true = sub["red_di_flag"].astype(int)
    y_pred = sub["our_di_flag"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    print("\n── Data Integrity Flag Agreement ───────────────────────────")
    print(f"  Documents compared: {len(sub)}")
    print(f"  Redica positive (DI present): {int(y_true.sum())} / {len(sub)}")
    print(f"  Ours positive  (DI flagged):  {int(y_pred.sum())} / {len(sub)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    fn_docs = sub[(y_true == 1) & (y_pred == 0)][["fei", "insp_date"]]
    if len(fn_docs):
        print(f"\n  False negatives (Redica flagged DI, our LLM did not):")
        print(fn_docs.to_string(index=False))

    return {
        "metric": ["di_precision", "di_recall", "di_f1"],
        "value":  [round(prec, 4), round(recall, 4), round(f1, 4)],
        "n_docs": [len(sub)] * 3,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    our    = load_our(OUR_CSV)
    redica = load_redica(REDICA_CSV)

    print(f"Our pipeline: {len(our)} obs, {our['fei'].nunique()} FEIs")
    print(f"Redica:       {len(redica)} obs, {redica['fei'].nunique()} FEIs")

    # ── document-level overlap ───────────────────────────────────────────────
    our_docs    = agg_our(our)
    redica_docs = agg_redica(redica)

    docs = our_docs.merge(redica_docs, on=["fei", "insp_date"], how="inner")
    print(f"\nOverlapping documents (FEI+date): {len(docs)}")
    print(f"  Our obs in overlap:    {docs['our_n_obs'].sum()}")
    print(f"  Redica obs in overlap: {docs['red_n_obs'].sum()}")

    if len(docs) == 0:
        print("No overlap — cannot validate. Check that 00_load_redica_obs.py has been run.")
        return

    # ── run three validation tasks ───────────────────────────────────────────
    r1 = task1_severity(docs)
    r2 = task2_violation_cat(docs)
    r3 = task3_di_flag(docs)

    # ── save outputs ─────────────────────────────────────────────────────────
    docs.to_csv(OUT_DOC, index=False)
    print(f"\nDocument-level results → {OUT_DOC}")

    summary_rows = []
    for result in [r1, r2, r3]:
        for m, v, n in zip(result["metric"], result["value"], result["n_docs"]):
            summary_rows.append({"metric": m, "value": v, "n_docs": n})
    pd.DataFrame(summary_rows).to_csv(OUT_SUMM, index=False)
    print(f"Summary metrics → {OUT_SUMM}")

    # ── interpretation note ──────────────────────────────────────────────────
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interpretation note
  Severity and violation category are validated at the dominant-tier
  level because observations from the two sources cannot be matched
  1:1 (no shared observation ID). A document with 5 observations in
  Redica and 8 in ours is expected — different extraction methods.
  Agreement on the dominant tier and on the Crit+Major share is the
  most meaningful cross-source comparison available.

  DI flag is validated as a document-level binary: did any observation
  in this document trigger the flag? This is the cleanest possible
  ground-truth comparison.

  Signals NOT validated here (no Redica ground truth):
    contamination_flag_llm, root_cause_type, remediation_signal,
    patient_risk_flag_llm, investigation_flag_llm, scope.
  The quality of these signals must be assessed via human review
  (see expert review document).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
