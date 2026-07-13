"""
validate_vs_redica.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates LLM-extracted features against Redica's pre-structured
categorizations.

Two validation tracks
─────────────────────

  TRACK A — Observation-level (primary, most reliable)
    Input: redica_483_obs_llm_signals_anthropic.csv
    The LLM was run on the SAME Redica observation text. Comparison
    is row-by-row — no matching problem, no ambiguity about which
    observations correspond. This directly answers: do our prompt
    rules reproduce Redica's categorization?

    Requires: run  python3 01_extract_observation_signals.py --source redica
    before this script.

  TRACK B — Document-level cross-source (secondary)
    Input: fdapdf_483_obs_llm_signals_anthropic.csv  (PDF+LLM)
           redica_483_observations.csv               (Redica structured)
    Observations from the two sources for the same 483 (matched by
    FEI + date) cannot be aligned row-by-row. Comparison is at the
    document level (dominant tier, Crit+Major share, any-DI flag).
    This cross-source track is useful for checking consistency
    between PDF text and Redica text, but is less conclusive than
    Track A.

Comparable fields (columns Redica provides → our LLM equivalents)
  redica_severity  (Critical/Major/Other)    ↔  severity_tier (collapsed)
  redica_vc        (QSL-mapped, 8-class)     ↔  violation_category
  redica_di_flag   (DI Labels non-empty)     ↔  data_integrity_flag_llm

Fields with no Redica equivalent (not compared here):
  contamination_flag_llm, root_cause_type, remediation_signal,
  patient_risk_flag_llm, investigation_flag_llm, scope

Outputs (written to this eval/ folder)
  redica_validation_obs_level.csv    Track A: per-observation comparison
  redica_validation_doc_level.csv    Track B: per-document comparison
  redica_validation_summary.csv      Aggregate metrics from both tracks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE    = Path(__file__).resolve().parent   # eval/
PARENT  = HERE.parent                       # 99 - Outputs - Text Analysis/

LLM_PDF_CSV    = PARENT / "step01_fdapdf_483_obs_llm_signals_anthropic.csv"
LLM_REDICA_CSV = PARENT / "step01_redica_483_obs_llm_signals_anthropic.csv"
REDICA_CSV     = PARENT / "step00_redica_483_observations.csv"

OUT_OBS  = HERE / "redica_validation_obs_level.csv"
OUT_DOC  = HERE / "redica_validation_doc_level.csv"
OUT_SUMM = HERE / "redica_validation_summary.csv"

SEVERITY_COLLAPSE = {"Critical": "Critical", "Major": "Major",
                     "Moderate": "Other", "Minor": "Other"}


# ── helpers ──────────────────────────────────────────────────────────────────

def dominant(s: pd.Series) -> str:
    counts = s.dropna().value_counts()
    return counts.index[0] if len(counts) else ""

def cm_share(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.isin({"Critical", "Major"}).mean()) if len(s) else np.nan

def any_flag(s: pd.Series) -> bool:
    return bool(s.fillna(False).any())

def prf(y_true, y_pred):
    """Precision, recall, F1 for binary series."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p+r) if (p+r) else 0.0
    return p, r, f1, tp, fp, fn, tn


# ════════════════════════════════════════════════════════════════════════════
# TRACK A — observation-level (LLM run on Redica text)
# ════════════════════════════════════════════════════════════════════════════

def track_a(redica: pd.DataFrame) -> list[dict]:
    if not LLM_REDICA_CSV.exists():
        print("\n[Track A] redica_483_obs_llm_signals_anthropic.csv not found.")
        print("  Run: python3 01_extract_observation_signals.py --source redica")
        print("  Track A skipped.\n")
        return []

    llm = pd.read_csv(LLM_REDICA_CSV)
    llm = llm[llm["extraction_status"].isin(["ok", "partial"])].copy()
    llm["insp_date"] = pd.to_datetime(llm["insp_date"]).dt.strftime("%Y-%m-%d")
    llm["severity_collapsed"] = llm["severity_tier"].map(SEVERITY_COLLAPSE)

    # Join on (fei, insp_date, obs_num) — same source, should be 1:1
    merged = redica.merge(
        llm[["fei", "insp_date", "obs_num",
             "severity_collapsed", "violation_category",
             "data_integrity_flag_llm", "contamination_flag_llm",
             "patient_risk_flag_llm", "investigation_flag_llm",
             "scope", "root_cause_type", "remediation_signal",
             "confidence"]],
        on=["fei", "insp_date", "obs_num"],
        how="inner",
    )
    merged.to_csv(OUT_OBS, index=False)

    n = len(merged)
    print(f"\n{'━'*66}")
    print("TRACK A — Observation-level  (LLM on Redica text vs Redica labels)")
    print(f"{'━'*66}")
    print(f"  Matched observations: {n}  (of {len(redica)} Redica / {len(llm)} LLM)")
    if n == 0:
        print("  No matched rows — check fei/insp_date/obs_num alignment.")
        return []

    results = []

    # ── A1. Severity ──────────────────────────────────────────────────────
    sev = merged.dropna(subset=["redica_severity", "severity_collapsed"])
    agree_sev = (sev["redica_severity"] == sev["severity_collapsed"]).mean()
    labels = sorted(set(sev["redica_severity"]) | set(sev["severity_collapsed"]))
    print(f"\n  A1. Severity  (n={len(sev)})")
    print(f"      Agreement: {agree_sev:.1%}")
    print(f"      Redica distribution:  {sev['redica_severity'].value_counts().to_dict()}")
    print(f"      LLM distribution:     {sev['severity_collapsed'].value_counts().to_dict()}")
    from sklearn.metrics import confusion_matrix as _cm
    if len(sev) > 0:
        cm = _cm(sev["redica_severity"], sev["severity_collapsed"], labels=labels)
        print(f"      Confusion matrix (rows=Redica, cols=LLM):")
        print(f"      {'':12s}" + "  ".join(f"{l:>10s}" for l in labels))
        for i, rl in enumerate(labels):
            print(f"      {rl:12s}" + "  ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))
    results.append({"track": "A", "metric": "severity_agreement",
                    "value": round(agree_sev, 4), "n": len(sev)})

    # ── A2. Violation category ────────────────────────────────────────────
    vc = merged.dropna(subset=["redica_vc", "violation_category"])
    vc = vc[(vc["redica_vc"] != "") & (vc["violation_category"] != "")]
    agree_vc = (vc["redica_vc"] == vc["violation_category"]).mean()
    print(f"\n  A2. Violation category  (n={len(vc)})")
    print(f"      Agreement: {agree_vc:.1%}")
    vc_labels = sorted(set(vc["redica_vc"]) | set(vc["violation_category"]))
    if len(vc) > 0:
        cm = _cm(vc["redica_vc"], vc["violation_category"], labels=vc_labels)
        print(f"      Confusion matrix (rows=Redica, cols=LLM):")
        hdr = "      " + "".join(f"{l[:11]:>12s}" for l in vc_labels)
        print(hdr)
        for i, rl in enumerate(vc_labels):
            print(f"      {rl[:11]:11s} " + "".join(f"{cm[i,j]:>12d}" for j in range(len(vc_labels))))
    results.append({"track": "A", "metric": "violation_category_agreement",
                    "value": round(agree_vc, 4), "n": len(vc)})

    # ── A3. Data integrity flag ───────────────────────────────────────────
    di = merged.dropna(subset=["redica_di_flag", "data_integrity_flag_llm"])
    y_true = di["redica_di_flag"].astype(int)
    y_pred = di["data_integrity_flag_llm"].astype(int)
    p, r, f1, tp, fp, fn, tn = prf(y_true, y_pred)
    print(f"\n  A3. Data integrity flag  (n={len(di)})")
    print(f"      Redica positive: {int(y_true.sum())}   LLM positive: {int(y_pred.sum())}")
    print(f"      TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"      Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    for m, v in [("di_precision", p), ("di_recall", r), ("di_f1", f1)]:
        results.append({"track": "A", "metric": m, "value": round(v, 4), "n": len(di)})

    # ── A4. QSL area disagrement analysis ────────────────────────────────
    print(f"\n  A4. Top violation-category disagreements:")
    dis = vc[vc["redica_vc"] != vc["violation_category"]]
    if len(dis):
        top = (dis.groupby(["redica_vc", "violation_category"])
               .size().reset_index(name="count")
               .sort_values("count", ascending=False).head(10))
        print(top.to_string(index=False))
    else:
        print("      No disagreements.")

    return results


# ════════════════════════════════════════════════════════════════════════════
# TRACK B — document-level cross-source (PDF vs Redica)
# ════════════════════════════════════════════════════════════════════════════

def track_b(redica: pd.DataFrame) -> list[dict]:
    if not LLM_PDF_CSV.exists():
        print("\n[Track B] fdapdf_483_obs_llm_signals_anthropic.csv not found. Skipped.")
        return []

    pdf = pd.read_csv(LLM_PDF_CSV)
    pdf = pdf[pdf["extraction_status"].isin(["ok", "partial"])].copy()
    pdf["insp_date"] = pd.to_datetime(pdf["insp_date"]).dt.strftime("%Y-%m-%d")
    pdf["severity_collapsed"] = pdf["severity_tier"].map(SEVERITY_COLLAPSE)

    # aggregate to document level
    def agg_pdf(df):
        g = df.groupby(["fei", "insp_date"])
        return pd.DataFrame({
            "pdf_n_obs":    g.size(),
            "pdf_dom_sev":  g["severity_collapsed"].apply(dominant),
            "pdf_cm_share": g["severity_collapsed"].apply(cm_share),
            "pdf_dom_vc":   g["violation_category"].apply(dominant),
            "pdf_di_flag":  g["data_integrity_flag_llm"].apply(any_flag),
        }).reset_index()

    def agg_red(df):
        g = df.groupby(["fei", "insp_date"])
        return pd.DataFrame({
            "red_n_obs":    g.size(),
            "red_dom_sev":  g["redica_severity"].apply(dominant),
            "red_cm_share": g["redica_severity"].apply(cm_share),
            "red_dom_vc":   g["redica_vc"].apply(dominant),
            "red_di_flag":  g["redica_di_flag"].apply(any_flag),
        }).reset_index()

    docs = agg_pdf(pdf).merge(agg_red(redica), on=["fei", "insp_date"], how="inner")
    docs.to_csv(OUT_DOC, index=False)

    n = len(docs)
    print(f"\n{'━'*66}")
    print("TRACK B — Document-level cross-source  (PDF+LLM vs Redica labels)")
    print(f"{'━'*66}")
    print(f"  Overlapping documents (FEI+date): {n}")
    if n == 0:
        return []

    results = []

    # severity
    sev = docs.dropna(subset=["pdf_dom_sev", "red_dom_sev"])
    agree_sev = (sev["pdf_dom_sev"] == sev["red_dom_sev"]).mean()
    cm_sub = docs.dropna(subset=["pdf_cm_share", "red_cm_share"])
    rho, pval = spearmanr(cm_sub["pdf_cm_share"], cm_sub["red_cm_share"])
    print(f"\n  B1. Severity dominant-tier agreement:  {agree_sev:.1%}  (n={len(sev)})")
    print(f"      Crit+Major share Spearman ρ: {rho:.3f}  p={pval:.3f}  (n={len(cm_sub)})")
    results += [{"track": "B", "metric": "doc_severity_agreement",
                 "value": round(agree_sev, 4), "n": len(sev)},
                {"track": "B", "metric": "doc_cm_share_spearman_rho",
                 "value": round(rho, 4), "n": len(cm_sub)}]

    # violation category
    vc = docs.dropna(subset=["pdf_dom_vc", "red_dom_vc"])
    vc = vc[(vc["pdf_dom_vc"] != "") & (vc["red_dom_vc"] != "")]
    agree_vc = (vc["pdf_dom_vc"] == vc["red_dom_vc"]).mean()
    print(f"\n  B2. Violation category agreement:      {agree_vc:.1%}  (n={len(vc)})")
    results.append({"track": "B", "metric": "doc_vc_agreement",
                    "value": round(agree_vc, 4), "n": len(vc)})

    # DI flag
    di = docs.dropna(subset=["pdf_di_flag", "red_di_flag"])
    p, r, f1, tp, fp, fn, tn = prf(di["red_di_flag"].astype(int),
                                    di["pdf_di_flag"].astype(int))
    print(f"\n  B3. DI flag (doc-level): "
          f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  (n={len(di)})")
    for m, v in [("doc_di_precision", p), ("doc_di_recall", r), ("doc_di_f1", f1)]:
        results.append({"track": "B", "metric": m, "value": round(v, 4), "n": len(di)})

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    redica = pd.read_csv(REDICA_CSV)
    redica["insp_date"] = pd.to_datetime(redica["insp_date"]).dt.strftime("%Y-%m-%d")
    print(f"Redica observations loaded: {len(redica)}  ({redica['fei'].nunique()} FEIs)")

    results_a = track_a(redica)
    results_b = track_b(redica)

    # save summary
    all_results = results_a + results_b
    if all_results:
        pd.DataFrame(all_results).to_csv(OUT_SUMM, index=False)
        print(f"\nSummary → {OUT_SUMM}")

    print(f"\n{'━'*66}")
    print("Columns NOT compared (no Redica ground truth):")
    print("  contamination_flag_llm, patient_risk_flag_llm,")
    print("  investigation_flag_llm, scope, root_cause_type, remediation_signal")
    print("  → validate these via expert review (see expert review document).")
    print(f"{'━'*66}\n")


if __name__ == "__main__":
    main()
