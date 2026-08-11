"""
eval/human_eval_02_score.py

Scores a completed labeling_template_v2.xlsx (RA labels) against
DO_NOT_SHARE_answer_key_v2.csv (model predictions), joined on (fei, obs_num).

Computes per-field accuracy, macro-F1, and per-class precision/recall/F1 for
every categorical and binary field, and reports two qualitative free-text
checks rather than trying to score prose automatically:
  - human_patient_risk_why vs. the model's own patient_risk_rationale
    (side-by-side comparison -- the model has a dedicated rationale field here)
  - human_contamination_why on rows where contamination_flag or
    contamination_risk_flag disagree (disagreement audit trail -- the model
    has no dedicated rationale field for this split)

Outputs:
  eval/human_eval_metrics_v2.md  -- per-field metrics table (paper audit trail)

Run:
  python human_eval_02_score.py
  python human_eval_02_score.py --labels <path-to-completed-xlsx>
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_LABELS_XLSX = HERE / "labeling_template_v2.xlsx"
ANSWER_KEY_CSV       = HERE / "DO_NOT_SHARE_answer_key_v2.csv"
METRICS_MD           = HERE / "human_eval_metrics_v2.md"

# (metric name, model column in answer key, human column in labels, kind)
# kind: "categorical" -> accuracy + macro-F1 + per-class P/R/F1
#       "bool"        -> accuracy + P/R/F1 for the True class only
FIELDS = [
    ("violation_category",      "violation_category",           "human_violation_category",      "categorical"),
    ("severity_tier",           "severity_tier",                "human_severity_tier",            "categorical"),
    ("scope",                   "scope",                        "human_scope",                    "categorical"),
    ("root_cause_type",         "root_cause_type",              "human_root_cause_type",          "categorical"),
    ("remediation_signal",      "remediation_signal",           "human_remediation_signal",        "categorical"),
    ("repeat_flag",             "repeat_flag_llm",              "human_repeat_flag",               "bool"),
    ("patient_risk_flag",       "patient_risk_flag_llm",        "human_patient_risk_flag",         "bool"),
    ("contamination_flag",      "contamination_flag_llm",       "human_contamination_flag",        "bool"),
    ("contamination_risk_flag", "contamination_risk_flag_llm",  "human_contamination_risk_flag",   "bool"),
    ("investigation_flag",      "investigation_flag_llm",       "human_investigation_flag",        "bool"),
    ("data_integrity_flag",     "data_integrity_flag_llm",      "human_data_integrity_flag",       "bool"),
]


def _normalize(v) -> str:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return "true"
    if s in ("false", "0", "no"):
        return "false"
    return s


def precision_recall_f1(y_true: list, y_pred: list, label: str) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"label": label, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(y_true: list, y_pred: list, classes: list[str]) -> float:
    f1s = [precision_recall_f1(y_true, y_pred, c)["f1"] for c in classes]
    return sum(f1s) / len(f1s) if f1s else 0.0


def field_metrics(merged: pd.DataFrame, human_col: str, model_col: str, kind: str) -> dict:
    sub = merged[[human_col, model_col]].copy()
    # Filter to rows the RA actually filled in. Use notna() rather than
    # stringifying and comparing to "nan" -- pandas >= 3.0's default string
    # dtype makes .astype(str) a no-op on missing values (it no longer forces
    # them to the literal text "nan" the way object-dtype columns used to),
    # so a string-based emptiness check silently lets every blank row through.
    filled_mask = sub[human_col].notna() & (sub[human_col].astype(str).str.strip() != "")
    sub = sub[filled_mask]
    sub[human_col] = sub[human_col].astype(str).str.strip()
    if len(sub) == 0:
        return {"n": 0}

    y_true = [_normalize(v) for v in sub[human_col]]
    y_pred = [_normalize(v) for v in sub[model_col]]
    classes = ["true", "false"] if kind == "bool" else sorted(set(y_true) | set(y_pred))

    result = {
        "n": len(sub),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, classes), 4),
        "per_class": [precision_recall_f1(y_true, y_pred, c) for c in classes],
    }
    return result


def _load_labels(path: Path) -> pd.DataFrame:
    # keep_default_na=False for the same reason as the answer-key read above:
    # if the RA picks "None" from the human_remediation_signal dropdown (a
    # real category value, not "I skipped this"), pandas' default NA-sniffing
    # would otherwise silently convert it back to NaN and the row would look
    # unlabeled.
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name="Labeling", keep_default_na=False, na_values=[""])
    else:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df


def main():
    parser = argparse.ArgumentParser(description="Score completed RA labels against the model answer key.")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS_XLSX),
                         help="Path to the completed labeling_template_v2.xlsx (or a CSV export of it).")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"Completed labels file not found: {labels_path}")
    if not ANSWER_KEY_CSV.exists():
        sys.exit(f"Answer key not found: {ANSWER_KEY_CSV}\nRun human_eval_01_generate_template.py first.")

    labels = _load_labels(labels_path)
    # keep_default_na=False so the literal string "None" (a real
    # remediation_signal category, not a missing marker) survives the read --
    # pandas' default NA-sniffing list includes the bare word "None" and
    # would otherwise silently turn it back into NaN. na_values=[""] keeps
    # genuinely empty cells (e.g. an unset evidence_quote) treated as missing.
    answers = pd.read_csv(ANSWER_KEY_CSV, keep_default_na=False, na_values=[""])

    n_filled = labels["human_violation_category"].notna().sum() if "human_violation_category" in labels.columns else 0
    print(f"Rows in labels file: {len(labels)}  |  rows with a violation_category label: {n_filled}")
    if n_filled == 0:
        print("[WARN] No human labels found yet. Have the RA fill in the "
              "'Labeling' tab, then re-run this script.")
        return

    merged = labels.merge(answers, on=["fei", "obs_num"], how="inner", suffixes=("", "_answer"))
    print(f"Rows joined to answer key: {len(merged)}")

    all_metrics = {}
    for field_name, model_col, human_col, kind in FIELDS:
        if human_col not in merged.columns or model_col not in merged.columns:
            continue
        m = field_metrics(merged, human_col, model_col, kind)
        all_metrics[field_name] = m
        if m.get("n", 0) > 0:
            print(f"  {field_name:26s}  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  n={m['n']}")
        else:
            print(f"  {field_name:26s}  no labels yet")

    # Qualitative check for patient_risk_why vs. the model's own rationale --
    # not scored automatically; printed for manual read-through.
    if "human_patient_risk_why" in merged.columns:
        both_true = merged[
            (merged["human_patient_risk_flag"].astype(str).str.upper() == "TRUE")
            & (merged["patient_risk_flag_llm"].astype(str).str.upper() == "TRUE")
        ]
        print(f"\nRows where both RA and model set patient_risk = TRUE: {len(both_true)} "
              "(see human_eval_metrics_v2.md for the side-by-side text)")

    # Qualitative check for contamination_why -- the model has no dedicated
    # rationale field for this split (unlike patient_risk), so this is a
    # disagreement audit trail rather than a side-by-side comparison: every
    # row where the RA's contamination_flag or contamination_risk_flag
    # differs from the model's, with the RA's stated reasoning attached.
    if "human_contamination_why" in merged.columns:
        contam_disagree = merged[
            (merged["human_contamination_flag"].astype(str).str.upper()
             != merged["contamination_flag_llm"].astype(str).str.upper())
            | (merged["human_contamination_risk_flag"].astype(str).str.upper()
               != merged["contamination_risk_flag_llm"].astype(str).str.upper())
        ]
        contam_disagree = contam_disagree[contam_disagree["human_contamination_why"].notna()]
        print(f"Rows where RA and model disagree on contamination_flag or "
              f"contamination_risk_flag: {len(contam_disagree)} "
              "(see human_eval_metrics_v2.md for the RA's stated reasoning)")

    _write_metrics_md(all_metrics, merged)
    print(f"\nMetrics written to: {METRICS_MD}")


def _write_metrics_md(all_metrics: dict, merged: pd.DataFrame):
    today = date.today().isoformat()
    lines = [
        "# Human Labeling vs. LLM (v2) -- Agreement Metrics",
        f"\n> Generated: {today}  |  Source: `eval/human_eval_02_score.py`\n",
        "## Summary Table\n",
        "| Field | Accuracy | Macro F1 | N |",
        "|---|---|---|---|",
    ]
    for field, m in all_metrics.items():
        if m.get("n", 0) == 0:
            lines.append(f"| {field} | no labels | -- | 0 |")
            continue
        lines.append(f"| {field} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['n']} |")

    lines.append("\n## Per-Class Breakdown\n")
    for field, m in all_metrics.items():
        if m.get("n", 0) == 0:
            continue
        lines.append(f"### {field}\n")
        lines.append("| Class | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for pc in m.get("per_class", []):
            lines.append(f"| {pc['label']} | {pc['precision']:.3f} | {pc['recall']:.3f} "
                         f"| {pc['f1']:.3f} | {pc['tp']} | {pc['fp']} | {pc['fn']} |")
        lines.append("")

    if "human_patient_risk_why" in merged.columns:
        lines.append("## patient_risk_flag Rationale Comparison (qualitative)\n")
        lines.append("Side-by-side text for every row where the RA marked patient_risk = TRUE "
                     "(regardless of what the model said) -- read through for agreement in "
                     "reasoning, not just the boolean flag.\n")
        ra_true = merged[merged["human_patient_risk_flag"].astype(str).str.upper() == "TRUE"]
        for _, r in ra_true.iterrows():
            lines.append(f"**FEI {r['fei']}, obs {r['obs_num']}**")
            lines.append(f"- RA: {r.get('human_patient_risk_why', '')}")
            lines.append(f"- Model ({r.get('patient_risk_flag_llm', '')}): "
                         f"{r.get('patient_risk_rationale', '')}")
            lines.append("")

    if "human_contamination_why" in merged.columns:
        lines.append("## contamination_flag / contamination_risk_flag Disagreements "
                     "(qualitative)\n")
        lines.append("The model has no dedicated rationale field for this split (unlike "
                     "patient_risk), so this is a disagreement audit trail, not a side-by-side "
                     "comparison: every row where the RA's contamination_flag or "
                     "contamination_risk_flag differs from the model's, with the RA's stated "
                     "reasoning. Use this to tell a genuine misread of the confirmed-event-vs-"
                     "control-risk split apart from a defensible close call.\n")
        contam_disagree = merged[
            (merged["human_contamination_flag"].astype(str).str.upper()
             != merged["contamination_flag_llm"].astype(str).str.upper())
            | (merged["human_contamination_risk_flag"].astype(str).str.upper()
               != merged["contamination_risk_flag_llm"].astype(str).str.upper())
        ]
        contam_disagree = contam_disagree[contam_disagree["human_contamination_why"].notna()]
        if len(contam_disagree) == 0:
            lines.append("*No disagreements with a stated rationale yet.*\n")
        for _, r in contam_disagree.iterrows():
            lines.append(f"**FEI {r['fei']}, obs {r['obs_num']}**")
            lines.append(f"- RA: contamination_flag={r.get('human_contamination_flag', '')}, "
                         f"contamination_risk_flag={r.get('human_contamination_risk_flag', '')} "
                         f"-- {r.get('human_contamination_why', '')}")
            lines.append(f"- Model: contamination_flag={r.get('contamination_flag_llm', '')}, "
                         f"contamination_risk_flag={r.get('contamination_risk_flag_llm', '')}")
            lines.append("")

    lines.append("## Notes for Paper\n")
    lines.append(
        "- Field definitions match `483_Labeling_Rules_v2.docx` (rules) and "
        "`483_Background_Reference_Guide.docx` (onboarding/background, not shown to the "
        "model) -- the rules document is a plain-language rewrite of the v2 LLM prompt in "
        "`01_extract_observation_signals.py`.\n"
        "- Labeling was blind: the RA never saw model predictions while labeling "
        "(see `human_eval_01_generate_template.py`).\n"
        "- `patient_risk_flag` carries a rationale comparison above -- treat the flag's "
        "F1 alongside a read of the reasoning, not the F1 alone, given how narrowly it is "
        "scoped.\n"
        "- This is a complementary check to the existing LLM-vs-Redica convergent-validity "
        "comparison (`eval/validate_llm_vs_redica.py`); Redica is itself human-coded, so "
        "this file is the actual independent ground-truth check.\n"
    )
    METRICS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
