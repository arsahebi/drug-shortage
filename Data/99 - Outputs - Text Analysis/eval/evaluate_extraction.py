"""
eval/evaluate_extraction.py

Computes precision, recall, and F1 for each extracted field by comparing
the model's predictions in fei_observation_signals.csv against the
human labels in labeling_template.csv (after you fill in the human_* columns).

Also compares model severity_tier vs the deterministic keyword baseline
(severity_tier_baseline) to quantify how much the LLM adds over the rule-based
approach — supporting the paper's claims about LLM value.

Outputs:
  eval/extraction_metrics.md  — per-field metrics table (for paper audit trail)

Run:
  python eval/evaluate_extraction.py
"""

import sys
from pathlib import Path

import pandas as pd

EVAL_DIR     = Path(__file__).parent
TEMPLATE_CSV = EVAL_DIR / "labeling_template.csv"
METRICS_MD   = EVAL_DIR / "extraction_metrics.md"


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize(v) -> str:
    return str(v).strip().lower()


def precision_recall_f1(y_true: list, y_pred: list, label: str) -> dict:
    """Compute P/R/F1 for a single class label (one-vs-rest)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"label": label, "precision": p, "recall": r, "f1": f,
            "tp": tp, "fp": fp, "fn": fn}


def macro_f1(y_true: list, y_pred: list, classes: list[str]) -> float:
    f1s = [precision_recall_f1(y_true, y_pred, c)["f1"] for c in classes]
    return sum(f1s) / len(f1s) if f1s else 0.0


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def field_metrics(df: pd.DataFrame, human_col: str, model_col: str,
                  baseline_col: str | None = None) -> dict:
    """Return accuracy, macro-F1, and per-class P/R/F1 for one field."""
    sub = df[[human_col, model_col]].dropna()
    sub = sub[sub[human_col].str.strip() != ""]

    if len(sub) == 0:
        return {"n": 0, "note": "no human labels for this field"}

    y_true  = [_normalize(v) for v in sub[human_col]]
    y_pred  = [_normalize(v) for v in sub[model_col]]
    classes = sorted(set(y_true))

    result = {
        "n":           len(sub),
        "accuracy":    round(accuracy(y_true, y_pred), 4),
        "macro_f1":    round(macro_f1(y_true, y_pred, classes), 4),
        "per_class":   [precision_recall_f1(y_true, y_pred, c) for c in classes],
    }

    if baseline_col and baseline_col in df.columns:
        sub_b = df[[human_col, baseline_col]].dropna()
        sub_b = sub_b[sub_b[human_col].str.strip() != ""]
        if len(sub_b) > 0:
            yb_true = [_normalize(v) for v in sub_b[human_col]]
            yb_pred = [_normalize(v) for v in sub_b[baseline_col]]
            result["baseline_accuracy"] = round(accuracy(yb_true, yb_pred), 4)
            result["baseline_macro_f1"] = round(macro_f1(yb_true, yb_pred, classes), 4)

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not TEMPLATE_CSV.exists():
        sys.exit(
            f"Labeling template not found: {TEMPLATE_CSV}\n"
            "Run eval/generate_labeling_template.py to create it, "
            "then fill in the human_* columns."
        )

    df = pd.read_csv(TEMPLATE_CSV)
    print(f"Template rows loaded: {len(df)}")

    # Check if any human labels have been filled in
    human_cols = [
        "human_violation_category", "human_severity_tier",
        "human_root_cause_type", "human_remediation_signal",
        "human_repeat_flag", "human_systemic_flag", "human_patient_risk_flag",
    ]
    n_labeled = df[human_cols[0]].notna().sum()
    if n_labeled == 0:
        print("\n[WARN] No human labels found in labeling_template.csv.")
        print("Fill in the human_* columns and re-run this script.")
        print(f"Writing empty metrics template to: {METRICS_MD}")
        _write_empty_metrics()
        return

    print(f"Rows with human labels: {n_labeled}")

    # ── Per-field evaluation ───────────────────────────────────────────────
    FIELDS = [
        ("violation_category", "model_violation_category", "human_violation_category",
         None),
        ("severity_tier",      "model_severity_tier",      "human_severity_tier",
         "model_severity_tier_baseline"),
        ("root_cause_type",    "model_root_cause_type",    "human_root_cause_type",
         None),
        ("remediation_signal", "model_remediation_signal", "human_remediation_signal",
         None),
        ("repeat_flag",        "model_repeat_flag",        "human_repeat_flag",
         None),
        ("systemic_flag",      "model_systemic_flag",      "human_systemic_flag",
         None),
        ("patient_risk_flag",  "model_patient_risk_flag",  "human_patient_risk_flag",
         None),
    ]

    all_metrics = {}
    for field_name, model_col, human_col, baseline_col in FIELDS:
        m = field_metrics(df, human_col, model_col, baseline_col)
        all_metrics[field_name] = m
        print(f"\n  {field_name:25s}  acc={m.get('accuracy',0):.3f}  "
              f"macro-F1={m.get('macro_f1',0):.3f}  n={m.get('n',0)}")
        if "baseline_accuracy" in m:
            print(f"    vs keyword baseline:  acc={m['baseline_accuracy']:.3f}  "
                  f"macro-F1={m.get('baseline_macro_f1',0):.3f}")

    # ── Write metrics markdown ─────────────────────────────────────────────
    _write_metrics_md(all_metrics)
    print(f"\nMetrics written to: {METRICS_MD}")


def _write_empty_metrics():
    content = """\
# LLM Extraction Metrics

> **Status:** Template not yet filled in.
> Run `eval/generate_labeling_template.py`, complete the `human_*` columns,
> then re-run `eval/evaluate_extraction.py`.

## Summary Table

| Field | Accuracy | Macro F1 | Baseline Acc | Baseline F1 | N |
|---|---|---|---|---|---|
| violation_category | — | — | — | — | — |
| severity_tier | — | — | — | — | — |
| root_cause_type | — | — | — | — | — |
| remediation_signal | — | — | — | — | — |
| repeat_flag | — | — | — | — | — |
| systemic_flag | — | — | — | — | — |
| patient_risk_flag | — | — | — | — | — |
"""
    METRICS_MD.write_text(content)


def _write_metrics_md(all_metrics: dict):
    from datetime import date
    today = date.today().isoformat()

    lines = [
        "# LLM Extraction Metrics",
        f"\n> Generated: {today}  |  Source: `eval/evaluate_extraction.py`\n",
        "## Summary Table\n",
        "| Field | Accuracy | Macro F1 | Baseline Acc | Baseline F1 | N |",
        "|---|---|---|---|---|---|",
    ]

    for field, m in all_metrics.items():
        if m.get("n", 0) == 0:
            lines.append(f"| {field} | no labels | — | — | — | 0 |")
            continue
        b_acc = f"{m['baseline_accuracy']:.3f}" if "baseline_accuracy" in m else "—"
        b_f1  = f"{m['baseline_macro_f1']:.3f}"  if "baseline_macro_f1" in m else "—"
        lines.append(
            f"| {field} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} "
            f"| {b_acc} | {b_f1} | {m['n']} |"
        )

    lines.append("\n## Per-Class Breakdown\n")

    for field, m in all_metrics.items():
        if m.get("n", 0) == 0:
            continue
        lines.append(f"### {field}\n")
        lines.append("| Class | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for pc in m.get("per_class", []):
            lines.append(
                f"| {pc['label']} | {pc['precision']:.3f} | {pc['recall']:.3f} "
                f"| {pc['f1']:.3f} | {pc['tp']} | {pc['fp']} | {pc['fn']} |"
            )
        lines.append("")

    lines.append("## Notes for Paper\n")
    lines.append(
        "- Evaluation set: ~40 stratified observations (High / Moderate / Low severity).\n"
        "- Baseline (`severity_tier_baseline`) is a deterministic keyword classifier "
        "(see `05_extract_signals_langgraph.py`, `keyword_severity_baseline()`).\n"
        "- Macro F1 is computed across all classes including rare ones; "
        "interpret with caution for classes with < 5 examples.\n"
        "- Root cause type (`Capital / Cultural / Mixed / Unclear`) is the "
        "paper's theoretical core variable — treat its F1 as the primary metric.\n"
    )

    METRICS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
