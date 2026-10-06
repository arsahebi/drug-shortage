"""
eval/code/04_score_models_vs_human.py

Three-way comparison on Abdul's 50 blind-labeled observations: how does
each model (Claude Sonnet 5, GPT-5-mini) score against the human label, not
just against each other? This is what actually explains a low cross-model
agreement number: it can mean "both models are genuinely uncertain here" or
it can mean "one model is simply wrong here and the other is not" --
03_score_cross_model_agreement.py alone cannot tell those apart.

Unlike eval/validation_data/483_observation_context_signals_sample50_gpt5mini_v2.csv
(dated 2026-08-05, predates the round-1 prompt fixes), this pulls both
models' predictions for the same 50 rows from the current, full, post-fix
canonical extraction files (step01_redica_..._v2.csv), so both sides of the
comparison reflect the validated v2 prompt.

severity_tier is scored with Major+Moderate collapsed (see
20260916_LLM_Extraction_Validation_Report.docx, Section 2): the raw 4-tier
confusion is concentrated at that boundary, not Critical-vs-Major, and this
is the standard used for severity_tier in analysis going forward.

Run from this folder:
  python 04_score_models_vs_human.py

Output:
  eval/results_and_notes/models_vs_human_agreement.csv
"""

from pathlib import Path
import pandas as pd

HERE      = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA      = EVAL_ROOT.parent

LABELS_XLSX = EVAL_ROOT / "sent_to_abdul" / "labeling_template_v2.xlsx"
CLAUDE_CSV  = DATA / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
GPT_CSV     = DATA / "step01_redica_483_obs_llm_signals_openai_v2.csv"
OUT_CSV     = EVAL_ROOT / "results_and_notes" / "models_vs_human_agreement.csv"

JOIN_KEY = ["fei", "insp_date", "obs_num"]

# model field -> human field
FIELD_MAP = {
    "violation_category": "human_violation_category",
    "severity_tier":       "human_severity_tier",
    "scope":                "human_scope",
    "root_cause_type":      "human_root_cause_type",
    "remediation_signal":   "human_remediation_signal",
}

_SEVERITY_COLLAPSE = {"Major": "Major_or_Moderate", "Moderate": "Major_or_Moderate"}


def _score(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    valid = a.notna() & b.notna()
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float((a[valid] == b[valid]).mean()), n


def main() -> None:
    xl = pd.ExcelFile(LABELS_XLSX)
    labels = xl.parse("Labeling")
    labels["key"] = (labels["fei"].astype(str) + "_" + labels["insp_date"].astype(str)
                      + "_" + labels["obs_num"].astype(str))

    claude = pd.read_csv(CLAUDE_CSV)
    gpt    = pd.read_csv(GPT_CSV)
    for df in (claude, gpt):
        df["key"] = (df["fei"].astype(str) + "_" + df["insp_date"].astype(str)
                      + "_" + df["obs_num"].astype(str))

    print(f"Human-labeled rows: {len(labels)}")

    rows = []
    for model_field, human_field in FIELD_MAP.items():
        j = labels[["key", human_field]].merge(
            claude[["key", model_field]].rename(columns={model_field: "claude"}), on="key", how="left"
        ).merge(
            gpt[["key", model_field]].rename(columns={model_field: "gpt"}), on="key", how="left"
        )
        h = j[human_field]
        c = j["claude"]
        g = j["gpt"]
        if model_field == "severity_tier":
            h = h.replace(_SEVERITY_COLLAPSE)
            c = c.replace(_SEVERITY_COLLAPSE)
            g = g.replace(_SEVERITY_COLLAPSE)
            label = "severity_tier (Major+Moderate collapsed)"
        else:
            label = model_field

        claude_acc, n_c = _score(c, h)
        gpt_acc, n_g    = _score(g, h)
        cg_acc, n_cg    = _score(c, g)
        rows.append({
            "field": label,
            "claude_vs_human": round(claude_acc, 4) if n_c else float("nan"),
            "n_claude": n_c,
            "gpt_vs_human": round(gpt_acc, 4) if n_g else float("nan"),
            "n_gpt": n_g,
            "claude_vs_gpt": round(cg_acc, 4) if n_cg else float("nan"),
        })

    result = pd.DataFrame(rows)
    print("\n" + result.to_string(index=False))
    print(
        "\nNote: remediation_signal is blank on most observations (both human and "
        "model); its n above reflects only rows where a human label exists, and is "
        "usually too small to draw a conclusion from."
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
